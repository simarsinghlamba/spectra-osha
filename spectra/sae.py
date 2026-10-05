"""Gemma 3 reader with layer hooks, Gemma Scope 2 JumpReLU SAE loader, activation extraction."""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

class JumpReLUSAE(torch.nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        super().__init__()
        for k, v in dict(W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec, threshold=threshold).items():
            self.register_buffer(k, v.float())
    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold)
    def encode_one(self, x, i):
        pre = x @ self.W_enc[:, i] + self.b_enc[i]
        return torch.relu(pre) * (pre > self.threshold[i])
    def decode(self, f):
        return f @ self.W_dec + self.b_dec

def _pick(p, names):
    for n in names:
        if n in p:
            return p[n]
    raise KeyError(f"Expected one of {names}; file has {list(p)}")

def load_sae(layer, width, l0, repo, d_model, device="cuda"):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    folder = f"resid_post/layer_{layer}_width_{width}_l0_{l0}"
    cfg = json.loads(Path(hf_hub_download(repo, f"{folder}/config.json")).read_text())
    p = {k.lower(): v for k, v in load_file(hf_hub_download(repo, f"{folder}/params.safetensors")).items()}
    print(folder, "| config:", cfg, "| tensors:", {k: tuple(v.shape) for k, v in p.items()})
    W_enc, W_dec = _pick(p, ["w_enc", "encoder.weight"]), _pick(p, ["w_dec", "decoder.weight"])
    if W_enc.shape[0] != d_model: W_enc = W_enc.T
    if W_dec.shape[1] != d_model: W_dec = W_dec.T
    b_enc, b_dec = _pick(p, ["b_enc", "encoder.bias"]), _pick(p, ["b_dec", "decoder.bias"])
    thr = p["threshold"] if "threshold" in p else _pick(p, ["log_threshold"]).exp()
    sae = JumpReLUSAE(W_enc, W_dec, b_enc, b_dec, thr).to(device).eval()
    assert tuple(sae.W_enc.shape) == (d_model, cfg["width"]), sae.W_enc.shape
    return sae, cfg

class Stop(Exception):
    pass

class GemmaReader:
    """Runs Gemma only up to the deepest layer needed; captures block outputs; optional edit at one layer."""
    def __init__(self, model_id, layers, max_len=160, device="cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="eager").to(device).eval()
        c = self.model.config
        self.n_layers, self.d_model = c.num_hidden_layers, c.hidden_size
        print(f"Gemma loaded: {self.n_layers} layers, hidden {self.d_model}, dtype {self.model.dtype}")
        self.blocks = self.model.model.layers
        self.layers, self.max_len, self.device = list(layers), max_len, device
        self.cache, self.stop_at, self.edit, self.mask = {}, max(layers), None, None
        for L in self.layers:
            self.blocks[L].register_forward_hook(self._hook(L))

    def _hook(self, L):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.edit is not None and self.edit[0] == L:
                h = self.edit[1](h)
                out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.cache[L] = h.detach()
            if L == self.stop_at:
                raise Stop()
            return out
        return fn

    @torch.no_grad()
    def run(self, texts, stop_at=None):
        enc = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_len).to(self.device)
        assert (enc["input_ids"][:, 0] == self.tok.bos_token_id).all(), "expected BOS at position 0"
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0
        self.mask = mask.float()
        self.cache, self.stop_at = {}, (stop_at or max(self.layers))
        try:
            self.model(**enc, use_cache=False)
        except Stop:
            pass
        return self.cache, mask, enc["input_ids"]

def extract(reader, saes, texts, outdir, layers, batch=32, chunk=2048):
    """Saves per-narrative: dense mean-pooled residual + SAE max-pooled features (sparse). Resumable."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    stop = max(layers)
    for c0 in range(0, len(texts), chunk):
        if (outdir / f"part_{c0:07d}.done").exists():
            continue
        dense = {L: [] for L in layers}; smax = {L: [] for L in layers}; ntok = []
        for b0 in range(c0, min(c0 + chunk, len(texts)), batch):
            cache, mask, _ = reader.run(texts[b0:b0 + batch], stop_at=stop)
            m = mask.unsqueeze(-1).float(); denom = m.sum(1).clamp(min=1)
            ntok.append(mask.sum(1).cpu().numpy())
            for L in layers:
                h = cache[L].float()
                dense[L].append(((h * m).sum(1) / denom).cpu().numpy())
                f = saes[L].encode(h) * m
                smax[L].append(sp.csr_matrix(f.max(1).values.cpu().numpy()))
        for L in layers:
            np.save(outdir / f"dense_L{L}_{c0:07d}.npy", np.concatenate(dense[L]).astype(np.float32))
            sp.save_npz(outdir / f"saemax_L{L}_{c0:07d}.npz", sp.vstack(smax[L]).tocsr())
        np.save(outdir / f"ntok_{c0:07d}.npy", np.concatenate(ntok))
        (outdir / f"part_{c0:07d}.done").touch()
        print(f"{outdir.name}: {min(c0 + chunk, len(texts))}/{len(texts)}")

def load_acts(outdir, kind, layer=None):
    """kind: 'dense' | 'saemax' | 'ntok'."""
    outdir = Path(outdir)
    pat = f"{kind}_L{layer}_*" if layer is not None else f"{kind}_*"
    files = sorted(outdir.glob(pat + (".npz" if kind == "saemax" else ".npy")))
    assert files, f"no {pat} files in {outdir}"
    if kind == "saemax":
        return sp.vstack([sp.load_npz(f) for f in files]).tocsr()
    return np.concatenate([np.load(f) for f in files])
