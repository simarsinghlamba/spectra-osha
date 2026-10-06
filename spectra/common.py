"""Shared config, paths and helpers for SPECTRA. Imported by every notebook."""
import json, random, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

GH_USER    = "simarsinghlamba"
GH_REPO    = "spectra-osha"
GIT_NAME   = "Simar Singh Lamba"
GIT_EMAIL  = "257922671+simarsinghlamba@users.noreply.github.com "
HF_USER    = "Simar123456"
HF_DATASET = "spectra-osha-sir"
HF_MODEL   = "spectra-deberta-oiics"
HF_SPACE   = "spectra-demo"

DRIVE  = Path("/content/drive/MyDrive")
REPO   = DRIVE / "spectra"
WORK   = DRIVE / "spectra_work"
RAW, DATA, ACTS, MODELS, PREDS, PUB = (WORK / d for d in ["raw", "data", "acts", "models", "preds", "publish"])
TABLES = REPO / "results" / "tables"
FIGS   = REPO / "results" / "figures"
SEED   = 42

CFG = dict(
    sir_zip_url="https://www.osha.gov/sites/default/files/January2015toNovember2025.zip",
    min_words=15, min_class_count=200,
    train_sample=40000, val_sample=5000, masked_train_n=20000,
    gemma="google/gemma-3-1b-pt", sae_repo="google/gemma-scope-2-1b-pt",
    sae_layers=[13, 17], sae_main_layer=13, sae_width="16k", sae_l0="medium",
    max_len=160, batch=32, deberta="microsoft/deberta-v3-base", n_random_ablation=100,
)

def setup_dirs():
    for p in [RAW, DATA, ACTS, MODELS, PREDS, PUB, TABLES, FIGS]:
        p.mkdir(parents=True, exist_ok=True)

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def save_table(df, name):
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    print("saved results/tables/" + path.name)
    return path

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print("saved", path)

def load_json(path):
    return json.loads(Path(path).read_text())

def load_split(name):
    return pd.read_parquet(DATA / f"{name}.parquet")

def save_preds(model, task, split, ids, y, pred, proba=None):
    df = pd.DataFrame({"id": np.asarray(ids), "y": np.asarray(y), "pred": np.asarray(pred)})
    if proba is not None:
        proba = np.asarray(proba, dtype=np.float32)
        df["proba"] = list(proba) if proba.ndim == 2 else proba
    df.to_parquet(PREDS / f"{model}__{task}__{split}.parquet", index=False)

def load_preds(model, task, split):
    return pd.read_parquet(PREDS / f"{model}__{task}__{split}.parquet")

def sh(cmd, secret=None):
    """Run a shell command; hide `secret` in the printed output."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if secret:
        out = out.replace(secret, "***").replace(secret.strip(), "***")
    if out:
        print(out)
    return r.returncode

def commit(msg):
    """Local commit only (no network, no token)."""
    sh(f'cd "{REPO}" && git add -A && git commit -q -m "{msg}"; git -C "{REPO}" log --oneline -3')
