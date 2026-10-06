# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/", "height": 688, "referenced_widgets": ["cf34779931ff47f9b892abb7d2f03982", "c0293df362ad419aaa689c1ae6fef4d7", "547872bad41c4990a4e52a17345aa17c", "85a4ce1cfc1b43e9837b2fa1f9216fde", "c30671956ffa43ebb804d811e9b76eba", "383e8e82647242a3844705820aeeea84", "d9fbd6713e4b48e38806060edcb17b9e", "a4658d8c0cfa46ce8cfac87cf0b76acb", "0b4468cf9df84c9ca3f3b908e1fd845f", "bcdebeba869e46a7992af07c52d4e5ec", "9799e2ea97814e98b1275857cbe9b4bf", "21db6e7c0cce4a44af0f9c75f7cfd497", "cfb1896546ed4cbaafa9c4fab687a7dd", "3d5cf09bd40b45ad8105d5a37d9bbb1a", "f71333f0874642fe9db9846820deb93b", "a0edbf801db4430e9e85064d203a584e", "0994313935a64e4abfc941ee5fb80292", "1b08545953b541a88ac4a68847f49e71", "8fc6c49982b84641ba48c5fac273ae8f", "28949ed9ffa3492fb770a1aca2f4ffde", "914af23436bc4a57934393bc9e45baf1", "a9630d5d66f048f0a07d6ccbbf4ba8b5", "fcd728ea06ab4032b01aedeca6a09ec1", "debafda3ea5d4afeae87d8578a0abc19", "2bb4588018cd425e9e6048280b10af4f", "caf1dde9cb2e49ebaec2be4ffe7586ac", "ea9d87c05e544f0c97b04597e6e2cc17", "ad1e5c0c3b834fe08772dcfe0f716c30", "29eaea50289f4517aeb64cb6f5af8ccc", "b0c396464eff41838a282e87f9d86db6", "d1c10d05fa414c1d93b67c6b4ff9ed98", "c1e12f38a3f1400a942ce691aab18f6a", "7824e69211d14ddb8261747e08660209", "ebd066d4a81240dbac8054596f6891de", "933004f1749a4986b375bde3de45eb2d", "8fa389e6c3e341ccb878a25402d8ce8e", "b8529919515740cf82e489dba44ce651", "5e909f74a08042e587747d70b6c8275c", "d2461de390ad4c55a0d05a8ca24015f9", "9c6d01ef7f4c4fddba3605c2d0861e85", "76a5153cacb84279ab724546a5fca4d9", "5c429d23124a45348d953842cc729988", "d38676821e6d427ea05dbe4e72bddabf", "e7d633c4289b418ebdb2ce3f902faf50", "e01523ca74b04a9298163e7faa41ed49", "c03e0f3559ef4696a4b70efdcc1a56b1", "8b81e1fe3f5a4c0bbf50c692f53cfadf", "8a7acdc8108044f1985023135c66eb8b", "b05f3cfbc30f454c96f77bf73f13ba98", "aa4e5ff5090340df9601fa8c6f0e1d3a", "3241426c93104d64adbf630c70ab1075", "6e759be594094f36a25f53131d2b4366", "f0338d002b9a4953b7fa1bbe58ccb5ee", "9c4fb46f5ebc4b2bb18f9264e3207cda", "361b0f7ba8244389ac1959fdc0e395bd", "ea46310d18da4a0dbb5f1cff21484fa5", "66351a99f4e9451bb944ec68bc6a2fcf", "4334bc4b7df84de1a068eb1661b8a1a0", "dff71da8379c43af9d3349311ab091ab", "ca6a0d3d73474dfc994b21c9fd443606", "1ea521af53224af69acb55e0f0829278", "2d1feaa469284a11bbce5ce388bf61c6", "b40ca992233f4e34a04857fa0e0431be", "1c1bb846ecec49f08e1d6d317bc37058", "22133ade2793455cb660c20bc87d94b2", "41719528475748d1abb473e8b948cfad", "d0609682bd604f2d97334fb16254ecca", "4df8d88bc87442afae66c17014876fe5", "9a99cf22573b40faa0192a8aba9beb23", "f2fc99dc88594dcebd672b9518f9dbc8", "d03c7f633d9346a0b0ed0dfa7f36175e", "8122c64fc8ae4fc6b49482d6eb87a4cd", "7485bffb83974468bce26937fd460155", "df6d2772a6374881a193ac37d5424373", "9d70973a36ed400aac59187c0a33a01f", "a0a412b1fd8f49f393afa56b33d9db71", "07d439f6ea8940b88bca61936b25cab1", "0485b09670274b3a85f7a98a8e8cd178", "0199f39f8e814aeebd3cfff9f6a69072", "1ec30f239ed1412cac05fde8b50e7bf9", "e6ff7b5f8f17404fb627225abe9aa4a5", "71f9e8edf224441ca92f97b74a93840d", "0b87782c277c4617a812f27ca114c8ef", "33962866fafb4f40a12af09294440a44", "ff25cc4f15394415aa96b97fb5c6218c", "b7e6bedeee9440e285bed0fece4f06cb", "22167c3946f943c78e7cfc92421b916f", "6166526ac2b5453d82c0aa2e4d7146fb", "fd3e2e5beb6e4a3a98e6364c2f23448b", "b1c30aec098a457f96a74e71a7388b28", "ed55db17512f4132adec70fb93c5d4d6", "198d7892e90e4a72bfd1ce84c29d48ac", "907fa2f07d1945d58bc1a89c2a89cb50", "2038a1f924ea4400aff6d615b70b567b", "d6b05a575cec420fb5fe5d44976fb195", "8d371265954b46c58261c7409a8c78a7", "1f77daf9d89448f99378f98b29906b98", "f5f42c6fd61d4650a133ae34fd93d5ec", "4b35339fcae048f5a5eb6e447aff9a6c", "a72198d4242841d4a81bf57b529d6bbe", "c1e28bea51a64666bbea052828b3a7c1", "37d2ee79ce3e482aa0c33f375f84d5cc", "3d2218209a374138bf336ec09c505742", "29ca8d76385a433b9748bb66d61c0901", "2077d5e98bd44c228f59388c0dcdcf3b", "69586107a1304bfc87d3e47f564226db", "d3746b278f6243f781eb53a6a1735174", "bfbac8aa92264c88ab7a896ee15930e2", "2f65ed876ddd4e6593a3a75bf6cd9e4e", "673b9551e03f40e0a5783526b4646116", "98479fb194bc4eb69a61d1f51ef62666", "df30f4ca7b6d4c95b4669b486f44b43c", "29967d6cd65f4e0eaec6dfe8ff0341ce", "5d58ec0cc1f74705b047799822544d41", "68ed33fb2f344baeba5bbff9a7cbb994", "2f0ab77d926247e4bfa1e97e1ac35bfa", "5598ad4bdc1241578570002683b3547a", "746a67a2d5ad4f93a0335ff450fe05ea", "d47c1faf42c24656918a10a226c7bd14", "1d4813a896b9423aae277abb254e809d", "52292450dcce4aa8a5b6f73eba0ebbab", "cf035375c15c43fb9e2d22c253cc11e4", "af42fc2220f94a9b9087f27af2073383", "896410725e184626804e0aeaa3a8101a", "dbbc0e2380984dd981c12414315f8f49", "5333df280b36479095040a702d2aac26", "0bc14d12674d44daa68d6f8959ada05d", "e61a7ede9631493fb5b63d5e755d3b23", "389c2a6d6e9f4e84af3b660ef707a5ad", "7a53ef5fbb2e49a38048e44f349a884b", "acda0a082e4c439a9714c819ef82061c", "a3212641802e4cb08c14b93d0c44e408", "c9a19b24c5cb41f09ece0e12bd19f1b9", "ce846fb834634f90af6cf083892cacae", "6cbf18eb7c6149b3959da481db1b0e7e", "5de44dc1435448de991f0128a7035436", "93fbfe0bcabe4271a6035f03243262eb", "d6b1e1f15e1845ca945d1fa5a6f7c9e3", "febe379673534196b74a609478a09db5", "83dc17ce59df43338b0ee743ccbc5a95", "30a4078825024d6490a13296c4c4909b", "e1161add46244f09a21ad3c96203cf7e", "9fe3e23841374f03a5dbeadd4f9e59f3", "6199f1d861084288aae338c3486ab4a4", "ca61fe7216744f66a2a78f95f67a1e04", "43ab229887754d1383bb41697eea5330", "fd84e93d17824e6fbcf9189c815402b3", "222bf4d7ba114214978947eae8c13536", "a684dbb4cf304f3d9f5096fd5e27d74e", "4293dbf7ae0b4d268e00da1eabb9c379", "f96fd73496b641ee9012dd8c8abefa1e", "46d7c6d1dbcf4c7baab3f8682ab7da79", "5d95f5e3ee314cadbd0999fda6733d07", "f1cee7b881ac43c1a86cbb8d7d360f20", "7ca0a951705049068f024f2685fb612d", "0bb24cff0b5a449c811300d90eb053cb", "b94b17d04d584acf9395a3210f231b28", "5b1f275621ac41e5ba71ab7e647498fa", "dc86c24289fc40d3b704a7b2ef8af802", "af3b8b45d2a144328e8e959b457e4b17", "e1e12929f96b47af86b0451a26cf4a7a", "701d7cd5372a4347a6f74878578e1fd0", "2d7b2b0bb0f04753b4619052ae7e98fb", "4e1aa26e26744afeb3b134f7d146708e", "81e12689666f4323a6278607ea95812f", "4126bb0df9774caa916d6c587e5f737f", "888333a5128a4303a2b560672a2ade6d", "d61c1363ee754d1aa86d927b9ae72ab1", "9c288413bb49459880ce8742fd668f00", "aa639fa56f7b42c7a9cd2fdac9976b39", "d5914e470db444db9a09c937f1de2add", "ea5794bc53814be882d00c57b9aa2089", "3dc1f4b183f94a4eac5aad4c5f2f44f2", "6d80bad6843546b88445eebd1cae2b1e", "8f921551938e43438194ace7620750ee", "f741a99621444d2ca9b413a1e5dbe305"]} executionInfo={"elapsed": 126808, "status": "ok", "timestamp": 1791267189890, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="EWgkb1Hf1Srb" outputId="3cf83b11-aeac-4d4a-c037-a05ab538335b"
# !pip -q install -U transformers accelerate safetensors huggingface_hub statsmodels
from google.colab import drive, userdata
drive.mount("/content/drive")
import sys; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
setup_dirs(); seed_all()

# make sure Gemma always loads in fp32 with the new transformers version
sp = REPO / "spectra" / "sae.py"; t = sp.read_text()
t = t.replace("torch_dtype=torch.float32", "dtype=torch.float32")
t = t.replace('attn_implementation="eager").to(device).eval()', 'attn_implementation="eager").float().to(device).eval()')
sp.write_text(t)

import torch, time, numpy as np
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN").strip())
from spectra.sae import GemmaReader, load_sae, extract, load_acts
train, val, test, shift = (load_split(s) for s in ["train", "val", "test", "shift"])
LAYERS = CFG["sae_layers"]

reader = GemmaReader(CFG["gemma"], LAYERS, max_len=CFG["max_len"])
assert reader.n_layers == 26 and reader.d_model == 1152, (reader.n_layers, reader.d_model)
assert next(reader.model.parameters()).dtype == torch.float32
saes = {L: load_sae(L, CFG["sae_width"], CFG["sae_l0"], CFG["sae_repo"], reader.d_model)[0] for L in LAYERS}
print(round(torch.cuda.memory_allocated() / 1e9, 2), "GB on GPU")

# --- sanity check ---
texts = test.narrative.sample(32, random_state=SEED).tolist()
t0 = time.time(); cache, mask, _ = reader.run(texts); dt = time.time() - t0
m = mask.bool()
for L in LAYERS:
    h = cache[L].float(); f = saes[L].encode(h); rec = saes[L].decode(f)
    hv, rv = h[m], rec[m]
    fve = 1 - ((hv - rv) ** 2).sum() / ((hv - hv.mean(0)) ** 2).sum()
    l0 = (f[m] > 0).float().sum(-1).mean()
    print(f"L{L}: variance explained={fve:.3f} | mean L0={l0:.1f} | NaN={torch.isnan(h).any().item()}")
n_total = len(train) + len(val) + len(test) + len(shift) + len(test) + CFG["masked_train_n"]
print(f"one batch: {dt:.2f}s -> full extraction estimate ~{dt * n_total / 32 / 60:.0f} min")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2709301, "status": "ok", "timestamp": 1791269955530, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="IVnfcgdr1loq" outputId="3d5e5283-be7c-4524-b625-b5d03d2c6e3e"
for name, d in [("test", test), ("shift", shift), ("val", val), ("train", train)]:
    extract(reader, saes, d.narrative.tolist(), ACTS / f"raw_{name}", LAYERS, batch=CFG["batch"])
print("raw extraction done")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 790961, "status": "ok", "timestamp": 1791272276165, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="JHaMq1J92RAu" outputId="f9005b80-bdd8-464e-d650-1ca09d7225d7"
n = CFG["masked_train_n"]
extract(reader, saes, test.narrative_masked.tolist(), ACTS / "masked_test", [13], batch=CFG["batch"])
extract(reader, saes, train.narrative_masked.tolist()[:n], ACTS / "masked_train", [13], batch=CFG["batch"])
print("masked extraction done")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 65520, "status": "ok", "timestamp": 1791272384901, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="S_2y3n5oGb46" outputId="504ae65c-16b4-4ddb-da73-8ce12ae8b93a"
for name, d in [("train", train), ("val", val), ("test", test), ("shift", shift)]:
    ntok = load_acts(ACTS / f"raw_{name}", "ntok")
    assert len(ntok) == len(d), name
    print(name, len(d), "| truncated at max_len:", f"{(ntok >= CFG['max_len'] - 1).mean():.1%}")
for name, k in [("masked_test", len(test)), ("masked_train", CFG["masked_train_n"])]:
    assert len(load_acts(ACTS / name, "ntok")) == k, name
X = load_acts(ACTS / "raw_test", "saemax", 13)
print("test SAE matrix", X.shape, "| avg active features per narrative:", round(X.getnnz(1).mean(), 1))

# !cd /content/drive/MyDrive/spectra && python -m pytest -q tests

# %% id="B9NSGNrkJnkR"
