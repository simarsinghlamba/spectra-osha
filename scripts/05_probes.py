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

# %% colab={"base_uri": "https://localhost:8080/"} id="iE4fEjViKV-U" executionInfo={"status": "ok", "timestamp": 1791272589503, "user_tz": -330, "elapsed": 42060, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}} outputId="40581470-cc2a-489b-a2db-2556a2f0b828"
# !pip -q install statsmodels
from google.colab import drive, userdata
drive.mount("/content/drive")
import sys; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
from spectra.stats import *
from spectra.sae import load_acts
setup_dirs(); seed_all()
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline

train, val, test, shift = (load_split(s) for s in ["train", "val", "test", "shift"])
ytr, yva, yte = train.label_id.values, val.label_id.values, test.label_id.values
lm = load_json(DATA / "label_map.json")
classes, divs, class_div = lm["classes"], lm["divisions"], lm["class_div"]
A = lambda tag, kind, L: load_acts(ACTS / tag, kind, L)

def tune(make, Xtr, ytr, Xva, yva, Cs):
    best = None
    for C in Cs:
        m = make(C).fit(Xtr, ytr); s = macro_f1(yva, m.predict(Xva)); print(f"  C={C}: val macro-F1 {s:.4f}")
        if best is None or s > best[0]: best = (s, C, m)
    return best[2]

def rep(name, split, task, y, p):
    r = clf_report(name, split, y, p); r["task"] = task; return r
print("ready")

# %% colab={"base_uri": "https://localhost:8080/"} id="dhQMrklOKfMh" executionInfo={"status": "ok", "timestamp": 1791276953892, "user_tz": -330, "elapsed": 4355162, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}} outputId="11a5b143-19a0-4331-d4e4-7b3af3f27838"
rows = []
for L in CFG["sae_layers"]:
    for kind, scaler in [("dense", StandardScaler), ("saemax", MaxAbsScaler)]:
        print(f"Layer {L} {kind}")
        Xtr, Xva, Xte, Xsh = (A(f"raw_{s}", kind, L) for s in ["train", "val", "test", "shift"])
        probe = tune(lambda C: make_pipeline(scaler(), LogisticRegression(C=C, max_iter=2000)),
                     Xtr, ytr, Xva, yva, [0.01, 0.1, 1.0])
        assert list(probe.classes_) == list(range(len(classes)))
        joblib.dump(probe, MODELS / f"probe_{kind}_L{L}_event.joblib")
        name, key = f"P-{'dense' if kind == 'dense' else 'SAE'} L{L}", f"{kind}_L{L}"
        for split, X, d in [("test", Xte, test), ("shift", Xsh, shift)]:
            P = probe.predict_proba(X); p = P.argmax(1)
            save_preds(key, "event", split, d.id, d.label_id, p, P)
            if split == "test":
                rows.append(rep(name, split, "event", d.label_id.values, p))
            pdiv, Q = div_from_proba(P, classes, class_div, divs)
            save_preds(key, "div", split, d.id, d.div_id, pdiv, Q)
            rows.append(rep(name, split, "division", d.div_id.values, pdiv))
res = pd.DataFrame(rows)[["model", "task", "split", "n", "macro_f1", "macro_f1_lo", "macro_f1_hi", "weighted_f1", "accuracy"]]
save_table(res, "probes_event")
print(res.round(3).to_string(index=False))

# %% colab={"base_uri": "https://localhost:8080/", "height": 669} id="GrRGZYqbKruJ" executionInfo={"status": "ok", "timestamp": 1791277875381, "user_tz": -330, "elapsed": 843991, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}} outputId="dbdef5f5-a656-40d3-f199-ac68e9e2ab46"
L = CFG["sae_main_layer"]
sc = MaxAbsScaler().fit(A("raw_train", "saemax", L))
Str, Sva, Ste = (sc.transform(A(f"raw_{s}", "saemax", L)) for s in ["train", "val", "test"])
def topk_by_class(X, y, k):
    feats = set()
    for c in np.unique(y):
        diff = np.asarray(X[y == c].mean(0)).ravel() - np.asarray(X[y != c].mean(0)).ravel()
        feats.update(np.argsort(-diff)[:k].tolist())
    return np.array(sorted(feats))
curve = []
for k in [1, 5, 20, 100]:
    idx = topk_by_class(Str, ytr, k)
    m = tune(lambda C: LogisticRegression(C=C, max_iter=2000), Str[:, idx], ytr, Sva[:, idx], yva, [0.1, 1.0, 10.0])
    r = clf_report(f"SAE top-{k}/class", "test", yte, m.predict(Ste[:, idx])); r.update(k=k, n_features=len(idx)); curve.append(r)
curve = pd.DataFrame(curve); save_table(curve, "ksparse_curve")
ref = res.query("split=='test' and task=='event'").set_index("model").macro_f1
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(curve.n_features, curve.macro_f1, "o-", label="SAE top-k per class", color="#c2410c")
ax.axhline(ref[f"P-dense L{L}"], ls="--", color="#4a6fa5", label="dense probe (all 1152 dims)")
ax.axhline(ref[f"P-SAE L{L}"], ls=":", color="#555", label="SAE probe (all 16k features)")
ax.set_xscale("log"); ax.set_xlabel("number of SAE features used"); ax.set_ylabel("test macro-F1"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "ksparse_curve.png", dpi=150)
print(curve[["model", "k", "n_features", "macro_f1", "macro_f1_lo", "macro_f1_hi"]].round(3).to_string(index=False))

# %% colab={"base_uri": "https://localhost:8080/"} id="jbFmG6t1bl6A" executionInfo={"status": "ok", "timestamp": 1791277996313, "user_tz": -330, "elapsed": 99311, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}} outputId="8881d280-7d99-436a-cdd5-6140370638ad"
n = CFG["masked_train_n"]
ya_tr, ya_te = train.amputation_bin.values[:n], test.amputation_bin.values
rows_amp = []
for variant in ["raw", "masked"]:
    for kind, scaler in [("dense", StandardScaler), ("saemax", MaxAbsScaler)]:
        Xtr, Xte = A(f"{variant}_train", kind, 13)[:n], A(f"{variant}_test", kind, 13)
        pipe = make_pipeline(scaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")).fit(Xtr, ya_tr)
        s = pipe.predict_proba(Xte)[:, 1]
        save_preds(f"{kind}_L13_{variant}", "amputation", "test", test.id, ya_te, (s > 0.5).astype(int), s)
        rows_amp.append(bin_report(f"P-{'dense' if kind == 'dense' else 'SAE'} L13 ({variant})", "test", ya_te, s))
        if kind == "saemax":
            joblib.dump(pipe, MODELS / f"probe_saemax_L13_amp_{variant}.joblib")
amp = pd.DataFrame(rows_amp); save_table(amp, "probes_amputation_raw_vs_masked")
print(amp.round(3).to_string(index=False))

# %% colab={"base_uri": "https://localhost:8080/", "height": 349} id="gGnRdc7qe5Pd" executionInfo={"status": "error", "timestamp": 1791278660691, "user_tz": -330, "elapsed": 28782, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}} outputId="5665a0d8-6ac6-4b6c-f8fb-301cd26175d8"
commit("Probes: dense vs SAE (L13, L17), k-sparse curve, amputation raw vs masked in Gemma")
gh = userdata.get("GH_TOKEN").strip()
sh(f'cd "{REPO}" && git push https://{GH_USER}:{gh}@github.com/{GH_USER}/{GH_REPO}.git main', secret=gh)

# %% colab={"base_uri": "https://localhost:8080/"} id="7RpcSs4yhsqc" executionInfo={"status": "ok", "timestamp": 1791280548230, "user_tz": -330, "elapsed": 5724, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}} outputId="54de96c4-1588-4515-b90d-74bcf9b543c4"
gh = userdata.get("GH_TOKEN").strip()
sh(f'cd "{REPO}" && git push https://{GH_USER}:{gh}@github.com/{GH_USER}/{GH_REPO}.git main', secret=gh)

# %% id="3hRtd1nFo_Hc"
