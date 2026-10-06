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

# %% id="LmqHaGzoomJC"
# !pip -q install datasketch statsmodels pytest
from google.colab import drive
drive.mount("/content/drive")
import sys; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
setup_dirs(); seed_all()

# one-time: add the class->division helper to stats.py (used here and in NB05/NB07)
sp_path = REPO / "spectra" / "stats.py"
if "def div_from_proba" not in sp_path.read_text():
    with open(sp_path, "a") as fh:
        fh.write(r'''

def div_from_proba(P, classes, class_div, divs):
    """Sum 2-digit class probabilities into divisions ('other' ignored, then renormalised)."""
    P = np.asarray(P, dtype=float)
    M = np.zeros((len(classes), len(divs)))
    for i, c in enumerate(classes):
        if class_div.get(c) is not None:
            M[i, divs.index(class_div[c])] = 1.0
    Q = P @ M
    Q = Q / np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return Q.argmax(1), Q
''')
    print("added div_from_proba to stats.py")

from spectra.stats import *
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
train, val, test, shift = (load_split(s) for s in ["train", "val", "test", "shift"])
lm = load_json(DATA / "label_map.json")
classes, divs, class_div = lm["classes"], lm["divisions"], lm["class_div"]
print({k: len(v) for k, v in dict(train=train, val=val, test=test, shift=shift).items()}, "|", len(classes), "classes,", len(divs), "divisions")


# %% id="1XhETZMvotf3"
def rep(name, split, task, y, p):
    r = clf_report(name, split, y, p); r["task"] = task; return r

def div_eval(name, key, split, d, P):
    pdiv, Q = div_from_proba(P, classes, class_div, divs)
    save_preds(key, "div", split, d.id, d.div_id, pdiv, Q)
    return rep(name, split, "division", d.div_id.values, pdiv)

rows = []
# --- B0 majority ---
maj, majd = train.label_id.mode().iat[0], train.div_id.mode().iat[0]
p = np.full(len(test), maj); save_preds("B0", "event", "test", test.id, test.label_id, p)
rows.append(rep("B0 majority", "test", "event", test.label_id.values, p))
for split, d in [("test", test), ("shift", shift)]:
    p = np.full(len(d), majd); save_preds("B0", "div", split, d.id, d.div_id, p)
    rows.append(rep("B0 majority", split, "division", d.div_id.values, p))

# --- B1 TF-IDF + LR (tuned on val macro-F1) ---
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True)
Xtr = vec.fit_transform(train.narrative)
Xva, Xte, Xsh = (vec.transform(d.narrative) for d in (val, test, shift))
best = None
for C in [1.0, 4.0, 16.0]:
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(Xtr, train.label_id)
    s = macro_f1(val.label_id, clf.predict(Xva)); print(f"C={C}: val macro-F1 {s:.4f}")
    if best is None or s > best[0]: best = (s, C, clf)
b1 = best[2]; assert list(b1.classes_) == list(range(len(classes)))
for split, X, d in [("test", Xte, test), ("shift", Xsh, shift)]:
    P = b1.predict_proba(X); p = P.argmax(1)
    save_preds("B1", "event", split, d.id, d.label_id, p, P)
    if split == "test":
        rows.append(rep("B1 TF-IDF+LR", split, "event", d.label_id.values, p))
    rows.append(div_eval("B1 TF-IDF+LR", "B1", split, d, P))
res = pd.DataFrame(rows)[["model", "task", "split", "n", "macro_f1", "macro_f1_lo", "macro_f1_hi", "weighted_f1", "accuracy"]]
save_table(res, "baselines_event")
print(res.round(3).to_string(index=False))

# --- top TF-IDF terms per class ---
names = np.array(vec.get_feature_names_out())
top = pd.DataFrame([{"class": classes[k], "title": lm["titles"][classes[k]],
                     "top_terms": ", ".join(names[np.argsort(-b1.coef_[k])[:12]])} for k in range(len(classes))])
save_table(top, "tfidf_top_terms")

# --- amputation: raw vs masked (cheap shortcut test) ---
rows_amp = []
for variant, col in [("raw", "narrative"), ("masked", "narrative_masked")]:
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True)
    m = LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced").fit(v.fit_transform(train[col]), train.amputation_bin)
    for split, d in [("test", test), ("shift", shift)]:
        s = m.predict_proba(v.transform(d[col]))[:, 1]
        save_preds(f"B1_{variant}", "amputation", split, d.id, d.amputation_bin, (s > 0.5).astype(int), s)
        rows_amp.append(bin_report(f"B1 TF-IDF ({variant})", split, d.amputation_bin.values, s))
    vn = np.array(v.get_feature_names_out())
    print(f"\nTop {variant} amputation terms:", ", ".join(vn[np.argsort(-m.coef_[0])[:15]]))
amp = pd.DataFrame(rows_amp); save_table(amp, "baselines_amputation_raw_vs_masked")
print(amp.round(3).to_string(index=False))

# %% id="heJk5NiBo8fg"
from google.colab import userdata

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 63815, "status": "ok", "timestamp": 1791264804178, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="Pf3utgJ2sTor" outputId="3c323a66-c857-4979-f9a3-965326312a2c"
from google.colab import drive, userdata
drive.mount("/content/drive")
import sys; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
from spectra.stats import *
import numpy as np, pandas as pd, importlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1) mask the outcome word together with its surrounding grammar ("was ...", "an ... of", "partial ...")
dp = REPO / "spectra" / "data.py"
if "MASK_RE" not in dp.read_text():
    with open(dp, "a") as fh:
        fh.write(r'''

# Stronger masking: also remove the grammar around an outcome word, so no "the of" holes are left.
_PRE = r"(?:\b(?:was|were|been|be|is|are|had|has|have)\s+)?(?:\b(?:an?|the|his|her|their|partial(?:ly)?|traumatic|surgical(?:ly)?|complete(?:ly)?)\s+)*"
_POST = r"(?:\s+(?:of|to|at|off)\b)?"
MASK_RE = re.compile(_PRE + "(?:" + "|".join(LEAK_PATTERNS) + ")" + _POST, flags=re.IGNORECASE)

def mask_text(t):
    return re.sub(r"\s+", " ", MASK_RE.sub(" ", t)).strip()
''')
import spectra.data as sd; importlib.reload(sd)

# 2) re-mask all saved splits
for name in ["train", "val", "test", "shift"]:
    p = DATA / f"{name}.parquet"; x = pd.read_parquet(p)
    x["narrative_masked"] = x.narrative.map(sd.mask_text); x.to_parquet(p, index=False)
train, test, shift = load_split("train"), load_split("test"), load_split("shift")
ex = train[train.amputation_bin == 1].head(3)
for a, b in zip(ex.narrative, ex.narrative_masked): print("RAW:", a[:160], "\nMSK:", b[:160], "\n")

# 3) rerun the raw vs masked amputation test
rows_amp = []
for variant, col in [("raw", "narrative"), ("masked", "narrative_masked")]:
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True)
    m = LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced").fit(v.fit_transform(train[col]), train.amputation_bin)
    for split, d in [("test", test), ("shift", shift)]:
        s = m.predict_proba(v.transform(d[col]))[:, 1]
        save_preds(f"B1_{variant}", "amputation", split, d.id, d.amputation_bin, (s > 0.5).astype(int), s)
        rows_amp.append(bin_report(f"B1 TF-IDF ({variant})", split, d.amputation_bin.values, s))
    vn = np.array(v.get_feature_names_out())
    print(f"Top {variant} terms:", ", ".join(vn[np.argsort(-m.coef_[0])[:20]]), "\n")
amp = pd.DataFrame(rows_amp); save_table(amp, "baselines_amputation_raw_vs_masked")
print(amp.round(3).to_string(index=False))

# !cd /content/drive/MyDrive/spectra && python -m pytest -q tests

# %% id="gvg8vDxKtT8L"
