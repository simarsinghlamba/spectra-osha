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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 43944, "status": "ok", "timestamp": 1791287742643, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="gI4FV9GTEKmi" outputId="57a57256-4de7-4000-ecc8-8e9e8e363d2f"
# !pip -q install statsmodels
from google.colab import drive, userdata
drive.mount("/content/drive")
import sys; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
from spectra.stats import *
from spectra.sae import load_acts
setup_dirs(); seed_all()
import numpy as np, pandas as pd, scipy.sparse as sp, matplotlib.pyplot as plt
from scipy.stats import spearmanr
train, val, test, shift = (load_split(s) for s in ["train", "val", "test", "shift"])
lm = load_json(DATA / "label_map.json"); classes, divs = lm["classes"], lm["divisions"]

# cosmetic fix: class titles from pre-2024 (OIICS 2) rows only
pre_all = pd.concat([train, val, test])
for c in classes[:-1]:
    g = pre_all[pre_all.event2 == c]; exact = g[g.event_full == c]
    lm["titles"][c] = exact.event_title.mode().iat[0].strip() if len(exact) else "e.g. " + g.event_title.mode().iat[0].strip()
save_json(lm, DATA / "label_map.json"); titles = lm["titles"]
print({c: titles[c] for c in classes})

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 101875, "status": "ok", "timestamp": 1791287863841, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="wMkeYm5WESPS" outputId="245cff98-df77-48a0-8552-19d9fecab7cb"
NAMES = {"B1": "B1 TF-IDF+LR", "M1": "M1 DeBERTa-v3", "dense_L13": "P-dense L13", "saemax_L13": "P-SAE L13",
         "dense_L17": "P-dense L17", "saemax_L17": "P-SAE L17"}
rows = []
for key, name in NAMES.items():
    d = load_preds(key, "event", "test"); r = clf_report(name, "test", d.y.values, d.pred.values); r["task"] = "event"; rows.append(r)
    for split in ["test", "shift"]:
        d = load_preds(key, "div", split); r = clf_report(name, split, d.y.values, d.pred.values); r["task"] = "division"; rows.append(r)
main = pd.DataFrame(rows)[["model", "task", "split", "n", "macro_f1", "macro_f1_lo", "macro_f1_hi", "weighted_f1", "accuracy"]]
save_table(main, "main_results_event")
wide = main[main.task == "division"].pivot(index="model", columns="split", values="macro_f1")
wide["drop_test_to_shift"] = wide["test"] - wide["shift"]
save_table(wide.reset_index(), "shift_drop")
print(main.round(3).to_string(index=False)); print(); print(wide.round(3).sort_values("test", ascending=False))

pairs = [("M1", "B1"), ("saemax_L13", "dense_L13"), ("M1", "saemax_L13"), ("saemax_L13", "B1")]
out = []
for a, b in pairs:
    da, db = load_preds(a, "event", "test"), load_preds(b, "event", "test")
    assert (da.id.values == db.id.values).all()
    out.append(dict(A=a, B=b, **mcnemar_test(da.y, da.pred, db.pred),
                    **{f"macroF1_{k}": v for k, v in paired_boot_diff(da.y, da.pred, db.pred).items()}))
paired = pd.DataFrame(out); save_table(paired, "paired_tests")
print(); print(paired.round(4).to_string(index=False))

# %% colab={"base_uri": "https://localhost:8080/", "height": 863} executionInfo={"elapsed": 127812, "status": "ok", "timestamp": 1791288007560, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="7XSPf-5uEhsG" outputId="8bc28795-d1fb-40b4-b3be-614eb92e8926"
L = CFG["sae_main_layer"]
X_pre = sp.vstack([load_acts(ACTS / "raw_train", "saemax", L), load_acts(ACTS / "raw_test", "saemax", L)]).tocsr()
pre = pd.concat([train, test]).reset_index(drop=True)
X_post = load_acts(ACTS / "raw_shift", "saemax", L)

def stability(level, col, groups, title_of):
    res = []
    for g in groups:
        ypre, ypost = (pre[col].values == g), (shift[col].values == g)
        if ypre.sum() < 50 or ypost.sum() < 50: continue
        r_pre, _ = point_biserial(X_pre, ypre); r_post, _ = point_biserial(X_post, ypost)
        t_pre, t_post = np.argsort(-r_pre)[:50], np.argsort(-r_post)[:50]
        res.append(dict(level=level, group=g, title=title_of(g), n_pre=int(ypre.sum()), n_post=int(ypost.sum()),
                        overlap_top50=len(set(t_pre) & set(t_post)) / 50,
                        spearman_top50=float(spearmanr(r_pre[t_pre], r_post[t_pre]).correlation)))
    return res
stab = pd.DataFrame(
    stability("division", "y_div", divs, lambda g: lm["div_titles"][g]) +
    stability("2-digit code", "event2", [c for c in classes if c != "other"], lambda g: titles[g]))
save_table(stab, "concept_stability")
print(stab.groupby("level")[["overlap_top50", "spearman_top50"]].median().round(2))
print(stab.sort_values(["level", "overlap_top50"]).round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.5))
s2 = stab.sort_values(["level", "overlap_top50"])
ax.barh(s2.level.str[:3] + " " + s2.group, s2.overlap_top50, color=np.where(s2.level == "division", "#4a6fa5", "#c2410c"))
ax.set_xlabel("top-50 concept overlap, 2015-23 vs 2024+"); ax.set_title("Concept stability: divisions (blue) vs 2-digit codes (orange)")
fig.tight_layout(); fig.savefig(FIGS / "concept_stability.png", dpi=150)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 56779, "status": "ok", "timestamp": 1791288075434, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="jCjIco6yE-cM" outputId="13617824-9515-47b0-d7ce-fe2b4f3cd080"
test["len_tercile"] = pd.qcut(test.n_words, 3, labels=["short", "medium", "long"])
rows = []
for key in ["B1", "M1", "saemax_L13"]:
    d = load_preds(key, "event", "test").merge(test[["id", "sector", "len_tercile"]], on="id")
    for col in ["sector", "len_tercile"]:
        for g, dd in d.groupby(col, observed=True):
            if len(dd) >= 100:
                r = clf_report(NAMES[key], "test", dd.y.values, dd.pred.values); r.update(group_by=col, group=g); rows.append(r)
sub = pd.DataFrame(rows); save_table(sub, "subgroups")
print(sub.pivot_table(index=["group_by", "group"], columns="model", values="macro_f1").round(3))

m1 = load_preds("M1", "event", "test").merge(test[["id", "narrative"]], on="id")
err = m1[m1.y != m1.pred].sample(100, random_state=SEED).copy()
err["true"] = err.y.map(lambda i: f"{classes[i]} {titles[classes[i]]}")
err["pred_label"] = err.pred.map(lambda i: f"{classes[i]} {titles[classes[i]]}")
err["error_type"] = ""   # 1 label noise | 2 vague | 3 multi-event | 4 sibling (42 vs 43) | 5 jargon | 6 negation | 7 other
err[["id", "narrative", "true", "pred_label", "error_type"]].to_csv(REPO / "results" / "errors_to_tag.csv", index=False)
test.sample(150, random_state=SEED)[["id", "narrative", "event_full", "event_title"]].assign(looks_miscoded="") \
    .to_csv(REPO / "results" / "label_audit_150.csv", index=False)
print("sheets saved: results/errors_to_tag.csv, results/label_audit_150.csv")

# %% id="9uPN_VgvGJ27"
