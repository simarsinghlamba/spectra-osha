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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21500, "status": "ok", "timestamp": 1791263199138, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="lyn4bhY4lwxB" outputId="7e25d2c4-5f1b-4956-c9b5-0f5401bb1eca"
# !pip -q install datasketch statsmodels pytest
from google.colab import drive, userdata
drive.mount("/content/drive")
import sys; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
from spectra.data import *
import zipfile, glob
import pandas as pd, numpy as np, matplotlib.pyplot as plt
setup_dirs(); seed_all()

with zipfile.ZipFile(RAW / "sir.zip") as z:
    z.extractall(RAW / "unzipped")
f = [p for p in glob.glob(str(RAW / "unzipped" / "**" / "*"), recursive=True) if p.lower().endswith(".csv")][0]
try:
    raw = pd.read_csv(f, low_memory=False)
except UnicodeDecodeError:
    raw = pd.read_csv(f, low_memory=False, encoding="latin-1")

df = add_fields(standardize(raw))
seen = set(df.loc[df.year <= 2022, "event_full"].dropna())
novel = df.groupby("year").event_full.apply(lambda s: float((~s.isin(seen)).mean()))
amp = df[df.amputation_bin == 1]
facts = {
    "source_file": Path(f).name, "raw_rows": len(raw), "raw_columns": list(raw.columns),
    "date_min": str(df.date.min()), "date_max": str(df.date.max()),
    "has_loss_of_eye_column": "loss_of_eye" in df.columns,
    "rows_per_year": df.year.value_counts().sort_index().to_dict(),
    "event_code_length_counts": df.event_full.str.len().value_counts().to_dict(),
    "narrative_words_median": float(df.n_words.median()),
    "share_novel_event_codes_by_year": novel.to_dict(),
    "amputation_rows_with_amputat_word": float(amp.narrative.str.contains(r"amputat", case=False).mean()),
}
save_json(facts, TABLES / "data_facts.json")
{k: v for k, v in facts.items() if k != "raw_columns"}

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 66051, "status": "ok", "timestamp": 1791263314478, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="RVKtLTKvmvpk" outputId="7ef4891c-6144-48a7-c9a3-d7c3127cbb33"
n_raw = len(df)
df = df[df.year.notna() & (df.n_words >= CFG["min_words"]) & df.event2.notna()].copy()
n_filtered = len(df)
df = df.sort_values("date").reset_index(drop=True)
df["norm"] = df.narrative.map(norm_text)
df = df.drop_duplicates("norm", keep="first").reset_index(drop=True)
n_exact = len(df)
df = df[near_dup_keep(df.narrative.tolist(), threshold=0.9)].reset_index(drop=True)
print(f"raw {n_raw} -> filtered {n_filtered} -> exact-dedup {n_exact} -> near-dedup {len(df)}")

df["narrative_masked"] = df.narrative.map(mask_text)
df["split"] = df.year.map(year_to_split)
train_full = df[df.split == "train"]
train = train_full.sample(n=min(CFG["train_sample"], len(train_full)), random_state=SEED).copy()
val_all = df[df.split == "val"]
val = val_all.sample(n=min(CFG["val_sample"], len(val_all)), random_state=SEED).copy()
test, shift = df[df.split == "test"].copy(), df[df.split == "shift"].copy()

counts = train.event2.value_counts()
keep_classes = sorted(counts[counts >= CFG["min_class_count"]].index)
classes = keep_classes + ["other"]; cid = {c: i for i, c in enumerate(classes)}
for d in (train, val, test, shift):
    d["y_event"] = d.event2.where(d.event2.isin(keep_classes), "other")
    d["label_id"] = d.y_event.map(cid).astype(int)

titles = {}
for c in keep_classes:
    g = df[df.event2 == c]; exact = g[g.event_full == c]
    titles[c] = exact.event_title.mode().iat[0] if len(exact) else "e.g. " + g.event_title.mode().iat[0]
titles["other"] = "Other (rare classes merged)"
unmapped_shift = shift.loc[~shift.event2.isin(keep_classes), "event2"].value_counts().to_dict()
save_json({"classes": classes, "titles": titles, "unmapped_shift_codes": unmapped_shift}, DATA / "label_map.json")

COLS = ["id", "date", "year", "split", "narrative", "narrative_masked", "event_full", "event2",
        "event_title", "y_event", "label_id", "amputation_bin", "hospitalized_bin", "loss_of_eye_bin",
        "nature_title", "part_title", "source_title", "naics", "sector", "state", "n_words"]
splits = dict(train=train, val=val, test=test, shift=shift)
for name, d in splits.items():
    d[[c for c in COLS if c in d.columns]].reset_index(drop=True).to_parquet(DATA / f"{name}.parquet", index=False)

stats = {
    "rows": {k: len(v) for k, v in splits.items()},
    "train_full_rows": len(train_full), "val_full_rows": len(val_all), "n_classes": len(classes),
    "pipeline_counts": dict(raw=n_raw, filtered=n_filtered, exact_dedup=n_exact, near_dedup=len(df)),
    "amputation_prevalence": {k: float(v.amputation_bin.mean()) for k, v in splits.items()},
    "shift_rows_mapped_to_other": int(sum(unmapped_shift.values())),
}
save_json(stats, TABLES / "data_stats.json")
save_table(pd.DataFrame({"class": classes, "title": [titles[c] for c in classes],
                         **{k: [int((v.y_event == c).sum()) for c in classes] for k, v in splits.items()}}),
           "class_distribution")

fig, ax = plt.subplots(figsize=(8, 3))
df.year.value_counts().sort_index().plot.bar(ax=ax, color="#4a6fa5")
ax.set_title("Severe injury reports per year (after cleaning)"); ax.set_xlabel(""); fig.tight_layout()
fig.savefig(FIGS / "rows_per_year.png", dpi=150)
fig, ax = plt.subplots(figsize=(8, 3))
novel.plot(marker="o", ax=ax, color="#c2410c")
ax.set_title("Share of event codes never seen in 2015-2022 (OIICS switch check)"); fig.tight_layout()
fig.savefig(FIGS / "oiics_switch_check.png", dpi=150)

print(len(classes), "classes:", {c: titles[c] for c in classes})
stats

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 192, "status": "ok", "timestamp": 1791263402995, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="Sae-hWVUnA74" outputId="a5f1f3aa-9127-4bc9-bf21-9bb39d247537"
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
# 1) near-dedupe sanity: an exact copy with one extra word should be dropped
t0, t1 = df.narrative.iloc[0], df.narrative.iloc[1]
print("dedupe check (want [True, False, True]):", near_dup_keep([t0, t0 + " today", t1]).tolist())

# 2) how 2-digit codes changed after the OIICS switch
pre, post = df[df.split != "shift"], df[df.split == "shift"]
mode = lambda x: x.mode().iat[0] if len(x.mode()) else ""
tab = pd.DataFrame({"pre_share": pre.event2.value_counts(normalize=True),
                    "post_share": post.event2.value_counts(normalize=True)}).fillna(0).round(3)
tab["pre_title"] = pre.groupby("event2").event_title.agg(mode)
tab["post_title"] = post.groupby("event2").event_title.agg(mode)
tab["in_classes"] = tab.index.isin(classes)
print(tab.sort_values("post_share", ascending=False).head(25))

# 3) 1-digit division (falls=4, contact=6, ...) before vs after
div = pd.DataFrame({"pre": pre.event2.str[0].value_counts(normalize=True),
                    "post": post.event2.str[0].value_counts(normalize=True)}).round(3)
print(div)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3630, "status": "ok", "timestamp": 1791263474988, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="SA6wcd8wnmnz" outputId="f200d4fe-384c-494c-daa3-c3d5b1cd6dca"
DIV_TITLES = {"1": "Violence and other injuries by persons or animals", "2": "Transportation incidents",
              "3": "Fires and explosions", "4": "Falls, slips, trips",
              "5": "Exposure to harmful substances or environments", "6": "Contact with objects and equipment",
              "7": "Overexertion and bodily reaction", "9": "Nonclassifiable"}
divs = sorted(DIV_TITLES); did = {d: i for i, d in enumerate(divs)}
for name in ["train", "val", "test", "shift"]:
    p = DATA / f"{name}.parquet"; x = pd.read_parquet(p)
    x["y_div"] = x.event2.str[0]; x["div_id"] = x.y_div.map(did).astype(int)
    x.to_parquet(p, index=False)
    print(name, x.y_div.value_counts().sort_index().to_dict())

lm = load_json(DATA / "label_map.json")
lm.update(divisions=divs, div_titles=DIV_TITLES,
          class_div={c: (c[0] if c != "other" else None) for c in lm["classes"]},
          note="OIICS 3 (2024+) renumbered 2-digit codes; shift split is scored at division (1-digit) level.")
save_json(lm, DATA / "label_map.json")

save_table(tab.reset_index(), "oiics_code_meaning_shift")
save_table(div.reset_index(names="division"), "division_shares")

with open(REPO / "tests" / "test_data.py", "a") as fh:
    fh.write('''
def test_division_labels(d):
    for s in SPLITS:
        assert d[s].div_id.notna().all() and d[s].y_div.isin(list("12345679")).all(), s
''')
print("division labels added")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24771, "status": "ok", "timestamp": 1791263537194, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="g9MLcTY8n3XU" outputId="4aadaff2-ad40-4c7b-ad87-ac4d9896e299"
# !cd /content/drive/MyDrive/spectra && python -m pytest -q tests

# %% id="t6hr4EHGoBY2"
