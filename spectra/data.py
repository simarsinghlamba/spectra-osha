"""Cleaning, labels, leak-word masking and near-duplicate removal."""
import re
import numpy as np
import pandas as pd

COL_CANDIDATES = {
    "id": ["ID"],
    "date": ["EventDate", "Event Date"],
    "narrative": ["Final Narrative", "FinalNarrative", "Narrative"],
    "event": ["Event"],
    "event_title": ["EventTitle", "Event Title"],
    "nature_title": ["NatureTitle", "Nature Title"],
    "part_title": ["Part of Body Title", "PartofBodyTitle"],
    "source_title": ["SourceTitle", "Source Title"],
    "hospitalized": ["Hospitalized"],
    "amputation": ["Amputation"],
    "loss_of_eye": ["Loss of Eye", "LossOfEye"],
    "naics": ["Primary NAICS", "PrimaryNAICS", "NAICS"],
    "state": ["State"],
}
PRIVATE_COLS = ["Employer", "Address1", "Address2", "City", "Zip", "Latitude", "Longitude",
                "employer", "address1", "address2", "city", "zip", "latitude", "longitude"]

LEAK_PATTERNS = [
    r"\bamputat\w*", r"\bamputee\w*",
    r"\bhospitali[sz]\w*", r"\bin-?patient\b", r"\badmitted\b", r"\badmission\b",
    r"\bsever(?:ed|ing|s)?\b",
    r"\benucleat\w*",
    r"\b(?:loss|lost) of (?:an? |the |his |her |their )?(?:left |right )?eye\b",
    r"\blost (?:an? |the |his |her |their )?(?:left |right )?eye\b",
]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS), flags=re.IGNORECASE)

def _key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def standardize(raw):
    lookup = {_key(c): c for c in raw.columns}
    out = {}
    for new, cands in COL_CANDIDATES.items():
        for c in cands:
            if _key(c) in lookup:
                out[new] = raw[lookup[_key(c)]]
                break
    missing = [k for k in ["date", "narrative", "event"] if k not in out]
    assert not missing, f"Missing required columns {missing}. Found: {list(raw.columns)}"
    df = pd.DataFrame(out)
    if "id" not in df:
        df["id"] = np.arange(len(df))
    return df

def code_str(x):
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return None

def event2(x):
    s = code_str(x)
    return s[:2] if s and len(s) >= 2 else None

def sector(naics):
    s = code_str(naics)
    if not s:
        return "unknown"
    return {"23": "construction", "31": "manufacturing", "32": "manufacturing", "33": "manufacturing",
            "48": "transport_warehousing", "49": "transport_warehousing"}.get(s[:2], "other")

def add_fields(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["narrative"] = df["narrative"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["n_words"] = df["narrative"].str.split().str.len()
    df["event_full"] = df["event"].map(code_str)
    df["event2"] = df["event"].map(event2)
    for c in ["amputation", "hospitalized", "loss_of_eye"]:
        if c in df:
            df[c + "_bin"] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(int)
    if "naics" in df:
        df["naics"] = df["naics"].map(code_str)
        df["sector"] = df["naics"].map(sector)
    df["id"] = df["id"].astype(str)
    return df

def mask_text(t):
    return re.sub(r"\s+", " ", LEAK_RE.sub(" ", t)).strip()

def norm_text(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()

def year_to_split(y):
    if 2015 <= y <= 2021: return "train"
    if y == 2022: return "val"
    if y == 2023: return "test"
    if y >= 2024: return "shift"
    return None

def _shingles(t, k=3):
    w = re.findall(r"[a-z0-9]+", t.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}

def near_dup_keep(texts, threshold=0.9, num_perm=128):
    """Keep the first occurrence of each near-duplicate group (texts must be sorted by date)."""
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = np.ones(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        m.update_batch([s.encode() for s in _shingles(t)])
        if lsh.query(m):
            keep[i] = False
        else:
            lsh.insert(str(i), m)
        if i % 20000 == 0:
            print(f"  dedupe {i}/{len(texts)}")
    return keep
