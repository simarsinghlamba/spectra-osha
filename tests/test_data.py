"""Data-integrity tests. Run from Colab: !cd /content/drive/MyDrive/spectra && python -m pytest -q tests"""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectra.common import DATA, TABLES, ACTS, PUB, CFG, load_json
from spectra.data import LEAK_RE, PRIVATE_COLS, norm_text

SPLITS = ["train", "val", "test", "shift"]

@pytest.fixture(scope="module")
def d():
    if not (DATA / "train.parquet").exists():
        pytest.skip("splits not built yet (Notebook 01)")
    return {s: pd.read_parquet(DATA / f"{s}.parquet") for s in SPLITS}

def test_years(d):
    assert d["train"].year.between(2015, 2021).all()
    assert (d["val"].year == 2022).all()
    assert (d["test"].year == 2023).all()
    assert (d["shift"].year >= 2024).all()

def test_no_text_overlap(d):
    sets = {s: set(d[s].narrative.map(norm_text)) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not (sets[a] & sets[b]), f"same narrative in {a} and {b}"

def test_no_id_overlap(d):
    seen = set()
    for s in SPLITS:
        ids = set(d[s].id)
        assert not (ids & seen), f"duplicate ids in {s}"
        seen |= ids

def test_masked_has_no_leak_words(d):
    for s in SPLITS:
        assert not d[s].narrative_masked.str.contains(LEAK_RE).any(), s

def test_no_private_columns(d):
    for s in SPLITS:
        assert not set(PRIVATE_COLS) & set(d[s].columns), s

def test_min_words(d):
    for s in SPLITS:
        assert (d[s].n_words >= CFG["min_words"]).all(), s

def test_labels(d):
    for s in SPLITS:
        assert d[s].label_id.notna().all(), s
    counts = d["train"].y_event.value_counts()
    assert (counts.drop("other", errors="ignore") >= CFG["min_class_count"]).all()

def test_counts_match_stats(d):
    stats = load_json(TABLES / "data_stats.json")
    for s in SPLITS:
        assert len(d[s]) == stats["rows"][s], s

def test_activations_aligned(d):
    from spectra.sae import load_acts
    for s in SPLITS:
        folder = ACTS / f"raw_{s}"
        if not folder.exists():
            pytest.skip("activations not extracted yet")
        assert load_acts(folder, "ntok").shape[0] == len(d[s]), s

def test_published_dataset_clean():
    folder = PUB / "hf_dataset"
    if not folder.exists():
        pytest.skip("not published yet")
    for f in folder.glob("*.parquet"):
        cols = set(pd.read_parquet(f).columns)
        assert not set(PRIVATE_COLS) & cols, f.name
