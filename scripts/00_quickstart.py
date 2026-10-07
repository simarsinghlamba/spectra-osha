# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # SPECTRA quickstart
# Run this first. It copies the project into **your** Google Drive, installs packages, downloads the OSHA data and checks it is the exact file used in the study.
#
# **When you run the first cell, Google asks permission to connect your Drive.** Choose your account and **tick all the permission boxes** (or *Select all*), then Continue. If you skip a box, the connection fails.

# %%
from google.colab import drive
import os
try:
    drive.mount("/content/drive")
except Exception as e:
    raise RuntimeError(
        "Google Drive did not connect (" + str(e) + ").\n"
        "Fix: run this cell again. In the Google pop-up, choose your account and tick ALL the permission boxes, "
        "then Continue. Still failing? Allow pop-ups for colab.research.google.com, use Chrome, or "
        "Runtime > Disconnect and delete runtime, then retry.") from None
if not os.path.exists("/content/drive/MyDrive/spectra"):
    # !git clone https://github.com/simarsinghlamba/spectra-osha.git /content/drive/MyDrive/spectra
else:
    print("Found an existing 'My Drive/spectra' folder - using it.")
# !pip -q install datasketch statsmodels pytest
print("Step 1 done: the project is in My Drive/spectra")

# %%
import os
if not os.path.exists("/content/drive/MyDrive/spectra/spectra/common.py"):
    raise RuntimeError("Run the cell above first: Google Drive is not connected or the project was not copied.") from None
import sys, hashlib, requests; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
setup_dirs()
EXPECTED = "a3f7f434e200fb956131f12277378e592993a25db3f328716fbece106f846bb0"
MIRROR = "https://github.com/simarsinghlamba/spectra-osha/releases/download/data-v1/January2015toNovember2025.zip"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/126.0 Safari/537.36", "Accept": "*/*"}
dest = RAW / "sir.zip"
ok = lambda b: b[:2] == b"PK" and hashlib.sha256(b).hexdigest() == EXPECTED
if not (dest.exists() and ok(dest.read_bytes())):
    for name, url in [("OSHA website", CFG["sir_zip_url"]), ("GitHub mirror", MIRROR)]:
        try:
            r = requests.get(url, headers=HEADERS, timeout=300)
            print(f"{name}: HTTP {r.status_code}, {len(r.content) / 1e6:.1f} MB")
            if r.status_code == 200 and ok(r.content):
                dest.write_bytes(r.content); print("Saved from", name); break
            if r.status_code == 200:
                print(f"{name}: not the exact file used in the study (OSHA may have updated it) - trying next source")
        except Exception as e:
            print(f"{name}: failed ({e})")
if dest.exists() and ok(dest.read_bytes()):
    print("Data ready: the exact file used in the study (SHA-256 verified).")
else:
    print("Automatic download failed. Download the ZIP from", MIRROR, "and upload it to My Drive/spectra_work/raw/sir.zip")
print("Note: the other folders in My Drive/spectra_work stay empty until notebooks 01-07 fill them.")

# %% [markdown]
# ## Done. What next?
# This notebook copied the project into **your** Google Drive (`My Drive/spectra`) and saved the data in `My Drive/spectra_work`. Now run the stages **in order**. Click a link, run all cells top to bottom, then come back for the next one.
#
# | Step | Notebook | Runtime | What it does |
# |---|---|---|---|
# | 1 | [Open 01_clean_split.ipynb](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/01_clean_split.ipynb) | CPU | clean, dedupe, mask, time splits |
# | 2 | [Open 02_baselines.ipynb](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/02_baselines.ipynb) | CPU | TF-IDF baselines |
# | 3 | [Open 03_deberta.ipynb](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/03_deberta.ipynb) | **T4 GPU** | DeBERTa reference model |
# | 4 | [Open 04_gemma_sae.ipynb](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/04_gemma_sae.ipynb) | **T4 GPU** | Gemma + SAE features (needs `HF_TOKEN` secret) |
# | 5 | [Open 05_probes.ipynb](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/05_probes.ipynb) | CPU | dense vs SAE probes |
# | 6 | [Open 06_concepts_ablation.ipynb](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/06_concepts_ablation.ipynb) | **T4 GPU** | concepts + causal ablation (needs `HF_TOKEN`) |
# | 7 | [Open 07_shift_stats_errors.ipynb](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/07_shift_stats_errors.ipynb) | CPU | statistics + shift audit |
#
# **Before step 4:** accept the Gemma licence at https://huggingface.co/google/gemma-3-1b-pt, create a Hugging Face *read* token, and add it in Colab: key icon (Secrets) in the left bar → `HF_TOKEN`.
# **GPU steps:** Runtime → Change runtime type → T4 GPU.
# **Do not run** `00_setup.ipynb`: it is the original scaffold notebook and would overwrite the helper code.
#
# Prefer Drive? The same notebooks are in `My Drive/spectra/notebooks`; double-click to open in Colab.
# Only want to check the numbers? All results are in `results/tables/` in the repo; nothing needs rerunning.
