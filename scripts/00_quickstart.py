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
# Run this first. It clones the repo into your Google Drive, installs packages, downloads the OSHA data and checks it is the exact file used in the paper. Then open notebooks 01 → 07 from `MyDrive/spectra/notebooks` (T4 GPU for 03, 04, 06).

# %%
from google.colab import drive
drive.mount("/content/drive")
import os
if not os.path.exists("/content/drive/MyDrive/spectra"):
    # !git clone https://github.com/simarsinghlamba/spectra-osha.git /content/drive/MyDrive/spectra
# !pip -q install datasketch statsmodels pytest

# %%
import sys, hashlib, requests; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
setup_dirs()
EXPECTED = "a3f7f434e200fb956131f12277378e592993a25db3f328716fbece106f846bb0"
dest = RAW / "sir.zip"
if not dest.exists():
    r = requests.get(CFG["sir_zip_url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
    if r.status_code == 200 and r.content[:2] == b"PK": dest.write_bytes(r.content)
    else: print("Blocked: download the ZIP from https://www.osha.gov/severe-injury-reports and upload it to", dest)
if dest.exists():
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    print("SHA-256 matches the published file:", sha == EXPECTED)

# %% [markdown]
# **Next:** open `01_clean_split.ipynb`. Skip the last *commit / push* cell in each notebook; it needs the author's GitHub token. Do **not** run `00_setup.ipynb`: it is the original scaffold notebook and rewrites the helper package from scratch.
