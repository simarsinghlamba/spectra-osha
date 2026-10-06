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
