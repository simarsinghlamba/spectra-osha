# SPECTRA
**Sparse Probing, Explanation, Causal Testing & Robustness Audit of AI injury coders**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
[![Open quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/00_quickstart.ipynb)

> When an AI model reads a workplace-injury report, does it see the **hazard**, or a **word that gives the answer away**?

US employers must report every severe work injury (amputation, in-patient hospitalization, loss of an eye) to OSHA within
24 hours. Each report gets an official **OIICS event code** (fall, struck-by, caught-in ...), and agencies increasingly use
AI auto-coders to assign them. SPECTRA opens up a language model while it reads **101,227 real OSHA Severe Injury Reports
(Jan 2015 - Nov 2025)** and audits what it relies on.

**How:** Gemma 3 1B reads each report; Google DeepMind's pre-trained **Gemma Scope 2 sparse autoencoders** split its internal
state into 16,384 readable concepts; we find which concepts predict each injury type, **switch them off** to test which ones
the model actually uses, catch **shortcut** concepts with raw-vs-masked text, and check whether concepts survive OSHA's
**2024 coding change (OIICS 2 -> 3)**. A fine-tuned DeBERTa-v3 and a TF-IDF model are the accuracy references.

## Key findings
1. **The giveaway word is everywhere.** 95.8% of amputation reports literally contain "amputat...".
2. **...but it is redundant.** Removing all outcome words barely changes amputation detection: TF-IDF AUROC
   0.997 -> 0.988; Gemma SAE probe 0.997 -> 0.994.
   Body-part and mechanism words ("fingertip", "thumb") carry the signal.
3. **The model still takes the shortcut when it can.** With outcome words present, **3/10** of Gemma's top amputation
   concepts fire on the outcome word itself; with them masked, **0/10**. Only 2/10 concepts are shared, so
   the model switches route while the score stays flat.
4. **Readable concepts carry the signal.** An SAE probe on 16k concepts scores macro-F1 0.713 [0.697–0.726]
   vs 0.696 [0.681–0.712] for a dense probe (paired bootstrap p = 0.054);
   just **273 concepts (1.7% of the dictionary)** reach 0.690.
5. **Concepts are causally used.** **10 of 16** tested concepts pass an ablation test against
   random concepts (BH-adjusted p < 0.05). Strongest: concept #3448 for *Exposure to environmental heat*; switching it off lowers
   true-class confidence by 0.176, 131x the random-concept effect.
6. **Accuracy anchor.** DeBERTa-v3 reaches macro-F1 0.728 [0.717–0.738], significantly above TF-IDF
   (McNemar p < 0.001); the SAE probe ties TF-IDF (McNemar p = 0.341).
7. **The 2024 code change silently moves meanings.** OIICS 3 renumbered 2-digit codes (e.g. code 43: *Other fall to lower level, unspecified* ->
   *Fall on same level due to slip or trip*), so we score 2024+ at division level. All models drop
   3.6-6.2 points. Concept overlap across eras is 84% for divisions but 36%
   for 2-digit codes (24, 63, 64: 0%).

## Results
| Model | Event macro-F1, 2023 [95% CI] | Division macro-F1, 2023 | Division macro-F1, 2024+ |
|---|---|---|---|
| M1 DeBERTa-v3 | 0.728 [0.717–0.738] | 0.812 | 0.775 |
| B1 TF-IDF+LR | 0.712 [0.698–0.726] | 0.807 | 0.762 |
| P-SAE L13 | 0.713 [0.697–0.726] | 0.821 | 0.760 |
| P-dense L13 | 0.696 [0.681–0.712] | 0.794 | 0.746 |
| P-SAE L17 | 0.708 [0.693–0.722] | 0.816 | 0.768 |
| P-dense L17 | 0.696 [0.682–0.710] | 0.783 | 0.745 |

All numbers come from [`results/tables/`](results/tables); figures are in [`results/figures/`](results/figures).

## Pipeline
```
OSHA ZIP -> clean / dedupe / mask outcome words -> time splits (train 2015-21 | val 22 | test 23 | shift 24+)
  -> TF-IDF baseline | DeBERTa-v3 reference
  -> Gemma 3 1B (fp32), layers 13 / 17 -> Gemma Scope 2 SAE (16k, medium L0)
  -> dense vs SAE probes -> concept discovery -> causal ablation -> shortcut + shift audit
```

## Reproduce
Everything runs on **free Google Colab** (T4 GPU for notebooks 03, 04, 06).
1. Accept the Gemma licence at [google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt) and add a Hugging Face
   read token as the Colab secret `HF_TOKEN` (needed for 04 and 06).
2. Open **`00_quickstart`** below and run it: it clones this repo to `MyDrive/spectra`, downloads the OSHA data and checks
   its SHA-256 against the published file.
3. Run notebooks **01 -> 07** in order. Each saves its outputs to Drive; large files go to `MyDrive/spectra_work` (never in git).
4. Run `python -m pytest -q tests` to check the data.

| Notebook | What it does | Runtime | Time |
|---|---|---|---|
| [`00_quickstart.ipynb`](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/00_quickstart.ipynb) | Clone, install, download + verify data (**start here**) | CPU | ~5 min |
| [`01_clean_split.ipynb`](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/01_clean_split.ipynb) | Clean, dedupe, mask outcome words, time splits, tests | CPU | ~15 min |
| [`02_baselines.ipynb`](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/02_baselines.ipynb) | Majority + TF-IDF baselines; amputation raw vs masked | CPU | ~10 min |
| [`03_deberta.ipynb`](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/03_deberta.ipynb) | DeBERTa-v3 fine-tune (accuracy reference) | T4 GPU | ~25 min |
| [`04_gemma_sae.ipynb`](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/04_gemma_sae.ipynb) | Gemma 3 1B activations + Gemma Scope 2 SAE features | T4 GPU | ~80 min |
| [`05_probes.ipynb`](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/05_probes.ipynb) | Dense vs SAE probes, k-sparse curve, raw vs masked | CPU | ~45 min |
| [`06_concepts_ablation.ipynb`](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/06_concepts_ablation.ipynb) | Concept discovery, snippets, causal ablation | T4 GPU | ~60 min |
| [`07_shift_stats_errors.ipynb`](https://colab.research.google.com/github/simarsinghlamba/spectra-osha/blob/main/notebooks/07_shift_stats_errors.ipynb) | Statistics, shift audit, concept stability, subgroups | CPU | ~10 min |

To check results without rerunning anything, read the CSVs in `results/tables/`. Package versions are in
[`requirements.txt`](requirements.txt); GPU runs can differ in the last decimal place. `scripts/` holds a Python version of
every notebook. `00_setup.ipynb` is the original scaffold notebook and should not be rerun.

## Repository
```
spectra/     helper package (config, data cleaning + masking, Gemma reader + SAE loader, statistics)
notebooks/   one notebook per stage (Colab)          scripts/   the same stages as Python files
tests/       data-integrity tests                    results/   tables (CSV/JSON) and figures
data/        how to download the data (no data stored here)
```

## Honest scope and limitations
- SAE probes are not expected to beat strong classifiers (Kantamneni et al., ICML 2025); SPECTRA uses them for **discovery and auditing**.
- One model (Gemma 3 1B), one main layer (13), one SAE width (16k). Concept labels are human hypotheses.
- Outcome-word masking is regex-based; a few weak traces remain (e.g. "had to").
- Ablation uses 50 random concepts per test, so the smallest possible p is about 0.02.
- Narratives are employer-written and cover federal-OSHA states only.

## Responsible use
Research and decision support for injury coders only, not for enforcement, employer ranking or individual claims.
Employer, address and location columns are never published.

## Licences and credits
Code: MIT. Data: US DOL/OSHA Severe Injury Reports (public domain; no endorsement implied). Gemma 3: Gemma Terms of Use.
Gemma Scope 2 SAEs: Google DeepMind, CC-BY-4.0. DeBERTa-v3: MIT. See [`CITATION.cff`](CITATION.cff) to cite this work.

## Coming next
Public Hugging Face dataset, model and an interactive demo page; human-labelled concept cards.

Author: **Simar Singh Lamba** · [github.com/simarsinghlamba](https://github.com/simarsinghlamba)
