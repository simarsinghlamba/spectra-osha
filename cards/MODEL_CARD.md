# SPECTRA DeBERTa-v3 OIICS event coder
A fine-tuned `microsoft/deberta-v3-base` that reads a US OSHA severe-injury narrative and predicts its **2-digit OIICS
(v2.01) event code**: how the injury happened (fall, struck by, caught in ...). 19 classes; rare codes
are merged into `other`. It is the accuracy reference for the SPECTRA interpretability audit: https://github.com/simarsinghlamba/spectra-osha

## Results
| Test set | Macro-F1 [95% CI] | Accuracy |
|---|---|---|
| 2023 reports, 19 event classes | 0.728 [0.717-0.738] | 0.877 |
| 2023 reports, 8 divisions | 0.812 | 0.953 |
| 2024+ reports, 8 divisions (after the code change) | 0.775 | 0.937 |

## Use
```python
from transformers import pipeline
coder = pipeline("text-classification", model="Simar123456/spectra-deberta-oiics")
coder("An employee was climbing a 12-foot extension ladder when he fell to the concrete floor below.")
```
Labels are 2-digit codes (e.g. `43`); titles are in the dataset's `label_map.json`: https://huggingface.co/datasets/Simar123456/spectra-osha-sir

## Training
Data: 40,000 narratives from 2015-2021, validation 5,000 from 2022 (https://huggingface.co/datasets/Simar123456/spectra-osha-sir). 3 epochs, learning rate 2e-5,
batch 32, max length 160 tokens, 225 warm-up steps, weight decay 0.01, fp16 mixed precision on one T4 GPU, seed 42;
best epoch chosen by validation macro-F1.

## Intended use
Research on, and decision support for, OIICS event coding of US severe-injury narratives. **Not for** enforcement,
ranking employers, individual claims, or non-US / non-English text without re-evaluation.

## Limitations
Predicts **OIICS 2.01** codes: OSHA renumbered many codes in 2024 (OIICS 3), so 2-digit predictions on 2024+ reports must
be mapped, e.g. to divisions. Narratives are employer-written and federal-OSHA only; training labels contain coder noise.

## Licences
Model weights: MIT (as DeBERTa-v3). Training data: US federal government work (public domain).
