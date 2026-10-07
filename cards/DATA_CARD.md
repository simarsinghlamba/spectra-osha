# SPECTRA: OSHA Severe Injury Report splits
Cleaned, de-duplicated, time-split narratives from US OSHA **Severe Injury Reports** (1 Jan 2015 - 30 Nov 2025), with
official **OIICS event** labels and a second copy of every narrative with give-away outcome words removed.
Built for the SPECTRA study: https://github.com/simarsinghlamba/spectra-osha

## Splits
| Split | Years | Rows |
|---|---|---|
| train | 2015-2021 (random sample of 66,151) | 40,000 |
| validation | 2022 (random sample of 8,787) | 5,000 |
| test | 2023 | 8,762 |
| shift_2024plus | Jan 2024 - Nov 2025 (after the OIICS 2 -> 3 change) | 17,527 |

## Main columns
- `narrative`: the employer-written injury description. `narrative_masked`: the same text with outcome words removed
  (amputat*, hospitali*, admitted, severed, loss of eye ...) and the grammar around them.
- `y_event` / `label_id`: 2-digit OIICS event code, 19 classes (codes with < 200 training examples merged into `other`).
- `y_div` / `div_id`: 1-digit OIICS division (8 broad groups). **Use this for the 2024+ split** (see below).
- `event_full`, `event_title`: the original full code and title. `amputation_bin`, `hospitalized_bin`, `loss_of_eye_bin`: outcomes.
- `nature_title`, `part_title`, `source_title`, `naics`, `sector`, `state`, `date`, `year`, `n_words`, `id`.

## Important: the 2024 code change
From 2024 OSHA codes with **OIICS 3**, which **renumbered many 2-digit codes** (e.g. code 43 meant *fall to lower level*
before 2024 and *fall on same level* after). 2-digit labels in `shift_2024plus` are **not comparable** with earlier years;
the 1-digit divisions are. `label_map.json` holds class and division titles.

## How it was made
Narratives under 15 words dropped; exact and near-duplicate narratives removed (MinHash, Jaccard >= 0.9, earliest kept);
time-based splits so no report leaks across years. Removed columns: Employer, Address1/2, City, Zip, Latitude, Longitude,
UPA, Inspection, FederalState.

## Classes
| Code | Title (pre-2024, OIICS 2) |
|---|---|
| 11 | e.g. Hitting, kicking, beating, shoving |
| 13 | e.g. Other animal bites, nonvenomous |
| 24 | e.g. Pedestrian struck by vehicle in nonroadway area, unspecified |
| 27 | e.g. Part of occupant s body caught between vehicle and other object in nonroadway transport incident |
| 31 | e.g. Ignition of vapors, gases, or liquids |
| 32 | e.g. Explosion of pressure vessel, piping, or tire |
| 40 | Fall, slip, trip, unspecified |
| 42 | e.g. Fall on same level due to slipping |
| 43 | e.g. Other fall to lower level, unspecified |
| 51 | e.g. Direct exposure to electricity, unspecified |
| 53 | e.g. Exposure to environmental heat |
| 55 | e.g. Exposure through intact skin, eyes, or other exposed tissue |
| 60 | Contact with objects and equipment, unspecified |
| 62 | e.g. Injured by slipping or swinging object held by injured worker |
| 63 | e.g. Struck against moving part of machinery or equipment |
| 64 | e.g. Caught in running equipment or machinery during regular operation |
| 71 | e.g. Overexertion in lifting-single episode |
| 99 | e.g. Nonclassifiable |
| other | Other (rare classes merged) |

| Division | Title |
|---|---|
| 1 | Violence and other injuries by persons or animals |
| 2 | Transportation incidents |
| 3 | Fires and explosions |
| 4 | Falls, slips, trips |
| 5 | Exposure to harmful substances or environments |
| 6 | Contact with objects and equipment |
| 7 | Overexertion and bodily reaction |
| 9 | Nonclassifiable |

## Known issues
Employer-written text; federal-OSHA states only; labels from human coders contain some noise; class imbalance;
outcome-word masking is rule-based and leaves a few faint traces.

## Source and licence
US Department of Labor / OSHA Severe Injury Reports, file `January2015toNovember2025.zip` (SHA-256 `a3f7f434e200fb956131f12277378e592993a25db3f328716fbece106f846bb0`;
unchanged mirror: https://github.com/simarsinghlamba/spectra-osha/releases/tag/data-v1). US federal government work, generally public domain; no endorsement by
the Department of Labor is implied.

## Load
```python
from datasets import load_dataset
ds = load_dataset("Simar123456/spectra-osha-sir")
```
