# RIQE

**R**adiology **I**mage **Q**uality **E**valuator, for computed tomography.

RIQE is a NIQE-style no-reference image quality model fitted on CT instead of
natural photographs. It is released together with its validation, and with
everything needed to rebuild it from public data.

NIQE (Mittal, Soundararajan and Bovik, IEEE SPL 20(3):209–212, 2013) measures
how far an image deviates from the statistical regularity of a reference
corpus. Its official reference model was fitted on 125 natural photographs, so
on a CT image it returns a number whose relation to CT quality is unknown. A 2025 study
(*Computers* 14(1):18) fitted NIQE on CT but did not publish the model
parameters, the code or the HU-to-luminance conversion. RIQE publishes all
three and tests the result.

The model file itself is simple: a 36-dimensional mean and a 36×36 covariance.
The contribution is **the validation**. It checks two things: whether the
model ranks a degraded image worse than its source, and whether its
preferences among filtered images follow the signal of small low-contrast
lesions. The first holds; the second does not.

> **The RIQE score is a distance from the declared reference model, not a
> measure of diagnostic quality.** The model does not know what a lesion is.
> Its values are not comparable with NIQE values in the literature.

## What it is good for, and what it is not

**In one sentence:** RIQE ranks a degraded CT image worse than its source
(simulated dose reduction, added noise, blur), but it **should not be the sole
criterion to choose, compare or tune a denoiser**. It prefers an
edge-preserving filter even at strengths that leave less than a third of the
signal of a small low-contrast lesion, and does not prefer linear smoothing
that keeps more.

Results on the held-out test split (40 patients never used for fitting or model
selection). Intervals are 95% bootstrap intervals over patients.

| Test | Result |
|---|---|
| Simulated reduced-dose image ranked worse than full dose, same slice | chest 240/240 pairs (10 patients); abdomen 95.8% [91.2, 99.6] of 240 pairs (10 patients) |
| Noise added at +5% / +10% of the native noise, detected (chest / abdomen) | 85.0 / 89.2%, 92.5 / 98.3% |
| Noise added at ≥ +20% of the native noise, detected | 97.5–100% |
| Gaussian blur σ = 0.5 px / ≥ 1 px, detected | 49.2% chest, 92.5% abdomen / 99–100% |
| Filtered full-dose images scoring *better* than the unfiltered original | 41.2% [34.8, 48.3]; 27.8% counting only changes > 0.05 |
| Bilateral filter preferred to the unfiltered image, 4 mm +10 HU lesion (18 slices, 64 noise realisations) | 18/18 images up to 32 HU residual, where 29% [25, 32] of the lesion signal is left and d′ falls from 0.51 to 0.33 |
| Gaussian filter preferred to the unfiltered image | never, although at 32 HU it keeps 70% of the signal and d′ 0.48 (bilateral − Gaussian d′: −0.17 [−0.19, −0.14]) |
| Rank correlation with radiologists, LDCTIQAC 2023 (1000 images; exploratory, patients not identifiable, intervals over inferred source slices) | within one source slice: median −0.47 [−0.54, −0.30]; pooled: −0.24 [−0.32, −0.17] |
| Same code fitted on 119 CC0 photographs, RIQE settings: reduced dose ranked worse, abdomen | 31.2% |
| Same code fitted on the photographs with the NIQE parameters (P = 96, C = 1): reduced dose ranked worse, abdomen | 0% |
| Stability: rank Spearman between bootstrap models | 0.99 |
| Refit from the public data with `scripts/verify_model.py` | bit-identical in the recorded environment |

Using RIQE safely:

- Compare scores **only within one acquisition protocol** (same anatomy,
  kernel and slice thickness). Sub-models fitted on different kernels or
  anatomies diverge by 0.5 to 2.7 times as much as a simulated dose reduction
  on the same patients. RIQE is deliberately released as a single model, so
  absolute scores across protocols are not comparable.
- Use it for **paired ranking**: an image against a degraded or processed
  version of the same anatomy. It has not been validated as a stand-alone
  detector with a threshold on single images.
- A **better score after processing does not mean a better image**.

## The model

| | |
|---|---|
| Files | [artifacts/riqe-v1.0.npz](artifacts/riqe-v1.0.npz) (`nu`, `sigma`, `pinv_sigma`), [artifacts/riqe-v1.0.json](artifacts/riqe-v1.0.json) (specification, provenance) |
| Intensity mapping | HU clipped to [−1000, +1000] and mapped linearly to [0, 255], float32, not rounded. This mapping is part of the model. |
| Hyperparameters | patch P = 24 px, MSCN stabiliser C = 0.01, sharpness fraction p = 0.50 |
| Masks | patches fully inside the field of view and with ≥ 90% body |
| Fitting corpus | 121,213 patches from 3,792 full-dose slices of 158 patients |
| Minimum for a score | 72 valid patches per image; below that the score is undefined (NaN) |

The JSON file records the full intensity mapping, the hyperparameters and the
criterion that selected them. It also records every `PatientID`,
`SeriesInstanceUID` and `SOPInstanceUID` used with its SHA-256, the exclusion
counts per criterion, and the git commit and source hash of the code.

### Scoring an image

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```python
import json
import numpy as np
from riqe.dicomio import read_hu
from riqe.extract import MIN_PATCHES_FOR_SCORE, Spec, features_from_hu
from riqe.model import RCOND, mahalanobis_mixed

art = json.load(open("artifacts/riqe-v1.0.json"))
z = np.load("artifacts/riqe-v1.0.npz")
hp = art["hyperparameters"]
spec = Spec(P=hp["P"], C=hp["C"], p=hp["p"])

hu, padding, _ = read_hu("slice.dcm")
pf = features_from_hu(hu, padding, spec, fitting=False)   # no sharpness selection when scoring
f = pf.feat[pf.valid]
if f.shape[0] < MIN_PATCHES_FOR_SCORE:
    score = float("nan")                                  # too little body in the field of view
else:
    score = mahalanobis_mixed(z["nu"], z["sigma"], f.mean(axis=0), np.cov(f, rowvar=False), RCOND)
```

Lower means closer to the full-dose reference corpus. Report the value as a
distance from `riqe-v1.0`, stating the model version.

## Reproducing everything

### Data

The corpus is **not in this repository** (36 GB of DICOM). The repository
carries its *identity*: [corpus/corpus_manifest.json](corpus/corpus_manifest.json)
lists the UIDs of every slice used, and the model JSON carries their SHA-256.

- **Fitting and testing:** TCIA collection *LDCT-and-Projection-data*,
  version 7 ([doi:10.7937/9npb-2637](https://doi.org/10.7937/9npb-2637)),
  Chest and Liver/Abdomen subsets, CC BY 4.0. No registration is needed; the
  public NBIA REST API answers anonymously. The Head subset is under NIH
  controlled access and is not used.
- **Experiment 6:** LDCTIQAC 2023 training set
  ([doi:10.5281/zenodo.7833096](https://doi.org/10.5281/zenodo.7833096)),
  CC BY 4.0. Unpack it into `data/ldctiqac/` so that
  `data/ldctiqac/LDCTIQAG2023_train/train.json` exists.
- **Photographic baseline:** 119 CC0 photographs from Wikimedia Commons,
  frozen in [corpus/photo_corpus_manifest.json](corpus/photo_corpus_manifest.json)
  (title, URL, licence, SHA-256). They are downloaded automatically.

### Pipeline

```bash
.venv/bin/python scripts/download_corpus.py     # image series from TCIA (~36 GB, ~3.5 h)
.venv/bin/python scripts/build_slice_table.py   # metadata and statistics per slice
.venv/bin/python scripts/make_corpus.py         # inclusion criteria S1-S6, patient split, manifest
.venv/bin/python scripts/cache_features.py      # per-patch features for each (P, C)
bash scripts/run_pipeline.sh                    # everything else, resumable
```

`run_pipeline.sh` runs the following steps in order:

1. builds the validation bank;
2. searches the hyperparameters on an inner split of the fitting patients,
   and compares one model per protocol with the single model on that split;
3. fits the model;
4. runs the stratification experiment;
5. builds and scores the test bank;
6. runs the validation battery, the lesion experiment, the photographic
   baselines and LDCTIQAC;
7. computes the confidence intervals (`uncertainty.py`);
8. finally runs `verify_model.py`, which refits from the downloaded data and
   compares the result with the published numbers.

Outputs go to `experiments/`. After that, `scripts/make_tables.py` and
`scripts/make_figures.py` write the LaTeX tables and figures to
`paper/tables/` and `paper/figures/`.

The split is **by patient**, never by slice: 158 patients for fitting and
40 for testing, stratified by protocol. The test split was not touched until
the final battery.

### Design choices and deviations, declared

- **Inclusion criteria** (S1–S6, [riqe/inclusion.py](riqe/inclusion.py)) are
  pure operations on a per-slice table. The exclusions are counted in
  [corpus/exclusion_report.csv](corpus/exclusion_report.csv). 4,752
  full-dose slices from 198 patients are kept.
- **Reduced dose is simulated.** The reduced-dose reconstructions in the
  collection were produced by the data providers by inserting noise into the
  full-dose projection data (Moen et al., Med Phys 48(2), 2021). They exist
  for the 100 Siemens patients only (10% of routine dose for chest, 25% for
  abdomen). Each reduced-dose slice is scored on the masks of the full-dose
  slice at the same position.
- **Hyperparameter criterion revised once.** The criterion declared before the
  search ignored reduced-dose ordering. It selected P = 32, C = 0.01, p = 0.05,
  which on the test split ranks reduced-dose abdomen correctly in only 28.5% of
  214 scoreable pairs. After seeing validation data, and before touching the
  test split, reduced-dose ordering was made the first criterion. Both choices
  are recorded in the model JSON.
- **Filtered full-dose preference.** The criterion declared first required
  that a filtered full-dose image never score better than its source. A
  full-dose image still contains noise, so a preference for light filtering
  is not by itself an error: the rate is reported as a preference rate, and
  whether the preferred filtering removes signal is tested with inserted
  lesions.
- **Confidence intervals** resample patients (bootstrap, B = 2000), since the
  slices, pairs and filtered versions of one patient are not independent.
  LDCTIQAC does not identify its patients: its analysis is exploratory, with
  intervals over source slices inferred from image similarity and a
  sensitivity analysis on the grouping (`scripts/ldctiqac_groups_check.py`).
- **Denoisers:** Gaussian, total variation, bilateral, non-local means and
  wavelet (scikit-image). Their strength is calibrated per image to the same
  residual standard deviation, from 2 to 64 HU. BM3D is not used (GPL).
- **Masks.** GE images in this collection pad 21.3% of every slice with
  −3024 HU outside the field of view, and Siemens images do not. Without the
  FOV mask, the vendor difference RIQE would measure would be a DICOM
  convention, not physics.

## Repository contents

| Path | Content |
|---|---|
| [riqe/nss.py](riqe/nss.py) | MSCN, moment-matching GGD/AGGD estimators, 36 features per patch, sharpness selection |
| [riqe/hu.py](riqe/hu.py) | canonical HU → luminance mapping, FOV and body masks |
| [riqe/extract.py](riqe/extract.py) | `Spec`: everything that changes the numbers, in one place |
| [riqe/model.py](riqe/model.py) | MVG model, score, divergence between models |
| [riqe/inclusion.py](riqe/inclusion.py) | inclusion criteria S1–S6 |
| [riqe/degrade.py](riqe/degrade.py) | white and FBP-like noise, blur, calibrated denoisers, lesion insertion, d′ and signal retention |
| [riqe/bank.py](riqe/bank.py) | test bank as deterministic recipes rather than pixels |
| [riqe/cache.py](riqe/cache.py) | feature cache, with the threshold `p` applied downstream |
| [riqe/dosetest.py](riqe/dosetest.py) | full-dose / reduced-dose pairs matched by position |
| [riqe/evaluate.py](riqe/evaluate.py) | scoring from stored moments, sign tests, Holm, Kendall's W, patient-level bootstrap |
| [riqe/niqecfg.py](riqe/niqecfg.py) | the parameters published for NIQE (P = 96, C = 1, p = 0.75, whole image), for the photographic baseline |
| [riqe/figstyle.py](riqe/figstyle.py) | figure style |
| [scripts/](scripts/) | the pipeline above, one script per step |
| [corpus/](corpus/) | corpus identity: series list, manifest, split, exclusion report, photo manifest |
| [artifacts/](artifacts/) | the released model |

## Licence constraints respected

- **pyiqa / IQA-PyTorch is not used.** Its licences (PolyForm Noncommercial,
  NTU S-Lab) are incompatible with a permissive release.
- **The LIVE pristine model (`modelparameters.mat`) is not used**, not even
  as a default or a convenience reference. The photographic baselines are
  refits of this code on CC0 photographs.
- **The algorithm is implemented from the paper**, and no LIVE MATLAB code is
  incorporated. The paper is cited as the source of the algorithm.
- Dependencies are permissively licensed: see
  [requirements.txt](requirements.txt).

## Citation and licence

Code: [MIT](LICENSE). Model, manifests and results:
[CC BY 4.0](LICENSE-MODEL).

Any use of the model must also cite the data it was fitted on, as CC BY 4.0
requires:

> McCollough, C., Chen, B., Holmes III, D., Duan, X., Yu, Z., Yu, L., Leng,
> S., Fletcher, J. (2020). Low Dose CT Image and Projection Data
> (LDCT-and-Projection-data) (Version 7) [dataset]. The Cancer Imaging
> Archive. https://doi.org/10.7937/9npb-2637

> Clark, K., et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and
> Operating a Public Information Repository. Journal of Digital Imaging 26(6),
> 1045–1057.

It must also acknowledge NIBIB grants EB017095 and EB017185 (PI: Cynthia
McCollough).

Author: Fabio Mattiussi.
