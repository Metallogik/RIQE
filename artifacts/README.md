# Model artifact

Released under CC BY 4.0 (see [LICENSE-MODEL](../LICENSE-MODEL)).

| File | Content |
|---|---|
| `riqe-v1.0.npz` | `nu` (36), `sigma` (36×36), `pinv_sigma` computed with the threshold used |
| `riqe-v1.0.json` | everything else, described below |

The JSON file is not packaging around the numbers. It is the part that
existing work leaves out:

- the **intensity mapping** (HU window, scale, `C`, kernel, FOV and body mask
  rules), which *defines* the model: different windows produce mutually
  incompatible models;
- the **hyperparameters** and the criterion that selected them;
- the **identity of the corpus**: every `PatientID`, `SeriesInstanceUID` and
  `SOPInstanceUID` used, with its SHA-256, plus the exclusion counts for each
  criterion;
- the **identity of the code**: git commit, SHA-256 of the sources and
  library versions.

## Reproducing the fit

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_corpus.py      # ~36 GB from TCIA, CC BY 4.0
.venv/bin/python scripts/build_slice_table.py
.venv/bin/python scripts/make_corpus.py
.venv/bin/python scripts/cache_features.py
.venv/bin/python scripts/fit_model.py --P 24 --C 0.01 --p 0.5
.venv/bin/python scripts/verify_model.py --check-hashes
```

The last command checks that every declared slice is present and intact. It
then refits the model and compares it with the published numbers. The
published model is reproduced bit for bit.

## How to report the score

Always report it as a **distance from the declared model**:

> RIQE distance from model `riqe-v1.0`
> (HU [−1000, +1000] → [0, 255], C = 0.01, P = 24, p = 0.5)

Never report it as an absolute judgement of diagnostic quality. Never compare
it with NIQE values from the literature, which use a different scale. The
model does not know what a lesion is.

Compare scores only within one acquisition protocol (anatomy, kernel, slice
thickness).
