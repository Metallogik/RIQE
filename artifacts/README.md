# Artefatto del modello

Contenuto del deposito (Zenodo, CC BY 4.0):

| File | Cosa contiene |
|---|---|
| `riqe-v1.0.npz` | `nu` (36), `sigma` (36×36), `pinv_sigma` con la soglia usata |
| `riqe-v1.0.json` | tutto il resto — vedi sotto |
| `verify_ok.txt` | esito della verifica di riproducibilità al momento del rilascio |

Il file JSON non è un contorno: è la parte che manca al lavoro esistente.
Porta la **mappatura di intensità** (finestra HU, scala, `C`, kernel, regole
di maschera FOV e corpo), che *definisce* il modello — finestre diverse
producono modelli fra loro incompatibili; gli **iperparametri** e il criterio
che li ha scelti; l'**identità del corpus**, cioè ogni `PatientID`,
`SeriesInstanceUID` e `SOPInstanceUID` usato con il suo SHA-256, più i
conteggi di esclusione per ciascun criterio; e l'**identità del codice**,
cioè commit git, SHA-256 dei sorgenti e versioni delle librerie.

## Riprodurre il fit

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pydicom scikit-image pandas pyarrow PyWavelets
.venv/bin/python scripts/download_corpus.py      # ~36 GB da TCIA, CC BY 4.0
.venv/bin/python scripts/build_slice_table.py
.venv/bin/python scripts/make_corpus.py
.venv/bin/python scripts/cache_features.py
.venv/bin/python scripts/fit_model.py
.venv/bin/python scripts/verify_model.py --check-hashes
```

L'ultimo comando rifà il fit e lo confronta con i numeri pubblicati, dopo aver
verificato che ogni slice dichiarata sia presente e integra. La
riproducibilità è quindi controllabile, non promessa.

## Come riferire il punteggio

Sempre come **distanza relativa al modello dichiarato**:

> distanza RIQE rispetto al modello `riqe-v1.0`
> (corpus `sha256:…`, HU [−1000, +1000] → [0, 255], C = …, P = …, p = …)

Mai come giudizio assoluto di qualità diagnostica, e mai confrontato con
valori NIQE della letteratura: sono scale diverse. Il modello non sa cosa sia
una lesione.
