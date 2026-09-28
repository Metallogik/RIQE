# RIQE

**R**eference **I**mage **Q**uality **E**valuator per tomografia computerizzata.

Modello di riferimento no-reference in stile NIQE, fittato su TC anziche' su
fotografie naturali, con la validazione che manca al lavoro esistente.

NIQE (Mittal, Soundararajan, Bovik, IEEE SPL 20(3):209-212, 2013) misura
quanto un'immagine devia dalla regolarita' statistica attesa. Il suo modello
di riferimento ufficiale e' adattato su 125 fotografie naturali: applicato a
una TC produce un numero privo di significato clinico. Uno studio del 2025
(*Computers* 14(1):18) ha fatto il fitting su TC ma **non ha pubblicato i
parametri del modello, ne' il codice, ne' la conversione HU -> luminanza**.
Questo progetto colma quel buco, e soprattutto valida il risultato.

Il contributo non e' il file del modello — sono 36 numeri e una matrice 36x36,
chiunque puo' calcolarli. Il contributo e' **la validazione**: dimostrare che
il modello ordina correttamente le degradazioni e che non cade nella trappola
tipica di queste metriche, cioe' premiare un'immagine sovrafiltrata.

Il punteggio va sempre riferito come **distanza relativa al modello
dichiarato**, mai come giudizio assoluto di qualita' diagnostica. Il modello
non sa cosa sia una lesione.

---

## Stato

| Fase | Stato |
|---|---|
| Verifica accesso dati e licenze | fatta, vedi [docs/00](docs/00-data-access-and-license.md) |
| Specifica algoritmo dall'articolo primario | fatta, vedi [docs/02](docs/02-spec-niqe.md) |
| Disegno sperimentale concordato | fatto, vedi [docs/01](docs/01-proposta.md) |
| Estrattore, fitter, degradazioni, banco di prova | implementati |
| Corpus scaricato | 299 serie, 72.588 slice, 36 GB |
| Ricerca iperparametri, fitting, batteria | in corso |

Raffinamenti emersi dall'implementazione, con i numeri che li hanno imposti:
[docs/03](docs/03-scoperte-in-implementazione.md).

## Dati

Il corpus **non e' nel repository** e non deve esserci: sono 36 GB di DICOM.
Nel repository c'e' la sua *identita'* — `corpus/corpus_manifest.json` con UID
e SHA-256 di ogni slice usata — che rende il fit ricostruibile da chi scarica
gli stessi dati pubblici.

Sorgente: TCIA, collezione `LDCT-and-Projection-data`
([DOI 10.7937/9npb-2637](https://doi.org/10.7937/9npb-2637)), versione 7,
sottoinsiemi Chest e Liver, **CC BY 4.0**. Nessun accordo da firmare, nessuna
registrazione: l'API REST pubblica NBIA risponde in anonimo.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pydicom scikit-image matplotlib pandas pyarrow PyWavelets
.venv/bin/python scripts/download_corpus.py          # ~36 GB, ~3,5 h
```

Citazione obbligatoria del dataset e riconoscimento dei finanziamenti: vedi
[docs/00](docs/00-data-access-and-license.md) §1.2.

## Pipeline

```bash
.venv/bin/python scripts/download_corpus.py            # serie di immagini da TCIA
.venv/bin/python scripts/build_slice_table.py          # metadati + statistiche per slice
.venv/bin/python scripts/make_corpus.py                # criteri S1-S6, partizione, manifest
.venv/bin/python scripts/cache_features.py             # feature per patch, per (P, C)
.venv/bin/python scripts/prepare_bank.py --split val_inner
.venv/bin/python scripts/score_bank.py --bank experiments/bank_val_inner.json
.venv/bin/python scripts/search_hparams.py --moments experiments/bank_val_inner_moments.npz
```

La partizione e' **per paziente**, mai per slice, e TEST resta congelato fino
alla batteria finale.

## Pacchetto

| Modulo | Contenuto |
|---|---|
| [riqe/nss.py](riqe/nss.py) | MSCN, stime GGD/AGGD per momento, 36 feature per patch, selezione per nitidezza |
| [riqe/hu.py](riqe/hu.py) | mappatura canonica HU -> luminanza, maschere FOV e corpo |
| [riqe/extract.py](riqe/extract.py) | `Spec`: tutto cio' che cambia i numeri, in un solo posto |
| [riqe/model.py](riqe/model.py) | modello MVG, punteggio, divergenza fra modelli |
| [riqe/inclusion.py](riqe/inclusion.py) | criteri S1-S6, come operazioni pure su tabella |
| [riqe/degrade.py](riqe/degrade.py) | rumore bianco e tipo FBP, sfocatura, 5 denoiser calibrati, d' e ritenzione di segnale |
| [riqe/bank.py](riqe/bank.py) | banco di prova come ricette deterministiche, non pixel |
| [riqe/cache.py](riqe/cache.py) | cache delle feature con soglia `p` variabile a valle |

## Vincoli di licenza rispettati

- **pyiqa / IQA-PyTorch non usati** (PolyForm Noncommercial + NTU S-Lab,
  incompatibili con un rilascio permissivo).
- **Modello pristine di LIVE non usato**, nemmeno come default o riferimento
  di comodo: e' fittato su fotografie, ed e' cio' che stiamo sostituendo. La
  baseline fotografica di confronto e' un **ri-fitting del nostro stesso
  codice** su fotografie CC0/pubblico dominio.
- Algoritmo implementato **dall'articolo**, senza incorporare il codice MATLAB
  di LIVE. L'articolo e' citato come fonte dell'algoritmo.
- Dipendenze: numpy/scipy/scikit-image (BSD-3), pydicom (MIT), PyWavelets
  (MIT). Nessun BM3D (GPL o licenza incerta).

## Licenza

Codice: [MIT](LICENSE). Modello, manifest e risultati: CC BY 4.0
(vedi [LICENSE-MODEL](LICENSE-MODEL)), che compone con l'obbligo di
attribuzione ereditato dal dataset TCIA senza aggiungere restrizioni.
