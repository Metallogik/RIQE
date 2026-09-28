#!/usr/bin/env python3
"""Esperimento 5 - confronto con un modello fotografico.

Il modello pristine di LIVE non viene usato: e' fittato su fotografie, ed e'
esattamente cio' che stiamo sostituendo.  La baseline e' invece un
**ri-fitting del nostro stesso codice** su un corpus di fotografie in
pubblico dominio o CC0 da Wikimedia Commons, con gli stessi P, C, p.  Cambia
**solo il corpus**, quindi il confronto isola la variabile che interessa;
confrontarsi con il .mat di LIVE avrebbe confuso corpus, implementazione,
sigma del kernel e pre-elaborazione in un unico numero.

Tre modelli a confronto sulle stesse immagini TC:
  1. RIQE, corpus TC, maschere attive (il modello che pubblichiamo);
  2. RIQE, corpus TC, maschere disattivate (ablazione: isola l'effetto delle
     maschere dall'effetto del corpus);
  3. fotografico, corpus di fotografie CC0/PD, maschere disattivate.

Se il modello fotografico ordina le degradazioni TC come il nostro, il
fitting specifico per modalita' non serve, ed e' un risultato negativo
importante che va pubblicato con lo stesso rilievo dell'altro.

    .venv/bin/python scripts/exp5_photo_baseline.py --n-photos 125
"""

from __future__ import annotations

import os

# Un thread per processo: il parallelismo lo diamo con i processi, e lasciare
# che ogni worker apra i propri thread BLAS porta a oversubscription (load 90
# su 32 core, misurato) invece che a velocita'. Va fatto prima di numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.cache import FeatureCache  # noqa: E402
from riqe.dicomio import read_hu  # noqa: E402
from riqe.evaluate import attach_scores, load_moments, spearman, step_monotone  # noqa: E402
from riqe.extract import Spec, features_from_hu, masks_for  # noqa: E402
from riqe.model import fit_mvg, model_divergence  # noqa: E402
from riqe.nss import patch_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"
PHOTOS = ROOT / "data" / "photos"
API = "https://commons.wikimedia.org/w/api.php"
#: la policy di Wikimedia richiede uno User-Agent descrittivo con un
#: riferimento di contatto: senza, l'API risponde 429.  Usiamo l'URL pubblico
#: del repository, non un indirizzo personale.
UA = "RIQE/0.1 (https://github.com/Metallogik/RIQE) python-urllib"

#: pausa fra chiamate all'API, per non farsi limitare
API_SLEEP = 1.0

#: licenze accettate: solo pubblico dominio e CC0, per non ereditare vincoli
PERMISSIVE = ("cc0", "public domain", "pd-", "publicdomain", "cc-zero")

#: Il corpus deve essere di **fotografie naturali**: l'articolo NIQE dice
#: esplicitamente che il modello statistico "is violated when the images do
#: not derive from a natural source (e.g. computer graphics)".  Le immagini
#: in pubblico dominio di Commons sono pero' ricche di mappe, incisioni,
#: stampe e grafica generata, che sono PD proprio perche' antiche o
#: sintetiche.  Scartiamo per parole chiave nelle categorie e nel titolo, e
#: accettiamo solo JPEG: i formati senza perdita su Commons sono quasi sempre
#: grafica vettoriale rasterizzata, diagrammi o animazioni.
NON_PHOTO = (
    "map", "maps", "cartograph", "atlas", "chart", "diagram", "engraving",
    "etching", "lithograph", "woodcut", "drawing", "painting", "artwork",
    "illustration", "poster", "manuscript", "codex", "book", "page scan",
    "svg", "animation", "animated", "render", "3d model", "fractal", "logo",
    "coat of arms", "flag", "stamp", "banknote", "typography", "font",
    "comic", "cartoon", "graph", "plot", "schematic", "blueprint", "sheet music",
    # intrusi osservati nel bacino CC0: non sono grafica, ma non sono
    # nemmeno scene naturali, ed e' la naturalita' che l'assunzione NSS
    # richiede
    "sculpture", "statue", "radiograph", "x-ray", "calligraphy", "mosaic",
    "stained glass", "tapestry", "medal",
)
ACCEPTED_MIME = ("image/jpeg",)

#: al massimo questo numero di patch per fotografia, perche' una foto da 20
#: megapixel non pesi cento volte una slice TC nel fitting
MAX_PATCHES_PER_PHOTO = 500


def api(params: dict, retries: int = 5) -> dict:
    """Chiamata all'API di Commons, con attesa crescente sui 429."""
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            time.sleep(API_SLEEP)
            return out
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == retries - 1:
                raise
            print(f"    HTTP {e.code}, attendo {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("irraggiungibile")


def is_permissive(ext: dict) -> tuple[bool, str]:
    lic = (ext.get("LicenseShortName", {}).get("value", "") or "").lower()
    licname = (ext.get("License", {}).get("value", "") or "").lower()
    both = f"{lic} {licname}"
    return any(t in both for t in PERMISSIVE), lic or licname


def is_photograph(title: str, ext: dict, mime: str) -> tuple[bool, str]:
    """Filtro fotografico, con il motivo del rifiuto per la registrazione."""
    if mime not in ACCEPTED_MIME:
        return False, f"mime {mime}"
    cats = (ext.get("Categories", {}).get("value", "") or "").lower()
    hay = f"{title.lower()} {cats}"
    for t in NON_PHOTO:
        if t in hay:
            return False, f"categoria/titolo: {t}"
    return True, ""


#: Bacino da cui pescare.  NON "Featured pictures": il suo sottoinsieme in
#: pubblico dominio e' dominato da mappe, incisioni, monete e certificati
#: azionari -- quelle immagini sono PD *perche' antiche*, quindi sono
#: scansioni di documenti, non fotografie.  Misurato: su 1500 candidati, i
#: pochi permissivi superstiti erano azioni ferroviarie e ritratti dipinti.
#: "Quality images" e' invece una selezione di fotografie moderne curata per
#: qualita' tecnica, che e' esattamente cio' che serve a un corpus pristine.
#: `Category:CC-Zero` e' la categoria di *licenza*: pescando da li' la
#: licenza e' garantita al 100%, e resta solo da filtrare le fotografie.
#: Pescare da "Quality images" e filtrare per licenza dava una resa dello
#: 0,5%, misurata: il bacino sbagliato.
POOL_CATEGORY = "Category:CC-Zero"

#: categorie che segnalano una selezione per qualita' tecnica: a parita' di
#: licenza si preferiscono, perche' un corpus pristine deve esserlo davvero
QUALITY_MARKERS = ("quality images", "featured pictures", "valued images")


def harvest(n_wanted: int, seed: int) -> list[dict]:
    """Fotografie di qualita' con licenza permissiva da Wikimedia Commons."""
    members, cont = [], {}
    #: bastano ampiamente per sceglierne 125 dopo il filtro di licenza
    target_pool = max(4000, n_wanted * 30)
    for _ in range(20):
        r = api({"action": "query", "list": "categorymembers",
                 "cmtitle": POOL_CATEGORY,
                 "cmtype": "file", "cmlimit": "500", **cont})
        members += [m["title"] for m in r["query"]["categorymembers"]]
        if "continue" not in r or len(members) >= target_pool:
            break
        cont = r["continue"]
    rng = np.random.default_rng(seed)
    rng.shuffle(members)
    print(f"candidati da Commons: {len(members)}")

    out, rejected = [], {}
    for i in range(0, len(members), 40):
        batch = members[i : i + 40]
        r = api({"action": "query", "titles": "|".join(batch), "prop": "imageinfo",
                 "iiprop": "url|extmetadata|size|mime", "iiurlwidth": "1600"})
        for pg in r.get("query", {}).get("pages", {}).values():
            ii = (pg.get("imageinfo") or [None])[0]
            if not ii or not ii.get("mime", "").startswith("image/"):
                continue
            ext = ii.get("extmetadata", {})
            ok, lic = is_permissive(ext)
            if not ok:
                rejected["licenza non permissiva"] = rejected.get("licenza non permissiva", 0) + 1
                continue
            isphoto, why = is_photograph(pg["title"], ext, ii.get("mime", ""))
            if not isphoto:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            cats = (ext.get("Categories", {}).get("value", "") or "").lower()
            out.append({"title": pg["title"], "license": lic,
                        "url": ii.get("thumburl") or ii["url"],
                        "descriptionurl": ii.get("descriptionurl", ""),
                        "width": ii.get("thumbwidth", ii.get("width")),
                        "height": ii.get("thumbheight", ii.get("height")),
                        "quality_marked": any(q in cats for q in QUALITY_MARKERS)})
            if len(out) >= n_wanted:
                print(f"  scartate: {dict(sorted(rejected.items(), key=lambda t:-t[1])[:8])}")
                return out, rejected
        print(f"  raccolte {len(out)}/{n_wanted}", flush=True)
    print(f"  scartate: {dict(sorted(rejected.items(), key=lambda t:-t[1])[:8])}")
    return out, rejected


def download(rec: dict) -> dict | None:
    """Scarica una fotografia, con ritentativi su 429."""
    PHOTOS.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(rec["title"].encode()).hexdigest()[:16] + ".img"
    path = PHOTOS / name
    if not path.exists():
        req = urllib.request.Request(rec["url"], headers={"User-Agent": UA})
        delay, last = 3.0, None
        for _ in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    path.write_bytes(r.read())
                last = None
                break
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(delay)
                delay *= 2
        if last is not None:
            return {**rec, "error": str(last)}
    b = path.read_bytes()
    return {**rec, "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(b).hexdigest(),
            "bytes": len(b)}


def luminance_of(path: Path) -> np.ndarray | None:
    """Luma Rec.601 su scala 0-255, float32, senza quantizzare."""
    from PIL import Image

    try:
        im = Image.open(path)
        im = im.convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    a = np.asarray(im, dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def photo_features(job):
    path, P, C, p, seed = job
    lum = luminance_of(ROOT / path)
    if lum is None or min(lum.shape) < 4 * P:
        return None
    pf = patch_features(lum, P=P, C=C, fov=None, body=None, p=p)
    f = pf.selected
    if f.shape[0] == 0:
        return None
    if f.shape[0] > MAX_PATCHES_PER_PHOTO:
        idx = np.random.default_rng(seed).choice(f.shape[0], MAX_PATCHES_PER_PHOTO, replace=False)
        f = f[idx]
    return f.astype(np.float32)


def ct_features_nomask(job):
    path, P, C, p = job
    hu, pad, _ = read_hu(str(ROOT / path))
    pf = features_from_hu(hu, pad, Spec(P=P, C=C, p=p, use_masks=False), fitting=True)
    return pf.selected.astype(np.float32)


def verdicts(d: pd.DataFrame, col: str) -> dict:
    """I verdetti qualitativi che devono o non devono cambiare fra modelli."""
    orig = {r.slice_path: getattr(r, col) for r in
            d[(d.kind == "original") & (d.source == "full")].itertuples()}
    out = {}
    for kind, param in (("noise_white", "sigma_hu"), ("noise_fbp", "sigma_hu"), ("blur", "sigma_px")):
        g = d[(d.kind == kind) & (d.source == "full")]
        perfect = total = 0
        for sp, gg in g.groupby("slice_path"):
            gg = gg.sort_values(param)
            seq = np.concatenate([[orig.get(sp, np.nan)], gg[col].to_numpy()])
            if np.isfinite(seq).all():
                total += 1
                perfect += int(step_monotone(seq))
        out[f"monotone_{kind}"] = perfect / total if total else np.nan
    den = d[(d.kind == "denoise") & (d.source == "full")].copy()
    den["base"] = den["slice_path"].map(orig)
    ok = den[col].notna() & den["base"].notna()
    out["frazione_fallimenti_sovrafiltraggio"] = float((den.loc[ok, col] < den.loc[ok, "base"]).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-photos", type=int, default=125)
    ap.add_argument("--moments", default="experiments/bank_test_moments.npz")
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    ch_file = ROOT / "experiments" / "hparam_choice.json"
    ch = json.loads(ch_file.read_text()) if ch_file.exists() else {}
    P = args.P if args.P is not None else ch.get("P")
    C = args.C if args.C is not None else ch.get("C")
    p = args.p if args.p is not None else ch.get("p")
    if P is None:
        ap.error("servono --P --C --p, oppure experiments/hparam_choice.json")
    print(f"iperparametri comuni ai tre modelli: P={P} C={C:g} p={p:g}")

    # --- corpus fotografico ------------------------------------------------
    man_file = OUT / "photo_corpus_manifest.json"
    if man_file.exists():
        photos = json.loads(man_file.read_text())["photos"]
        print(f"corpus fotografico gia' presente: {len(photos)} immagini")
    else:
        print("raccolta da Wikimedia Commons (solo pubblico dominio / CC0)...")
        recs, rejected = harvest(args.n_photos, args.seed)
        photos = []
        with cf.ThreadPoolExecutor(8) as ex:
            for r in ex.map(download, recs):
                if r and "error" not in r:
                    photos.append(r)
        man_file.write_text(json.dumps({
            "source": f"Wikimedia Commons, {POOL_CATEGORY}",
            "license_filter": list(PERMISSIVE),
            "photo_filter": {"rejected_keywords": list(NON_PHOTO),
                             "accepted_mime": list(ACCEPTED_MIME),
                             "rejection_counts": rejected},
            "n": len(photos), "seed": args.seed, "photos": photos}, indent=1))
        print(f"scaricate {len(photos)} fotografie")

    t0 = time.time()
    jobs = [(r["path"], P, C, p, args.seed + i) for i, r in enumerate(photos)]
    parts = []
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for f in ex.map(photo_features, jobs, chunksize=2):
            if f is not None and f.shape[0]:
                parts.append(f)
    X = np.concatenate(parts)
    photo_model = fit_mvg(X, n_images=len(parts), n_patients=len(parts),
                          meta={"corpus": "Wikimedia Commons PD/CC0", "P": P, "C": C, "p": p})
    print(f"modello fotografico: {len(parts)} immagini, {photo_model.n_patches} patch, "
          f"cond={photo_model.cond():.3e}  ({(time.time()-t0)/60:.1f} min)")

    # --- modelli TC, con e senza maschere ---------------------------------
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"]
                 & corpus.patient_id.isin(set(split["fit"]))]
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
    ct_model = cache.fit(list(fit["path"]), p, n_patients=fit.patient_id.nunique())
    print(f"modello TC (maschere attive): {ct_model.n_patches} patch, cond={ct_model.cond():.3e}")

    t1 = time.time()
    jobs2 = [(q, P, C, p) for q in fit["path"]]
    parts2 = []
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for f in ex.map(ct_features_nomask, jobs2, chunksize=8):
            if f.shape[0]:
                parts2.append(f)
    ct_nomask = fit_mvg(np.concatenate(parts2), n_images=len(parts2),
                        n_patients=fit.patient_id.nunique(),
                        meta={"corpus": "TC, maschere disattivate", "P": P, "C": C, "p": p})
    print(f"modello TC (maschere disattivate): {ct_nomask.n_patches} patch, "
          f"cond={ct_nomask.cond():.3e}  ({(time.time()-t1)/60:.1f} min)")

    # --- divergenze fra i tre ---------------------------------------------
    print("\ndivergenze D fra modelli (stesse unita' del punteggio):")
    print(f"  TC maschere attive  vs  TC senza maschere : {model_divergence(ct_model, ct_nomask):.4f}")
    print(f"  TC maschere attive  vs  fotografico       : {model_divergence(ct_model, photo_model):.4f}")
    print(f"  TC senza maschere   vs  fotografico       : {model_divergence(ct_nomask, photo_model):.4f}")

    # --- punteggi sulle stesse immagini TC --------------------------------
    meta, NU, SG = load_moments(args.moments)
    meta = meta[(meta.P == P) & (meta.C == C)].copy()
    d = attach_scores(meta, ct_model, NU, SG, "riqe")
    d = attach_scores(d, ct_nomask, NU, SG, "ct_nomask")
    d = attach_scores(d, photo_model, NU, SG, "photo")

    rho_ph = spearman(d["riqe"], d["photo"])
    rho_nm = spearman(d["riqe"], d["ct_nomask"])
    print(f"\nSpearman fra ordinamenti su {len(d)} immagini TC:")
    print(f"  RIQE vs fotografico    : {rho_ph:+.4f}")
    print(f"  RIQE vs TC senza masch.: {rho_nm:+.4f}")

    print("\nverdetti qualitativi sotto ciascun modello:")
    rows = []
    for name, col in (("RIQE (TC, maschere)", "riqe"), ("TC senza maschere", "ct_nomask"),
                      ("fotografico PD/CC0", "photo")):
        v = verdicts(d, col)
        rows.append({"modello": name, **v})
    vt = pd.DataFrame(rows)
    print(vt.round(4).to_string(index=False))

    changed = not np.allclose(
        vt.iloc[0].drop("modello").astype(float).to_numpy(),
        vt.iloc[2].drop("modello").astype(float).to_numpy(), atol=0.02, equal_nan=True)
    concl = ("il fitting specifico per modalita' CAMBIA i verdetti"
             if changed or abs(rho_ph) < 0.95 else
             "RISULTATO NEGATIVO: il modello fotografico ordina le degradazioni TC "
             "come il nostro, e i verdetti non cambiano")
    print(f"\nCONCLUSIONE: {concl}")

    d.to_csv(OUT / "exp5_scores_three_models.csv", index=False)
    vt.to_csv(OUT / "exp5_verdicts.csv", index=False)
    np.savez_compressed(OUT / "exp5_photo_model.npz", nu=photo_model.nu, sigma=photo_model.sigma)
    (OUT / "exp5_summary.json").write_text(json.dumps({
        "P": P, "C": C, "p": p,
        "photo_corpus": {"n_images": len(parts), "n_patches": int(photo_model.n_patches),
                         "source": f"Wikimedia Commons {POOL_CATEGORY}, PD/CC0"},
        "D_ct_vs_nomask": model_divergence(ct_model, ct_nomask),
        "D_ct_vs_photo": model_divergence(ct_model, photo_model),
        "D_nomask_vs_photo": model_divergence(ct_nomask, photo_model),
        "spearman_riqe_vs_photo": rho_ph,
        "spearman_riqe_vs_nomask": rho_nm,
        "verdicts": rows,
        "conclusione": concl,
    }, indent=1, default=float))
    print(f"\nscritti {OUT}/exp5_*.{{csv,json,npz}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
