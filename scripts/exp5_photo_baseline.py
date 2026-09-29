#!/usr/bin/env python3
"""Experiment 5 - comparison with a photographic model.

The LIVE pristine model is not used: it is fitted on photographs, and it is
exactly what we are replacing. The baseline is instead a **refit of our own
code** on a corpus of public-domain or CC0 photographs from Wikimedia Commons,
with the same P, C, p. **Only the corpus** changes, so the comparison
isolates the variable of interest; comparing with the LIVE .mat would have
confounded corpus, implementation, kernel sigma and preprocessing into a
single number.

Three models compared on the same CT images:
  1. RIQE, CT corpus, masks on (the model we publish);
  2. RIQE, CT corpus, masks off (ablation: separates the effect of the masks
     from the effect of the corpus);
  3. photographic, CC0/PD photo corpus, masks off.

On top of the qualitative verdicts, RIQE and the photographic model are
compared on reduced-dose ordering and on detection of noise added relative to
the native noise, on the TEST split.

If the photographic model orders CT degradations like ours, modality-specific
fitting is unnecessary, and that is an important negative result to be
published with the same prominence as the other.

The photographic corpus is frozen in corpus/photo_corpus_manifest.json
(title, URL, licence and SHA-256 of every photograph). When the manifest is
present, exactly those files are downloaded instead of harvesting again, since
the contents of a Commons category change over time.

    .venv/bin/python scripts/exp5_photo_baseline.py --n-photos 125
"""

from __future__ import annotations

import os

# One thread per process: parallelism comes from processes, and letting every
# worker open its own BLAS threads causes oversubscription (load 90 on 32
# cores, measured) instead of speed. Must be set before importing numpy.
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

from riqe import degrade as dg  # noqa: E402
from riqe.cache import FeatureCache  # noqa: E402
from riqe import niqecfg  # noqa: E402
from riqe.dicomio import read_hu  # noqa: E402
from riqe.dosetest import dose_ordering, dose_pairs, low_cache_dir, pair_moments, summarize  # noqa: E402
from riqe.evaluate import attach_scores, load_moments, spearman, step_monotone  # noqa: E402
from riqe.extract import Spec, features_from_hu, luminance, masks_for  # noqa: E402
from riqe.model import fit_mvg, model_divergence  # noqa: E402
from riqe.nss import patch_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"
PHOTOS = ROOT / "data" / "photos"
API = "https://commons.wikimedia.org/w/api.php"
#: Wikimedia policy requires a descriptive User-Agent with a contact
#: reference: without it, the API answers 429. We use the public URL of the
#: repository, not a personal address.
UA = "RIQE/0.1 (https://github.com/Metallogik/RIQE) python-urllib"

#: pause between API calls, to avoid rate limiting
API_SLEEP = 1.0

#: accepted licences: public domain and CC0 only, so as not to inherit constraints
PERMISSIVE = ("cc0", "public domain", "pd-", "publicdomain", "cc-zero")

#: The corpus must consist of **natural photographs**: the NIQE paper states
#: explicitly that the statistical model "is violated when the images do not
#: derive from a natural source (e.g. computer graphics)". Public-domain
#: images on Commons, however, are rich in maps, engravings, prints and
#: generated graphics, which are PD precisely because they are old or
#: synthetic. We reject by keywords in categories and title, and accept only
#: JPEG: lossless formats on Commons are almost always rasterised vector
#: graphics, diagrams or animations.
NON_PHOTO = (
    "map", "maps", "cartograph", "atlas", "chart", "diagram", "engraving",
    "etching", "lithograph", "woodcut", "drawing", "painting", "artwork",
    "illustration", "poster", "manuscript", "codex", "book", "page scan",
    "svg", "animation", "animated", "render", "3d model", "fractal", "logo",
    "coat of arms", "flag", "stamp", "banknote", "typography", "font",
    "comic", "cartoon", "graph", "plot", "schematic", "blueprint", "sheet music",
    # intruders observed in the CC0 pool: not graphics, but not natural
    # scenes either, and naturalness is what the NSS assumption requires
    "sculpture", "statue", "radiograph", "x-ray", "calligraphy", "mosaic",
    "stained glass", "tapestry", "medal",
)
ACCEPTED_MIME = ("image/jpeg",)

#: at most this many patches per photograph, so that a 20-megapixel photo
#: does not weigh a hundred times a CT slice in the fit
MAX_PATCHES_PER_PHOTO = 500


def api(params: dict, retries: int = 5) -> dict:
    """Call to the Commons API, with increasing back-off on 429."""
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
            print(f"    HTTP {e.code}, waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def is_permissive(ext: dict) -> tuple[bool, str]:
    lic = (ext.get("LicenseShortName", {}).get("value", "") or "").lower()
    licname = (ext.get("License", {}).get("value", "") or "").lower()
    both = f"{lic} {licname}"
    return any(t in both for t in PERMISSIVE), lic or licname


def is_photograph(title: str, ext: dict, mime: str) -> tuple[bool, str]:
    """Photograph filter, with the rejection reason for the record."""
    if mime not in ACCEPTED_MIME:
        return False, f"mime {mime}"
    cats = (ext.get("Categories", {}).get("value", "") or "").lower()
    hay = f"{title.lower()} {cats}"
    for t in NON_PHOTO:
        if t in hay:
            return False, f"category/title: {t}"
    return True, ""


#: Pool to draw from. NOT "Featured pictures": its public-domain subset is
#: dominated by maps, engravings, coins and share certificates -- those
#: images are PD *because they are old*, so they are document scans, not
#: photographs. Measured: out of 1500 candidates, the few permissive
#: survivors were railway shares and painted portraits. "Quality images" is
#: a selection of modern photographs curated for technical quality, but
#: filtering it by licence gave a yield of 0.5%, measured: the wrong pool.
#: `Category:CC-Zero` is the *licence* category: drawing from it guarantees
#: the licence, and only the photograph filter remains.
POOL_CATEGORY = "Category:CC-Zero"

#: categories that signal a selection for technical quality: recorded, since
#: a pristine corpus should really be pristine
QUALITY_MARKERS = ("quality images", "featured pictures", "valued images")


def harvest(n_wanted: int, seed: int) -> list[dict]:
    """Permissively licensed photographs from Wikimedia Commons."""
    members, cont = [], {}
    #: amply enough to choose 125 after the licence and photograph filters
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
    print(f"candidates from Commons: {len(members)}")

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
                rejected["non-permissive licence"] = rejected.get("non-permissive licence", 0) + 1
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
                print(f"  rejected: {dict(sorted(rejected.items(), key=lambda t:-t[1])[:8])}")
                return out, rejected
        print(f"  collected {len(out)}/{n_wanted}", flush=True)
    print(f"  rejected: {dict(sorted(rejected.items(), key=lambda t:-t[1])[:8])}")
    return out, rejected


def download(rec: dict) -> dict | None:
    """Download one photograph, retrying on 429."""
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
    """Rec.601 luma on a 0-255 scale, float32, not quantised."""
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
    """The qualitative verdicts that should or should not change between models."""
    orig = {r.slice_path: getattr(r, col) for r in
            d[(d.kind == "original") & (d.source == "full")].itertuples()}
    out = {}
    for kind, param in (("noise_white", "sigma_hu"), ("noise_fbp", "sigma_hu"), ("blur", "sigma_px")):
        g = d[(d.kind == kind) & (d.source == "full")]
        if param == "sigma_hu":
            g = g[g[param].isin(dg.NOISE_SIGMAS_HU)]
        g = g.groupby(["slice_path", param], as_index=False).agg(**{col: (col, "mean")})
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
    out["overfilter_failure_fraction"] = float((den.loc[ok, col] < den.loc[ok, "base"]).mean())
    return out


def photo_features_niqe(job):
    """Photograph patches with the published NIQE parameters (riqe/niqecfg.py)."""
    path, seed = job
    lum = luminance_of(ROOT / path)
    if lum is None or min(lum.shape) < 2 * niqecfg.P:
        return None
    f = niqecfg.fitting_features(lum)
    if f.shape[0] > MAX_PATCHES_PER_PHOTO:
        idx = np.random.default_rng(seed).choice(f.shape[0], MAX_PATCHES_PER_PHOTO, replace=False)
        f = f[idx]
    return f.astype(np.float32) if f.shape[0] else None


def niqe_moments_of(path):
    """Whole-image moments of a CT slice with the NIQE parameters."""
    hu, _, _ = read_hu(str(ROOT / path))
    mu, cov, _ = niqecfg.image_moments(luminance(hu, Spec(use_masks=False)))
    return path, (None if mu is None else (mu, cov))


def dose_and_rel_noise(model, pairs, moments_pairs, d, col) -> tuple[dict, pd.DataFrame]:
    """Reduced-dose ordering and relative-noise detection (% correct) for the
    scores in column `col`; also returns the scored pairs."""
    scored = dose_ordering(model, pairs, moments_pairs)
    s = summarize(scored)
    o = d[(d.kind == "original") & (d.source == "full")].set_index("slice_path")[col]
    g = d[(d.kind == "noise_rel") & (d.source == "full")].copy()
    g["base"] = g.slice_path.map(o)
    g = g.dropna(subset=[col, "base"])
    rel = (g.assign(worse=g[col] > g.base).groupby("rel_increase")["worse"].mean() * 100).round(1)
    return ({"dose_chest": round(100 * s["correct_chest"], 1),
             "dose_abdomen": round(100 * s["correct_abdomen"], 1),
             **{f"noise_+{int(round(100 * k))}%": float(v) for k, v in rel.items()}}, scored)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-photos", type=int, default=125)
    ap.add_argument("--moments", default="experiments/bank_test_moments.npz")
    ap.add_argument("--niqe-moments", default="experiments/bank_test_moments_niqecfg.npz",
                    help="bank moments with the NIQE parameters (score_bank.py --niqe-config)")
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
        ap.error("need --P --C --p, or experiments/hparam_choice.json")
    print(f"hyperparameters shared by the three models: P={P} C={C:g} p={p:g}")

    # --- photographic corpus -------------------------------------------------
    man_file = ROOT / "corpus" / "photo_corpus_manifest.json"
    if man_file.exists():
        photos = json.loads(man_file.read_text())["photos"]
        print(f"photographic corpus already present: {len(photos)} images")
        missing = [r for r in photos if not (ROOT / r["path"]).exists()]
        if missing:
            print(f"  downloading {len(missing)} missing files listed in the manifest")
            with cf.ThreadPoolExecutor(8) as ex:
                for r in ex.map(download, missing):
                    if r.get("error") or r.get("sha256") != next(
                            q["sha256"] for q in photos if q["title"] == r["title"]):
                        print(f"  WARNING: {r['title']}: missing or changed upstream")
    else:
        print("harvesting from Wikimedia Commons (public domain / CC0 only)...")
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
        print(f"downloaded {len(photos)} photographs")

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
    print(f"photographic model: {len(parts)} images, {photo_model.n_patches} patches, "
          f"cond={photo_model.cond():.3e}  ({(time.time()-t0)/60:.1f} min)")

    # the same photographs with the parameters published for NIQE
    parts_n = []
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for f in ex.map(photo_features_niqe, [(r["path"], args.seed + i) for i, r in enumerate(photos)],
                        chunksize=2):
            if f is not None:
                parts_n.append(f)
    niqe_model = fit_mvg(np.concatenate(parts_n), n_images=len(parts_n), n_patients=len(parts_n),
                         meta={"corpus": "Wikimedia Commons PD/CC0", "P": niqecfg.P,
                               "C": niqecfg.C, "p": niqecfg.P_SELECT})
    print(f"photographic model with NIQE parameters (P={niqecfg.P}, C={niqecfg.C:g}, "
          f"p={niqecfg.P_SELECT:g}): {len(parts_n)} images, {niqe_model.n_patches} patches")

    # --- CT models, with and without masks ------------------------------------
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"]
                 & corpus.patient_id.isin(set(split["fit"]))]
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
    ct_model = cache.fit(list(fit["path"]), p, n_patients=fit.patient_id.nunique())
    print(f"CT model (masks on): {ct_model.n_patches} patches, cond={ct_model.cond():.3e}")

    t1 = time.time()
    jobs2 = [(q, P, C, p) for q in fit["path"]]
    parts2 = []
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for f in ex.map(ct_features_nomask, jobs2, chunksize=8):
            if f.shape[0]:
                parts2.append(f)
    ct_nomask = fit_mvg(np.concatenate(parts2), n_images=len(parts2),
                        n_patients=fit.patient_id.nunique(),
                        meta={"corpus": "CT, masks off", "P": P, "C": C, "p": p})
    print(f"CT model (masks off): {ct_nomask.n_patches} patches, "
          f"cond={ct_nomask.cond():.3e}  ({(time.time()-t1)/60:.1f} min)")

    # --- divergences between the three ------------------------------------------
    print("\ndivergences D between models (same units as the score):")
    print(f"  CT masks on   vs  CT masks off   : {model_divergence(ct_model, ct_nomask):.4f}")
    print(f"  CT masks on   vs  photographic   : {model_divergence(ct_model, photo_model):.4f}")
    print(f"  CT masks off  vs  photographic   : {model_divergence(ct_nomask, photo_model):.4f}")

    # --- scores on the same CT images -------------------------------------------
    meta, NU, SG = load_moments(args.moments)
    meta = meta[(meta.P == P) & (meta.C == C)].copy()
    d = attach_scores(meta, ct_model, NU, SG, "riqe")
    d = attach_scores(d, ct_nomask, NU, SG, "ct_nomask")
    d = attach_scores(d, photo_model, NU, SG, "photo")
    nmeta, NNU, NSG = load_moments(args.niqe_moments)
    nmeta = attach_scores(nmeta, niqe_model, NNU, NSG, "niqe_params")
    key = ["slice_path", "source", "label"]
    d = d.merge(nmeta[key + ["niqe_params"]].drop_duplicates(key), on=key, how="left")

    rho_ph = spearman(d["riqe"], d["photo"])
    rho_nm = spearman(d["riqe"], d["ct_nomask"])
    rho_nq = spearman(d["riqe"], d["niqe_params"])
    print(f"\nSpearman between rankings on {len(d)} CT images:")
    print(f"  RIQE vs photographic (same settings) : {rho_ph:+.4f}")
    print(f"  RIQE vs NIQE parameters, photographs : {rho_nq:+.4f}")
    print(f"  RIQE vs CT masks off                 : {rho_nm:+.4f}")

    print("\nqualitative verdicts under each model:")
    rows = []
    for name, col in (("RIQE (CT, masks)", "riqe"), ("CT without masks", "ct_nomask"),
                      ("photographic PD/CC0", "photo"), ("NIQE parameters, PD/CC0", "niqe_params")):
        v = verdicts(d, col)
        rows.append({"model": name, **v})
    vt = pd.DataFrame(rows)
    print(vt.round(4).to_string(index=False))

    # --- reduced dose and relative noise ----------------------------------------
    pairs = dose_pairs(corpus, split["test"])
    moments_pairs = pair_moments(cache, FeatureCache(low_cache_dir(P, C)), pairs)
    with cf.ProcessPoolExecutor(args.workers) as ex:
        niqe_pairs = dict(ex.map(niqe_moments_of, list(pairs.path_full) + list(pairs.path_low),
                                 chunksize=8))
    dr, scored = {}, []
    for label, model, mom, col in (("RIQE (CT)", ct_model, moments_pairs, "riqe"),
                                   ("photographic", photo_model, moments_pairs, "photo"),
                                   ("NIQE parameters", niqe_model, niqe_pairs, "niqe_params")):
        dr[label], sc = dose_and_rel_noise(model, pairs, mom, d, col)
        scored.append(sc.assign(model=label))
    print("\nreduced dose ordered correctly and relative noise detected (% of TEST cases):")
    print(pd.DataFrame(dr).T.to_string())
    pd.concat(scored, ignore_index=True).to_csv(OUT / "exp5_dose_pairs.csv", index=False)

    changed = not np.allclose(
        vt.iloc[0].drop("model").astype(float).to_numpy(),
        vt.iloc[2].drop("model").astype(float).to_numpy(), atol=0.02, equal_nan=True)
    concl = ("modality-specific fitting CHANGES the verdicts"
             if changed or abs(rho_ph) < 0.95 else
             "NEGATIVE RESULT: the photographic model orders CT degradations "
             "like ours, and the verdicts do not change")
    print(f"\nCONCLUSION: {concl}")

    d.to_csv(OUT / "exp5_scores_three_models.csv", index=False)
    vt.to_csv(OUT / "exp5_verdicts.csv", index=False)
    np.savez_compressed(OUT / "exp5_photo_model.npz", nu=photo_model.nu, sigma=photo_model.sigma)
    np.savez_compressed(OUT / "exp5_photo_model_niqecfg.npz", nu=niqe_model.nu, sigma=niqe_model.sigma)
    (OUT / "exp5_dose_and_relnoise.json").write_text(json.dumps(dr, indent=1))
    (OUT / "exp5_summary.json").write_text(json.dumps({
        "P": P, "C": C, "p": p,
        "photo_corpus": {"n_images": len(parts), "n_patches": int(photo_model.n_patches),
                         "source": f"Wikimedia Commons {POOL_CATEGORY}, PD/CC0"},
        "niqe_parameters_model": {"P": niqecfg.P, "C": niqecfg.C, "p": niqecfg.P_SELECT,
                                  "n_images": len(parts_n), "n_patches": int(niqe_model.n_patches)},
        "spearman_riqe_vs_niqe_params": rho_nq,
        "D_ct_vs_nomask": model_divergence(ct_model, ct_nomask),
        "D_ct_vs_photo": model_divergence(ct_model, photo_model),
        "D_nomask_vs_photo": model_divergence(ct_nomask, photo_model),
        "spearman_riqe_vs_photo": rho_ph,
        "spearman_riqe_vs_nomask": rho_nm,
        "verdicts": rows,
        "dose_and_rel_noise": dr,
        "conclusion": concl,
    }, indent=1, default=float))
    print(f"\nwrote {OUT}/exp5_*.{{csv,json,npz}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
