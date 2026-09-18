"""Build ONE cohort table that every later step reads.

The models are trained on SMDG-19, but from here on the project reports a single
unified cohort: every eye is pushed through the complete pipeline
(U-Net -> post-processing -> CDR -> EfficientNet-B0 risk) and joined, in one row,
with whatever reference data exists for it - expert disc/cup contours, clinical
diagnosis, IOP, age and the Humphrey 30-2 visual-field Mean Defect.

    python scripts/build_unified_cohort.py                 # PAPILA + SMDG test eyes
    python scripts/build_unified_cohort.py --sources papila
    python scripts/build_unified_cohort.py --limit 40      # quick smoke run

Output: data/cohort_master.csv  - columns:
    source split image patient eye age iop diagnosis label
    vcdr hcdr acdr expert_vcdr n_experts dice_disc dice_cup
    glaucoma_probability mean_defect
Rows with mean_defect are the structure-function subset; rows with expert
contours or masks are the segmentation-accuracy subset. Nothing else in the
project reads the two datasets separately any more.
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config  # noqa: E402
from src import papila as pap  # noqa: E402
from src.cdr import compute_cdr  # noqa: E402
from src.inference import VisionTrackPipeline  # noqa: E402
from src.metrics import dice  # noqa: E402
from src.preprocessing import read_rgb  # noqa: E402

COLUMNS = ["source", "split", "image", "path", "patient", "eye", "age", "iop", "axial_length",
           "pachymetry", "diagnosis", "label",
           "vcdr", "hcdr", "acdr", "expert_vcdr", "n_experts", "dice_disc", "dice_cup",
           "rim_inferior", "rim_superior", "rim_nasal", "rim_temporal", "min_rim_ratio",
           "isnt_respected", "rim_to_disc_area", "disc_area_mm2", "rim_area_mm2",
           "glaucoma_probability", "prediction", "mean_defect"]


def rim_columns(result):
    """Flatten the rim / physical blocks into cohort columns."""
    c1 = result.get("component1") or {}
    rim, phys = c1.get("rim") or {}, c1.get("physical") or {}
    ratio = rim.get("sector_rim_ratio") or {}
    return {
        "rim_inferior": ratio.get("inferior"), "rim_superior": ratio.get("superior"),
        "rim_nasal": ratio.get("nasal"), "rim_temporal": ratio.get("temporal"),
        "min_rim_ratio": rim.get("min_rim_to_disc_ratio"),
        "isnt_respected": rim.get("isnt_respected"),
        "rim_to_disc_area": rim.get("rim_to_disc_area_ratio"),
        "disc_area_mm2": phys.get("disc_area_mm2"), "rim_area_mm2": phys.get("rim_area_mm2"),
    }


def run_one(pipe, path, eye="OD", axial_length=None):
    rgb = read_rgb(path)
    res, disc, cup = pipe.analyze(rgb, Path(path).stem, eye=eye, axial_length_mm=axial_length)
    cdr = res["component1"]["cdr"]
    cls = res["classification"] or {}
    return rgb, disc, cup, cdr, cls, res


def do_papila(pipe, root, limit=None):
    root = Path(root)
    if not root.exists():
        print(f"[skip] PAPILA not found at {root}")
        return []
    images = pap.image_index(root)
    clin = pap.clinical(root)
    contours = pap.contour_files(root)
    print(f"PAPILA: {len(images)} images, {len(clin)} clinical records, {len(contours)} contoured eyes")
    rows, names = [], sorted(images)[: limit or None]
    for name in tqdm(names, desc="PAPILA"):
        m = pap.NAME.match(name)
        if not m:
            continue
        pid, eye = int(m.group(1)), m.group(2).upper()
        path = images[name]
        c = clin.get((pid, eye), {})
        al = c.get("axial_length")
        al = None if al is None or pd.isna(al) else float(al)
        rgb, disc, cup, cdr, cls, res = run_one(pipe, path, eye, al)
        e_disc, e_cup, n_exp = pap.expert_masks(contours.get((pid, eye), {}), rgb.shape[:2])
        expert_vcdr = dd = dc = None
        if e_disc is not None:
            expert_vcdr = compute_cdr(e_disc, e_cup)["vertical_cdr"]
            dd, dc = dice(disc, e_disc), dice(cup, e_cup)
        dx = c.get("diagnosis")
        label = None if dx is None or pd.isna(dx) else (1 if int(dx) == 1 else (0 if int(dx) == 0 else None))
        rows.append({
            **rim_columns(res),
            "source": "PAPILA", "split": "cohort", "image": path.name, "path": str(path),
            "patient": f"PAPILA-{pid:03d}", "eye": eye,
            "age": c.get("age"), "iop": c.get("iop"), "axial_length": al,
            "pachymetry": c.get("pachymetry"), "diagnosis": dx, "label": label,
            "vcdr": cdr["vertical_cdr"], "hcdr": cdr["horizontal_cdr"], "acdr": cdr["area_cdr"],
            "expert_vcdr": expert_vcdr, "n_experts": n_exp, "dice_disc": dd, "dice_cup": dc,
            "glaucoma_probability": cls.get("glaucoma_probability"), "prediction": cls.get("prediction"),
            "mean_defect": c.get("mean_defect"),
        })
    return rows


def do_smdg(pipe, splits_dir, limit=None):
    splits_dir = Path(splits_dir)
    frames = []
    for kind in ("seg", "cls"):
        f = splits_dir / f"{kind}_test.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["in_seg_test"] = kind == "seg"
            frames.append(d)
    if not frames:
        print(f"[skip] no test splits in {splits_dir}")
        return []
    df = pd.concat(frames, ignore_index=True)
    df["disc"] = df.get("disc", "").fillna("")
    df["cup"] = df.get("cup", "").fillna("")
    df = df.sort_values("in_seg_test", ascending=False).drop_duplicates("name").reset_index(drop=True)
    if limit:
        df = df.head(limit)
    print(f"SMDG held-out test eyes: {len(df)} ({int(df.in_seg_test.sum())} with expert masks)")
    rows = []
    for r in tqdm(list(df.itertuples()), desc="SMDG"):
        path = Path(r.fundus)
        if not path.exists():
            continue
        rgb, disc, cup, cdr, cls, res = run_one(pipe, path)
        expert_vcdr = dd = dc = None
        n_exp = 0
        if r.disc and r.cup and Path(r.disc).exists() and Path(r.cup).exists():
            gd = (cv2.imread(str(r.disc), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            gc = (cv2.imread(str(r.cup), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            if gd.shape != disc.shape:
                gd = cv2.resize(gd, (disc.shape[1], disc.shape[0]), interpolation=cv2.INTER_NEAREST)
                gc = cv2.resize(gc, (cup.shape[1], cup.shape[0]), interpolation=cv2.INTER_NEAREST)
            if gd.sum() > 0:
                expert_vcdr = compute_cdr(gd, gc)["vertical_cdr"]
                dd, dc = dice(disc, gd), dice(cup, gc)
                n_exp = 1
        label = int(r.label) if r.label in (0, 1) else None
        rows.append({
            **rim_columns(res),
            "source": "SMDG", "split": "test", "image": path.name, "path": str(path),
            "patient": f"SMDG-{r.name}", "eye": "NA",
            "age": None, "iop": None, "axial_length": None, "pachymetry": None,
            "diagnosis": label, "label": label,
            "vcdr": cdr["vertical_cdr"], "hcdr": cdr["horizontal_cdr"], "acdr": cdr["area_cdr"],
            "expert_vcdr": expert_vcdr, "n_experts": n_exp, "dice_disc": dd, "dice_cup": dc,
            "glaucoma_probability": cls.get("glaucoma_probability"), "prediction": cls.get("prediction"),
            "mean_defect": None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papila", default=str(config.PAPILA_DIR))
    ap.add_argument("--splits", default=str(config.SPLITS_DIR))
    ap.add_argument("--sources", default="papila,smdg")
    ap.add_argument("--limit", type=int, default=None, help="cap images per source (smoke test)")
    ap.add_argument("--out", default=str(config.ROOT / "data" / "cohort_master.csv"))
    args = ap.parse_args()

    pipe = VisionTrackPipeline()
    print(f"device: {pipe.device}  thresholds: {pipe.thr}")
    want = {s.strip().lower() for s in args.sources.split(",")}
    rows = []
    if "papila" in want:
        rows += do_papila(pipe, args.papila, args.limit)
    if "smdg" in want:
        rows += do_smdg(pipe, args.splits, args.limit)

    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\ncohort rows: {len(df)}")
    print(df.groupby("source").agg(eyes=("image", "count"), with_md=("mean_defect", "count"),
                                   with_expert=("expert_vcdr", "count"),
                                   with_label=("label", "count")).to_string())
    if df.mean_defect.notna().any():
        v = df.dropna(subset=["mean_defect", "vcdr"])
        print(f"structure-function subset: n = {len(v)}, r(CDR, MD) = {v.vcdr.corr(v.mean_defect):.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
