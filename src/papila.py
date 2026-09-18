"""PAPILA readers: clinical sheets, expert disc/cup contours, image index.

Factored out of scripts/validate_structure_function.py so the unified-cohort
builder and the evaluation share exactly the same parsing.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
CONTOUR = re.compile(r"RET(\d{3})_?(OD|OS)_?(cup|disc)_?exp(\d)", re.I)
NAME = re.compile(r"^RET(\d{3})(OD|OS)$", re.I)


def image_index(root):
    """{'RET002OD': Path(...)} for every PAPILA fundus image."""
    root = Path(root)
    img_dir = next((d for d in root.rglob("FundusImages") if d.is_dir()), root)
    return {p.stem.upper(): p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT}


def _read_sheet(xlsx):
    raw = pd.read_excel(xlsx, header=None)

    def is_header(r):
        cells = [str(v).strip().lower() for v in r.values]
        return any(c == "id" or "id" in c.split() for c in cells) or \
            (any("age" in c for c in cells) and any("diagn" in c for c in cells))

    hdr = next((i for i, r in raw.iterrows() if is_header(r)), 0)
    df = pd.read_excel(xlsx, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _pick(df, *keys, exact=False):
    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("_", "")
        if (lc in keys) if exact else any(k in lc for k in keys):
            return c
    return None


def clinical(root):
    """{(patient, eye): {age, diagnosis, iop, mean_defect}} from the ClinicalData sheets."""
    root = Path(root)
    clin_dir = next((d for d in root.rglob("ClinicalData") if d.is_dir()), root)
    out = {}
    for x in sorted(clin_dir.glob("*.xls*")):
        eye = "OS" if "os" in x.stem.lower() else "OD"
        df = _read_sheet(x)
        c_id = _pick(df, "id", "patientid", "patient", exact=True) or df.columns[0]
        c_md = _pick(df, "meandefect", "vfmd", "md", exact=True) or _pick(df, "meandefect", "defect")
        c_dx, c_age = _pick(df, "diagnosis"), _pick(df, "age")
        c_iop = _pick(df, "pneumatic", "iop")
        c_al = _pick(df, "axiallength", "axial")
        c_pach = _pick(df, "pachymetry")
        c_sex = _pick(df, "gender", "sex")
        for _, r in df.iterrows():
            digits = "".join(ch for ch in str(r[c_id]) if ch.isdigit())
            if not digits:
                continue
            md = pd.to_numeric(str(r[c_md]).replace(",", "."), errors="coerce") if c_md else None
            out[(int(digits), eye)] = {
                "age": pd.to_numeric(r[c_age], errors="coerce") if c_age else None,
                "diagnosis": pd.to_numeric(r[c_dx], errors="coerce") if c_dx else None,
                "iop": pd.to_numeric(r[c_iop], errors="coerce") if c_iop else None,
                "mean_defect": None if md is None or pd.isna(md) else float(md),
                "axial_length": pd.to_numeric(r[c_al], errors="coerce") if c_al else None,
                "pachymetry": pd.to_numeric(r[c_pach], errors="coerce") if c_pach else None,
                "gender": r[c_sex] if c_sex else None,
            }
    return out


def _read_contour(path):
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", path.read_text(errors="ignore"))]
    if len(nums) < 6:
        return None
    return np.array(nums[: len(nums) // 2 * 2]).reshape(-1, 2)


def contour_files(root):
    """{(patient, eye): {expert: {'disc': Path, 'cup': Path}}}"""
    files = {}
    for f in Path(root).rglob("*"):
        if f.is_file() and f.suffix.lower() in (".txt", ".csv", ".dat", ""):
            m = CONTOUR.search(f.name)
            if m:
                key = (int(m.group(1)), m.group(2).upper())
                files.setdefault(key, {}).setdefault(m.group(4), {})[m.group(3).lower()] = f
    return files


def expert_masks(experts, shape):
    """Mean expert disc/cup masks (majority vote) plus each expert's vertical CDR."""
    h, w = shape
    discs, cups = [], []
    for parts in experts.values():
        if "cup" not in parts or "disc" not in parts:
            continue
        m = {}
        for k in ("disc", "cup"):
            pts = _read_contour(parts[k])
            if pts is None:
                break
            mk = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mk, [np.round(pts).astype(np.int32)], 1)
            m[k] = mk
        if len(m) == 2 and m["disc"].sum() > 0:
            discs.append(m["disc"])
            cups.append(m["cup"])
    if not discs:
        return None, None, 0
    n = len(discs)
    disc = (np.mean(discs, axis=0) >= 0.5).astype(np.uint8)
    cup = (np.mean(cups, axis=0) >= 0.5).astype(np.uint8)
    return disc, cup, n


__all__ = ["image_index", "clinical", "contour_files", "expert_masks", "NAME", "IMG_EXT"]
