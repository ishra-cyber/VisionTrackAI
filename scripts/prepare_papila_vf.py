"""Build the visual-field matched dataset for Component 2 (next phase).

Runs Component 1 on PAPILA fundus images and joins the predicted CDR with the
clinical data (diagnosis, IOP, Mean Defect of the visual field).

    python scripts/prepare_papila_vf.py --papila data/papila
Output: data/papila_vf_matched.csv  (only rows with a Mean Defect value)
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import pandas as pd

import config
from src.inference import VisionTrackPipeline
from src.preprocessing import read_rgb


def read_clinical(xlsx):
    raw = pd.read_excel(xlsx, header=None)
    def is_header(r):
        cells = [str(v).strip().lower() for v in r.values]
        return any(c == "id" or "id" in c.split() for c in cells) or \
            (any("age" in c for c in cells) and any("diagn" in c for c in cells))
    hdr = next((i for i, r in raw.iterrows() if is_header(r)), 0)
    df = pd.read_excel(xlsx, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def pick(df, *keys, exact=False):
    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("_", "")
        if (lc in keys) if exact else any(k in lc for k in keys):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papila", default=str(config.PAPILA_DIR))
    ap.add_argument("--out", default=str(config.ROOT / "data" / "papila_vf_matched.csv"))
    args = ap.parse_args()
    root = Path(args.papila)
    img_dir = next((d for d in root.rglob("FundusImages") if d.is_dir()), root)
    clin_dir = next((d for d in root.rglob("ClinicalData") if d.is_dir()), root)
    pipe = VisionTrackPipeline()

    rows = []
    for x in sorted(clin_dir.glob("*.xls*")):
        eye = "OS" if "os" in x.stem.lower() else "OD"
        df = read_clinical(x)
        c_id = pick(df, "id", "patientid", "patient", exact=True) or df.columns[0]
        c_md = pick(df, "meandefect", "vfmd", "md", exact=True) or pick(df, "meandefect", "defect")
        c_dx, c_age = pick(df, "diagnosis"), pick(df, "age")
        c_iop = pick(df, "pneumatic", "iop")
        if c_md is None:
            print(f"[warn] no Mean Defect column in {x.name}: {list(df.columns)}")
            continue
        for _, r in df.iterrows():
            md = pd.to_numeric(str(r[c_md]).replace(",", "."), errors="coerce")
            if pd.isna(r[c_id]) or pd.isna(md):
                continue
            pid = str(r[c_id]).strip().lstrip("#")
            digits = "".join(ch for ch in pid if ch.isdigit())
            if not digits:
                continue
            num = int(digits)
            img = next(iter(img_dir.glob(f"RET{num:03d}{eye}.*")), None)
            if img is None:
                continue
            res, _, _ = pipe.analyze(read_rgb(img), img.stem)
            cdr = res["component1"]["cdr"]
            cls = res["classification"] or {}
            rows.append({
                "image": img.name, "patient": num, "eye": eye,
                "age": r[c_age] if c_age else None,
                "diagnosis": r[c_dx] if c_dx else None,  # 0 healthy, 1 glaucoma, 2 suspect
                "iop": r[c_iop] if c_iop else None,
                "mean_defect": float(md),
                "vertical_cdr": cdr["vertical_cdr"], "horizontal_cdr": cdr["horizontal_cdr"],
                "area_cdr": cdr["area_cdr"],
                "glaucoma_probability": cls.get("glaucoma_probability"),
            })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    if len(out):
        print(out.groupby("diagnosis").size().rename("cases").to_string())
        v = out.dropna(subset=["vertical_cdr"])
        if len(v) > 2:
            print(f"correlation vertical CDR vs Mean Defect: {v.vertical_cdr.corr(v.mean_defect):.3f}")
    print(f"{len(out)} PAPILA cases matched with visual-field Mean Defect -> {args.out}")


if __name__ == "__main__":
    main()
