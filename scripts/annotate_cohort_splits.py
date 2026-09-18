"""Tag every cohort row with the split it belongs to — and stop us quoting
numbers measured on the models' own training data.

PAPILA is one of the 19 datasets pooled into SMDG-19, so all 488 PAPILA images
also live inside SMDG and most of them were used to train the U-Net. Without this
step, `evaluate_cohort.py` would average Dice and CDR error over eyes the model
had already seen, and report a flattering number.

Adds two columns to data/cohort_master.csv:
  seg_split   train / val / test / none   (segmentation splits)
  cls_split   train / val / test / none   (classification splits)
Matching is by SMDG image name; PAPILA rows are matched through their
`PAPILA-<patient>` name inside SMDG.

    python scripts/annotate_cohort_splits.py
No GPU and no model needed - it only reads CSVs.
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import pandas as pd

import config  # noqa: E402


def load_splits(splits_dir, kind):
    out = {}
    for k in ("train", "val", "test"):
        f = Path(splits_dir) / f"{kind}_{k}.csv"
        if f.exists():
            for n in pd.read_csv(f)["name"].astype(str):
                out[n] = k
    return out


def cohort_key(row):
    """The name this eye has inside SMDG's split files."""
    if row.source == "SMDG":
        return Path(str(row.image)).stem
    digits = "".join(ch for ch in str(row.patient).split("-")[-1] if ch.isdigit())
    return f"PAPILA-{int(digits)}" if digits else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(config.ROOT / "data" / "cohort_master.csv"))
    ap.add_argument("--splits", default=str(config.SPLITS_DIR))
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    seg, cls = load_splits(args.splits, "seg"), load_splits(args.splits, "cls")
    keys = [cohort_key(r) for r in df.itertuples()]
    df["split_key"] = keys
    df["seg_split"] = [seg.get(k, "none") for k in keys]
    df["cls_split"] = [cls.get(k, "none") for k in keys]
    df["seg_heldout"] = df.seg_split.isin(["test", "none"])
    df["cls_heldout"] = df.cls_split.isin(["test", "none"])
    df.to_csv(args.cohort, index=False)

    print(f"{len(df)} rows -> {args.cohort}\n")
    for kind in ("seg", "cls"):
        print(f"{kind}_split by source:")
        print(pd.crosstab(df.source, df[f"{kind}_split"]).to_string(), "\n")

    seg_ref = df.dropna(subset=["expert_vcdr"])
    print(f"eyes with an expert reference: {len(seg_ref)} "
          f"({int((seg_ref.seg_split == 'test').sum())} held out for segmentation, "
          f"{int(seg_ref.seg_split.isin(['train', 'val']).sum())} seen in training)")
    lab = df.dropna(subset=["label"])
    print(f"eyes with a diagnosis label: {len(lab)} "
          f"({int((lab.cls_split == 'test').sum())} held out for classification, "
          f"{int(lab.cls_split.isin(['train', 'val']).sum())} seen in training)")
    vf = df.dropna(subset=["mean_defect"])
    print(f"eyes with a visual field: {len(vf)} "
          f"({int((vf.seg_split == 'test').sum())} held out for segmentation, "
          f"{int(vf.seg_split.isin(['train', 'val']).sum())} seen in training)")
    print("\nFrom here on, evaluate_cohort.py reports held-out eyes as the headline "
          "and shows the training-set figures separately, labelled.")


if __name__ == "__main__":
    main()
