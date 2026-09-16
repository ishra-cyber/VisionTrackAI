"""Scan the SMDG-19 dataset and write train/val/test split CSVs.

Usage:
    python scripts/prepare_data.py --data data/smdg
Outputs data/splits/{seg,cls}_{train,val,test}.csv
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import pandas as pd
from sklearn.model_selection import train_test_split

import config

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def index_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {p.stem: str(p) for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT}


def find_metadata(root):
    cands = sorted(root.glob("*.csv")) + sorted(root.parent.glob("*.csv"))
    for c in cands:
        if "metadata" in c.name.lower():
            return c
    return cands[0] if cands else None


def load_labels(meta_path):
    df = pd.read_csv(meta_path)
    cols = {c.lower().strip(): c for c in df.columns}
    name_col = next((cols[k] for k in ("names", "name", "image", "filename", "fundus") if k in cols), df.columns[0])
    type_col = next((cols[k] for k in ("types", "type", "label", "labels") if k in cols), None)
    if type_col is None:
        raise ValueError(f"No label column found in {meta_path}: {list(df.columns)}")
    stems = df[name_col].astype(str).map(lambda s: Path(s.replace("\\", "/")).stem)
    return dict(zip(stems, pd.to_numeric(df[type_col], errors="coerce")))


def split(df, strat_col, seed):
    strat = df[strat_col] if strat_col and df[strat_col].nunique() > 1 else None
    test_n = config.TEST_FRAC
    trainval, test = train_test_split(df, test_size=test_n, random_state=seed, stratify=strat)
    strat_tv = trainval[strat_col] if strat is not None else None
    val_rel = config.VAL_FRAC / (1 - test_n)
    train, val = train_test_split(trainval, test_size=val_rel, random_state=seed, stratify=strat_tv)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--out", default=str(config.SPLITS_DIR))
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.out)
    if not (root / "full-fundus").exists():  # Kaggle zips sometimes add a parent folder
        found = next((d.parent for d in root.rglob("full-fundus") if d.is_dir()), None)
        if found is not None:
            print(f"using dataset root {found}")
            root = found
    out.mkdir(parents=True, exist_ok=True)

    fundus = index_folder(root / "full-fundus")
    discs = index_folder(root / "optic-disc")
    cups = index_folder(root / "optic-cup")
    if not fundus:
        raise SystemExit(f"No images in {root / 'full-fundus'} - check --data path")
    meta = find_metadata(root)
    labels = load_labels(meta) if meta else {}
    print(f"fundus={len(fundus)} disc_masks={len(discs)} cup_masks={len(cups)} metadata={meta}")

    rows = []
    for stem, f in sorted(fundus.items()):
        lab = labels.get(stem)
        rows.append({
            "name": stem,
            "source": stem.rsplit("-", 1)[0] if "-" in stem else "unknown",
            "fundus": f,
            "disc": discs.get(stem, ""),
            "cup": cups.get(stem, ""),
            "label": int(lab) if lab is not None and pd.notna(lab) else -9,
        })
    df = pd.DataFrame(rows)

    seg = df[(df.disc != "") & (df.cup != "")].reset_index(drop=True)
    cls = df[df.label.isin([0, 1])].reset_index(drop=True)
    print(f"paired segmentation samples: {len(seg)}")
    print(f"labelled classification samples: {len(cls)} (glaucoma={int((cls.label == 1).sum())}, "
          f"non-glaucoma={int((cls.label == 0).sum())})")

    seg["strat"] = seg["source"]
    small = seg["strat"].map(seg["strat"].value_counts()) < 10
    seg.loc[small, "strat"] = "other"
    for name, part in zip(("train", "val", "test"), split(seg, "strat", config.SEED)):
        part.drop(columns="strat").to_csv(out / f"seg_{name}.csv", index=False)
        print(f"seg_{name}: {len(part)}")

    cls["strat"] = cls["source"] + "_" + cls["label"].astype(str)
    small = cls["strat"].map(cls["strat"].value_counts()) < 10
    cls.loc[small, "strat"] = cls.loc[small, "label"].astype(str)
    for name, part in zip(("train", "val", "test"), split(cls, "strat", config.SEED)):
        part.drop(columns="strat").to_csv(out / f"cls_{name}.csv", index=False)
        print(f"cls_{name}: {len(part)}")
    df.to_csv(out / "all_images.csv", index=False)


if __name__ == "__main__":
    main()
