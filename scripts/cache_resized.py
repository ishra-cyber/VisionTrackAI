"""Pre-resize the classification images once so each training epoch is much faster.

    python scripts/cache_resized.py --size 320
    python scripts/train_classifier.py --splits data/splits_fast --size 320 ...

Writes resized copies to data/cache_<size>/ and new split CSVs to data/splits_fast/
(segmentation CSVs are copied unchanged so evaluate.py can use the same --splits).
"""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _path  # noqa: F401
import cv2
import pandas as pd
from tqdm import tqdm

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--splits", default=str(config.SPLITS_DIR))
    ap.add_argument("--out-splits", default=str(config.ROOT / "data" / "splits_fast"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    src, dst = Path(args.splits), Path(args.out_splits)
    cache = config.ROOT / "data" / f"cache_{args.size}"
    cache.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)

    def work(path):
        out = cache / (Path(path).stem + ".png")
        if not out.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return path, None
            img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), img)
        return path, str(out)

    for split in ("train", "val", "test"):
        df = pd.read_csv(src / f"cls_{split}.csv")
        with ThreadPoolExecutor(args.threads) as ex:
            mapping = dict(tqdm(ex.map(work, df["fundus"]), total=len(df), desc=f"cls_{split}"))
        bad = [k for k, v in mapping.items() if v is None]
        if bad:
            print(f"  skipped {len(bad)} unreadable images")
        df["fundus"] = df["fundus"].map(mapping)
        df.dropna(subset=["fundus"]).to_csv(dst / f"cls_{split}.csv", index=False)
    for f in src.glob("seg_*.csv"):
        shutil.copy(f, dst / f.name)
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
