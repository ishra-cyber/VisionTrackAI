import shutil
from pathlib import Path

import _path  # noqa: F401
import pandas as pd

import config

TARGETS = [("visit1_low_cdr", 0.30), ("visit2_mid_cdr", 0.50), ("visit3_high_cdr", 0.72)]


def main():
    per = pd.read_csv(config.REPORTS_DIR / "seg_per_image.csv")
    paths = None
    for splits in (config.SPLITS_DIR, config.ROOT / "data" / "splits_fast"):
        f = Path(splits) / "seg_test.csv"
        if f.exists():
            paths = pd.read_csv(f)[["name", "fundus"]]
            break
    df = per.merge(paths, on="name").dropna(subset=["gt_vcdr", "pred_vcdr"])
    good = df[(df.dice_disc > 0.93) & (df.dice_cup > 0.85) & ((df.pred_vcdr - df.gt_vcdr).abs() < 0.05)]
    if len(good) < 3:
        good = df
    out = config.ROOT / "demo_images"
    out.mkdir(exist_ok=True)
    used = set()
    for label, target in TARGETS:
        cand = good[~good.name.isin(used)].copy()
        cand["gap"] = (cand.gt_vcdr - target).abs()
        r = cand.sort_values(["gap", "dice_cup"], ascending=[True, False]).iloc[0]
        used.add(r["name"])
        dst = out / f"{label}_{r['name']}{Path(r['fundus']).suffix}"
        shutil.copy(r["fundus"], dst)
        print(f"{dst.name:45s} expert CDR {r.gt_vcdr:.2f} | predicted {r.pred_vcdr:.2f} | "
              f"Dice disc {r.dice_disc:.2f} cup {r.dice_cup:.2f}")
    print(f"\nCopied to {out}")


if __name__ == "__main__":
    main()
