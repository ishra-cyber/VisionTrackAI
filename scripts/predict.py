"""Run the Phase-1 pipeline on one image or a folder from the command line.

    python scripts/predict.py path/to/fundus.png
    python scripts/predict.py path/to/folder --out reports/predictions
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
import cv2
import pandas as pd

from src.inference import VisionTrackPipeline, make_overlay
from src.preprocessing import read_rgb

EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="reports/predictions")
    args = ap.parse_args()
    src, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXT)
    pipe = VisionTrackPipeline()
    rows = []
    for f in files:
        rgb = read_rgb(f)
        res, disc, cup = pipe.analyze(rgb, f.stem)
        cv2.imwrite(str(out / f"{f.stem}_overlay.png"), cv2.cvtColor(make_overlay(rgb, disc, cup), cv2.COLOR_RGB2BGR))
        (out / f"{f.stem}.json").write_text(json.dumps(res, indent=2))
        c, k = res["component1"]["cdr"], res["classification"] or {}
        rows.append({"image": f.name, **{x: c[x] for x in ("vertical_cdr", "horizontal_cdr", "area_cdr")},
                     "glaucoma_probability": k.get("glaucoma_probability"), "prediction": k.get("prediction"),
                     "risk_level": k.get("risk_level"), "flags": "; ".join(res["flags"])})
        print(json.dumps(rows[-1]))
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
