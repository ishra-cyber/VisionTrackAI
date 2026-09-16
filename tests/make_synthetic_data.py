"""Generate a tiny synthetic dataset in SMDG-19 layout for smoke-testing the code
(NOT for real training).

    python tests/make_synthetic_data.py --out data/synthetic --n 60
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def make(rng, size=384):
    img = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[:size, :size]
    c = size / 2
    r = np.hypot(xx - c, yy - c)
    fov = r < size * 0.47
    shade = np.clip(1 - r / (size * 0.6), 0, 1)
    img[..., 0] = (150 + 80 * shade) * fov
    img[..., 1] = (60 + 40 * shade) * fov
    img[..., 2] = (20 + 15 * shade) * fov
    cx = int(c + rng.uniform(-0.25, 0.25) * size)
    cy = int(c + rng.uniform(-0.1, 0.1) * size)
    dr = int(size * rng.uniform(0.07, 0.1))
    ratio = rng.uniform(0.25, 0.8)
    cr = int(dr * ratio)
    for _ in range(6):  # vessels
        ang = rng.uniform(0, 2 * np.pi)
        cv2.line(img, (cx, cy), (int(cx + np.cos(ang) * size), int(cy + np.sin(ang) * size)),
                 (110, 25, 15), int(rng.integers(2, 5)))
    img[~fov] = 0
    disc = np.zeros((size, size), np.uint8)
    cup = np.zeros((size, size), np.uint8)
    cv2.ellipse(disc, (cx, cy), (dr, int(dr * 1.08)), 0, 0, 360, 255, -1)
    cv2.ellipse(cup, (cx, cy), (cr, int(cr * 1.1)), 0, 0, 360, 255, -1)
    img[disc > 0] = (235, 170, 90)
    img[cup > 0] = (250, 225, 170)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = rng.normal(0, 4, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, disc, cup, int(ratio > 0.55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    for d in ("full-fundus", "optic-disc", "optic-cup"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(args.n):
        src = ["SYNA", "SYNB"][i % 2]
        name = f"{src}-{i}"
        img, disc, cup, lab = make(rng)
        cv2.imwrite(str(out / "full-fundus" / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i % 5 != 4:  # some images without masks, like the real dataset
            cv2.imwrite(str(out / "optic-disc" / f"{name}.png"), disc)
            cv2.imwrite(str(out / "optic-cup" / f"{name}.png"), cup)
        rows.append({"names": name, "types": lab if i % 17 else -1, "fundus": f"/full-fundus/{name}.png"})
    pd.DataFrame(rows).to_csv(out / "metadata - standardized.csv", index=False)
    print(f"wrote {args.n} synthetic images to {out}")


if __name__ == "__main__":
    main()
