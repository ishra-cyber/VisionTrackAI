"""Create the figures shown on the Phase-1 slides from your own trained models.

    python scripts/phase1_summary.py            # results panel + 3 demo figures
    python scripts/phase1_summary.py --n 5      # more demo figures

Outputs (reports/):
  phase1_results_panel.png   Structural analysis / CDR estimation / Glaucoma classification
  demo_<name>.png            Original fundus, GT disc, GT cup, predicted disc, cup, disc+cup
Run scripts/evaluate.py --tune first (it writes reports/phase1_results.json).
"""
import argparse
import json

import _path  # noqa: F401
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

NAVY, GREEN = "#0f1f4b", "#1f8a4c"


def results_panel(res, path):
    seg, cls = res.get("segmentation", {}), res.get("classification", {})
    cdr = seg.get("cdr", {}).get("vertical", {})
    boxes = [
        ("STRUCTURAL ANALYSIS\n(U-NET SEGMENTATION)", [
            ("Disc Dice Score", seg.get("disc_dice")), ("Cup Dice Score", seg.get("cup_dice")),
            ("Overall Dice Score", seg.get("overall_dice"))], "{:.4f}"),
        ("CDR ESTIMATION\n(vertical CDR)", [
            ("Mean Predicted CDR", cdr.get("mean_pred")), ("Mean Ground Truth CDR", cdr.get("mean_true")),
            ("MAE", cdr.get("mae")), ("R2", cdr.get("r2"))], "{:.4f}"),
        ("GLAUCOMA CLASSIFICATION\n(EFFICIENTNET-B0)", [
            ("Accuracy", cls.get("accuracy")), ("Sensitivity", cls.get("sensitivity")),
            ("Specificity", cls.get("specificity")), ("ROC-AUC", cls.get("roc_auc"))], None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, rows, fmt) in zip(axes, boxes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, ec="#c9d1e3", lw=1.5))
        ax.text(0.5, 0.86, title, ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
        for i, (k, v) in enumerate(rows):
            y = 0.62 - i * 0.14
            if v is None:
                s = "—"
            elif fmt:
                s = fmt.format(v)
            else:
                s = f"{v * 100:.2f}%" if k != "ROC-AUC" else f"{v:.4f}"
            ax.text(0.08, y, k, fontsize=11, va="center")
            ax.text(0.92, y, ": " + s, fontsize=11, va="center", ha="right", fontweight="bold", color=GREEN)
    fig.suptitle("VisionTrack AI – Phase 1 Output / Results (test set)", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def demo_figures(n, out):
    from src.inference import VisionTrackPipeline
    from src.preprocessing import read_mask, read_rgb
    from src.visualize import make_overlay

    pipe = VisionTrackPipeline()
    df = pd.read_csv(config.SPLITS_DIR / "seg_test.csv").sample(n, random_state=7)
    paths = []
    for _, r in df.iterrows():
        rgb = read_rgb(r["fundus"])
        cup_gt = read_mask(r["cup"], rgb.shape)
        disc_gt = np.maximum(read_mask(r["disc"], rgb.shape), cup_gt)
        res, disc, cup = pipe.analyze(rgb, r["name"])
        dim = (rgb * 0.35).astype(np.uint8)
        pd_img = make_overlay(dim, disc, np.zeros_like(cup), alpha=0.8)
        pc_img = make_overlay(dim, np.zeros_like(disc), cup, alpha=0.8)
        panels = [(rgb, "Original Fundus"), (disc_gt, "Ground Truth Disc"), (cup_gt, "Ground Truth Cup"),
                  (pd_img, "Predicted Disc"), (pc_img, "Predicted Cup"),
                  (make_overlay(rgb, disc, cup), "Predicted Disc + Cup")]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
        for ax, (im, t) in zip(axes.ravel(), panels):
            ax.imshow(im, cmap="viridis" if im.ndim == 2 else None)
            ax.set_title(t, fontsize=12)
            ax.axis("off")
        c = res["component1"]["cdr"]
        k = res["classification"] or {}
        gt_v = None
        ys = np.nonzero(disc_gt)[0]
        yc = np.nonzero(cup_gt)[0]
        if len(ys) and len(yc):
            gt_v = (yc.max() - yc.min() + 1) / (ys.max() - ys.min() + 1)
        sub = f"{r['name']}  |  predicted vCDR {c['vertical_cdr']}"
        if gt_v is not None:
            sub += f"  (ground truth {gt_v:.3f})"
        if k:
            sub += f"  |  glaucoma probability {k['glaucoma_probability']:.2f} ({k['risk_level']} risk)"
        fig.suptitle(sub, fontsize=12)
        fig.tight_layout(h_pad=2.5)
        p = out / f"demo_{r['name']}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of demo figures")
    args = ap.parse_args()
    out = config.REPORTS_DIR
    out.mkdir(exist_ok=True)
    rp = out / "phase1_results.json"
    if rp.exists():
        res = json.loads(rp.read_text())
        results_panel(res, out / "phase1_results_panel.png")
        seg, cls = res.get("segmentation", {}), res.get("classification", {})
        cdr = seg.get("cdr", {}).get("vertical", {})
        print("STRUCTURAL ANALYSIS (U-NET SEGMENTATION)")
        print(f"  Disc Dice Score        : {seg.get('disc_dice', float('nan')):.4f}")
        print(f"  Cup Dice Score         : {seg.get('cup_dice', float('nan')):.4f}")
        print(f"  Overall Dice Score     : {seg.get('overall_dice', float('nan')):.4f}")
        print("CDR ESTIMATION")
        print(f"  Mean Predicted CDR     : {cdr.get('mean_pred', float('nan')):.4f}")
        print(f"  Mean Ground Truth CDR  : {cdr.get('mean_true', float('nan')):.4f}")
        print(f"  MAE                    : {cdr.get('mae', float('nan')):.4f}")
        print(f"  R2                     : {cdr.get('r2', float('nan')):.4f}")
        if cls:
            print("GLAUCOMA CLASSIFICATION (EFFICIENTNET-B0)")
            print(f"  Accuracy               : {cls['accuracy'] * 100:.2f}%")
            print(f"  Sensitivity            : {cls['sensitivity'] * 100:.2f}%")
            print(f"  Specificity            : {cls['specificity'] * 100:.2f}%")
            print(f"  ROC-AUC                : {cls['roc_auc']:.4f}")
        print(f"-> {out / 'phase1_results_panel.png'}")
    else:
        print("reports/phase1_results.json not found - run: python scripts/evaluate.py --tune")
    if args.n > 0 and config.SEG_CKPT.exists():
        for p in demo_figures(args.n, out):
            print(f"-> {p}")


if __name__ == "__main__":
    main()
