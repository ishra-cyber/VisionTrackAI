"""Evaluate Phase-1 models and write reports.

    python scripts/evaluate.py --tune        # tune thresholds on validation, then test
    python scripts/evaluate.py               # test with saved thresholds

Writes to reports/:
  seg_per_image.csv            per-image Dice/IoU/CDR  (error log)
  failure_analysis.md          worst cases & edge cases
  worst_cases.png              visual grid of worst segmentations
  cls_per_image.csv, cls_misclassified.csv
  roc_curve.png, confusion_matrix.png, cdr_scatter.png
  phase1_results.json          all headline numbers
"""
import argparse

import _path  # noqa: F401
import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src.cdr import compute_cdr  # noqa: E402
from src.datasets import ClsDataset, SegDataset  # noqa: E402
from src.inference import load_thresholds  # noqa: E402
from src.metrics import classification_metrics, dice, iou, regression_metrics, youden_threshold  # noqa: E402
from src.models import load_classifier, load_unet  # noqa: E402
from src.postprocess import clean_masks  # noqa: E402
from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.utils import save_json  # noqa: E402


@torch.no_grad()
def seg_probs(model, ds, device, batch=8, workers=2):
    dl = DataLoader(ds, batch, shuffle=False, num_workers=workers)
    P, Y, X = [], [], []
    for x, y in tqdm(dl, desc="segmenting", leave=False):
        xd = x.to(device)
        p = torch.sigmoid(model(xd))
        p = (p + torch.flip(torch.sigmoid(model(torch.flip(xd, dims=[3]))), dims=[3])) / 2
        P.append(p.cpu().numpy().astype(np.float16))
        Y.append(y.numpy().astype(np.uint8))
        X.append(x.numpy()[:, :, ::4, ::4].astype(np.float16))  # thumbnails for figures
    return np.concatenate(P), np.concatenate(Y), np.concatenate(X)


@torch.no_grad()
def cls_probs(model, ds, device, batch=32, workers=2):
    dl = DataLoader(ds, batch, shuffle=False, num_workers=workers)
    out = []
    for x, _ in tqdm(dl, desc="classifying", leave=False):
        xd = x.to(device)
        p = torch.sigmoid(model(xd)).squeeze(1)
        p = (p + torch.sigmoid(model(torch.flip(xd, dims=[3]))).squeeze(1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def tune_seg(P, Y):
    grid = np.round(np.arange(0.3, 0.75, 0.05), 2)
    best_d = max(grid, key=lambda t: np.mean([dice(p[0] >= t, y[0]) for p, y in zip(P, Y)]))
    scores = {}
    for t in grid:
        scores[t] = np.mean([dice(clean_masks(p[0], p[1], best_d, t)[1], y[1]) for p, y in zip(P, Y)])
    best_c = max(scores, key=scores.get)
    return float(best_d), float(best_c)


def denorm(x):
    x = x.astype(np.float32).transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x, 0, 1)


def evaluate_seg(P, Y, X, names, thr):
    rows = []
    raw_d, raw_c = [], []
    for i, (p, y) in enumerate(zip(P, Y)):
        p = p.astype(np.float32)
        raw_d.append(dice(p[0] >= thr["disc"], y[0]))
        raw_c.append(dice(p[1] >= thr["cup"], y[1]))
        disc, cup = clean_masks(p[0], p[1], thr["disc"], thr["cup"])
        pc, gc = compute_cdr(disc, cup), compute_cdr(y[0], y[1])
        rows.append({
            "name": names[i],
            "dice_disc": dice(disc, y[0]), "dice_cup": dice(cup, y[1]),
            "iou_disc": iou(disc, y[0]), "iou_cup": iou(cup, y[1]),
            "pred_vcdr": pc["vertical_cdr"], "gt_vcdr": gc["vertical_cdr"],
            "pred_hcdr": pc["horizontal_cdr"], "gt_hcdr": gc["horizontal_cdr"],
            "pred_acdr": pc["area_cdr"], "gt_acdr": gc["area_cdr"],
            "disc_detected": bool(disc.sum() > 0), "cup_detected": bool(cup.sum() > 0),
        })
    df = pd.DataFrame(rows)
    df["overall_dice"] = (df.dice_disc + df.dice_cup) / 2
    valid = df.dropna(subset=["pred_vcdr", "gt_vcdr"])
    df["abs_vcdr_error"] = (df.pred_vcdr - df.gt_vcdr).abs()
    res = {
        "n_images": len(df),
        "disc_dice": float(df.dice_disc.mean()), "cup_dice": float(df.dice_cup.mean()),
        "overall_dice": float(df.overall_dice.mean()),
        "disc_iou": float(df.iou_disc.mean()), "cup_iou": float(df.iou_cup.mean()),
        "raw_disc_dice_no_postprocess": float(np.mean(raw_d)),
        "raw_cup_dice_no_postprocess": float(np.mean(raw_c)),
        "disc_not_detected": int((~df.disc_detected).sum()),
        "cup_not_detected": int((~df.cup_detected).sum()),
    }
    cdr = {}
    for k, (a, b) in {"vertical": ("pred_vcdr", "gt_vcdr"), "horizontal": ("pred_hcdr", "gt_hcdr"),
                      "area": ("pred_acdr", "gt_acdr")}.items():
        v = df.dropna(subset=[a, b])
        if len(v):
            cdr[k] = regression_metrics(v[a], v[b])
    res["cdr"] = cdr
    res["cdr_n_valid"] = len(valid)
    return df, res


def seg_figures(df, P, Y, X, thr, out):
    # CDR scatter
    v = df.dropna(subset=["pred_vcdr", "gt_vcdr"])
    if len(v):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(v.gt_vcdr, v.pred_vcdr, s=10, alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="Ground-truth vertical CDR", ylabel="Predicted vertical CDR", xlim=(0, 1), ylim=(0, 1),
               title="CDR agreement (test set)")
        fig.tight_layout()
        fig.savefig(out / "cdr_scatter.png", dpi=120)
        plt.close(fig)
    # worst cases grid
    worst = df.sort_values("overall_dice").head(6).index.tolist()
    if worst:
        fig, axes = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(worst):
            p = P[i].astype(np.float32)
            disc, cup = clean_masks(p[0], p[1], thr["disc"], thr["cup"])
            axes[r, 0].imshow(denorm(X[i]))
            axes[r, 0].set_title(df.loc[i, "name"], fontsize=8)
            axes[r, 1].imshow(Y[i][0] + Y[i][1], vmin=0, vmax=2)
            axes[r, 1].set_title("ground truth", fontsize=8)
            axes[r, 2].imshow(disc + cup, vmin=0, vmax=2)
            axes[r, 2].set_title(f"pred  disc {df.loc[i, 'dice_disc']:.2f} cup {df.loc[i, 'dice_cup']:.2f}",
                                 fontsize=8)
            for a in axes[r]:
                a.axis("off")
        fig.tight_layout()
        fig.savefig(out / "worst_cases.png", dpi=100)
        plt.close(fig)


def failure_report(df, res, out, source_col=None):
    lines = ["# Failure analysis - Component 1 (test set)\n",
             f"- Images evaluated: {res['n_images']}",
             f"- Disc not detected: {res['disc_not_detected']}",
             f"- Cup not detected: {res['cup_not_detected']}",
             f"- Dice without post-processing: disc {res['raw_disc_dice_no_postprocess']:.4f}, "
             f"cup {res['raw_cup_dice_no_postprocess']:.4f}",
             f"- Dice with post-processing: disc {res['disc_dice']:.4f}, cup {res['cup_dice']:.4f}\n"]
    if source_col is not None:
        lines.append("## Dice by source dataset\n")
        g = df.groupby(source_col)[["dice_disc", "dice_cup"]].agg(["mean", "count"]).round(4)
        lines.append("```\n" + g.to_string() + "\n```")
        lines.append("")
    lines.append("## 10 worst cup segmentations\n")
    cols = ["name", "dice_disc", "dice_cup", "pred_vcdr", "gt_vcdr"]
    lines.append("```\n" + df.sort_values("dice_cup").head(10)[cols].round(4).to_string(index=False) + "\n```")
    big = df[df.abs_vcdr_error > 0.15]
    lines.append(f"\n## Large CDR errors (|error| > 0.15): {len(big)} images\n")
    if len(big):
        lines.append("```\n" + big.sort_values("abs_vcdr_error", ascending=False)[cols + ["abs_vcdr_error"]]
                     .head(15).round(4).to_string(index=False) + "\n```")
    (out / "failure_analysis.md").write_text("\n".join(lines))


def cls_figures(y, p, thr, out):
    from sklearn.metrics import roc_curve
    if len(set(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title="ROC - EfficientNet-B0")
        fig.tight_layout()
        fig.savefig(out / "roc_curve.png", dpi=120)
        plt.close(fig)
    m = classification_metrics(y, p, thr)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Non-G", "Glaucoma"])
    ax.set_yticks([0, 1], ["Non-G", "Glaucoma"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="tune thresholds on the validation split first")
    ap.add_argument("--splits", default=str(config.SPLITS_DIR))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-cls", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = config.REPORTS_DIR
    out.mkdir(exist_ok=True)
    thr = load_thresholds()
    results = {}

    # ---------------- segmentation + CDR ----------------
    seg, cfg = load_unet(config.SEG_CKPT, device)
    size = cfg.get("img_size", config.SEG_IMG_SIZE)
    if args.tune:
        va = SegDataset(f"{args.splits}/seg_val.csv", size)
        Pv, Yv, _ = seg_probs(seg, va, device, workers=args.workers)
        thr["disc"], thr["cup"] = tune_seg(Pv, Yv)
        print(f"tuned seg thresholds: disc={thr['disc']} cup={thr['cup']}")
        del Pv, Yv
    te = SegDataset(f"{args.splits}/seg_test.csv", size)
    P, Y, X = seg_probs(seg, te, device, workers=args.workers)
    df, res = evaluate_seg(P, Y, X, te.df["name"].tolist(), thr)
    df["source"] = te.df["source"].values
    df.to_csv(out / "seg_per_image.csv", index=False)
    seg_figures(df, P, Y, X, thr, out)
    failure_report(df, res, out, "source")
    results["segmentation"] = res
    print(f"Disc Dice {res['disc_dice']:.4f} | Cup Dice {res['cup_dice']:.4f} | Overall {res['overall_dice']:.4f}")
    if "vertical" in res["cdr"]:
        c = res["cdr"]["vertical"]
        print(f"CDR mean pred {c['mean_pred']:.4f} | mean GT {c['mean_true']:.4f} | MAE {c['mae']:.4f} | R2 {c['r2']:.4f}")
    del P, Y, X

    # ---------------- classification ----------------
    if not args.skip_cls and config.CLS_CKPT.exists():
        cls, ccfg = load_classifier(config.CLS_CKPT, device)
        csize = ccfg.get("img_size", config.CLS_IMG_SIZE)
        if args.tune:
            va = ClsDataset(f"{args.splits}/cls_val.csv", csize)
            pv = cls_probs(cls, va, device, workers=args.workers)
            if len(set(va.labels)) == 2:
                thr["cls"] = youden_threshold(va.labels, pv)
            print(f"tuned classification threshold: {thr['cls']:.4f}")
        te = ClsDataset(f"{args.splits}/cls_test.csv", csize)
        p = cls_probs(cls, te, device, workers=args.workers)
        y = te.labels
        m = classification_metrics(y, p, thr["cls"])
        results["classification"] = m
        cdf = te.df[["name", "source", "label"]].copy()
        cdf["prob"] = p
        cdf["pred"] = (p >= thr["cls"]).astype(int)
        cdf.to_csv(out / "cls_per_image.csv", index=False)
        cdf[cdf.pred != cdf.label].to_csv(out / "cls_misclassified.csv", index=False)
        cls_figures(y, p, thr["cls"], out)
        print(f"Accuracy {m['accuracy']:.4f} | Sens {m['sensitivity']:.4f} | Spec {m['specificity']:.4f} "
              f"| AUC {m['roc_auc']:.4f}")

    results["thresholds"] = thr
    if args.tune:
        save_json({k: float(v) for k, v in thr.items()}, config.CHECKPOINT_DIR / "thresholds.json")
    save_json(results, out / "phase1_results.json")
    print(f"reports written to {out}")


if __name__ == "__main__":
    main()
