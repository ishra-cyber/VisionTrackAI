"""Clinical validation requested by the reviewers.

Part A - Is the CDR accurate?
  * SMDG test set (431 images, expert masks): MAE, bias, Pearson r, ICC, Bland-Altman,
    % of eyes within +/-0.1 of the expert CDR.
  * PAPILA (independent clinic): our CDR vs the two ophthalmologists' contours.

Part B - Does CDR relate to visual-field (functional) damage?
  * PAPILA eyes with Humphrey 30-2 Mean Defect (MD).
  * Pearson / Spearman correlation with p-values (ours and expert CDR).
  * CDR across Hodapp-Parrish-Anderson severity stages (Kruskal-Wallis + trend test).
  * ROC: how well CDR identifies functional damage (MD < -6 dB).

    python scripts/validate_structure_function.py
Outputs: reports/structure_function/ (figures, summary.md, summary.json)
         app/static/validation/ (figures shown on the app's Visual Field page)
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import _path  # noqa: F401
import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src.cdr import compute_cdr  # noqa: E402

NAVY, ORANGE, GREY, RED = "#0b4f8a", "#e08a1e", "#6b7c8f", "#b3261e"
STAGES = ["Early (MD ≥ −6)", "Moderate (−6 to −12)", "Advanced (MD < −12)"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def icc_2_1(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single rater."""
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def agreement(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    d = pred - ref
    r, p = stats.pearsonr(pred, ref)
    return {"n": int(len(d)), "mean_pred": float(pred.mean()), "mean_ref": float(ref.mean()),
            "mae": float(np.abs(d).mean()), "bias": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)), "loa_high": float(d.mean() + 1.96 * d.std(ddof=1)),
            "pearson_r": float(r), "pearson_p": float(p), "icc": icc_2_1(pred, ref),
            "within_0_1": float((np.abs(d) <= 0.1).mean())}


def bland_altman(pred, ref, ag, title, path):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.scatter(ref, pred, s=12, alpha=0.55, color=NAVY, edgecolor="none")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Expert vertical CDR", ylabel="VisionTrack vertical CDR",
           title=f"Agreement  (r = {ag['pearson_r']:.2f}, ICC = {ag['icc']:.2f}, MAE = {ag['mae']:.3f})")
    ax = axes[1]
    m, d = (pred + ref) / 2, pred - ref
    ax.scatter(m, d, s=12, alpha=0.55, color=NAVY, edgecolor="none")
    for y, ls, lab in [(ag["bias"], "-", f"bias {ag['bias']:+.3f}"),
                       (ag["loa_low"], "--", f"−1.96 SD {ag['loa_low']:+.3f}"),
                       (ag["loa_high"], "--", f"+1.96 SD {ag['loa_high']:+.3f}")]:
        ax.axhline(y, ls=ls, color=RED if ls == "--" else GREY, lw=1)
        ax.text(0.99, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5)
    ax.set(xlabel="Mean of VisionTrack and expert CDR", ylabel="Difference (VisionTrack − expert)",
           title="Bland–Altman")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- PAPILA experts
CONTOUR = re.compile(r"RET(\d{3})_?(OD|OS)_?(cup|disc)_?exp(\d)", re.I)


def read_contour(path):
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", path.read_text(errors="ignore"))]
    if len(nums) < 6:
        return None
    return np.array(nums[: len(nums) // 2 * 2]).reshape(-1, 2)


def expert_cdrs(papila_root, images):
    """Vertical CDR from each expert's contours, averaged over available experts."""
    files = {}
    for f in papila_root.rglob("*"):
        if f.is_file() and f.suffix.lower() in (".txt", ".csv", ".dat", ""):
            m = CONTOUR.search(f.name)
            if m:
                key = (int(m.group(1)), m.group(2).upper())
                files.setdefault(key, {}).setdefault(m.group(4), {})[m.group(3).lower()] = f
    out = {}
    for (pid, eye), experts in files.items():
        img = images.get(f"RET{pid:03d}{eye}")
        if img is None:
            continue
        h, w = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE).shape[:2]
        vals = []
        for parts in experts.values():
            if "cup" not in parts or "disc" not in parts:
                continue
            masks = {}
            for k in ("disc", "cup"):
                pts = read_contour(parts[k])
                if pts is None:
                    break
                mk = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mk, [np.round(pts).astype(np.int32)], 1)
                masks[k] = mk
            if len(masks) == 2 and masks["disc"].sum() > 0:
                v = compute_cdr(masks["disc"], masks["cup"])["vertical_cdr"]
                if v is not None:
                    vals.append(v)
        if vals:
            out[(pid, eye)] = float(np.mean(vals))
    return out


def stage(md):
    return 0 if md >= -6 else (1 if md >= -12 else 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papila", default=str(config.PAPILA_DIR))
    ap.add_argument("--vf", default=str(config.ROOT / "data" / "papila_vf_matched.csv"))
    ap.add_argument("--seg", default=str(config.REPORTS_DIR / "seg_per_image.csv"))
    args = ap.parse_args()
    out = config.REPORTS_DIR / "structure_function"
    out.mkdir(parents=True, exist_ok=True)
    S = {}
    md_lines = ["# VisionTrack AI – CDR validation and structure–function analysis\n"]

    # ---------------- A1: SMDG test set
    seg = pd.read_csv(args.seg).dropna(subset=["pred_vcdr", "gt_vcdr"])
    a1 = agreement(seg.pred_vcdr, seg.gt_vcdr)
    S["smdg_agreement"] = a1
    bland_altman(seg.pred_vcdr, seg.gt_vcdr, a1, f"A. CDR accuracy – SMDG test set (n = {a1['n']})",
                 out / "A1_cdr_agreement_smdg.png")
    md_lines += ["## A. Is the CDR accurate?\n",
                 f"**SMDG held-out test set (n = {a1['n']}, expert masks)**\n",
                 f"- Mean CDR: VisionTrack {a1['mean_pred']:.3f} vs expert {a1['mean_ref']:.3f}",
                 f"- MAE {a1['mae']:.3f}; bias {a1['bias']:+.3f}; 95% limits of agreement "
                 f"{a1['loa_low']:+.3f} to {a1['loa_high']:+.3f}",
                 f"- Pearson r = {a1['pearson_r']:.3f} ({fmt_p(a1['pearson_p'])}); ICC(2,1) = {a1['icc']:.3f}",
                 f"- {a1['within_0_1'] * 100:.1f}% of eyes within ±0.10 of the expert CDR\n"]

    # ---------------- load PAPILA matched data
    vf = pd.read_csv(args.vf).dropna(subset=["mean_defect", "vertical_cdr"]).reset_index(drop=True)
    root = Path(args.papila)
    img_dir = next((d for d in root.rglob("FundusImages") if d.is_dir()), root)
    images = {p.stem.upper(): p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")}
    exp = expert_cdrs(root, images) if root.exists() else {}
    vf["expert_vcdr"] = [exp.get((int(r.patient), r.eye)) for r in vf.itertuples()]
    vf["stage"] = vf.mean_defect.map(stage)
    vf.to_csv(out / "papila_structure_function.csv", index=False)

    # ---------------- A2: PAPILA vs experts
    ve = vf.dropna(subset=["expert_vcdr"])
    if len(ve) > 5:
        a2 = agreement(ve.vertical_cdr, ve.expert_vcdr)
        S["papila_agreement"] = a2
        bland_altman(ve.vertical_cdr, ve.expert_vcdr, a2,
                     f"A. CDR accuracy – PAPILA vs ophthalmologists (n = {a2['n']})",
                     out / "A2_cdr_agreement_papila.png")
        md_lines += [f"**PAPILA, independent clinic (n = {a2['n']}, mean of two ophthalmologists)**\n",
                     f"- Mean CDR: VisionTrack {a2['mean_pred']:.3f} vs experts {a2['mean_ref']:.3f}",
                     f"- MAE {a2['mae']:.3f}; bias {a2['bias']:+.3f}; r = {a2['pearson_r']:.3f} "
                     f"({fmt_p(a2['pearson_p'])}); ICC = {a2['icc']:.3f}",
                     f"- {a2['within_0_1'] * 100:.1f}% within ±0.10\n"]
    else:
        md_lines.append("_PAPILA expert contours not found – expert comparison skipped._\n")
        print("[info] PAPILA expert contour files not found; skipping expert comparison")

    # ---------------- B1: correlation
    n = len(vf)
    pr, pp = stats.pearsonr(vf.vertical_cdr, vf.mean_defect)
    sr, sp = stats.spearmanr(vf.vertical_cdr, vf.mean_defect)
    slope, icpt, *_ = stats.linregress(vf.vertical_cdr, vf.mean_defect)
    B = {"n": n, "pearson_r": float(pr), "pearson_p": float(pp), "spearman_rho": float(sr),
         "spearman_p": float(sp), "slope": float(slope), "intercept": float(icpt), "r2": float(pr ** 2)}
    if len(ve) > 5:
        er, ep = stats.pearsonr(ve.expert_vcdr, ve.mean_defect)
        esr, esp = stats.spearmanr(ve.expert_vcdr, ve.mean_defect)
        B.update({"expert_pearson_r": float(er), "expert_pearson_p": float(ep),
                  "expert_spearman_rho": float(esr), "expert_spearman_p": float(esp), "expert_n": int(len(ve))})
    S["correlation"] = B

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for eye, col in (("OD", NAVY), ("OS", ORANGE)):
        d = vf[vf.eye == eye]
        ax.scatter(d.vertical_cdr, d.mean_defect, s=22, alpha=0.75, color=col, edgecolor="white", lw=0.5, label=eye)
    xs = np.linspace(vf.vertical_cdr.min(), vf.vertical_cdr.max(), 50)
    ax.plot(xs, slope * xs + icpt, "--", color=GREY, lw=1.5, label=f"fit: MD = {slope:.1f}·CDR {icpt:+.1f}")
    ax.axhline(-6, color=RED, lw=0.8, ls=":")
    ax.axhline(-12, color=RED, lw=0.8, ls=":")
    ax.set(xlabel="Vertical CDR (VisionTrack)", ylabel="Visual-field Mean Defect (dB)",
           title=f"B. Structure vs function (n = {n})\nPearson r = {pr:.2f} ({fmt_p(pp)}), "
                 f"Spearman ρ = {sr:.2f} ({fmt_p(sp)})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "B1_cdr_vs_md.png", dpi=150)
    plt.close(fig)

    # ---------------- B2: severity stages
    groups = [vf.loc[vf.stage == s, "vertical_cdr"].values for s in range(3)]
    nonempty = [g for g in groups if len(g)]
    kw_h, kw_p = stats.kruskal(*nonempty) if len(nonempty) > 1 else (float("nan"), float("nan"))
    tr, tp = stats.spearmanr(vf.stage, vf.vertical_cdr)
    B2 = {"stages": [{"stage": STAGES[s], "n": int(len(g)),
                      "mean_cdr": float(np.mean(g)) if len(g) else None,
                      "sd_cdr": float(np.std(g, ddof=1)) if len(g) > 1 else None,
                      "median_cdr": float(np.median(g)) if len(g) else None} for s, g in enumerate(groups)],
          "kruskal_h": float(kw_h), "kruskal_p": float(kw_p), "trend_rho": float(tr), "trend_p": float(tp)}
    S["severity"] = B2
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    pos = [i for i, g in enumerate(groups) if len(g)]
    bp = ax.boxplot([groups[i] for i in pos], positions=pos, widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#8cc9a3", "#ecc373", "#e68e87"]):
        patch.set_facecolor(c)
    for i in pos:
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(groups[i]))
        ax.scatter(i + jitter, groups[i], s=10, color="#333", alpha=0.5, zorder=3)
    ax.set_xticks(range(3), [f"{STAGES[i]}\nn = {len(groups[i])}" for i in range(3)], fontsize=8.5)
    ax.set(ylabel="Vertical CDR (VisionTrack)",
           title=f"B. CDR by visual-field severity (Hodapp–Parrish–Anderson)\n"
                 f"Kruskal–Wallis {fmt_p(kw_p)}; trend ρ = {tr:.2f} ({fmt_p(tp)})")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "B2_cdr_by_severity.png", dpi=150)
    plt.close(fig)

    # ---------------- B3: ROC for functional damage
    y = (vf.mean_defect < -6).astype(int).values
    B3 = None
    if 0 < y.sum() < len(y):
        auc = roc_auc_score(y, vf.vertical_cdr)
        fpr, tpr, thr = roc_curve(y, vf.vertical_cdr)
        j = int(np.argmax(tpr - fpr))
        B3 = {"auc": float(auc), "cutoff": float(thr[j]), "sensitivity": float(tpr[j]),
              "specificity": float(1 - fpr[j]), "n_damage": int(y.sum()), "n": int(len(y))}
        if vf.glaucoma_probability.notna().all():
            B3["auc_model_probability"] = float(roc_auc_score(y, vf.glaucoma_probability))
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.plot(fpr, tpr, color=NAVY, lw=2, label=f"Vertical CDR (AUC {auc:.2f})")
        if "auc_model_probability" in B3:
            f2, t2, _ = roc_curve(y, vf.glaucoma_probability)
            ax.plot(f2, t2, color=ORANGE, lw=1.5, label=f"Model probability (AUC {B3['auc_model_probability']:.2f})")
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
        ax.scatter([fpr[j]], [tpr[j]], color=RED, zorder=3,
                   label=f"CDR ≥ {thr[j]:.2f}: sens {tpr[j]:.2f}, spec {1 - fpr[j]:.2f}")
        ax.set(xlabel="1 − Specificity", ylabel="Sensitivity",
               title=f"B. Detecting functional damage (MD < −6 dB)\n{int(y.sum())} of {len(y)} eyes")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "B3_roc_functional_damage.png", dpi=150)
        plt.close(fig)
    S["functional_roc"] = B3

    # ---------------- report
    md_lines += ["## B. Does CDR relate to visual-field damage?\n",
                 f"**PAPILA eyes with Humphrey 30-2 Mean Defect (n = {n})**\n",
                 f"- Pearson r = {pr:.3f} ({fmt_p(pp)}); Spearman ρ = {sr:.3f} ({fmt_p(sp)}); R² = {pr ** 2:.3f}",
                 f"- Each +0.1 in vertical CDR ≈ {slope / 10:.2f} dB change in MD"]
    if "expert_pearson_r" in B:
        md_lines.append(f"- Reference – ophthalmologists' CDR vs MD: r = {B['expert_pearson_r']:.3f} "
                        f"({fmt_p(B['expert_pearson_p'])}), n = {B['expert_n']}")
    md_lines += ["", "| Severity stage | n | Mean CDR ± SD | Median |", "|---|---|---|---|"]
    for s in B2["stages"]:
        ms = f"{s['mean_cdr']:.3f} ± {s['sd_cdr']:.3f}" if s["sd_cdr"] is not None else "—"
        med = f"{s['median_cdr']:.3f}" if s["median_cdr"] is not None else "—"
        md_lines.append(f"| {s['stage']} | {s['n']} | {ms} | {med} |")
    md_lines += ["", f"- Kruskal–Wallis {fmt_p(kw_p)}; trend across stages ρ = {tr:.3f} ({fmt_p(tp)})"]
    if B3:
        md_lines.append(f"- Vertical CDR identifies MD < −6 dB with AUC {B3['auc']:.3f}; best cut-off "
                        f"{B3['cutoff']:.2f} (sensitivity {B3['sensitivity']:.2f}, specificity {B3['specificity']:.2f})")
    md_lines += ["", "## Limitations",
                 "- PAPILA images are also part of SMDG-19, so some may have been seen during U-Net training.",
                 "- Cross-sectional data (one visit per eye); progression over time is Phase II.",
                 "- CDR explains only part of the variance in MD, supporting the Phase II plan to combine "
                 "structural and functional inputs."]
    (out / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")

    static = config.ROOT / "app" / "static" / "validation"
    static.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        shutil.copy(f, static / f.name)
    shutil.copy(out / "summary.json", static / "summary.json")

    print("\n".join(md_lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
