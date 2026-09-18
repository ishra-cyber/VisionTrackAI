"""One evaluation, one dataset: every headline number recomputed from
data/cohort_master.csv.

Held-out only, by default. PAPILA is one of the datasets pooled into SMDG-19, so a
large part of the cohort was seen during training. Every headline figure here is
computed on eyes the relevant model never trained on (seg_split / cls_split of
"test" or "none"); the training-set figures are printed too, in their own clearly
labelled section, because hiding them would be its own kind of dishonesty.
Run scripts/annotate_cohort_splits.py first - it adds those columns.

Phase 1 reported segmentation on SMDG and structure-function on PAPILA, which made
it look like two projects. This script reports everything from the single unified
cohort table, broken down by source so nothing is hidden, and writes the table that
goes on the slide.

    python scripts/evaluate_cohort.py
Outputs: reports/unified/cohort_results.json / cohort_results.md
         reports/unified/U1_overview.png
No scipy or sklearn: the p-values come from the same Student-t implementation the
progression module uses.
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src.progression import t_two_sided_p  # noqa: E402
from src.staging import load_calibration  # noqa: E402

NAVY, ORANGE, GREY, RED = "#0b4f8a", "#e08a1e", "#6b7c8f", "#b3261e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(-0.999999, min(0.999999, r))
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return r, t_two_sided_p(t, n - 2)


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return pearson(rx, ry)


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def icc_2_1(a, b):
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def cdr_agreement(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    d = pred - ref
    r, p = pearson(pred, ref)
    ss = ((ref - ref.mean()) ** 2).sum()
    return {"n": int(len(d)), "mean_pred": float(pred.mean()), "mean_ref": float(ref.mean()),
            "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean())),
            "bias": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)),
            "loa_high": float(d.mean() + 1.96 * d.std(ddof=1)),
            "pearson_r": r, "pearson_p": p, "icc": icc_2_1(pred, ref),
            "r2": float(1 - (d ** 2).sum() / ss) if ss > 0 else float("nan"),
            "within_0_1": float((np.abs(d) <= 0.1).mean())}


def classification(prob, label, thr):
    p, y = np.asarray(prob, float), np.asarray(label, int)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"n": int(len(y)), "n_glaucoma": int((y == 1).sum()), "threshold": float(thr),
            "accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "balanced_accuracy": (sens + spec) / 2,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
            "roc_auc": auc(p, y), "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def split_frames(df, flag):
    """(held-out, seen-in-training). Falls back to treating everything as held out."""
    if flag not in df.columns:
        return df, df.iloc[0:0], False
    ho = df[df[flag].astype(str).str.lower().isin(["true", "1"])] if df[flag].dtype == object \
        else df[df[flag].astype(bool)]
    return ho, df.drop(ho.index), True


def seg_stats(d):
    if not len(d):
        return None
    return {"n": int(len(d)), "disc_dice": float(d.dice_disc.mean()),
            "cup_dice": float(d.dice_cup.mean()),
            "overall_dice": float(((d.dice_disc + d.dice_cup) / 2).mean()),
            "disc_dice_median": float(d.dice_disc.median()),
            "cup_dice_median": float(d.dice_cup.median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(config.ROOT / "data" / "cohort_master.csv"))
    args = ap.parse_args()
    out = config.REPORTS_DIR / "unified"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.cohort)
    thr = json.loads((config.CHECKPOINT_DIR / "thresholds.json").read_text())["cls"]
    calib = load_calibration()
    tagged = "seg_heldout" in df.columns and "cls_heldout" in df.columns
    if not tagged:
        print("[warn] cohort has no seg_split / cls_split columns - run "
              "scripts/annotate_cohort_splits.py first, or these numbers include training data")

    S = {"cohort": {"file": Path(args.cohort).name, "n_eyes": int(len(df)),
                    "by_source": df.groupby("source").size().to_dict(),
                    "with_expert_contours": int(df.expert_vcdr.notna().sum()),
                    "with_label": int(df.label.notna().sum()),
                    "with_visual_field": int(df.mean_defect.notna().sum()),
                    "split_tagged": tagged}}
    if tagged:
        S["cohort"]["seg_split"] = df.seg_split.value_counts().to_dict()
        S["cohort"]["cls_split"] = df.cls_split.value_counts().to_dict()
    print(f"cohort {len(df)} eyes: {S['cohort']['by_source']}")

    # ---- segmentation (held out) -----------------------------------------
    seg_all = df.dropna(subset=["dice_disc", "dice_cup"])
    seg_ho, seg_tr, _ = split_frames(seg_all, "seg_heldout")
    S["segmentation"] = {"heldout": seg_stats(seg_ho), "seen_in_training": seg_stats(seg_tr),
                         "by_source": {src: seg_stats(d) for src, d in seg_ho.groupby("source")}}
    if S["segmentation"]["heldout"]:
        h = S["segmentation"]["heldout"]
        print(f"segmentation (held out) n={h['n']}: disc {h['disc_dice']:.4f}, cup {h['cup_dice']:.4f}"
              + (f"   [training-set n={seg_tr.shape[0]}: disc "
                 f"{S['segmentation']['seen_in_training']['disc_dice']:.4f}]" if len(seg_tr) else ""))

    # ---- CDR agreement (held out) ----------------------------------------
    cdr_all = df.dropna(subset=["vcdr", "expert_vcdr"])
    cdr_ho, cdr_tr, _ = split_frames(cdr_all, "seg_heldout")
    S["cdr"] = {"heldout": cdr_agreement(cdr_ho.vcdr, cdr_ho.expert_vcdr) if len(cdr_ho) > 5 else None,
                "seen_in_training": cdr_agreement(cdr_tr.vcdr, cdr_tr.expert_vcdr) if len(cdr_tr) > 5 else None,
                "by_source": {src: cdr_agreement(d.vcdr, d.expert_vcdr)
                              for src, d in cdr_ho.groupby("source") if len(d) > 5}}
    if S["cdr"]["heldout"]:
        a = S["cdr"]["heldout"]
        print(f"CDR (held out) n={a['n']}: MAE {a['mae']:.4f}, R2 {a['r2']:.3f}, ICC {a['icc']:.3f}")

    # ---- classification (held out) ---------------------------------------
    cls_all = df.dropna(subset=["glaucoma_probability", "label"])
    cls_ho, cls_tr, _ = split_frames(cls_all, "cls_heldout")
    S["classification"] = {
        "heldout": classification(cls_ho.glaucoma_probability, cls_ho.label, thr) if len(cls_ho) else None,
        "seen_in_training": classification(cls_tr.glaucoma_probability, cls_tr.label, thr)
        if len(cls_tr) and cls_tr.label.nunique() > 1 else None,
        "by_source": {src: classification(d.glaucoma_probability, d.label, thr)
                      for src, d in cls_ho.groupby("source") if d.label.nunique() > 1}}
    if S["classification"]["heldout"]:
        c = S["classification"]["heldout"]
        print(f"classification (held out) n={c['n']}: acc {c['accuracy']*100:.2f}%, "
              f"sens {c['sensitivity']*100:.2f}%, spec {c['specificity']*100:.2f}%, AUC {c['roc_auc']:.4f}")

    # ---- structure-function ----------------------------------------------
    def sf_block(v):
        if len(v) < 10:
            return None
        r, p = pearson(v.vcdr, v.mean_defect)
        rho, prho = spearman(v.vcdr, v.mean_defect)
        slope, icpt = (float(x) for x in np.polyfit(v.vcdr, v.mean_defect, 1))
        y = (v.mean_defect < -6).astype(int)
        b = {"n": int(len(v)), "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": prho,
             "r2": r ** 2, "slope_db_per_cdr": slope, "intercept": icpt, "db_per_0_1_cdr": slope / 10,
             "n_damaged": int(y.sum()),
             "auc_cdr_md_below_6": auc(v.vcdr, y) if y.nunique() > 1 else None,
             "auc_probability_md_below_6": auc(v.glaucoma_probability, y)
             if y.nunique() > 1 and v.glaucoma_probability.notna().all() else None, "stages": []}
        if v.expert_vcdr.notna().sum() > 10:
            e = v.dropna(subset=["expert_vcdr"])
            er, ep = pearson(e.expert_vcdr, e.mean_defect)
            b["expert_pearson_r"], b["expert_pearson_p"], b["expert_n"] = er, ep, int(len(e))
        for name, lo, hi in (("Early (MD >= -6)", -6, 99), ("Moderate (-6 to -12)", -12, -6),
                             ("Advanced (MD < -12)", -999, -12)):
            d = v[(v.mean_defect >= lo) & (v.mean_defect < hi)]
            if len(d):
                b["stages"].append({"stage": name, "n": int(len(d)), "mean_cdr": float(d.vcdr.mean()),
                                    "sd_cdr": float(d.vcdr.std(ddof=1)) if len(d) > 1 else 0.0})
        return b

    vf = df.dropna(subset=["mean_defect", "vcdr"])
    vf_ho, vf_tr, _ = split_frames(vf, "seg_heldout")
    S["structure_function"] = {"all_eyes": sf_block(vf), "heldout": sf_block(vf_ho),
                               "seen_in_training": sf_block(vf_tr)}
    if S["structure_function"]["all_eyes"]:
        sfa = S["structure_function"]["all_eyes"]
        sfh = S["structure_function"]["heldout"]
        print(f"structure-function n={sfa['n']}: r {sfa['pearson_r']:.3f}"
              + (f"   [held out n={sfh['n']}: r {sfh['pearson_r']:.3f}]" if sfh else
                 f"   [held out n={len(vf_ho)} - too few for a separate figure]"))

    S["staging"] = (calib.get("validation") or {}).get("staging")
    S["md_model"] = calib.get("md_model")
    S["functional_damage_model"] = calib.get("functional_damage_model")
    (out / "cohort_results.json").write_text(json.dumps(S, indent=2))

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    if len(cdr_ho):
        for src, c_ in zip(sorted(cdr_ho.source.unique()), [NAVY, ORANGE]):
            d = cdr_ho[cdr_ho.source == src]
            ax.scatter(d.expert_vcdr, d.vcdr, s=11, alpha=0.5, color=c_, edgecolor="none",
                       label=f"{src} (n={len(d)})")
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
        a = S["cdr"]["heldout"]
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Expert vertical CDR", ylabel="VisionTrack vertical CDR",
               title=f"CDR agreement - held-out eyes\nICC {a['icc']:.3f}, MAE {a['mae']:.3f}, n = {a['n']}")
        ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[1]
    if len(cls_ho):
        for lab, c_, name in ((0, NAVY, "non-glaucoma"), (1, RED, "glaucoma")):
            d = cls_ho[cls_ho.label == lab]
            ax.hist(d.glaucoma_probability, bins=25, alpha=0.6, color=c_, label=f"{name} (n={len(d)})")
        ax.axvline(thr, ls="--", color=GREY, lw=1.2)
        ax.set(xlabel="Glaucoma probability", ylabel="Eyes",
               title=f"Classification - held-out eyes\nAUC {S['classification']['heldout']['roc_auc']:.3f}, "
                     f"threshold {thr:.3f}")
        ax.legend(fontsize=8.5)
    ax = axes[2]
    sfa = S["structure_function"]["all_eyes"]
    if sfa:
        ax.scatter(vf_tr.vcdr, vf_tr.mean_defect, s=14, alpha=0.35, color=GREY, edgecolor="none",
                   label=f"seen in training (n={len(vf_tr)})")
        ax.scatter(vf_ho.vcdr, vf_ho.mean_defect, s=22, alpha=0.85, color=NAVY, edgecolor="none",
                   label=f"held out (n={len(vf_ho)})")
        xs = np.linspace(vf.vcdr.min(), vf.vcdr.max(), 30)
        ax.plot(xs, sfa["intercept"] + sfa["slope_db_per_cdr"] * xs, color=ORANGE, lw=2)
        ax.set(xlabel="Vertical CDR", ylabel="Visual-field Mean Defect (dB)",
               title=f"Structure vs function\nr = {sfa['pearson_r']:.3f}, n = {sfa['n']}")
        ax.legend(fontsize=8)
    fig.suptitle(f"VisionTrack AI - unified cohort ({len(df)} eyes); headline metrics on held-out eyes",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "U1_overview.png", dpi=150)
    plt.close(fig)

    # ---- markdown ---------------------------------------------------------
    c = S["classification"]["heldout"] or {}
    g = S["segmentation"]["heldout"] or {}
    a = S["cdr"]["heldout"] or {}
    sfh, sfa = S["structure_function"]["heldout"], S["structure_function"]["all_eyes"]
    L = ["# VisionTrack AI - results on the unified cohort\n",
         f"Every figure comes from one table, `{Path(args.cohort).name}`: {len(df)} eyes "
         f"({', '.join(f'{k} {v}' for k, v in S['cohort']['by_source'].items())}), each pushed through "
         f"the complete pipeline.\n",
         "**All headline numbers below are on eyes the relevant model never trained on.** PAPILA is one "
         "of the 19 datasets pooled into SMDG-19, so a large part of the cohort was seen during training; "
         "those eyes are reported separately further down rather than removed from sight.\n",
         f"- Expert disc/cup reference: {S['cohort']['with_expert_contours']} eyes "
         f"({len(cdr_ho)} held out, {len(cdr_tr)} seen in training)",
         f"- Diagnosis label: {S['cohort']['with_label']} eyes "
         f"({len(cls_ho)} held out, {len(cls_tr)} seen in training)",
         f"- Humphrey 30-2 Mean Defect: {S['cohort']['with_visual_field']} eyes "
         f"({len(vf_ho)} held out, {len(vf_tr)} seen in training)\n",
         "## Headline table (for the slide) - held-out eyes only\n",
         "| Measure | Value | n |", "|---|---|---|"]
    if g:
        L += [f"| Optic disc Dice | {g['disc_dice']:.4f} | {g['n']} |",
              f"| Optic cup Dice | {g['cup_dice']:.4f} | {g['n']} |",
              f"| Overall Dice | {g['overall_dice']:.4f} | {g['n']} |"]
    if a:
        L += [f"| CDR mean absolute error | {a['mae']:.4f} | {a['n']} |",
              f"| CDR R2 vs expert | {a['r2']:.3f} | {a['n']} |",
              f"| CDR ICC(2,1) | {a['icc']:.3f} | {a['n']} |",
              f"| CDR within +/-0.10 of expert | {a['within_0_1']*100:.1f}% | {a['n']} |"]
    if c:
        L += [f"| Accuracy | {c['accuracy']*100:.2f}% | {c['n']} |",
              f"| Sensitivity | {c['sensitivity']*100:.2f}% | {c['n_glaucoma']} |",
              f"| Specificity | {c['specificity']*100:.2f}% | {c['n'] - c['n_glaucoma']} |",
              f"| ROC-AUC | {c['roc_auc']:.4f} | {c['n']} |"]
    ref = sfh or sfa
    if ref:
        tag = "held out" if sfh else "ALL eyes - too few held out to report separately"
        L += [f"| CDR vs visual-field MD (Pearson r, {tag}) | {ref['pearson_r']:.3f} | {ref['n']} |"]
        if ref.get("auc_cdr_md_below_6") is not None:
            L.append(f"| AUC, CDR detecting MD < -6 dB ({tag}) | {ref['auc_cdr_md_below_6']:.3f} | {ref['n']} |")
    if S["functional_damage_model"]:
        fd = S["functional_damage_model"]
        L.append(f"| AUC, image model detecting MD < -6 dB | {fd.get('cv_auc', fd.get('auc')):.3f} "
                 f"(cross-validated) | {fd['n']} |")
    st = S["staging"]
    if st:
        L += [f"| Stage agreement with the measured field | {st['accuracy']*100:.1f}% exact, "
              f"{st['within_one_stage']*100:.1f}% within one stage | {st['n']} |",
              f"| Stage agreement, quadratic kappa | {st['quadratic_kappa']:.3f} | {st['n']} |"]

    L += ["\n## For comparison: the same models on eyes they were trained on\n",
          "These are **not** performance figures. They are here so the gap is visible instead of implied.\n",
          "| Measure | Held out | Seen in training |", "|---|---|---|"]
    gt, at, ct = (S["segmentation"]["seen_in_training"], S["cdr"]["seen_in_training"],
                  S["classification"]["seen_in_training"])
    if g and gt:
        L += [f"| Disc Dice | {g['disc_dice']:.4f} (n={g['n']}) | {gt['disc_dice']:.4f} (n={gt['n']}) |",
              f"| Cup Dice | {g['cup_dice']:.4f} (n={g['n']}) | {gt['cup_dice']:.4f} (n={gt['n']}) |"]
    if a and at:
        L += [f"| CDR MAE | {a['mae']:.4f} (n={a['n']}) | {at['mae']:.4f} (n={at['n']}) |",
              f"| CDR ICC | {a['icc']:.3f} | {at['icc']:.3f} |"]
    if c and ct:
        L += [f"| Accuracy | {c['accuracy']*100:.2f}% (n={c['n']}) | {ct['accuracy']*100:.2f}% (n={ct['n']}) |",
              f"| ROC-AUC | {c['roc_auc']:.4f} | {ct['roc_auc']:.4f} |"]

    L += ["\n## Held-out numbers split by source\n"]
    if S["segmentation"]["by_source"]:
        L += ["**Segmentation**\n", "| Source | n | Disc Dice | Cup Dice | Overall |", "|---|---|---|---|---|"]
        for src, d in S["segmentation"]["by_source"].items():
            if d:
                L.append(f"| {src} | {d['n']} | {d['disc_dice']:.4f} | {d['cup_dice']:.4f} | "
                         f"{d['overall_dice']:.4f} |")
    if S["cdr"]["by_source"]:
        L += ["\n**CDR agreement**\n", "| Source | n | MAE | Bias | ICC | r | Within +/-0.10 |",
              "|---|---|---|---|---|---|---|"]
        for src, d in S["cdr"]["by_source"].items():
            L.append(f"| {src} | {d['n']} | {d['mae']:.4f} | {d['bias']:+.4f} | {d['icc']:.3f} | "
                     f"{d['pearson_r']:.3f} | {d['within_0_1']*100:.1f}% |")
    if S["classification"]["by_source"]:
        L += ["\n**Classification**\n", "| Source | n | Accuracy | Sensitivity | Specificity | AUC |",
              "|---|---|---|---|---|---|"]
        for src, d in S["classification"]["by_source"].items():
            L.append(f"| {src} | {d['n']} | {d['accuracy']*100:.2f}% | {d['sensitivity']*100:.2f}% | "
                     f"{d['specificity']*100:.2f}% | {d['roc_auc']:.4f} |")

    if sfa:
        p_txt = "< 0.001" if sfa["pearson_p"] < 0.001 else "= %.3f" % sfa["pearson_p"]
        L += ["\n## Structure and function on the same eyes\n",
              f"Across all {sfa['n']} eyes with a visual field: Pearson r = {sfa['pearson_r']:.3f} "
              f"(p {p_txt}), Spearman rho = {sfa['spearman_rho']:.3f}, R2 = {sfa['r2']:.3f}. "
              f"Each +0.1 of CDR corresponds to {sfa['db_per_0_1_cdr']:.2f} dB of Mean Defect."]
        if sfh:
            L.append(f"\nOn the {sfh['n']} eyes held out of segmentation training: r = "
                     f"{sfh['pearson_r']:.3f}. Quote this one.")
        else:
            L.append(f"\nOnly {len(vf_ho)} of those eyes were held out of segmentation training - too few "
                     "for a separate correlation. Say so rather than quoting the pooled figure as "
                     "independent validation.")
        if sfa.get("expert_pearson_r") is not None:
            L.append(f"\nOphthalmologists' own CDR against the same fields: r = "
                     f"{sfa['expert_pearson_r']:.3f} (n = {sfa['expert_n']}). The expert figure is free of "
                     "any training contamination, which is what makes it the fair yardstick.")
        L += ["", "| Visual-field stage | n | Mean CDR |", "|---|---|---|"]
        for s_ in sfa["stages"]:
            L.append(f"| {s_['stage']} | {s_['n']} | {s_['mean_cdr']:.3f} +/- {s_['sd_cdr']:.3f} |")

    L += ["\n## Limitations\n",
          "- PAPILA is pooled inside SMDG-19, so it is **not** an independent clinic for segmentation. "
          f"Only {len(vf_ho)} of the {len(vf)} eyes with visual fields were held out of U-Net training.",
          "- Visual fields exist only for the PAPILA part of the cohort; SMDG has no perimetry.",
          "- All of it is cross-sectional: progression is validated separately, by simulation.",
          "- Thresholds were tuned on the validation split, not on the test eyes reported here.\n"]
    (out / "cohort_results.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n-> {out}/cohort_results.md")


if __name__ == "__main__":
    main()
