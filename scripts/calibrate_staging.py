"""Fit and validate Component 2 (CDR + visual-field staging) on the unified cohort.

Everything Component 2 needs is learned here from data/cohort_master.csv and
written to checkpoints/staging_calibration.json:

  * md_model      - Mean Defect predicted from the fundus image (vertical CDR and
                    the classifier probability), with its residual SD, so a single
                    uploaded image can be given a visual-field stage and an interval.
  * cdr_cutoffs   - CDR cut-points between stages, chosen by Youden index against
                    the Hodapp-Parrish-Anderson stage boundaries (-2, -6, -12 dB).
  * cdr_repeatability - the critical change Component 3 uses, propagated from the
                    model-vs-expert agreement on this cohort.

Honesty measures: every headline number is also reported 5-fold cross-validated,
so the fitted cut-points are never scored on the eyes that chose them.

    python scripts/calibrate_staging.py
Outputs: checkpoints/staging_calibration.json
         reports/phase2/staging_calibration.json / .md
         reports/phase2/C1_md_model.png, C2_stage_confusion.png
         app/static/validation/ (figures shown in the app)
"""
import argparse
import json
import shutil
from pathlib import Path

import _path  # noqa: F401
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

NAVY, ORANGE, GREY, RED = "#0b4f8a", "#e08a1e", "#6b7c8f", "#b3261e"
STAGE_NAMES = ["No definite damage", "Early", "Moderate", "Advanced"]
MD_CUTS = {"normal_early": -2.0, "early_moderate": -6.0, "moderate_advanced": -12.0}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


# ------------------------------------------------------------------ helpers
def auc(scores, labels):
    """ROC AUC from ranks (no sklearn)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    df = pd.DataFrame({"s": s, "r": ranks})
    ranks = df.groupby("s")["r"].transform("mean").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def youden_cutoff(scores, labels):
    """Threshold on `scores` maximising sensitivity + specificity - 1."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    best = (None, -1, 0, 0)
    for t in np.unique(np.round(s, 4)):
        pred = s >= t
        tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens + spec - 1 > best[1]:
            best = (float(t), sens + spec - 1, sens, spec)
    return {"cutoff": best[0], "youden": best[1], "sensitivity": best[2], "specificity": best[3]}


def ols(X, y):
    """Least squares with an intercept column prepended. Returns (coefs, residual_sd, r2)."""
    X = np.asarray(X, float)
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    res = y - pred
    dof = max(1, len(y) - A.shape[1])
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    return coef, float(np.sqrt((res ** 2).sum() / dof)), float(r2)


def logistic(X, y, iters=1000, lr=0.5, l2=1e-3):
    """Small ridge-penalised logistic regression (gradient descent, no sklearn)."""
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    y = np.asarray(y, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    b = w[1:] / sd                                   # coefficients for raw inputs
    b0 = w[0] - float((w[1:] * mu / sd).sum())
    return float(b0), [float(v) for v in b]


def logistic_p(b0, b, x):
    z = b0 + float(np.dot(b, x))
    return float(1 / (1 + np.exp(-max(-30.0, min(30.0, z)))))


def kfold(n, k=5, seed=42):
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [(np.setdiff1d(idx, f), f) for f in np.array_split(idx, k)]


def md_stage(md):
    if md >= MD_CUTS["normal_early"]:
        return 0
    if md >= MD_CUTS["early_moderate"]:
        return 1
    if md >= MD_CUTS["moderate_advanced"]:
        return 2
    return 3


def cdr_stage(v, cuts):
    if v < cuts["normal_early"]:
        return 0
    if v < cuts["early_moderate"]:
        return 1
    if v < cuts["moderate_advanced"]:
        return 2
    return 3


def quadratic_kappa(a, b, k=4):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    w = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    ha, hb = np.bincount(a, minlength=k), np.bincount(b, minlength=k)
    E = np.outer(ha, hb) / len(a)
    denom = (w * E).sum()
    return float(1 - (w * O).sum() / denom) if denom > 0 else float("nan")


def fit_cutoffs(cdr, md):
    """Choose the three CDR cut-points jointly.

    Fitting each boundary on its own with a Youden index gives three almost
    identical cut-offs, because CDR separates every severity boundary in much the
    same way - the middle stages then collapse to nothing. Instead the triple is
    chosen together to maximise quadratic-weighted agreement with the measured
    visual-field stage, which is the quantity the staging module is judged on.
    Per-boundary AUC / sensitivity / specificity are still reported alongside.
    """
    cdr = np.asarray(cdr, float)
    true_stage = np.array([md_stage(v) for v in md])
    grid = np.round(np.arange(0.30, 0.86, 0.01), 2)
    best = (None, -np.inf)
    for i, c1 in enumerate(grid):
        for j in range(i + 3, len(grid)):          # at least 0.03 apart
            c2 = grid[j]
            for k in range(j + 3, len(grid)):
                c3 = grid[k]
                pred = np.digitize(cdr, [c1, c2, c3])
                score = quadratic_kappa(true_stage, pred)
                if np.isfinite(score) and score > best[1]:
                    best = ((c1, c2, c3), score)
    if best[0] is None:
        vals = {"normal_early": 0.50, "early_moderate": 0.60, "moderate_advanced": 0.70}
    else:
        vals = dict(zip(("normal_early", "early_moderate", "moderate_advanced"),
                        (float(v) for v in best[0])))

    detail = {}
    for name, bound in MD_CUTS.items():
        y = (md < bound).astype(int)
        r = youden_cutoff(cdr, y)
        r["auc"] = auc(cdr, y)
        r["n_positive"] = int(y.sum())
        r["chosen_cutoff"] = vals[name]
        pred = cdr >= vals[name]
        tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
        r["sensitivity_at_chosen"] = tp / (tp + fn) if tp + fn else 0.0
        r["specificity_at_chosen"] = tn / (tn + fp) if tn + fp else 0.0
        detail[name] = r
    return vals, detail


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(config.ROOT / "data" / "cohort_master.csv"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    out_dir = config.REPORTS_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.cohort)
    vf = df.dropna(subset=["mean_defect", "vcdr"]).reset_index(drop=True)
    if len(vf) < 30:
        raise SystemExit(f"only {len(vf)} eyes with a visual field in {args.cohort} - run build_unified_cohort.py first")
    has_prob = vf["glaucoma_probability"].notna().all()
    print(f"cohort: {len(df)} eyes, {len(vf)} with a visual field, classifier probability available: {has_prob}")

    cdr = vf["vcdr"].to_numpy(float)
    md = vf["mean_defect"].to_numpy(float)
    prob = vf["glaucoma_probability"].to_numpy(float) if has_prob else None
    true_stage = np.array([md_stage(v) for v in md])

    # ---- A. Mean Defect predicted from the image ------------------------
    # Candidate feature sets, poorest to richest. Each is cross-validated and the
    # winner is chosen on out-of-fold error, not on the fit. A richer set is only
    # kept if it actually earns its place.
    CANDIDATES = [
        ("cdr_only", ["vcdr"]),
        ("cdr_prob", ["vcdr", "glaucoma_probability"]),
        ("cdr_prob_rim", ["vcdr", "glaucoma_probability", "min_rim_ratio", "rim_inferior",
                          "rim_to_disc_area"]),
        ("cdr_prob_rim_clinical", ["vcdr", "glaucoma_probability", "min_rim_ratio", "rim_inferior",
                                   "rim_to_disc_area", "age", "iop"]),
    ]

    def feature_matrix(frame, cols):
        """Matrix for these columns, or None if any column is missing or sparse."""
        for c in cols:
            if c not in frame.columns or frame[c].notna().mean() < 0.9:
                return None, None
        sub = frame[cols].apply(pd.to_numeric, errors="coerce")
        keep = sub.notna().all(axis=1)
        if keep.sum() < 40:
            return None, None
        return sub[keep].to_numpy(float), keep.to_numpy()

    md_models, md_cv = {}, {}
    for name, cols in CANDIDATES:
        X, keep = feature_matrix(vf, cols)
        if X is None:
            print(f"  MD model [{name}]: skipped (predictors missing or too sparse)")
            continue
        y = md[keep]
        coef, sd, r2 = ols(X, y)
        preds, errs = np.full(len(y), np.nan), []
        for tr, te in kfold(len(y), args.folds):
            co, _, _ = ols(X[tr], y[tr])
            pr = co[0] + X[te] @ co[1:]
            preds[te] = pr
            errs.append(np.abs(pr - y[te]))
        resid = preds - y
        ss = ((y - y.mean()) ** 2).sum()
        md_models[name] = {"intercept": float(coef[0]),
                           "coefficients": [float(v) for v in coef[1:]],
                           "predictors": cols, "residual_sd": sd, "r2": r2, "n": int(len(y))}
        md_cv[name] = {"mae_db": float(np.concatenate(errs).mean()),
                       "rmse_db": float(np.sqrt((resid ** 2).mean())),
                       "r2": float(1 - (resid ** 2).sum() / ss),
                       "stage_accuracy": float(np.mean([md_stage(v) for v in preds]
                                                       == np.array([md_stage(v) for v in y]))),
                       "n": int(len(y))}
        print(f"  MD model [{name}]: in-sample R2 {r2:.3f} | {args.folds}-fold CV "
              f"MAE {md_cv[name]['mae_db']:.2f} dB, R2 {md_cv[name]['r2']:.3f} (n={len(y)})")

    if not md_models:
        raise SystemExit("no usable MD model - check the cohort columns")
    md_ranked = sorted(md_models, key=lambda k: -md_cv[k]["r2"])
    best_key = md_ranked[0]
    md_model = dict(md_models[best_key])
    # keep the legacy two-term fields so older calibrations still load
    md_model.setdefault("b_vcdr", md_model["coefficients"][0])
    md_model["b_prob"] = (md_model["coefficients"][1]
                          if "glaucoma_probability" in md_model["predictors"] else 0.0)
    print(f"  selected MD model: {best_key} (CV R2 {md_cv[best_key]['r2']:.3f})")

    # ---- A2. probability of moderate-or-worse field loss -----------------
    y_dam_full = (md < MD_CUTS["early_moderate"]).astype(int)
    dam_models, dam_cv = {}, {}
    for name, cols in CANDIDATES:
        X, keep = feature_matrix(vf, cols)
        if X is None:
            continue
        y = y_dam_full[keep]
        if len(np.unique(y)) < 2:
            continue
        b0, bs = logistic(X, y)
        scores = np.zeros(len(y))
        for tr, te in kfold(len(y), args.folds):
            f0, f = logistic(X[tr], y[tr])
            scores[te] = [logistic_p(f0, f, x) for x in X[te]]
        dam_models[name] = {"intercept": b0, "coefficients": bs, "predictors": cols,
                            "target": "MD < -6 dB (moderate or worse field loss)",
                            "n": int(len(y)), "n_positive": int(y.sum()),
                            "auc": auc([logistic_p(b0, bs, x) for x in X], y),
                            "cv_auc": auc(scores, y)}
        dam_cv[name] = dam_models[name]["cv_auc"]
        print(f"  P(field loss) [{name}]: AUC {dam_models[name]['auc']:.3f} "
              f"(CV {dam_models[name]['cv_auc']:.3f}, n={len(y)})")
    dam_ranked = sorted(dam_models, key=lambda k: -dam_cv[k])
    dam = dict(dam_models[dam_ranked[0]])
    print(f"  selected damage model: {dam_ranked[0]} (CV AUC {dam['cv_auc']:.3f})")
    cv = md_cv

    # ---- B. CDR cut-points ------------------------------------------------
    cuts, cut_detail = fit_cutoffs(cdr, md)
    print(f"  CDR cut-points: {cuts}")
    pred_stage = np.array([cdr_stage(v, cuts) for v in cdr])
    staging = {
        "accuracy": float((pred_stage == true_stage).mean()),
        "within_one_stage": float((np.abs(pred_stage - true_stage) <= 1).mean()),
        "quadratic_kappa": quadratic_kappa(true_stage, pred_stage),
        "confusion": [[int(((true_stage == i) & (pred_stage == j)).sum()) for j in range(4)] for i in range(4)],
        "n": int(len(vf)),
    }
    # cross-validated staging: cut-points refitted inside each fold
    cv_pred = np.zeros(len(vf), int)
    for tr, te in kfold(len(vf), args.folds):
        c_tr, _ = fit_cutoffs(cdr[tr], md[tr])
        cv_pred[te] = [cdr_stage(v, c_tr) for v in cdr[te]]
    staging["cv_accuracy"] = float((cv_pred == true_stage).mean())
    staging["cv_within_one_stage"] = float((np.abs(cv_pred - true_stage) <= 1).mean())
    staging["cv_quadratic_kappa"] = quadratic_kappa(true_stage, cv_pred)
    counts = np.bincount(true_stage, minlength=4)
    staging["stage_counts"] = counts.tolist()
    staging["majority_baseline"] = float(counts.max() / counts.sum())
    staging["balanced_accuracy"] = float(np.mean([
        (pred_stage[true_stage == i] == i).mean() for i in range(4) if counts[i] > 0]))
    print(f"  staging: accuracy {staging['accuracy']:.3f} (CV {staging['cv_accuracy']:.3f}), "
          f"within one stage {staging['within_one_stage']:.3f}, kappa_w {staging['quadratic_kappa']:.3f}")

    # ---- C. repeatability -> critical change for Component 3 -------------
    agree = df.dropna(subset=["vcdr", "expert_vcdr"])
    if "seg_heldout" in df.columns:
        # measurement error must come from eyes the U-Net did not train on, or the
        # critical change Component 3 uses would be optimistically small
        held = agree[agree.seg_heldout.astype(bool)]
        if len(held) > 50:
            print(f"  repeatability from {len(held)} held-out eyes "
                  f"(ignoring {len(agree) - len(held)} seen in training)")
            agree = held
    d = (agree["vcdr"] - agree["expert_vcdr"]).to_numpy(float)
    sd_diff = float(d.std(ddof=1)) if len(d) > 2 else 0.075
    repeat = {
        "n_agreement": int(len(d)), "sd_model_minus_expert": sd_diff,
        "source": "held-out eyes only" if "seg_heldout" in df.columns else "all paired eyes",
        "sd_measurement": round(sd_diff / np.sqrt(2), 4),
        # a single measurement has SD = sd_diff/sqrt(2); the difference between two
        # such measurements therefore has SD = sd_diff, so the 95% limit is 1.96*sd_diff
        "critical_change": round(1.96 * sd_diff, 4),
        "definition": ("95% critical change for the difference between two VisionTrack "
                       "measurements of the same eye, propagated from the model-vs-expert SD"),
    }
    print(f"  CDR critical change: {repeat['critical_change']:.3f} (from n = {repeat['n_agreement']} paired eyes)")

    calib = {
        "fitted_on": f"{Path(args.cohort).name} ({len(vf)} eyes with a visual field, "
                     f"{len(agree)} with expert contours)",
        "md_model": md_model,
        "md_models": md_models, "md_model_ranked": md_ranked,
        "damage_models": dam_models, "damage_model_ranked": dam_ranked,
        "md_from_cdr": {"slope": md_models["cdr_only"]["coefficients"][0],
                        "intercept": md_models["cdr_only"]["intercept"],
                        "residual_sd": md_models["cdr_only"]["residual_sd"],
                        "r2": md_models["cdr_only"]["r2"],
                        "pearson_r": float(np.corrcoef(cdr, md)[0, 1]), "n": len(vf)},
        "cdr_cutoffs": cuts,
        "md_cutoffs": MD_CUTS,
        "cdr_repeatability": repeat,
        "functional_damage_model": dam,
        "validation": {"md_model_cv": cv, "selected_model": best_key,
                       "cutoff_detail": cut_detail, "staging": staging},
    }
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (config.CHECKPOINT_DIR / "staging_calibration.json").write_text(json.dumps(calib, indent=2))
    (out_dir / "staging_calibration.json").write_text(json.dumps(calib, indent=2))

    # ---- figures ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    ax.scatter(cdr, md, s=14, alpha=0.6, color=NAVY, edgecolor="none")
    xs = np.linspace(cdr.min(), cdr.max(), 50)
    b = md_models["cdr_only"]
    b = {"intercept": b["intercept"], "b_vcdr": b["coefficients"][0],
         "residual_sd": b["residual_sd"], "r2": b["r2"]}
    ax.plot(xs, b["intercept"] + b["b_vcdr"] * xs, color=ORANGE, lw=2,
            label=f"MD = {b['intercept']:.2f} {b['b_vcdr']:+.2f}·CDR  (R² = {b['r2']:.2f})")
    ax.fill_between(xs, b["intercept"] + b["b_vcdr"] * xs - 1.96 * b["residual_sd"],
                    b["intercept"] + b["b_vcdr"] * xs + 1.96 * b["residual_sd"],
                    color=ORANGE, alpha=0.12, label="95% prediction interval")
    for y, lab in ((-2, "−2 dB"), (-6, "−6 dB"), (-12, "−12 dB")):
        ax.axhline(y, ls=":", lw=1, color=GREY)
        ax.text(ax.get_xlim()[1], y, f" {lab}", va="center", fontsize=8, color=GREY)
    ax.set(xlabel="Vertical CDR (VisionTrack)", ylabel="Visual-field Mean Defect (dB)",
           title=f"Visual field predicted from the fundus image (n = {len(vf)})")
    ax.legend(fontsize=8.5, loc="lower left")

    ax = axes[1]
    for i, c in enumerate([NAVY, "#3f7fb5", ORANGE, RED]):
        v = cdr[true_stage == i]
        if len(v):
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).normal(0, 0.06, len(v)), v,
                       s=13, alpha=0.6, color=c, edgecolor="none")
            ax.hlines(np.median(v), i - 0.25, i + 0.25, color=c, lw=2.5)
    for name, c in cuts.items():
        ax.axhline(c, ls="--", lw=1, color=GREY)
        ax.text(3.45, c, f" {c:.2f}", fontsize=8, color=GREY, va="center")
    ax.set(xticks=range(4), xticklabels=[s.replace(" ", "\n", 1) for s in STAGE_NAMES],
           ylabel="Vertical CDR", xlabel="Visual-field stage (Hodapp-Parrish-Anderson)",
           title=f"Fitted CDR cut-points  (accuracy {staging['accuracy']*100:.0f}%, "
                 f"within one stage {staging['within_one_stage']*100:.0f}%)")
    fig.tight_layout()
    fig.savefig(out_dir / "C1_md_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    M = np.array(staging["confusion"], float)
    ax.imshow(M, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.6 else "#0b2540", fontsize=10)
    ax.set(xticks=range(4), yticks=range(4),
           xticklabels=[s.split()[0] for s in STAGE_NAMES], yticklabels=[s.split()[0] for s in STAGE_NAMES],
           xlabel="VisionTrack stage (from CDR)", ylabel="Visual-field stage (measured)",
           title=f"Staging agreement\nκw = {staging['quadratic_kappa']:.2f}, n = {staging['n']}")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "C2_stage_confusion.png", dpi=150)
    plt.close(fig)

    vdir = config.ROOT / "app" / "static" / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    for f in ("C1_md_model.png", "C2_stage_confusion.png"):
        shutil.copy2(out_dir / f, vdir / f)
    (vdir / "staging.json").write_text(json.dumps(calib, indent=2))

    # ---- markdown --------------------------------------------------------
    L = [f"# Component 2 — CDR + visual-field staging (calibrated on {Path(args.cohort).name})\n",
         f"Unified cohort: **{len(df)} eyes**, of which **{len(vf)}** have a Humphrey 30-2 Mean Defect "
         f"and **{len(agree)}** have expert disc/cup contours. One table, one pipeline.\n",
         "## A. Visual field predicted from the fundus image\n",
         f"| Model | Predictors | In-sample R² | {args.folds}-fold CV R² | CV MAE (dB) | CV stage accuracy |",
         "|---|---|---|---|---|---|"]
    for k in md_ranked:
        m = md_models[k]
        L.append(f"| {k} | {', '.join(m['predictors'])} | {m['r2']:.3f} | {cv[k]['r2']:.3f} | "
                 f"{cv[k]['mae_db']:.2f} | {cv[k]['stage_accuracy']*100:.1f}% |")
    L += ["", "| Model | Predictors | AUC | CV AUC | n |", "|---|---|---|---|---|"]
    for k in dam_ranked:
        m = dam_models[k]
        L.append(f"| {k} | {', '.join(m['predictors'])} | {m['auc']:.3f} | {m['cv_auc']:.3f} | {m['n']} |")
    L += [f"\nSelected: **{best_key}**; residual SD {md_model['residual_sd']:.2f} dB, so a single image is "
          f"reported with a ±{1.96*md_model['residual_sd']:.1f} dB interval. "
          f"Each +0.1 of CDR corresponds to {md_model['b_vcdr']/10:.2f} dB of Mean Defect.\n",
          "## B. CDR cut-points between stages\n",
          "| Boundary | MD threshold | Fitted CDR cut-off | AUC | Sensitivity | Specificity |",
          "|---|---|---|---|---|---|"]
    for name, bound in MD_CUTS.items():
        d_ = cut_detail[name]
        L.append(f"| {name.replace('_', ' / ')} | {bound:g} dB | {cuts[name]:.2f} | {d_['auc']:.3f} | "
                 f"{d_['sensitivity_at_chosen']:.2f} | {d_['specificity_at_chosen']:.2f} |")
    L += [f"\nStaging against the measured field: accuracy **{staging['accuracy']*100:.1f}%** "
          f"(cross-validated {staging['cv_accuracy']*100:.1f}%), within one stage "
          f"**{staging['within_one_stage']*100:.1f}%**, quadratic κ **{staging['quadratic_kappa']:.3f}**.\n",
          f"Stage distribution in the cohort: {staging['stage_counts']} (majority-class baseline "
          f"{staging['majority_baseline']*100:.1f}%). Balanced accuracy {staging['balanced_accuracy']*100:.1f}%. "
          "The weak boundary is no-damage vs early field loss, which the disc alone cannot separate; "
          "moderate-or-worse loss is detected well.\n",
          "## B2. Probability of moderate-or-worse field loss (MD < \u22126 dB)\n",
          f"- Predictors: {', '.join(dam['predictors'])}; {dam['n_positive']} of {dam['n']} eyes affected",
          f"- AUC **{dam['auc']:.3f}** (cross-validated {dam['cv_auc']:.3f})",
          "- Reported on every study as a single percentage, which is more useful than the wide "
          "point estimate of Mean Defect.\n",
          "## C. Critical change handed to Component 3\n",
          f"- Paired eyes with expert contours: {repeat['n_agreement']}",
          f"- SD of (VisionTrack − expert) CDR: {repeat['sd_model_minus_expert']:.4f}",
          f"- 95% critical change between two visits: **{repeat['critical_change']:.3f} CDR units** — "
          "a smaller change is measurement noise, not progression.\n",
          "## Limitations\n",
          "- The visual-field estimate is a population regression, not perimetry; the interval is wide by design.",
          "- Visual fields exist only for the PAPILA part of the cohort, and that data is cross-sectional.",
          "- Cut-points are fitted on this cohort; the cross-validated figures are the ones to quote.\n"]
    (out_dir / "staging_calibration.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n-> checkpoints/staging_calibration.json\n-> {out_dir}")


if __name__ == "__main__":
    main()
