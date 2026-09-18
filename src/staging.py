"""Component 2 - CDR + visual-field staging.

Turns a structural measurement (vertical CDR from Component 1) and, when it is
available, a functional measurement (Humphrey 30-2 Mean Defect in dB) into a
single glaucoma stage.

Three things happen here:

1. `functional_stage`  - stages the visual field with the Hodapp-Parrish-Anderson
   criteria (the scheme used in the clinic).
2. `structural_grade`  - stages the optic nerve head from the vertical CDR, using
   cut-points fitted on the unified cohort (scripts/calibrate_staging.py), not
   textbook guesses.
3. `combined_stage`    - fuses the two. The stage is the worse of structure and
   function, and any disagreement between them is reported rather than hidden,
   because a structure/function mismatch is itself a clinical finding.

When no visual field has been done, `estimate_md` predicts the Mean Defect from
the CDR with the regression fitted on the cohort and reports a 95% prediction
interval. That estimate is explicitly labelled as an estimate everywhere it is
used - it does not replace perimetry.

Pure Python + the calibration file; no torch, no scipy.
"""
import json
import math
from pathlib import Path

import config

CALIB_PATH = config.CHECKPOINT_DIR / "staging_calibration.json"

# Fallback if the calibration file is missing. These are the values fitted on the
# PAPILA visual-field cohort on 2026-09-16; calibrate_staging.py overwrites them.
DEFAULT_CALIBRATION = {
    "functional_damage_model": {"intercept": -4.0, "coefficients": [6.0],
                                "predictors": ["vertical_cdr"], "auc": 0.80, "cv_auc": 0.80,
                                "target": "MD < -6 dB (moderate or worse field loss)"},
    "fitted_on": "fallback (PAPILA, n=164)",
    "md_from_cdr": {"slope": -18.309, "intercept": 4.206, "residual_sd": 3.55,
                    "r2": 0.243, "pearson_r": -0.493, "n": 164},
    "md_model": {"intercept": 4.206, "b_vcdr": -18.309, "b_prob": 0.0,
                 "residual_sd": 3.55, "r2": 0.243, "n": 164,
                 "predictors": ["vertical_cdr"]},
    "cdr_cutoffs": {"normal_early": 0.50, "early_moderate": 0.6166, "moderate_advanced": 0.70},
    "md_cutoffs": {"normal_early": -2.0, "early_moderate": -6.0, "moderate_advanced": -12.0},
    "cdr_repeatability": {"sd_measurement": 0.047, "critical_change": 0.13},
}

STAGE_NAMES = {0: "No definite damage", 1: "Early", 2: "Moderate", 3: "Advanced"}
# Suggested review interval in months, by fused stage.
STAGE_FOLLOW_UP = {0: 12, 1: 6, 2: 4, 3: 3}
STAGE_NOTE = {
    0: "No structural or functional damage detected. Routine screening interval.",
    1: "Early damage. Baseline perimetry and a repeat field to establish the trend.",
    2: "Moderate damage. Confirm with perimetry and review treatment adequacy.",
    3: "Advanced damage. Priority referral; close functional monitoring.",
}

_cache = {}


def load_calibration(path=CALIB_PATH):
    """Read the fitted calibration, falling back to the built-in values."""
    key = str(path)
    if key not in _cache:
        calib = json.loads(json.dumps(DEFAULT_CALIBRATION))
        p = Path(path)
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                for k, v in loaded.items():
                    if isinstance(v, dict) and isinstance(calib.get(k), dict):
                        calib[k].update(v)
                    else:
                        calib[k] = v
            except (ValueError, OSError):
                pass
        _cache[key] = calib
    return _cache[key]


def clear_cache():
    _cache.clear()



def _select_model(models, ranked, features):
    """Best-ranked model whose predictors are all available for this eye.

    The calibration stores a family of models, from CDR alone up to CDR plus rim
    geometry plus age and IOP. A study that has only an image gets the simple one;
    a study with clinical data gets the better one. Nothing is imputed.
    """
    if not models:
        return None
    for name in (ranked or sorted(models)):
        m = models.get(name)
        if not m:
            continue
        preds = m.get("predictors") or []
        vals = [features.get(k) for k in preds]
        if preds and all(v is not None for v in vals):
            return {**m, "name": name, "values": [float(v) for v in vals]}
    return None


def _linear(m):
    return m["intercept"] + sum(c * v for c, v in zip(m["coefficients"], m["values"]))


def build_features(vcdr=None, glaucoma_probability=None, rim=None, age=None, iop=None, extra=None):
    """Assemble the predictor dict the staging models expect."""
    f = {"vcdr": vcdr, "glaucoma_probability": glaucoma_probability,
         "age": age, "iop": iop}
    if rim:
        ratio = rim.get("sector_rim_ratio") or {}
        f.update({"min_rim_ratio": rim.get("min_rim_to_disc_ratio"),
                  "rim_inferior": ratio.get("inferior"), "rim_superior": ratio.get("superior"),
                  "rim_nasal": ratio.get("nasal"), "rim_temporal": ratio.get("temporal"),
                  "rim_to_disc_area": rim.get("rim_to_disc_area_ratio")})
    if extra:
        f.update({k: v for k, v in extra.items() if v is not None})
    return {k: v for k, v in f.items() if v is not None}


# --------------------------------------------------------------- functional
def _md_code(md):
    c = load_calibration()["md_cutoffs"]
    if md >= c["normal_early"]:
        return 0
    if md >= c["early_moderate"]:
        return 1
    if md >= c["moderate_advanced"]:
        return 2
    return 3


def functional_stage(md):
    """Hodapp-Parrish-Anderson stage from the visual-field Mean Defect (dB)."""
    if md is None:
        return None
    md = float(md)
    code = _md_code(md)
    return {"code": code, "label": STAGE_NAMES[code], "md": round(md, 2),
            "criteria": "Hodapp-Parrish-Anderson"}


# --------------------------------------------------------------- structural
def structural_grade(vcdr):
    """Optic nerve head grade from the vertical CDR, using the fitted cut-points."""
    if vcdr is None:
        return None
    vcdr = float(vcdr)
    c = load_calibration()["cdr_cutoffs"]
    if vcdr < c["normal_early"]:
        code = 0
    elif vcdr < c["early_moderate"]:
        code = 1
    elif vcdr < c["moderate_advanced"]:
        code = 2
    else:
        code = 3
    return {"code": code, "label": STAGE_NAMES[code], "vcdr": round(vcdr, 4),
            "cutoffs": dict(c), "criteria": "CDR cut-points fitted on the unified cohort"}


def estimate_md(vcdr, glaucoma_probability=None, features=None):
    """Predict this eye's visual-field Mean Defect, with a 95% interval.

    Uses the richest calibrated model whose inputs this study actually has - CDR
    alone if that is all there is, or CDR plus rim geometry plus age and IOP when
    they were entered. This is what lets the app show a visual-field stage for a
    single uploaded fundus image with no perimetry.
    """
    if vcdr is None:
        return None
    calib = load_calibration()
    f = dict(features or {})
    f.setdefault("vcdr", vcdr)
    if glaucoma_probability is not None:
        f.setdefault("glaucoma_probability", glaucoma_probability)

    m = _select_model(calib.get("md_models"), calib.get("md_model_ranked"), f)
    if m:
        md, sd, r2, n = _linear(m), m["residual_sd"], m["r2"], m["n"]
        predictors, name = ", ".join(m["predictors"]), m["name"]
        slope = m["coefficients"][0]
    else:
        u = calib["md_from_cdr"]
        md = u["slope"] * float(vcdr) + u["intercept"]
        sd, r2, n = u["residual_sd"], u["r2"], u["n"]
        predictors, name, slope = "vertical CDR", "cdr_only", u["slope"]
    half = 1.96 * sd
    return {"md_estimate": round(md, 2),
            "ci_low": round(md - half, 2), "ci_high": round(md + half, 2),
            "r2": round(r2, 3), "n_fit": n, "residual_sd": round(sd, 2),
            "predictors": predictors, "model": name,
            "per_0_1_cdr_db": round(slope / 10.0, 2),
            "stage": STAGE_NAMES[_md_code(md)], "stage_code": _md_code(md),
            "note": "Model estimate from the fundus image - does not replace perimetry."}


# --------------------------------------------------------------- fusion
def damage_probability(vcdr, glaucoma_probability=None, features=None):
    """P(visual-field Mean Defect < -6 dB) for this eye, from what the study has."""
    if vcdr is None:
        return None
    calib = load_calibration()
    f = dict(features or {})
    f.setdefault("vcdr", vcdr)
    if glaucoma_probability is not None:
        f.setdefault("glaucoma_probability", glaucoma_probability)

    m = _select_model(calib.get("damage_models"), calib.get("damage_model_ranked"), f)
    if m is None:
        legacy = calib.get("functional_damage_model") or {}
        coefs, preds = legacy.get("coefficients") or [], legacy.get("predictors") or []
        vals = [f.get("vcdr") if k in ("vcdr", "vertical_cdr") else f.get(k) for k in preds]
        if not coefs or any(v is None for v in vals):
            return None
        m = {**legacy, "name": "legacy", "values": [float(v) for v in vals]}
    z = m["intercept"] + sum(c * v for c, v in zip(m["coefficients"], m["values"]))
    p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    return {"probability": round(p, 4), "percent": round(p * 100, 1),
            "auc": m.get("cv_auc") or m.get("auc"), "model": m.get("name"),
            "predictors": ", ".join(m.get("predictors") or []),
            "target": m.get("target", "MD < -6 dB")}


def combined_stage(vcdr, md=None, glaucoma_probability=None, features=None):
    """Fuse structure and function into one stage.

    vcdr - vertical cup-to-disc ratio from Component 1
    md   - measured visual-field Mean Defect in dB, or None
    glaucoma_probability - classifier output, used only as supporting evidence
    """
    struct = structural_grade(vcdr)
    func = functional_stage(md)
    est = estimate_md(vcdr, glaucoma_probability, features) if md is None else None

    if struct is None and func is None:
        return {"available": False,
                "reason": "No optic disc measurement and no visual field available."}

    # The fused stage uses the measured field when there is one. When there is
    # not, the CDR cut-points carry the stage: the regression estimate of Mean
    # Defect is reported for context but is far too wide to stage an eye with.
    codes = [s["code"] for s in (struct, func) if s is not None]
    code = max(codes)
    source = ("structure and measured visual field" if struct and func
              else "structure + estimated visual field" if struct and est
              else "structure only" if struct else "visual field only")

    concordance = None
    if struct and func:
        d = struct["code"] - func["code"]
        if d == 0:
            concordance = {"agree": True, "delta": 0, "label": "Structure and function agree"}
        else:
            concordance = {
                "agree": False, "delta": d,
                "label": ("Structural damage exceeds field loss (pre-perimetric pattern)"
                          if d > 0 else "Field loss exceeds structural damage - review disc and repeat field"),
            }

    evidence = []
    if struct:
        evidence.append(f"Vertical CDR {struct['vcdr']:.3f} -> {struct['label'].lower()} structural grade")
    if func:
        evidence.append(f"Visual-field MD {func['md']:+.2f} dB -> {func['label'].lower()} (Hodapp-Parrish-Anderson)")
    elif est:
        evidence.append(f"No perimetry on file; MD estimated at {est['md_estimate']:+.2f} dB "
                        f"({est['ci_low']:+.2f} to {est['ci_high']:+.2f})")
    if glaucoma_probability is not None:
        evidence.append(f"Classifier glaucoma probability {glaucoma_probability * 100:.1f}%")

    return {
        "available": True,
        "stage": code,
        "stage_label": STAGE_NAMES[code],
        "basis": source,
        "structural": struct,
        "functional": func,
        "estimated_functional": est,
        "damage_probability": (damage_probability(vcdr, glaucoma_probability, features)
                               if func is None else None),
        "concordance": concordance,
        "confidence": ("measured visual field" if func else
                       "visual field estimated from the fundus image - confirm with perimetry"),
        "follow_up_months": STAGE_FOLLOW_UP[code],
        "recommendation": STAGE_NOTE[code],
        "evidence": evidence,
    }


def stage_from_result(result, md=None, age=None, iop=None):
    """Convenience wrapper over a Component-1 result dict (schema v2.0)."""
    c1 = result.get("component1") or {}
    cdr, rim = c1.get("cdr") or {}, c1.get("rim")
    cls = result.get("classification") or {}
    if md is None:
        md = ((result.get("visual_field") or {}).get("mean_defect_db"))
    feats = build_features(cdr.get("vertical_cdr"), cls.get("glaucoma_probability"), rim, age, iop)
    return combined_stage(cdr.get("vertical_cdr"), md, cls.get("glaucoma_probability"), feats)


__all__ = ["load_calibration", "clear_cache", "functional_stage", "structural_grade", "build_features",
           "estimate_md", "damage_probability", "combined_stage", "stage_from_result", "STAGE_NAMES",
           "STAGE_FOLLOW_UP", "CALIB_PATH"]
