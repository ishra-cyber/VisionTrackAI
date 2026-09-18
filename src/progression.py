"""Component 3 - visual-field and CDR progression monitoring.

Given a patient's visits over time, decide whether the optic nerve head and the
visual field are stable or deteriorating.

Two independent criteria are combined, which is how progression is judged
clinically:

* Trend analysis - ordinary least squares of the measurement against time in
  years, with the slope tested against zero (Student t on n-2 degrees of
  freedom). This answers "is there a real rate of change?".
* Event analysis - has the measurement moved further from baseline than the
  measurement error of the method itself? The critical change for CDR is not a
  guess: it comes from the agreement study in
  reports/structure_function/summary.json, propagated to the difference of two
  independent measurements. For the visual field it is the accepted 2 dB
  test-retest limit for Mean Defect.

A verdict is only issued when both the number of visits and the follow-up span
support it; otherwise the module says so instead of inventing a trend.

Pure Python - no numpy, no scipy - so the Flask app can call it per request.
"""
import math
from datetime import date, datetime

from .staging import load_calibration

# Rate thresholds. CDR: a tenth of a ratio per year is a fast structural change.
# MD: -1 dB/year is the accepted "fast progression" rate in glaucoma practice.
METRICS = {
    "vcdr": {
        "label": "Vertical CDR", "unit": "", "direction": -1,   # increasing = worse
        "critical_change": 0.13, "fast_rate": 0.05, "limit": 0.80, "decimals": 3,
    },
    "md": {
        "label": "Visual-field MD", "unit": " dB", "direction": 1,  # decreasing = worse
        "critical_change": 2.0, "fast_rate": 1.0, "limit": -12.0, "decimals": 2,
    },
}

MIN_VISITS = 3
MIN_SPAN_YEARS = 0.5


# ------------------------------------------------------------------ statistics
def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p-value of Student's t statistic."""
    if df <= 0:
        return 1.0
    return float(_betai(df / 2.0, 0.5, df / (df + t * t)))


def _to_date(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v)[:10]
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def linear_trend(years, values):
    """OLS of value against time in years. Returns slope per year and its p-value."""
    n = len(values)
    if n < 2:
        return None
    mx = sum(years) / n
    my = sum(values) / n
    sxx = sum((x - mx) ** 2 for x in years)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(years, values))
    syy = sum((y - my) ** 2 for y in values)
    slope = sxy / sxx
    intercept = my - slope * mx
    df = n - 2
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(years, values))
    r2 = 1.0 - ss_res / syy if syy > 0 else 0.0
    if df > 0 and ss_res > 0:
        se = math.sqrt(ss_res / df / sxx)
        t = slope / se if se > 0 else 0.0
        p = t_two_sided_p(t, df)
    else:
        se, t, p = 0.0, 0.0, 1.0 if df <= 0 else 0.0
    return {"slope_per_year": slope, "intercept": intercept, "se": se, "t": t, "p": p,
            "r2": max(0.0, min(1.0, r2)), "n": n, "df": df}


# ------------------------------------------------------------------ analysis
def analyze_series(visits, metric="vcdr"):
    """visits: iterable of {'date': ..., 'value': ...} for one eye and one metric."""
    spec = METRICS[metric]
    pts = []
    for v in visits:
        d = _to_date(v.get("date"))
        val = v.get("value")
        if d is None or val is None:
            continue
        pts.append((d, float(val)))
    pts.sort(key=lambda p: p[0])
    out = {"metric": metric, "label": spec["label"], "unit": spec["unit"],
           "n_visits": len(pts), "points": [{"date": d.isoformat(), "value": val} for d, val in pts]}
    if len(pts) < 2:
        out.update(status="Insufficient data", status_code="insufficient",
                   detail=f"{len(pts)} visit(s) on record; at least {MIN_VISITS} are needed for a trend.")
        return out

    t0 = pts[0][0]
    years = [(d - t0).days / 365.25 for d, _ in pts]
    values = [v for _, v in pts]
    span = years[-1]
    baseline, latest = values[0], values[-1]
    change = latest - baseline
    worsening_change = -change * spec["direction"] if spec["direction"] == -1 else -change
    # worsening is positive when the eye got worse, for either metric
    worsening_change = (latest - baseline) if spec["direction"] == -1 else (baseline - latest)

    fit = linear_trend(years, values)
    worsening_rate = fit["slope_per_year"] if spec["direction"] == -1 else -fit["slope_per_year"]

    out.update({
        "first_visit": pts[0][0].isoformat(), "last_visit": pts[-1][0].isoformat(),
        "span_years": round(span, 2), "baseline": round(baseline, spec["decimals"]),
        "latest": round(latest, spec["decimals"]), "change": round(change, spec["decimals"]),
        "worsening_change": round(worsening_change, spec["decimals"]),
        "rate_per_year": round(fit["slope_per_year"], spec["decimals"] + 1),
        "worsening_rate_per_year": round(worsening_rate, spec["decimals"] + 1),
        "p_value": round(fit["p"], 4), "r2": round(fit["r2"], 3),
        "critical_change": spec["critical_change"],
        "fit": {"slope": fit["slope_per_year"], "intercept": fit["intercept"]},
    })

    if len(pts) < MIN_VISITS or span < MIN_SPAN_YEARS:
        out.update(status="Insufficient data", status_code="insufficient",
                   detail=(f"{len(pts)} visits over {span:.1f} years. A verdict needs at least "
                           f"{MIN_VISITS} visits spanning {MIN_SPAN_YEARS} years."))
        return out

    event = worsening_change >= spec["critical_change"]
    trend = fit["p"] < 0.05 and worsening_rate > 0
    fast = worsening_rate >= spec["fast_rate"]

    if trend and event:
        code, status = "progressing", "Progressing"
    elif trend or event:
        code, status = "possible", "Possible progression"
    elif worsening_rate < 0 and fit["p"] < 0.05:
        code, status = "improving", "Improving"
    else:
        code, status = "stable", "Stable"

    reasons = []
    reasons.append(("Change from baseline {:+.{d}f}{u} "
                    + ("exceeds" if event else "is within") +
                    " the {c:.{d}f}{u} measurement-error limit").format(
        change, d=spec["decimals"], u=spec["unit"], c=spec["critical_change"]))
    reasons.append("Trend {:+.{d}f}{u}/year (p = {:.3f}, R² = {:.2f}){}".format(
        fit["slope_per_year"], fit["p"], fit["r2"],
        " - faster than the {:.2f}{}/year fast-progression rate".format(spec["fast_rate"], spec["unit"])
        if fast and trend else "", d=spec["decimals"] + 1, u=spec["unit"]))

    projection = None
    if code in ("progressing", "possible") and worsening_rate > 1e-6:
        remaining = ((spec["limit"] - latest) if spec["direction"] == -1 else (latest - spec["limit"]))
        if remaining > 0:
            projection = {"limit": spec["limit"], "years": round(remaining / worsening_rate, 1)}

    out.update({"status": status, "status_code": code, "event_criterion": bool(event),
                "trend_criterion": bool(trend), "fast": bool(fast and trend),
                "reasons": reasons, "projection": projection})
    return out


def analyze_patient(rows):
    """rows: dicts with date, eye, vcdr and optional md. Returns a per-eye analysis."""
    by_eye = {}
    for r in rows:
        eye = (r.get("eye") or "OD").upper()
        by_eye.setdefault(eye, {"vcdr": [], "md": []})
        if r.get("vcdr") is not None:
            by_eye[eye]["vcdr"].append({"date": r.get("date") or r.get("visit_date"), "value": r["vcdr"]})
        if r.get("md") is not None:
            by_eye[eye]["md"].append({"date": r.get("date") or r.get("visit_date"), "value": r["md"]})

    out, worst = {}, None
    order = {"progressing": 3, "possible": 2, "stable": 1, "improving": 1, "insufficient": 0}
    for eye, series in sorted(by_eye.items()):
        analyses = {m: analyze_series(series[m], m) for m in ("vcdr", "md") if series[m]}
        codes = [a.get("status_code", "insufficient") for a in analyses.values()]
        eye_code = max(codes, key=lambda c: order.get(c, 0)) if codes else "insufficient"
        out[eye] = {"metrics": analyses, "status_code": eye_code,
                    "status": next((a["status"] for a in analyses.values()
                                    if a.get("status_code") == eye_code), "Insufficient data")}
        if worst is None or order.get(eye_code, 0) > order.get(out[worst]["status_code"], 0):
            worst = eye

    summary = None
    if worst is not None:
        w = out[worst]
        summary = {"eye": worst, "status": w["status"], "status_code": w["status_code"]}
    return {"eyes": out, "summary": summary}


def chart(analysis, w=760, h=230, left=46, right=14, top=12, bottom=28):
    """Plot-ready coordinates for the trend chart drawn by the app (no matplotlib)."""
    pts = analysis.get("points") or []
    if len(pts) < 2:
        return None
    spec = METRICS[analysis["metric"]]
    dates = [_to_date(p["date"]) for p in pts]
    vals = [p["value"] for p in pts]
    t0, t1 = dates[0], dates[-1]
    span_days = max(1, (t1 - t0).days)

    if analysis["metric"] == "vcdr":
        lo, hi = 0.0, 1.0
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        lo = min(-30.0, min(vals) - 3)
        hi = max(2.0, max(vals) + 2)
        step = 5 if hi - lo <= 40 else 10
        ticks = [lo + i * step for i in range(int((hi - lo) / step) + 1)]

    def X(d):
        return left + (d - t0).days / span_days * (w - left - right)

    def Y(v):
        return top + (hi - v) / (hi - lo) * (h - top - bottom)

    out = {"w": w, "h": h, "left": left, "right": right, "top": top, "bottom": bottom,
           "points": [{"x": round(X(d), 1), "y": round(Y(v), 1), "date": d.isoformat(),
                       "value": round(v, spec["decimals"])} for d, v in zip(dates, vals)],
           "ticks": [{"y": round(Y(t), 1), "label": ("%.1f" % t) if analysis["metric"] == "vcdr" else ("%g" % t)}
                     for t in ticks],
           "xlabels": [{"x": round(X(d), 1), "label": d.isoformat()} for d in (dates[0], dates[-1])]}

    fit = analysis.get("fit")
    if fit:
        y0 = fit["intercept"]
        y1 = fit["intercept"] + fit["slope"] * (span_days / 365.25)
        out["line"] = {"x1": round(X(t0), 1), "y1": round(Y(y0), 1),
                       "x2": round(X(t1), 1), "y2": round(Y(y1), 1)}
    base = vals[0]
    worse = base + spec["critical_change"] if spec["direction"] == -1 else base - spec["critical_change"]
    out["noise"] = {"y_base": round(Y(base), 1), "y_limit": round(Y(worse), 1),
                    "label": f"measurement-error limit ({spec['critical_change']:g}{spec['unit']})"}
    return out


def critical_change_from_agreement(sd_difference):
    """CDR change that exceeds measurement error.

    sd_difference is the SD of (model - expert) from the agreement study. The
    expert contributes roughly as much variance as the model, so the model's own
    SD is sd_difference / sqrt(2); the difference between two model measurements
    then has SD = sd_difference, and the 95% critical change is 1.96 x that.
    """
    return 1.96 * float(sd_difference)


def refresh_thresholds():
    """Adopt the CDR critical change stored in the staging calibration file."""
    rep = load_calibration().get("cdr_repeatability") or {}
    if rep.get("critical_change"):
        METRICS["vcdr"]["critical_change"] = float(rep["critical_change"])
    return METRICS["vcdr"]["critical_change"]


__all__ = ["analyze_series", "analyze_patient", "linear_trend", "t_two_sided_p", "chart",
           "critical_change_from_agreement", "refresh_thresholds", "METRICS",
           "MIN_VISITS", "MIN_SPAN_YEARS"]
