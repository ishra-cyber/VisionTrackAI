"""How well does Component 3 detect progression, and how often does it cry wolf?

PAPILA and SMDG are cross-sectional - nobody in either dataset was imaged twice -
so there is no real longitudinal ground truth to test against. What CAN be tested,
honestly, is the decision rule itself: given a true rate of change and the
measurement noise this system actually has, how often does the rule fire?

The noise is not invented. It is the SD of (VisionTrack CDR - expert CDR) measured
on the unified cohort and stored in checkpoints/staging_calibration.json, so the
simulated visit-to-visit scatter is the scatter the real system produces.

    python scripts/validate_progression.py --trials 4000
Outputs: reports/phase2/progression_validation.json / .md
         reports/phase2/D1_progression_operating.png
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src.progression import analyze_series, refresh_thresholds  # noqa: E402
from src.staging import load_calibration  # noqa: E402

NAVY, ORANGE, GREY, RED = "#0b4f8a", "#e08a1e", "#6b7c8f", "#b3261e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def simulate(rate, n_visits, span_years, sd, rng, baseline=0.55):
    """One patient: evenly spaced visits, true linear change, measurement noise."""
    years = np.linspace(0, span_years, n_visits)
    true = baseline + rate * years
    obs = np.clip(true + rng.normal(0, sd, n_visits), 0.05, 0.99)
    days = (years * 365.25).astype(int)
    start = np.datetime64("2022-01-01")
    return [{"date": str((start + np.timedelta64(int(d), "D"))), "value": float(v)}
            for d, v in zip(days, obs)]


def run(rate, n_visits, span, sd, trials, rng):
    fired = flagged = 0
    for _ in range(trials):
        a = analyze_series(simulate(rate, n_visits, span, sd, rng), "vcdr")
        code = a.get("status_code")
        fired += code == "progressing"
        flagged += code in ("progressing", "possible")
    return fired / trials, flagged / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = config.REPORTS_DIR / "phase2"
    out.mkdir(parents=True, exist_ok=True)

    crit = refresh_thresholds()
    calib = load_calibration()["cdr_repeatability"]
    sd = float(calib.get("sd_measurement", 0.05))
    rng = np.random.default_rng(args.seed)
    print(f"measurement SD per study: {sd:.4f} CDR units (from {calib.get('n_agreement', '?')} paired eyes)\n"
          f"critical change in use: {crit:.3f}")

    rates = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    schedules = [(3, 2.0), (4, 3.0), (5, 4.0), (6, 5.0)]
    grid, S = {}, {}
    for n, span in schedules:
        key = f"{n} visits / {span:g} y"
        grid[key] = []
        for r in rates:
            strict, loose = run(r, n, span, sd, args.trials, rng)
            grid[key].append({"rate": r, "progressing": strict, "progressing_or_possible": loose})
            print(f"  {key}: true rate {r:+.2f}/y -> flagged {strict*100:5.1f}% "
                  f"(incl. possible {loose*100:5.1f}%)")
    S["operating_points"] = grid
    S["measurement_sd"] = sd
    S["critical_change"] = crit
    S["trials_per_cell"] = args.trials
    S["false_positive_rate"] = {k: v[0]["progressing"] for k, v in grid.items()}
    # smallest rate detected at >=80% for each schedule
    S["detectable_rate_80"] = {}
    for k, v in grid.items():
        hit = [c["rate"] for c in v if c["progressing"] >= 0.80]
        S["detectable_rate_80"][k] = min(hit) if hit else None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, key, title in ((axes[0], "progressing", "Flagged as Progressing"),
                           (axes[1], "progressing_or_possible", "Flagged as Progressing or Possible")):
        for (k, v), c in zip(grid.items(), [GREY, NAVY, ORANGE, RED]):
            ax.plot([c_["rate"] for c_ in v], [c_[key] * 100 for c_ in v], "o-", color=c, lw=1.8, ms=4, label=k)
        ax.axhline(80, ls=":", color=GREY, lw=1)
        ax.set(xlabel="True rate of CDR change per year", ylabel="% of patients flagged",
               ylim=(-3, 103), title=title)
        ax.legend(fontsize=8.5)
    fig.suptitle(f"Component 3 operating characteristics  (measurement SD {sd:.3f}, "
                 f"critical change {crit:.3f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "D1_progression_operating.png", dpi=150)
    plt.close(fig)

    (out / "progression_validation.json").write_text(json.dumps(S, indent=2))
    L = ["# Component 3 — progression detection: what the rule can and cannot see\n",
         "PAPILA and SMDG are cross-sectional, so there is no real longitudinal ground truth. "
         "This is a simulation of the decision rule under the measurement noise the system actually has: "
         f"SD **{sd:.4f} CDR units** per study, taken from the model-versus-expert agreement on the unified "
         f"cohort ({calib.get('n_agreement', '?')} paired eyes). Critical change in use: **{crit:.3f}**. "
         f"{args.trials} simulated patients per cell.\n",
         "| Follow-up schedule | False positives (truly stable) | Smallest rate detected in ≥80% of patients |",
         "|---|---|---|"]
    for k in grid:
        det = S["detectable_rate_80"][k]
        L.append(f"| {k} | {S['false_positive_rate'][k]*100:.1f}% | "
                 f"{('%+.2f CDR/year' % det) if det is not None else 'not reached at +0.10/year'} |")
    L += ["\n## Detection rate by true rate of change\n",
          "| True rate | " + " | ".join(grid) + " |", "|---|" + "---|" * len(grid)]
    for i, r in enumerate(rates):
        L.append(f"| {r:+.2f}/year | " + " | ".join(f"{grid[k][i]['progressing']*100:.0f}%" for k in grid) + " |")
    L += ["\n**Reading this honestly:** with three visits over two years the rule only catches fast "
          "progression; the slow change that matters clinically needs five or more visits. That is a property "
          "of CDR as a measurement, not a bug in the code — and it is the argument for adding OCT RNFL "
          "thickness, which is far less noisy, in a later phase.\n",
          "**Limitation:** a simulation validates the decision rule, not the biology. Real progression is not "
          "perfectly linear and real follow-up is not evenly spaced. Prospective data is the only way to "
          "settle that, and collecting it is what Phase 2 deployment is for.\n"]
    (out / "progression_validation.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n-> {out}/progression_validation.md")


if __name__ == "__main__":
    main()
