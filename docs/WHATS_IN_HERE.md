# VisionTrack AI — Phase 2 files (18 Sep 2026)

Drop this folder's contents over your `VisionTrackAI/` project folder, keeping the
subfolder structure. Everything here is already on your machine at
`Downloads\VisionTrackAI\VisionTrackAI\` — this copy is for backup, sharing or a second machine.

## New files

| Path | What it is |
|---|---|
| `src/staging.py` | **Component 2** — CDR + visual-field staging, structure/function fusion, estimated Mean Defect, probability of field loss. Pure Python, no torch or scipy. |
| `src/progression.py` | **Component 3** — trend + event progression rule, Student-t p-value without scipy, chart coordinates for the app. |
| `src/papila.py` | PAPILA readers (clinical sheets, expert contours, image index), shared by the cohort builder. |
| `scripts/build_unified_cohort.py` | Runs the whole pipeline over PAPILA + SMDG test eyes → `data/cohort_master.csv`. **Run this first.** |
| `scripts/calibrate_staging.py` | Fits and cross-validates Component 2 → `checkpoints/staging_calibration.json`, `reports/phase2/`. |
| `scripts/evaluate_cohort.py` | Recomputes every headline metric from the one cohort → `reports/unified/`. |
| `scripts/validate_progression.py` | Validates the progression rule under real measurement noise → `reports/phase2/`. |
| `scripts/seed_demo_patients.py` | Creates demo patients with several visits each, for the Progression page. |
| `app/templates/progression.html` | The new Progression page. |
| `checkpoints/staging_calibration.json` | **Interim** calibration, fitted on the existing per-image reports so the app works before you rebuild the cohort. `calibrate_staging.py` overwrites it with the real one. |
| `docs/DEMO_RUN_SHEET.md` | Demonstration order, numbers to quote, reviewer questions and answers, troubleshooting. |
| `docs/VisionTrack_Demo_Video_Script.md` | Spoken narration script, about 7 minutes. Numbers marked **[confirm]** need updating after `evaluate_cohort.py`. |

## Changed files

| Path | What changed |
|---|---|
| `app/app.py` | Database migrated in place (`md`, `stage`, `stage_label`, `md_source`, `damage_prob`); staging computed on every analysis; JSON is now schema v2.0 with `component2`; new `/progression`, `/api/progression/<patient_id>`, `/api/calibration` routes. |
| `app/templates/index.html` | Optional Mean Defect field on upload. |
| `app/templates/result.html` | "Glaucoma stage — structure and visual field" panel. |
| `app/templates/history.html` | Progression verdict banner, MD and stage columns. |
| `app/templates/base.html` | Progression added to the navigation; footer updated. |
| `app/static/css/style.css` | Styles for the stage panel, stage ladder and progression charts. |
| `README.md` | Phase 2 sections, the unified-cohort explanation, updated commands and JSON schema. |

## Run order

```
python scripts/build_unified_cohort.py          # 20-30 min on the GPU
python scripts/calibrate_staging.py
python scripts/evaluate_cohort.py
python scripts/validate_progression.py --trials 4000
python scripts/seed_demo_patients.py --reset
python app/app.py
```

Nothing from Phase 1 was modified — the trained weights, the tuned thresholds, the existing
reports and `evaluate.py` are all untouched.
