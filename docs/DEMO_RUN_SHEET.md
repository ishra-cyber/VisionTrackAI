# VisionTrack AI — demonstration run sheet

Everything below is either a command to run before the demo, a screen to open during it,
or a number to have ready when someone asks. Nothing here needs to be memorised; keep this
page open on a second screen.

---

## Part 0 — Before you present (run in order)

| # | Command | Takes | What it produces |
|---|---|---|---|
| 1 | `python scripts/build_unified_cohort.py` | 16 min (GPU) | `data/cohort_master.csv` — the single cohort table |
| 2 | `python scripts/annotate_cohort_splits.py` | seconds | tags every row train / val / test, so nothing is scored on training data |
| 3 | `python scripts/calibrate_staging.py` | < 1 min | `checkpoints/staging_calibration.json` + `reports/phase2/` |
| 4 | `python scripts/evaluate_cohort.py` | < 1 min | `reports/unified/cohort_results.md` — the slide table |
| 5 | `python scripts/validate_progression.py --trials 4000` | 2–3 min | `reports/phase2/progression_validation.md` |
| 6 | `python scripts/seed_demo_patients.py --reset` | 2–3 min | demo patients with several visits each |
| 7 | `python app/app.py` | — | the app at http://127.0.0.1:5000 |

Steps 1–5 were run on 18 Sep 2026; the numbers in Part 3 are from that run.

Sanity check before you start talking: open http://127.0.0.1:5000/health — it should say
`"status": "ok"` with `"classifier_loaded": true`. Both dots in the top-right of the app
should be green.

Have ready in browser tabs: the app's Worklist, and one fundus image you have **not**
analysed yet, on the desktop, for the live upload.

---

## Part 1 — The one-sentence version

> VisionTrack AI takes a single fundus photograph and returns a measured cup-to-disc ratio,
> a glaucoma risk, a glaucoma **stage** that combines the optic nerve head with the visual
> field, and — once a patient has been seen a few times — whether they are getting worse.

---

## Part 2 — The demo path (about 8 minutes)

### Screen 1 — Worklist → live upload
Drag in the fresh fundus image. Enter a patient ID. **Leave the Mean Defect box empty** —
that is the point of the next screen.

Say: *"No perimetry has been done for this patient. Watch what the system does anyway."*

### Screen 2 — Study report
Walk down it in this order:

1. **Overlay** — toggle Overlay / Original / Disc / Cup. The disc and cup are segmented, not guessed.
2. **Measurements** — vertical, horizontal and area CDR, with the disc and cup dimensions in pixels.
3. **Glaucoma risk** — the EfficientNet-B0 probability and the tuned decision threshold.
4. **Glaucoma stage — structure and visual field** ← *this is Component 2, new today.*
   - Left column: the CDR and the structural grade, with the cut-points that were fitted on the cohort.
   - Right column: the visual field. Because none was entered, it is **estimated from the image**,
     with a 95% interval, plus the probability of moderate-or-worse field loss.
   - The stage badge, the ladder, and a suggested review interval.

Say: *"The estimate is deliberately labelled as an estimate and carries a wide interval. What
is actually useful is the single percentage underneath — the probability that this eye has
moderate or worse field loss — and that is accurate to an AUC of about 0.88."*

Then re-run the same image **with** a Mean Defect typed in, and show that the panel switches
to `measured`, and that when structure and function disagree the system says so instead of
hiding it.

### Screen 3 — Patients
Pick a DEMO- patient. The CDR trend across visits, the table of studies, and the banner at the
top with the progression verdict.

### Screen 4 — Progression ← *Component 3, new today*
Same patient. For each eye and each measurement:
- rate of change per year, change from baseline, the p-value of the trend, the number of visits;
- the chart, with the fitted trend and the shaded band that marks the measurement error of the method;
- the reasoning in plain sentences, and the projection to the next threshold.

Say: *"A verdict needs two things at once — a statistically significant trend **and** a change
bigger than the measurement error of our own CDR. The band on the chart is not a textbook
number; it is derived from how far our CDR sits from expert graders on this cohort."*

### Screen 5 — Visual Field
The cohort-level structure–function scatter and the clinical validation figures. This is the
answer to the reviewers' email.

### Screen 6 — Export JSON
Open the exported JSON on the report page. Point at `component1`, `component2`, `component3`
and `schema_version: 2.0`.

Say: *"This is the interface. Any hospital system, or the next component, consumes this
without knowing anything about our models."*

---

## Part 3 — Numbers to have ready

Unified cohort: **2,699 eyes** (PAPILA 488, SMDG 2,211). Every figure below is on eyes the
relevant model **never trained on** — PAPILA is pooled inside SMDG-19, so that distinction is
real and you should make it before anyone asks.

| Measure | Value | n (held out) |
|---|---|---|
| Optic disc Dice | **0.9493** | 505 |
| Optic cup Dice | **0.8258** | 505 |
| Overall Dice | 0.8876 | 505 |
| CDR mean absolute error | **0.0550** | 505 |
| CDR R² vs expert | 0.775 | 505 |
| CDR ICC(2,1) | **0.872** | 505 |
| CDR within ±0.10 of expert | 85.7% | 505 |
| Accuracy | **90.26%** | 1,950 |
| Sensitivity | 85.25% | 1,950 |
| Specificity | 93.36% | 1,950 |
| ROC-AUC | **0.9617** | 1,950 |
| CDR vs visual-field MD (Pearson r) | **−0.517** held out / −0.493 all | 32 / 164 |
| Ophthalmologists' own CDR vs MD | −0.474 | 164 |
| AUC, CDR detecting MD < −6 dB | 0.893 | 32 |
| AUC, image model detecting MD < −6 dB | **0.886** (CV 0.873) | 164 |
| Stage agreement with the measured field | 51.8% exact, **88.4% within one stage** | 164 |
| Stage agreement, quadratic κ | 0.590 | 164 |
| CDR critical change (Component 3) | **0.144** | from 505 held-out eyes |
| Progression false-positive rate | < 1% | simulation |
| Progression detected, 5 visits / 4 y, +0.08 CDR/y | 88.2% | simulation |

**The overfitting question, answered before it is asked.** Measured on eyes the models *did*
train on, disc Dice is 0.9467 against 0.9493 held out, and accuracy 89.90% against 90.26%. The
training-set numbers are no better — in places slightly worse. That is what generalisation looks
like, and it is why the contamination does not invalidate the earlier work.

Full detail: `reports/unified/cohort_results.md`.

---

## Part 4 — Questions you will get, and honest answers

**"Two datasets — are you mixing training and test data?"**
We checked this explicitly. PAPILA is one of the 19 datasets pooled into SMDG-19, so 414 of the
488 PAPILA eyes were seen during U-Net training, and only 32 of the 164 eyes with visual fields
were held out. Every headline number we report excludes training eyes, and the report prints the
training-set figures next to them so the gap is visible: disc Dice 0.9467 seen versus 0.9493 held
out. There is no meaningful gap, which is the point.

**"So your earlier claim of independent PAPILA validation was wrong?"**
Yes, and we corrected it rather than leaving it. The ICC of 0.93 on PAPILA stands as an
agreement figure, but it is not independent validation, because most of those eyes trained the
segmentation network. The independent figure is the 74 held-out PAPILA eyes: ICC 0.926, MAE
0.048, 90.5% within ±0.10.

**"Your staging is only about 50% exact — isn't that weak?"**
Exact agreement across four ordinal stages, against a majority-class baseline of about 45%.
The honest figures are within-one-stage agreement (about 88%) and quadratic κ. The boundary
the disc cannot resolve is no-damage versus early field loss; moderate-or-worse damage is
detected at AUC ≈ 0.88. We report both rather than only the flattering one.

**"You're predicting a visual field from a photograph. Is that safe?"**
It is labelled an estimate, it carries a ±10 dB interval on screen, and the report says
perimetry is still required. Its job is triage — deciding who needs a field test sooner —
not replacing one.

**"There is no longitudinal data. How can progression be validated?"**
It cannot be, on this data, and the report says so in the first paragraph. What is validated
is the decision rule, under the measurement noise the system actually has: false-positive rate
around 1% on truly stable eyes, and a detection rate that rises with the number of visits.
Collecting prospective data is exactly what Phase 2 deployment is for.

**"Why is the demo patient's series constructed?"**
Because nobody in either public dataset was photographed twice. Each visit in a DEMO- patient
is a different real eye, ordered by CDR; every measurement on screen is a genuine model output.
Say this before anyone asks — it costs nothing and it is the difference between a demo and a claim.

**"What is next?"**
OCT RNFL thickness, which is far less noisy than CDR and would sharpen progression detection
considerably; and prospective follow-up data from the clinic.

---

## Part 5 — If something breaks

| Symptom | Fix |
|---|---|
| Red dots top-right, "model checkpoint not found" | `checkpoints/` is missing `unet_disc_cup.pt` / `effnet_b0_glaucoma.pt` |
| Stage panel missing on the report | `checkpoints/staging_calibration.json` missing — re-run step 2 |
| Progression page says "insufficient data" | Fewer than 3 visits, or under 0.5 years of follow-up — use a DEMO- patient |
| Visual Field page says not ready | `data/papila_vf_matched.csv` missing — re-run `scripts/prepare_papila_vf.py` |
| Upload rejected | Only PNG / JPG / TIFF / BMP, maximum 20 MB |

**Never say** the system diagnoses glaucoma. It is decision support, and the disclaimer is on
every page for a reason.
