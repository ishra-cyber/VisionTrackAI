# VisionTrack AI — demonstration narration script

About 7 minutes spoken. Square brackets are actions, not words to say.
All numbers are final, measured on the unified cohort (2,699 eyes) and computed **only on eyes
the relevant model never trained on**. PAPILA is pooled inside SMDG-19, so that distinction
matters and is made explicitly in the script below.

---

## 0:00 — Opening (30 s)

[Title slide]

"Glaucoma is the leading cause of irreversible blindness, and it is silent — by the time a patient
notices, the damage is permanent. Two things decide the outcome: catching it early, and catching
the moment it starts getting worse.

VisionTrack AI does both from a single fundus photograph. I'll show you the system measuring an
optic nerve head, staging it against the visual field, and then deciding whether a patient is
progressing."

---

## 0:30 — What it is built from (45 s)

[Architecture slide]

"Three components. The first segments the optic disc and the optic cup with a U-Net and measures
the cup-to-disc ratio. The second combines that structural measurement with the visual field to
produce a glaucoma stage. The third watches a patient over time and reports whether they are stable
or deteriorating.

The models were trained on SMDG-19 — twelve thousand four hundred fundus images pooled from nineteen
public datasets — on a local RTX 3050 GPU. But everything we report comes from one unified cohort
table, where each eye carries its image, its expert reference contours, its diagnosis and its
visual field in a single row. No result in this presentation comes from one dataset while a
different result comes from another."

---

## 1:15 — Component 1 results (60 s)

[Results slide]

"Segmentation first, on five hundred and five eyes the model never trained on. Dice of 0.949 on
the optic disc, 0.826 on the cup, 0.888 overall.

The cup-to-disc ratio is where accuracy matters clinically, so we measured it against expert
graders directly. Mean absolute error 0.055, intraclass correlation 0.87, and 85.7 per cent of
eyes within a tenth of a ratio unit of the expert value.

For glaucoma classification, on nineteen hundred and fifty held-out eyes: 90.3 per cent accuracy,
85.3 sensitivity, 93.4 specificity, and an area under the ROC curve of 0.962.

One thing I want to be explicit about. PAPILA is one of the nineteen datasets pooled into SMDG,
so part of our cohort was seen during training. Every number I just gave you excludes those eyes.
When we do measure the models on eyes they trained on, the difference is under half a per cent —
disc Dice 0.947 versus 0.949 — which is the evidence that the network generalised rather than
memorised."

---

## 2:15 — The reviewers' question (45 s)

[Structure–function slide]

"Our reviewers asked a harder question: does the cup-to-disc ratio actually relate to functional
damage — to what the patient can see?

On the PAPILA eyes with Humphrey 30-2 visual fields, our automated CDR correlates with Mean Defect
at r equals minus 0.49. For comparison, the ophthalmologists' own hand-drawn CDR on the same eyes
gives minus 0.47. Our automated measurement tracks visual function as well as expert grading does.

Each additional tenth of cup-to-disc ratio corresponds to about 1.8 decibels of field loss."

---

## 3:00 — LIVE: one image, full report (90 s)

[Switch to the app. Worklist screen.]

"Here is the system running. I'm uploading a fundus image this model has not seen, and I am
deliberately leaving the visual field box empty — this patient has had no perimetry.

[Upload. Report page opens.]

The disc and cup are segmented — I can toggle the overlay, the raw image, and each mask. Vertical,
horizontal and area cup-to-disc ratios are measured. The classifier gives a glaucoma probability
against a tuned decision threshold.

And this panel is Component 2. On the left, the structural grade from the cup-to-disc ratio,
against cut-points fitted on our cohort. On the right, the visual field — and because no perimetry
exists for this patient, the system estimates it from the image, with a 95 per cent interval, and
says clearly that it is an estimate.

The number that matters is underneath: the probability that this eye has moderate or worse field
loss. That estimate is accurate to an area under the curve of **0.87 cross-validated**. It does not
replace a visual field test. It decides who needs one first."

---

## 4:30 — LIVE: structure and function together (45 s)

[Open a PAPILA case from the Visual Field page.]

"Now the same panel on a patient who does have a visual field. It switches from estimated to
measured, and the stage is the worse of structure and function.

When the two disagree — when the disc looks worse than the field, or the field worse than the disc
— the system says so on screen rather than quietly averaging them away. A structure–function
mismatch is a clinical finding in its own right, not an inconvenience."

---

## 5:15 — LIVE: progression (75 s)

[Progression page, a DEMO patient.]

"Component 3. This patient has five visits.

For each eye we fit the trend of cup-to-disc ratio against time: the rate of change per year, the
significance of that trend, and the change from baseline.

This shaded band is the important part. It is the measurement error of our own method — derived
from how far our cup-to-disc ratio sits from expert graders on this cohort, not taken from a
textbook — five hundred and five held-out eyes. A change inside the band is noise. Our limit is **0.144** ratio units.

A verdict of 'progressing' requires both a statistically significant trend and a change larger than
that band. One without the other gives 'possible progression'. Fewer than three visits gives
'insufficient data' rather than a guess.

I should be explicit about one thing: this demonstration series is assembled from different real
eyes ordered by cup-to-disc ratio, because no public dataset photographs the same patient twice.
Every measurement on the screen is a genuine model output. The sequence is constructed, and you
should know that."

---

## 6:30 — How the progression rule was validated (40 s)

[Validation slide]

"Since no longitudinal ground truth exists, we validated the decision rule itself, under the
measurement noise the system actually has.

On truly stable eyes it raises a false alarm **under one per cent** of the time. With five visits
over four years it catches progression at 0.08 ratio units per year in **88 per cent** of patients,
and at 0.10 per year in **97 per cent**. With only three visits it catches the fast cases and misses slow ones — which is a
property of the cup-to-disc ratio as a measurement, and precisely the argument for adding OCT
retinal nerve fibre layer thickness, which is far less noisy, in the next phase."

---

## 7:10 — Close (30 s)

[Interface / JSON slide]

"Every analysis exports as a structured JSON record — Component 1 measurements, Component 2 stage,
Component 3 trend — so a hospital system consumes the output without knowing anything about our
models.

To summarise: from one photograph, VisionTrack AI measures the optic nerve head to within a
tenth of a ratio unit of an expert, stages it against the visual field, flags who needs perimetry
first, and tells a clinician when a patient has genuinely changed rather than when the measurement
merely wobbled.

It is decision support, not a diagnosis. Thank you."

---

## Notes for the presenter

- Never say the system diagnoses glaucoma. It measures, stages and flags.
- Say the demo series is constructed **before** anyone asks. It costs nothing and it is the
  difference between a demonstration and an overclaim.
- If asked why staging is only about half exact across four stages: quote within-one-stage
  agreement and the quadratic kappa instead, and name the weak boundary honestly — no-damage
  versus early field loss, which the disc alone cannot resolve.
- If the app misbehaves, the troubleshooting table is in `docs/DEMO_RUN_SHEET.md`.
