# Component 2 — CDR + visual-field staging (calibrated on cohort_master.csv)

Unified cohort: **2699 eyes**, of which **164** have a Humphrey 30-2 Mean Defect and **505** have expert disc/cup contours. One table, one pipeline.

## A. Visual field predicted from the fundus image

| Model | Predictors | In-sample R² | 5-fold CV R² | CV MAE (dB) | CV stage accuracy |
|---|---|---|---|---|---|
| cdr_prob | vcdr, glaucoma_probability | 0.280 | 0.251 | 3.96 | 37.2% |
| cdr_prob_rim | vcdr, glaucoma_probability, min_rim_ratio, rim_inferior, rim_to_disc_area | 0.311 | 0.235 | 3.83 | 39.6% |
| cdr_only | vcdr | 0.243 | 0.221 | 4.04 | 30.5% |

| Model | Predictors | AUC | CV AUC | n |
|---|---|---|---|---|
| cdr_prob | vcdr, glaucoma_probability | 0.886 | 0.873 | 164 |
| cdr_prob_rim | vcdr, glaucoma_probability, min_rim_ratio, rim_inferior, rim_to_disc_area | 0.887 | 0.857 | 164 |
| cdr_only | vcdr | 0.800 | 0.787 | 164 |

Selected: **cdr_prob**; residual SD 5.35 dB, so a single image is reported with a ±10.5 dB interval. Each +0.1 of CDR corresponds to -1.29 dB of Mean Defect.

## B. CDR cut-points between stages

| Boundary | MD threshold | Fitted CDR cut-off | AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| normal / early | -2 dB | 0.58 | 0.641 | 0.39 | 0.92 |
| early / moderate | -6 dB | 0.62 | 0.800 | 0.62 | 0.92 |
| moderate / advanced | -12 dB | 0.65 | 0.865 | 0.72 | 0.92 |

Staging against the measured field: accuracy **51.8%** (cross-validated 48.8%), within one stage **88.4%**, quadratic κ **0.590**.

Stage distribution in the cohort: [74, 53, 19, 18] (majority-class baseline 45.1%). Balanced accuracy 44.6%. The weak boundary is no-damage vs early field loss, which the disc alone cannot separate; moderate-or-worse loss is detected well.

## B2. Probability of moderate-or-worse field loss (MD < −6 dB)

- Predictors: vcdr, glaucoma_probability; 37 of 164 eyes affected
- AUC **0.886** (cross-validated 0.873)
- Reported on every study as a single percentage, which is more useful than the wide point estimate of Mean Defect.

## C. Critical change handed to Component 3

- Paired eyes with expert contours: 505
- SD of (VisionTrack − expert) CDR: 0.0735
- 95% critical change between two visits: **0.144 CDR units** — a smaller change is measurement noise, not progression.

## Limitations

- The visual-field estimate is a population regression, not perimetry; the interval is wide by design.
- Visual fields exist only for the PAPILA part of the cohort, and that data is cross-sectional.
- Cut-points are fitted on this cohort; the cross-validated figures are the ones to quote.
