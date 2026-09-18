# VisionTrack AI - results on the unified cohort

Every figure comes from one table, `cohort_master.csv`: 2699 eyes (PAPILA 488, SMDG 2211), each pushed through the complete pipeline.

**All headline numbers below are on eyes the relevant model never trained on.** PAPILA is one of the 19 datasets pooled into SMDG-19, so a large part of the cohort was seen during training; those eyes are reported separately further down rather than removed from sight.

- Expert disc/cup reference: 1275 eyes (505 held out, 770 seen in training)
- Diagnosis label: 2623 eyes (1950 held out, 673 seen in training)
- Humphrey 30-2 Mean Defect: 164 eyes (32 held out, 132 seen in training)

## Headline table (for the slide) - held-out eyes only

| Measure | Value | n |
|---|---|---|
| Optic disc Dice | 0.9493 | 505 |
| Optic cup Dice | 0.8258 | 505 |
| Overall Dice | 0.8876 | 505 |
| CDR mean absolute error | 0.0550 | 505 |
| CDR R2 vs expert | 0.775 | 505 |
| CDR ICC(2,1) | 0.872 | 505 |
| CDR within +/-0.10 of expert | 85.7% | 505 |
| Accuracy | 90.26% | 1950 |
| Sensitivity | 85.25% | 746 |
| Specificity | 93.36% | 1204 |
| ROC-AUC | 0.9617 | 1950 |
| CDR vs visual-field MD (Pearson r, held out) | -0.517 | 32 |
| AUC, CDR detecting MD < -6 dB (held out) | 0.893 | 32 |
| AUC, image model detecting MD < -6 dB | 0.873 (cross-validated) | 164 |
| Stage agreement with the measured field | 51.8% exact, 88.4% within one stage | 164 |
| Stage agreement, quadratic kappa | 0.590 | 164 |

## For comparison: the same models on eyes they were trained on

These are **not** performance figures. They are here so the gap is visible instead of implied.

| Measure | Held out | Seen in training |
|---|---|---|
| Disc Dice | 0.9493 (n=505) | 0.9467 (n=770) |
| Cup Dice | 0.8258 (n=505) | 0.8152 (n=770) |
| CDR MAE | 0.0550 (n=505) | 0.0509 (n=770) |
| CDR ICC | 0.872 | 0.893 |
| Accuracy | 90.26% (n=1950) | 89.90% (n=673) |
| ROC-AUC | 0.9617 | 0.9375 |

## Held-out numbers split by source

**Segmentation**

| Source | n | Disc Dice | Cup Dice | Overall |
|---|---|---|---|---|
| PAPILA | 74 | 0.9376 | 0.7684 | 0.8530 |
| SMDG | 431 | 0.9513 | 0.8357 | 0.8935 |

**CDR agreement**

| Source | n | MAE | Bias | ICC | r | Within +/-0.10 |
|---|---|---|---|---|---|---|
| PAPILA | 74 | 0.0478 | -0.0067 | 0.926 | 0.927 | 90.5% |
| SMDG | 431 | 0.0562 | +0.0006 | 0.846 | 0.864 | 84.9% |

**Classification**

| Source | n | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| PAPILA | 102 | 90.20% | 71.88% | 98.57% | 0.9420 |
| SMDG | 1848 | 90.26% | 85.85% | 93.03% | 0.9627 |

## Structure and function on the same eyes

Across all 164 eyes with a visual field: Pearson r = -0.493 (p < 0.001), Spearman rho = -0.331, R2 = 0.243. Each +0.1 of CDR corresponds to -1.83 dB of Mean Defect.

On the 32 eyes held out of segmentation training: r = -0.517. Quote this one.

Ophthalmologists' own CDR against the same fields: r = -0.489 (n = 164). The expert figure is free of any training contamination, which is what makes it the fair yardstick.

| Visual-field stage | n | Mean CDR |
|---|---|---|
| Early (MD >= -6) | 127 | 0.425 +/- 0.142 |
| Moderate (-6 to -12) | 19 | 0.553 +/- 0.190 |
| Advanced (MD < -12) | 18 | 0.672 +/- 0.138 |

## Limitations

- PAPILA is pooled inside SMDG-19, so it is **not** an independent clinic for segmentation. Only 32 of the 164 eyes with visual fields were held out of U-Net training.
- Visual fields exist only for the PAPILA part of the cohort; SMDG has no perimetry.
- All of it is cross-sectional: progression is validated separately, by simulation.
- Thresholds were tuned on the validation split, not on the test eyes reported here.
