# VisionTrack AI – CDR validation and structure–function analysis

## A. Is the CDR accurate?

**SMDG held-out test set (n = 431, expert masks)**

- Mean CDR: VisionTrack 0.482 vs expert 0.481
- MAE 0.056; bias +0.001; 95% limits of agreement -0.147 to +0.148
- Pearson r = 0.864 (p < 0.001); ICC(2,1) = 0.846
- 84.9% of eyes within ±0.10 of the expert CDR

**PAPILA, independent clinic (n = 164, mean of two ophthalmologists)**

- Mean CDR: VisionTrack 0.467 vs experts 0.461
- MAE 0.050; bias +0.006; r = 0.926 (p < 0.001); ICC = 0.925
- 90.9% within ±0.10

## B. Does CDR relate to visual-field damage?

**PAPILA eyes with Humphrey 30-2 Mean Defect (n = 164)**

- Pearson r = -0.493 (p < 0.001); Spearman ρ = -0.331 (p < 0.001); R² = 0.243
- Each +0.1 in vertical CDR ≈ -1.83 dB change in MD
- Reference – ophthalmologists' CDR vs MD: r = -0.474 (p < 0.001), n = 164

| Severity stage | n | Mean CDR ± SD | Median |
|---|---|---|---|
| Early (MD ≥ −6) | 127 | 0.425 ± 0.142 | 0.426 |
| Moderate (−6 to −12) | 19 | 0.553 ± 0.190 | 0.617 |
| Advanced (MD < −12) | 18 | 0.672 ± 0.138 | 0.700 |

- Kruskal–Wallis p < 0.001; trend across stages ρ = 0.450 (p < 0.001)
- Vertical CDR identifies MD < −6 dB with AUC 0.800; best cut-off 0.62 (sensitivity 0.65, specificity 0.92)

## Limitations
- PAPILA images are also part of SMDG-19, so some may have been seen during U-Net training.
- Cross-sectional data (one visit per eye); progression over time is Phase II.
- CDR explains only part of the variance in MD, supporting the Phase II plan to combine structural and functional inputs.