# Component 3 — progression detection: what the rule can and cannot see

PAPILA and SMDG are cross-sectional, so there is no real longitudinal ground truth. This is a simulation of the decision rule under the measurement noise the system actually has: SD **0.0520 CDR units** per study, taken from the model-versus-expert agreement on the unified cohort (505 paired eyes). Critical change in use: **0.144**. 4000 simulated patients per cell.

| Follow-up schedule | False positives (truly stable) | Smallest rate detected in ≥80% of patients |
|---|---|---|
| 3 visits / 2 y | 0.3% | not reached at +0.10/year |
| 4 visits / 3 y | 0.3% | not reached at +0.10/year |
| 5 visits / 4 y | 0.6% | +0.08 CDR/year |
| 6 visits / 5 y | 0.4% | +0.05 CDR/year |

## Detection rate by true rate of change

| True rate | 3 visits / 2 y | 4 visits / 3 y | 5 visits / 4 y | 6 visits / 5 y |
|---|---|---|---|---|
| +0.00/year | 0% | 0% | 1% | 0% |
| +0.01/year | 1% | 2% | 2% | 4% |
| +0.02/year | 1% | 4% | 7% | 15% |
| +0.03/year | 2% | 7% | 17% | 35% |
| +0.05/year | 4% | 19% | 49% | 81% |
| +0.08/year | 11% | 45% | 88% | 100% |
| +0.10/year | 14% | 61% | 97% | 100% |

**Reading this honestly:** with three visits over two years the rule only catches fast progression; the slow change that matters clinically needs five or more visits. That is a property of CDR as a measurement, not a bug in the code — and it is the argument for adding OCT RNFL thickness, which is far less noisy, in a later phase.

**Limitation:** a simulation validates the decision rule, not the biology. Real progression is not perfectly linear and real follow-up is not evenly spaced. Prospective data is the only way to settle that, and collecting it is what Phase 2 deployment is for.
