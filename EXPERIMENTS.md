# Experiment journal — hyperplane-facial-entropy

One entry per run, appended in order. Verdicts are quoted from the run's `results.md`.
Protocol: `experiments/facial_entropy/protocol.md` (version 1.0, frozen before any test image was processed).

## 2026-10-08 — smoke: synthetic end-to-end validation (v0.2.0)

**Question.** Does the whole pipeline (Rust reuse, alignment, patches, quantizers, entropy estimators, inference, report) run offline on rendered pseudo-faces, and do the numerical controls pass?

**Verdict.** 67 controls: 0 integrity failures, 3 audit findings — `solver.update_direction_matches_energy_gradient`, `solver.stationarity_at_z_star`, `solver.scharr_vs_central_divergence_adjoint`. The implemented Rust update uses `-beta*div(...)` (paper/code) while the true gradient of the implemented discrete energy carries `+beta*div(...)`; at the returned `z*` the energy-gradient norm is 0.55x its initial value with the `converged` flag set. Synthetic endpoints (9 pseudo-subjects): binary conditional D = -0.047 bits, Gaussian-curvature D = -0.728 bits. No empirical claim about faces follows from this run.

**Artifacts.** `artifacts/facial-entropy/smoke/` (controls.json, primary_results.json, results.md, 11 figures).

## 2026-10-08 — quick: 50 FEI subjects (v0.3.0)

**Question.** First real-data pass of the protocol: 10 calibration / 40 test subjects, one frontal image each (FEI pose 11), all six experiment stages.

**Verdict.** Co-primary endpoints disagree in sign. `binary_cond_otsu`: central 0.109 vs peripheral 0.054 bits, mean D = +0.055 [0.042, 0.069], 90% of subjects positive, Holm p = 0.0002. `gaussian_curvature` (16 bins): central 2.452 vs peripheral 3.232 bits, D = -0.780 [-0.848, -0.712], 0% positive, Holm p = 0.0002. Experiment 6 (verification, 10 test identities): entropy-vs-utility rank agreement over regions rho = +0.10 (binary) and -0.40 (curvature). Resource finding: the surface cache grew to 3.7 GB because every perturbation solve was persisted; cache policy changed to persist baseline surfaces only before the main run.

**Artifacts.** `artifacts/facial-entropy/quick/`, bundle `facial-entropy-bundle-quick.zip` (release asset).
