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

## 2026-10-08 — main: all 200 FEI subjects (v1.0.0)

**Question.** Confirmatory run of the frozen protocol with 40 calibration / 160 test subjects (159 retained), the 50-image robustness subset, and verification on 200 identities x 5-6 photographs.

**Verdict.** `binary_cond_otsu`: central 0.107 vs peripheral 0.057 bits, D = +0.050 [0.044, 0.057], 89% positive, Holm p = 0.0002 — supports a central advantage for the binary conditional estimator in this population, stable to thresholds, block rotations and nuisance perturbations. `gaussian_curvature`: central 2.689 vs peripheral 3.382 bits, D = -0.693 [-0.734, -0.652], 0% positive — the settled surface's quantized curvature distribution is *less* entropic centrally in every subject, robust to bins (8/16/32), smoothing, estimator, Miller-Madow, all photometric perturbations, landmark jitter and resolution. Experiment 3: every solver variant (constant trust, alpha x0.5/x2, beta = 0, forged-only, corrected gradient) gives D within +/-0.01 of baseline; plain luminance -0.79, smoothed luminance -0.44 — no unique Forge contribution. Atlas: the K_G entropy maximum never falls inside the eyes/nose mask (0/159). Experiment 6: region-only verifiers rank eyes and forehead (AUROC 0.74-0.75) above cheeks and nose (0.59-0.65); rank agreement with entropy rho = +0.14 (binary), -0.36 (curvature); whole-face HOG 0.995, surface z* 0.97. Audit additions: the Rust core is not bit-deterministic with >1 Rayon thread (<= 3e-4 in z*), exported energy terms belong to the last logged iterate.

**Artifacts.** `artifacts/facial-entropy/main/` (1.6 GB on disk incl. 1.3 GB cache; results, figures and controls committed), bundle `facial-entropy-bundle.zip` (release asset).

## Synthesis after v1.0.0

The professor's hypothesis holds for the binary-image conditional-entropy estimator and is contradicted for the proposed surface-curvature estimator; the Hyperplane-Forge construction adds nothing beyond plain image structure to the regional contrast, and regional entropy of either kind does not predict identity utility for the tested decoder. The solver sign defect found in Experiment 0 was later fixed on the `feat/exact-energy-gradient-mobilenet` branch of the Hyperplane-Forge repository; this project's reported runs used the original (unmodified) solver, with the corrected variant included as a labelled ablation.
