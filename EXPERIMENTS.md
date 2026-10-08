# Experiment journal — hyperplane-facial-entropy

One entry per run, appended in order. Verdicts are quoted from the run's `results.md`.
Protocol: `experiments/facial_entropy/protocol.md` (version 1.0, frozen before any test image was processed).

## 2026-10-08 — smoke: synthetic end-to-end validation (v0.2.0)

**Question.** Does the whole pipeline (Rust reuse, alignment, patches, quantizers, entropy estimators, inference, report) run offline on rendered pseudo-faces, and do the numerical controls pass?

**Verdict.** 67 controls: 0 integrity failures, 3 audit findings — `solver.update_direction_matches_energy_gradient`, `solver.stationarity_at_z_star`, `solver.scharr_vs_central_divergence_adjoint`. The implemented Rust update uses `-beta*div(...)` (paper/code) while the true gradient of the implemented discrete energy carries `+beta*div(...)`; at the returned `z*` the energy-gradient norm is 0.55x its initial value with the `converged` flag set. Synthetic endpoints (9 pseudo-subjects): binary conditional D = -0.047 bits, Gaussian-curvature D = -0.728 bits. No empirical claim about faces follows from this run.

**Artifacts.** `artifacts/facial-entropy/smoke/` (controls.json, primary_results.json, results.md, 11 figures).
