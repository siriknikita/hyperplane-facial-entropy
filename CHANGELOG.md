# Changelog

All notable changes to this project. Format follows Keep a Changelog; newest first.

## [v1.0.0] - 2026-10-08
### Added
- Main profile on all 200 FEI subjects (159 test), 50-image robustness subset, verification on 200 identities; complete `results.md`, figures, atlas, results bundle.
- Source audit updated with thread nondeterminism and energy-logging findings; README documents `--threads 1` for bit-exact reproduction.
### Findings
- Binary conditional endpoint: D = +0.050 bits [0.044, 0.057], Holm p = 0.0002. Gaussian-curvature endpoint: D = -0.693 bits [-0.734, -0.652]. No unique Forge contribution (luminance and all solver variants agree). Entropy does not predict identity utility (region rank rho +0.14 / -0.36).


## [v0.3.0] - 2026-10-08
### Added
- Rust/API/data audit (`artifacts/facial-entropy/audit`), quick profile on 50 FEI subjects with all six experiments, results bundle.
### Findings
- Co-primary endpoints disagree in sign on real faces: binary conditional entropy higher centrally (+0.055 bits), Gaussian-curvature entropy lower centrally (-0.78 bits).
### Changed
- Surface cache persists baseline surfaces only (2 GB budget projection) after the quick run's cache reached 3.7 GB.


## [v0.2.0] - 2026-10-08
### Added
- Smoke profile run on synthetic pseudo-faces with explicit landmarks (`artifacts/facial-entropy/smoke`).
### Findings
- 0 integrity failures; 3 solver audit findings (divergence sign, non-stationary z*, non-adjoint operators). Reported, baseline preserved, corrected variant added as a labelled ablation.


## [v0.1.0] - 2026-10-08
### Added
- Protocol 1.0 (`experiments/facial_entropy/protocol.md`) and profile configs (smoke / quick / main).
- `facial_entropy` package: Rust adapter over the existing `forge-detect` core (path dependency), solver replica for the audit, alignment (face-alignment 2DFAN-4), fixed patches, geometry, calibration-frozen quantizers, entropy estimators and local maps, controls, Experiments 1-6, statistics, plotting, reporting, `forge-entropy` CLI.
- 26 tests (derivatives, probabilistic definitions, grouping/leakage, Rust parity, projection, cache invalidation).

