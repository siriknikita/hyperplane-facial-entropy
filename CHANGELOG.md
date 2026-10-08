# Changelog

All notable changes to this project. Format follows Keep a Changelog; newest first.

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

