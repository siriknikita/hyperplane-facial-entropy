# Changelog

All notable changes to this project. Format follows Keep a Changelog; newest first.

## [v0.1.0] - 2026-10-08
### Added
- Protocol 1.0 (`experiments/facial_entropy/protocol.md`) and profile configs (smoke / quick / main).
- `facial_entropy` package: Rust adapter over the existing `forge-detect` core (path dependency), solver replica for the audit, alignment (face-alignment 2DFAN-4), fixed patches, geometry, calibration-frozen quantizers, entropy estimators and local maps, controls, Experiments 1-6, statistics, plotting, reporting, `forge-entropy` CLI.
- 26 tests (derivatives, probabilistic definitions, grouping/leakage, Rust parity, projection, cache invalidation).

