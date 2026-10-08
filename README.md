# Hyperplane-Forge facial entropy

Standalone, reproducible experiment suite testing whether within-image feature entropy is
concentrated around the eyes, inner eye corners and nose rather than cheeks and forehead —
measured (1) on binarised faces and (2) on the Hyperplane-Forge settled surface built by the
**existing Rust implementation** in the sibling checkout `../deepfake-detection-v2`
(package `forge-detect`, extension `forge_detect._core`). Experiment 6 separately tests whether
high-entropy regions carry task-relevant (identity) information.

* Protocol: `experiments/facial_entropy/protocol.md`
* Source audit of the reused Rust/Python code: `experiments/facial_entropy/reports/source_audit.md`
* Profiles: `experiments/facial_entropy/configs/{smoke,quick,main}.toml`
* Code: `src/facial_entropy/` (console script `forge-entropy`), tests in `tests/`

## Setup (uv only)

```bash
cd hyperplane-facial-entropy
uv sync --extra facial-entropy --extra dev      # builds ../deepfake-detection-v2 with maturin (needs cargo)
uv run pytest -q                                # 26 tests
```

The Rust extension is compiled by `uv sync` through the path dependency
(`[tool.uv.sources] forge-detect = { path = "../deepfake-detection-v2" }`,
maturin features `pyo3/extension-module` + `python-extension` from that checkout's own
`pyproject.toml`). To rebuild after changing the Rust sources:
`uv sync --reinstall-package forge-detect`. There is no `maturin develop` step in this project;
`make build-release` inside the checkout builds its own editable copy.

## Commands (verified)

```bash
uv run forge-entropy audit    --output artifacts/facial-entropy/audit
uv run forge-entropy validate --profile smoke --seed 20261008 --output artifacts/facial-entropy/smoke
uv run forge-entropy all --profile smoke --seed 20261008 --output artifacts/facial-entropy/smoke
uv run forge-entropy all --profile quick --dataset auto --seed 20261008 --output artifacts/facial-entropy/quick
uv run forge-entropy all --profile main  --dataset auto --seed 20261008 --output artifacts/facial-entropy/main
```

Flags: `--profile smoke|quick|main`, `--seed`, `--output`, `--data-root`, `--dataset auto|manifest|lfw|fei`,
`--manifest`, `--download` (opt-in, 1 GB cap: landmark weights ~185 MB, LFW ~233 MB), `--threads`
(RAYON_NUM_THREADS, default 4), `--landmark-device cpu|mps`, `--n-subjects`, `--experiments`.
The Rust core is not bit-deterministic with >1 Rayon thread (differences ≤3e-4 in `z*`); pass `--threads 1` for bit-exact reproduction at ~3× solver cost.
`run` and `report` operate on a directory produced by `prepare`/`all`; surfaces and maps are cached
under `<output>/cache/` and resumed.

First real run needs the 68-point landmark model (face-alignment 2DFAN-4 + S3FD):

```bash
uv run forge-entropy all --profile quick --dataset auto --download --seed 20261008 --output artifacts/facial-entropy/quick
```

Opt-in external data (not used for the reported runs):

```bash
uv run forge-entropy all --profile quick --dataset lfw --download --seed 20261008 --output artifacts/facial-entropy/lfw-quick
```

## Data

FEI Face Database (`/Users/mykytasiryk/Development/datasets/fei-face-data-dataset`, 200 subjects × 14
poses, 640×480). `--dataset auto` finds it. The main profile targets 500 subjects; FEI supplies
200, which the run reports as such (no duplication to fill the count).

## Outputs

Per run directory: `run_manifest.json`, `protocol.lock.json`, `dataset_manifest.csv`,
`exclusions.csv`, `calibration.json`, `controls.json`, `image_metrics.csv`, `subject_metrics.csv`,
`primary_results.json`, `ablation_*.{csv,json}`, `robustness_*.{csv,json}`, `policy_*.{csv,json}`,
`verification_pairs.csv`, `verification_results.json`, `atlas.npz`, `figures/`, `results.md`.
