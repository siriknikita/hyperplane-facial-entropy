# Source audit — Hyperplane-Forge as reused by the facial-entropy project

Audited on 2026-10-08 against the sibling checkout
`/Users/mykytasiryk/Development/personal-projects/deepfake-detection-v2`
(package `forge-detect` 0.1.0, crate `hyperplane_core` 0.1.0).

| item | value |
| --- | --- |
| Git commit | `d2881b09d3388cb46c93790f0f6bdeb969f769bb` (branch `main`) |
| Dirty tree at audit time | yes: modified `.gitignore`, `python/forge_detect/serving_infer.py`, `serving/app.py`; untracked `demo-artifacts/`, `serving/artifacts.py`, `tree.txt`. None of these touch `src/*.rs`, `python/forge_detect/backends/`, `config.py`, `types.py` or `trust_map.py`, so the numerics reused here are the committed ones. The checkout was **not modified** by this project. |
| `AGENTS.md` / `CLAUDE.md` | none present in the checkout |
| Build system | `pyproject.toml`: maturin ≥1.5, `[tool.maturin] features = ["pyo3/extension-module", "python-extension"]`, `python-source = "python"`, `module-name = "forge_detect._core"`; `Cargo.toml` feature `python-extension = []` (off for `cargo test`). `Makefile` targets `dev-install` (`uv sync --extra dev && uv run maturin develop --release`), `build-release`, `test-rust` (`cargo test --release`), `test-python` (`uv run pytest`). |
| Existing tests run before any new code | `cargo test --release`: 88 passed. `uv run pytest tests/test_cpu_backend.py`: 7 passed (in the checkout's own `.venv`, Python 3.12.11, torch 2.2.2). |
| How this project consumes it | `[tool.uv.sources] forge-detect = { path = "../deepfake-detection-v2" }` → `uv sync` builds the wheel with maturin (release profile, `python-extension` feature set by the dependency's own `pyproject.toml`; no manual `maturin develop` is needed). Installed extension: `.venv/lib/python3.12/site-packages/forge_detect/_core.abi3.so`. |

## Actual paths and symbols (match the diploma table)

| file | symbols found | notes |
| --- | --- | --- |
| `src/lib.rs` | modules `energy, hyperplane, impact, luminance, pde, scharr, stencils, tensor`, private `util`; `py_bindings` behind `python-extension` | as documented |
| `src/luminance.rs` | `weighted_luminance`, `gaussian_blur`, `dog_band`, `dog_pyramid`, constants `WEIGHT_SUM_TOL=1e-3`, `SIGMA_MIN=1e-3` | Gaussian radius `ceil(3σ)`, mirror boundary |
| `src/scharr.rs` | `scharr_gradients`, `gradient_magnitude` | 3×3 Scharr /32, mirror boundary |
| `src/tensor.rs` | `structural_tensor`, `eigenvalues_2x2`, `classify`, `adaptive_thresholds`, `keypoints` | |
| `src/hyperplane.rs` | `LocalHyperplane{x_i,y_i,z_i,dz_dx,dz_dy}`, `local_albedo` (square median, radius 2 default), `build_hyperplanes` (`dz = −∇I/(ρ+ε)`, `z_i = c_z·I_w`), `forge_per_scale` (min over `(2r+1)²` neighbourhood, clamped at borders), `fill_uncovered`, `compose_max_over_scales` | neighbourhood/fallback semantics preserved by reuse; fallback for uncovered pixels is `c_z·I_w` |
| `src/stencils.rs` | `laplacian` (5-point), `biharmonic` (iterated), `divergence` (central differences) | mirror boundary via `util::mirror`: `z(−i)=z(i)`, `z(n+i−1)=z(n−i−1)` (numpy/torch "reflect") |
| `src/energy.rs` | `EnergyTerms{data,smoothness,consistency,total}`, `energy(...)` | `λΣ(z−z_f)² + αΣ(Δz)² + βΣ‖W⊙(∇I−K∇z)‖²`, unit cell area |
| `src/pde.rs` | `PdeParams{lambda=1, alpha=0.5, beta=5, max_iter=500, tol=1e-5, log_every=10}`, `SolveResult`, `jacobi_solve` | update `z ← z − τR(z)` with `R = λ(z−z_f) + αΔ²z − β div(W²K(∇I−K∇z))`, `τ₀ = 0.5/(λ+20α+16β·max(WK)²)`, backtracking (6 halvings, keeps the last trial even if energy rises), stop when `‖Δz‖/‖z‖ < tol` |
| `src/impact.rs` | `PipelineParams` (defaults below), `ImpactResult{z_forged,z_star,r,l,energy_final,energy_trace,iterations,converged}`, `run_pipeline` | `z_ideal = gaussian_blur(z_forged, sigma_ref=4)`, `R = z* − z_ideal`, `L = Δz*`; `energy_final = energy_trace.last()` |
| `src/py_bindings.rs` | `_core.solve_and_extract(rgb, w_cnn, *, 21 keyword parameters)` → dict `z_forged, z_star, R, L, energy_total/data/smoothness/consistency, energy_trace, iterations, converged` | float32 in/out, `(H, W, 3)` RGB, `(H, W)` trust, `allow_threads` around the Rust call |
| `python/forge_detect/backends/cpu.py` | `CpuBackend.solve(rgb, w_cnn, params)` → `SolveResult` | contiguity/dtype contract, `params.as_core_kwargs()` |
| `python/forge_detect/config.py` | `PdeParams`, `PipelineParams.as_core_kwargs()` | structured params reused (no manual 21-kwarg rebuild) |
| `python/forge_detect/types.py` | `SolveResult.from_core_dict` (fields `residual` = `R`, `laplacian` = `L`) | |
| `python/forge_detect/trust_map.py` | `heuristic_trust_map(rgb, sigma=2, gain=8)` = `exp(−8‖I − G₂∗I‖)` | the only training-free trust mode; `pipeline.detect` falls back to it. Used as the "standard available trust mode". |
| `python/forge_detect/pipeline.py` | `load_image` (PIL RGB → float32/255), `detect` | colour order RGB |

Defaults actually executed by the baseline (`PipelineParams::default()` = Python `PipelineParams()`): `w=(0.299,0.587,0.114)`, `sigma_base=1`, `k_ratio=√2`, `n_scales=4`, `tensor_window_radius=2`, `p_flat=0.30`, `p_edge=0.70`, `neigh_radius=6`, `albedo_window_radius=2`, `epsilon=1e-3`, `c_z=1`, `sigma_ref=4`, PDE `(λ,α,β)=(1,0.5,5)`, `max_iter=500`, `tol=1e-5`, `log_every=10`.

Axis order: arrays are `[y, x]`; `x` is the column index; `scharr_gradients` returns `(Ix, Iy)` with `Ix` along columns. Threads: rayon pool, `RAYON_NUM_THREADS` honoured (this project sets it to 4 before the first import).

## Exported vs private intermediates

Exported: `z_forged`, `z_star`, `R`, `L`, energy terms, trace, `iterations`, `converged`. Input-side: `W` (we supply it) and `I_w` (recomputed with the same weights). Not exported: `K` (local albedo), per-scale buffers, keypoints, `τ` history. Rather than modify the checkout, the audit recomputes `I_w` and `K` with the same formulas in NumPy and validates the replica by reproducing the Rust energy terms (`controls.json` → `solver.replica_energy_matches_rust`, relative error < 1e-3) and the exported `L` (`rust.laplacian_export_matches_replica`, < 1e-4).

## Differences between diploma/paper, code, and mathematics

1. **Divergence sign (paper §Phase 5 and `pde.rs` agree with each other, but not with the energy).** The paper's first variation is `2∫[... + β(W²⊙(∇I−K∇z))·(−K∇δz)]dA`, and integration by parts of `−∫(KW²r)·∇δz` gives `+∫div(KW²r)δz`, not `−`. The autograd gradient of the implemented discrete energy confirms `+β div(...)` (relative error 0.045 vs 0.22–0.6 for the implemented sign). Consequence: the implemented operator is **not** the energy gradient; the backtracking line search keeps the energy non-increasing only because the `λ`/`α` parts dominate early, and the `converged` flag fires when the step size collapses. At the returned `z*` the true gradient norm is 0.2–0.55× its initial value (real face crop: 0.21; synthetic: 0.55). The baseline is kept exactly as is; the diagnostic `corrected_gradient_torch` variant minimises the validated replica energy (gradient ratio < 1e-5) and is compared in Experiment 3.
2. **`div` is not the adjoint of the Scharr gradient** (nor of central differences under mirror boundaries): `⟨∇u,F⟩ ≠ ⟨u,−div F⟩` even on interior-supported fields (3.17 vs 19.9 in the control). The paper's "Jacobi on a symmetric positive-definite system" description does not hold for the implemented operator.
3. **`energy_final` is the last logged iterate**, not the energy at the returned `z*` (unless `iterations % log_every == 0`). Recorded as an audit finding; parity checks use `log_every=1`.
4. **Repeat runs are not bit-exact with more than one Rayon thread.** `energy.rs` and `pde::l2_norm` reduce with `par_iter().sum()`, whose summation order depends on the thread split. On real 256² faces with `RAYON_NUM_THREADS=4`, three repeats of the same input differed by up to 2.7e-4 in `z*` and by ±3 iterations (41/43/40); with `RAYON_NUM_THREADS=1` repeats are bit-identical. The reported runs used 4 threads (recorded in `run_manifest.json`); the measured tolerance is stored in `controls.json` (`rust.repeat_determinism_256px`). Use `--threads 1` for bit-exact reproduction.
5. **Convergence in practice is fast and shallow**: 256² real faces settle in ~15–45 iterations (≈0.08 s) with the flag set, far from the documented 500-iteration budget, because of the step-size collapse described above.
6. **Paper divergence formula indices** (`(F_x)_{i+1,j} − (F_x)_{i−1,j}`) name the axes opposite to the code (`fx[(y, x_p)] − fx[(y, x_m)]`); numerically the code differentiates `F_x` along `x`, which is the intended quantity.
7. The diploma lists `dog_pyramid` in `luminance.rs`; `impact.rs` uses its own private `build_dog_pyramid` with identical semantics.
8. Documented `python/forge_detect/facial_entropy/` layout does not apply: this is a **separate project** (`hyperplane-facial-entropy/src/facial_entropy/`) depending on `forge-detect` by path, per the user's instruction; the Hyperplane-Forge checkout is unmodified.

## Data availability (audited)

FEI Face Database at `/Users/mykytasiryk/Development/datasets/fei-face-data-dataset/originalimages_part{1..4}/`: 2800 JPEG, 640×480, 200 subjects × 14 poses (`<subject>-<pose>.jpg`), 348 MB. No manipulated images, no identity ambiguity (subject id in filename), no pose angles in metadata (pose index semantics documented in `datasets.py`). Landmark model assets (face-alignment 1.5.0: 2DFAN-4 95.6 MB, S3FD 89.8 MB) were absent and fetched with the explicit `--download` opt-in into `~/.cache/torch/hub/checkpoints`.
