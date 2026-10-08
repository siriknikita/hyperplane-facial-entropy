# Regional facial entropy on the Hyperplane-Forge surface — results (smoke profile)

**Run type:** SYNTHETIC-ONLY (rendered pseudo-faces; no real-data conclusion)  
**Seed:** 20261008  **Status:** {'exp0': 'done (0 integrity failures, 3 audit findings)', 'exp1': 'done (binary endpoints)', 'exp2': 'done (surface endpoints + maps + atlas)', 'exp3': 'done', 'exp4': 'done', 'exp5': 'done', 'exp6': 'not run: fewer than 15 subjects with >= 4 photographs; verification not run', 'prepare': 'done', 'report': 'done (11 figures)'}  **Output:** `/Users/mykytasiryk/Development/personal-projects/hyperplane-facial-entropy/artifacts/facial-entropy/smoke`

## 1. Findings

**Experiment 0 (controls):** 67 checks, 0 integrity failures, 3 audit findings.  
Audit findings (solver discrepancies, reported not corrected in the baseline): ['solver.update_direction_matches_energy_gradient', 'solver.stationarity_at_z_star', 'solver.scharr_vs_central_divergence_adjoint']  

**Solver audit.** The implemented update uses the divergence term with sign `-beta*div (paper/code)`; the true gradient of the implemented discrete energy carries `+beta*div`. At the Rust `z*` the energy-gradient norm is 0.554 × its value at `z_forged` (iterations 14, converged flag True). The baseline `z*` is therefore a damped partial descent, not a stationary point of E. The baseline is preserved unchanged; `corrected_gradient_torch` (float64 L-BFGS on the validated replica energy) is reported as a separately labelled diagnostic variant (gradient ratio 0.000005).

**Co-primary endpoints** (test split only; subjects = independent unit; 9 test subjects, 9 with all eight patches retained; 3 calibration subjects never used for inference).

| endpoint | n subj | central mean ± SD (bits) | peripheral mean ± SD | mean D | median D | 95% boot CI | d_z | P(D>0) | sign-flip p | Holm p | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_conditional (`binary_cond_otsu`) | 9 | 0.051 ± 0.006 | 0.098 ± 0.032 | -0.047 | -0.046 | [-0.070, -0.026] | -1.300 | 0.11 | 0.0078 | 0.0078 | 0.0391 |
| gaussian_curvature (`gaussian_curvature`) | 9 | 2.626 ± 0.080 | 3.353 ± 0.058 | -0.728 | -0.718 | [-0.799, -0.661] | -6.569 | 0.00 | 0.0039 | 0.0078 | 0.0039 |

- **binary_conditional:** estimated contrast is non-positive; Holm-adjusted p < 0.05. Interpretation: see decision rules (§6).
- **gaussian_curvature:** estimated contrast is non-positive; Holm-adjusted p < 0.05. Interpretation: see decision rules (§6).

**Secondary metrics (exploratory family, own Holm correction):**

| metric | n subj | central ± SD | peripheral ± SD | mean D | median D | 95% CI | d_z | P(D>0) | p | Holm p (secondary) | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_block4_otsu | 9 | 0.563 ± 0.037 | 0.817 ± 0.132 | -0.254 | -0.262 | [-0.348, -0.154] | -1.615 | 0.11 | 0.0078 | 0.0664 | 0.0391 |
| binary_cond_otsu_rot1 | 9 | 0.050 ± 0.007 | 0.099 ± 0.028 | -0.049 | -0.055 | [-0.068, -0.029] | -1.506 | 0.00 | 0.0039 | 0.0664 | 0.0039 |
| binary_cond_otsu_rot2 | 9 | 0.048 ± 0.007 | 0.093 ± 0.027 | -0.045 | -0.042 | [-0.063, -0.027] | -1.493 | 0.00 | 0.0039 | 0.0664 | 0.0039 |
| binary_cond_otsu_rot3 | 9 | 0.051 ± 0.005 | 0.094 ± 0.032 | -0.043 | -0.049 | [-0.063, -0.022] | -1.264 | 0.11 | 0.0078 | 0.0664 | 0.0391 |
| binary_cond_t0.35 | 9 | 0.044 ± 0.007 | 0.000 ± 0.000 | 0.044 | 0.046 | [0.040, 0.048] | 6.323 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| binary_cond_t0.50 | 9 | 0.051 ± 0.005 | 0.059 ± 0.019 | -0.008 | -0.003 | [-0.021, 0.005] | -0.368 | 0.44 | 0.3281 | 0.3281 | 1.0000 |
| binary_cond_t0.65 | 9 | 0.009 ± 0.003 | 0.000 ± 0.000 | 0.009 | 0.009 | [0.007, 0.011] | 2.920 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| binary_marginal_otsu | 9 | 0.333 ± 0.013 | 0.447 ± 0.042 | -0.114 | -0.126 | [-0.137, -0.081] | -2.429 | 0.11 | 0.0078 | 0.0664 | 0.0391 |
| codebook_16 | 9 | 3.483 ± 0.038 | 2.830 ± 0.047 | 0.652 | 0.650 | [0.615, 0.690] | 10.707 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| codebook_32 | 9 | 4.399 ± 0.047 | 3.850 ± 0.141 | 0.549 | 0.515 | [0.476, 0.636] | 4.154 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| grad_mag | 9 | 3.422 ± 0.025 | 2.111 ± 0.113 | 1.311 | 1.316 | [1.247, 1.382] | 11.668 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| joint_KG_HM_R_27 | 9 | 4.043 ± 0.027 | 3.935 ± 0.065 | 0.107 | 0.086 | [0.074, 0.143] | 1.879 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| laplacian_analysis | 9 | 3.601 ± 0.038 | 2.465 ± 0.167 | 1.136 | 1.118 | [1.053, 1.219] | 8.111 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| mean_curvature | 9 | 3.597 ± 0.043 | 3.566 ± 0.078 | 0.031 | 0.036 | [0.007, 0.061] | 0.686 | 0.78 | 0.0547 | 0.1094 | 0.1797 |
| normal_dir | 9 | 3.659 ± 0.057 | 3.201 ± 0.204 | 0.458 | 0.412 | [0.371, 0.559] | 2.882 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| residual | 9 | 3.471 ± 0.041 | 2.318 ± 0.161 | 1.154 | 1.154 | [1.066, 1.242] | 7.920 | 1.00 | 0.0039 | 0.0664 | 0.0039 |
| trust | 9 | 2.419 ± 0.215 | 1.379 ± 0.440 | 1.040 | 1.019 | [0.898, 1.191] | 4.257 | 1.00 | 0.0039 | 0.0664 | 0.0039 |

**Estimator / discretisation sensitivity of the surface endpoint:**

| variant | n | mean D | 95% CI | p |
|---|---|---|---|---|
| binary_cond_otsu__all | 9 | -0.047 | [-0.067, -0.025] | 0.0078 |
| gaussian_curvature__all | 9 | -0.726 | [-0.795, -0.662] | 0.0039 |
| gaussian_curvature__bins16 | 9 | -0.728 | [-0.799, -0.661] | 0.0039 |
| gaussian_curvature__bins32 | 9 | -0.972 | [-1.054, -0.895] | 0.0039 |
| gaussian_curvature__bins8 | 9 | -0.485 | [-0.539, -0.434] | 0.0039 |
| gaussian_curvature__miller_madow | 9 | -0.728 | [-0.799, -0.661] | 0.0039 |
| gaussian_curvature__sigma0.5 | 9 | -0.869 | [-0.925, -0.810] | 0.0039 |
| gaussian_curvature__sigma1.0 | 9 | -0.728 | [-0.799, -0.661] | 0.0039 |
| gaussian_curvature__sigma2.0 | 9 | -0.530 | [-0.614, -0.444] | 0.0039 |
| gaussian_curvature__surface_weighted | 9 | -1.652 | [-1.714, -1.587] | 0.0039 |

**Maximum-location claim (descriptive atlas, not a test):** fraction of test subjects whose highest supported local-entropy cell lies inside the prespecified features mask — K_G: 0.78, binary conditional: 0.00; features-mask area fraction of valid support: 0.23; top-decile enrichment (mean) K_G 1.538, binary 0.714. Ties, overlapping windows and non-uniform support complicate any uniform-area null; no naive significance test is attached.

### Experiment 3 — what the Forge construction adds

| variant | Δ_K common calib. (mean [CI]) | p | Δ_K within-variant calib. | paired Δ vs baseline [CI], p | map ρ vs baseline | top-decile overlap | patch-rank ρ | converged | runtime s |
|---|---|---|---|---|---|---|---|---|---|
| alpha_x0.5 | -0.691 [-0.756, -0.631] | 0.004 | -0.673 [-0.732, -0.616] | 0.037 [0.021, 0.053], p=0.008 | 0.994 | 0.793 | 0.981 | 1.00 | 0.067 |
| alpha_x2 | -0.792 [-0.856, -0.735] | 0.004 | -0.763 [-0.824, -0.710] | -0.064 [-0.083, -0.049], p=0.004 | 0.993 | 0.779 | 0.992 | 1.00 | 0.098 |
| baseline | -0.728 [-0.799, -0.661] | 0.004 | -0.728 [-0.799, -0.661] | — | n/a | n/a | n/a | 1.00 | 0.072 |
| beta_0 | -0.666 [-0.744, -0.595] | 0.004 | -0.674 [-0.744, -0.609] | 0.062 [0.021, 0.107], p=0.027 | 0.969 | 0.617 | 0.955 | 1.00 | 0.130 |
| constant_trust | -0.697 [-0.762, -0.635] | 0.004 | -0.616 [-0.675, -0.559] | 0.031 [0.016, 0.046], p=0.012 | 0.995 | 0.808 | 0.981 | 1.00 | 0.067 |
| corrected_gradient_torch | -0.939 [-0.991, -0.888] | 0.004 | -0.984 [-1.034, -0.938] | -0.211 [-0.262, -0.166], p=0.004 | 0.965 | 0.567 | 0.968 | 1.00 | 0.454 |
| forged_only | -0.658 [-0.723, -0.593] | 0.004 | -0.666 [-0.727, -0.606] | 0.070 [0.045, 0.096], p=0.004 | 0.975 | 0.617 | 0.960 | 1.00 | 0.076 |
| luminance | 0.058 [-0.036, 0.164] | 0.332 | 0.140 [0.011, 0.283] | 0.786 [0.655, 0.914], p=0.004 | 0.686 | 0.257 | 0.590 | 1.00 | 0.000 |
| smoothed_luminance | 1.852 [1.692, 2.023] | 0.004 | 1.292 [1.053, 1.556] | 2.580 [2.395, 2.770], p=0.004 | 0.086 | 0.197 | -0.091 | 1.00 | 0.001 |

Matched-smoothed luminance sigma chosen on calibration: 3.0 px. Common calibration = baseline quantizers applied to every variant; within = quantizers refitted per variant on calibration images (sensitivity).

### Experiment 4 — robustness (n = 3 prespecified images)

| perturbation | n | mean ΔD_K (bits) | SD | sign agreement | mean ΔD_binary | K_G map ρ | top-decile overlap | binary map ρ | K_G clip frac | retained |
|---|---|---|---|---|---|---|---|---|---|---|
| bins:b16 | 3 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| bins:b32 | 3 | -0.236 | 0.037 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| bins:b8 | 3 | 0.219 | 0.030 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| landmark_jitter:sd1.0 | 15 | -0.003 | 0.054 | 1.00 | 0.006 | nan | nan | nan | 0.011 | 1.00 |
| landmark_jitter:sd2.0 | 15 | -0.005 | 0.092 | 1.00 | 0.007 | nan | nan | nan | 0.010 | 1.00 |
| photometric:blur_s0.5 | 3 | 0.049 | 0.017 | 1.00 | 0.011 | 0.996 | 0.847 | 0.931 | 0.010 | 1.00 |
| photometric:blur_s1.0 | 3 | 0.175 | 0.030 | 1.00 | 0.057 | 0.960 | 0.625 | 0.862 | 0.009 | 1.00 |
| photometric:brightness_x0.8 | 3 | -0.120 | 0.023 | 1.00 | 0.009 | 0.990 | 0.789 | 0.992 | 0.012 | 1.00 |
| photometric:brightness_x1.2 | 3 | 0.100 | 0.011 | 1.00 | 0.002 | 0.992 | 0.799 | 0.998 | 0.010 | 1.00 |
| photometric:gains_0.95_1.05 | 3 | -0.010 | 0.004 | 1.00 | 0.002 | 1.000 | 0.961 | 0.995 | 0.011 | 1.00 |
| photometric:gains_1.05_0.95 | 3 | 0.007 | 0.008 | 1.00 | 0.004 | 1.000 | 0.957 | 0.995 | 0.010 | 1.00 |
| photometric:gamma_0.8 | 3 | 0.041 | 0.007 | 1.00 | 0.070 | 0.992 | 0.768 | 0.826 | 0.010 | 1.00 |
| photometric:gamma_1.2 | 3 | -0.087 | 0.041 | 1.00 | 0.005 | 0.991 | 0.762 | 0.837 | 0.011 | 1.00 |
| photometric:jpeg_q50 | 3 | 0.008 | 0.059 | 1.00 | 0.026 | 0.935 | 0.441 | 0.970 | 0.009 | 1.00 |
| photometric:jpeg_q75 | 3 | -0.033 | 0.052 | 1.00 | 0.018 | 0.956 | 0.558 | 0.971 | 0.009 | 1.00 |
| photometric:jpeg_q95 | 3 | -0.023 | 0.012 | 1.00 | 0.006 | 0.987 | 0.730 | 0.987 | 0.011 | 1.00 |
| photometric:noise_sd0.01 | 3 | 0.000 | 0.037 | 1.00 | -0.065 | 0.969 | 0.637 | 0.970 | 0.012 | 1.00 |
| photometric:noise_sd0.03 | 3 | -0.073 | 0.038 | 1.00 | -0.234 | 0.909 | 0.323 | 0.742 | 0.012 | 1.00 |
| resolution:res128 | 3 | 0.088 | 0.172 | 1.00 | 0.091 | nan | nan | nan | 0.002 | 0.33 |
| resolution:res512 | 3 | 0.363 | 0.032 | 1.00 | 0.027 | nan | nan | nan | 0.025 | 1.00 |
| smoothing:sigma0.5 | 3 | -0.167 | 0.061 | 1.00 | -0.008 | 0.973 | 0.633 | 0.923 | 0.015 | 1.00 |
| smoothing:sigma1.0 | 3 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| smoothing:sigma2.0 | 3 | 0.197 | 0.175 | 1.00 | 0.059 | 0.935 | 0.551 | 0.879 | 0.002 | 1.00 |
| window:w11 | 3 | 0.000 | 0.000 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| window:w21 | 3 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| window:w31 | 3 | 0.000 | 0.000 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |

Pose was not numerically filtered: FEI pose index 11 (frontal neutral) is the inclusion rule; a landmark yaw proxy is recorded descriptively in `alignment_qa.json`. Glasses/facial hair are not annotated in FEI; no such strata could be formed.

### Experiment 5 — sampling policies (full-image-guided; usefulness untested here)

| policy@q | n | achieved q | frac in central patches | frac in features mask | Jaccard stability |
|---|---|---|---|---|---|
| binary_cond_entropy@q0.1 | 3 | 0.079 | 0.11 | 0.14 | 0.63 |
| binary_cond_entropy@q0.2 | 3 | 0.164 | 0.12 | 0.24 | 0.76 |
| binary_cond_entropy@q0.4 | 3 | 0.330 | 0.20 | 0.28 | 0.85 |
| central_first@q0.1 | 3 | 0.079 | 0.94 | 0.95 | n/a |
| central_first@q0.2 | 3 | 0.164 | 0.76 | 0.87 | n/a |
| central_first@q0.4 | 3 | 0.330 | 0.41 | 0.67 | n/a |
| gradient_magnitude@q0.1 | 3 | 0.079 | 0.17 | 0.33 | n/a |
| gradient_magnitude@q0.2 | 3 | 0.164 | 0.28 | 0.42 | n/a |
| gradient_magnitude@q0.4 | 3 | 0.330 | 0.28 | 0.40 | n/a |
| kg_entropy@q0.1 | 3 | 0.079 | 0.18 | 0.37 | 0.78 |
| kg_entropy@q0.2 | 3 | 0.164 | 0.15 | 0.32 | 0.74 |
| kg_entropy@q0.4 | 3 | 0.330 | 0.12 | 0.23 | 0.87 |
| random@q0.1 | 15 | 0.079 | 0.17 | 0.28 | n/a |
| random@q0.2 | 15 | 0.164 | 0.17 | 0.30 | n/a |
| random@q0.4 | 15 | 0.330 | 0.16 | 0.29 | n/a |

### Experiment 6 — task-relevant information (face verification, unseen identities)

Not run: fewer than 15 subjects with >= 4 photographs; verification not run.  Subjects with ≥4 images: 0.

## 2. How it was measured

- Surface: existing Rust pipeline `forge_detect._core.solve_and_extract` via `CpuBackend` with unchanged defaults ({'albedo_window_radius': 2, 'c_z': 1.0, 'epsilon': 0.001, 'k_ratio': 1.4142135623730951, 'n_scales': 4, 'neigh_radius': 6, 'p_edge': 0.7, 'p_flat': 0.3, 'pde_alpha': 0.5, 'pde_beta': 5.0, 'pde_lambda': 1.0, 'pde_log_every': 10, 'pde_max_iter': 500, 'pde_tol': 1e-05, 'rayon_threads': 4, 'sigma_base': 1.0, 'sigma_ref': 4.0, 'tensor_window_radius': 2, 'trust_mode': 'heuristic', 'w_b': 0.114, 'w_g': 0.587, 'w_r': 0.299}); backend recorded per image in the cache metadata (CPU Rust; never MPS).
- Random variable: `X_{F,A} = Q(F(U))`, `U` uniform over valid pixels of a 32×32 patch; plug-in entropy in bits; 900 fixed seeded samples per patch (800 block anchors for binary blocks).
- Quantizer: calibration-frozen asinh/log1p transform, 16 equal-width bins between calibration quantiles 0.005/0.995 (`calibration.json`); 8/32 bins as sensitivity.
- Geometry: `z̃ = (z* − median)/s_z`, Gaussian σ=1 px analysis smoothing, central differences with h = 1/255, `K_G`, `H_M`, normals as in protocol §4.3.
- Patches: protocol §5.4 rules on iBUG-68 landmarks (face-alignment 2DFAN-4 + S3FD), similarity alignment to (84,100)/(172,100)/(128,184).
- Inference: subjects are the unit; 10,000 subject bootstrap resamples; 10,000 seeded sign-flips; Holm over the two confirmatory endpoints; sign test as a different estimand.
- Cohort: requested 12 subjects, available 12, primary images 12 (calibration 3, test 9), secondary images 0. Notes: []
- Exclusions: 0 rows (`exclusions.csv`); reasons: {}
- Power (planning only, pilot SD from calibration subjects): binary_conditional: pilot SD 0.016 (n=3), n for d=0.3 at 80% power ≈ 106; gaussian_curvature: pilot SD 0.098 (n=3), n for d=0.3 at 80% power ≈ 106
- Resource use: timings {'align_primary': 0.08657270902767777, 'align_secondary': 0.00012979097664356232, 'calibration': 0.8366612088866532, 'exp1_exp2': 9.865875499788672, 'exp3': 46.38168212492019, 'exp4': 27.848406958859414, 'exp5': 0.8677571248263121, 'exp6': 0.002070459071546793, 'manifest': 0.6559409997425973, 'primary_metrics': 9.577710416633636, 'qa': 0.09458379121497273}; output directory size 218 MB; threads RAYON_NUM_THREADS=4.
- Source: forge repo commit d2881b09d3388cb46c93790f0f6bdeb969f769bb (dirty=True); source digest 871de52f768b2ab6.

## 3. Experiment 0 — control details

| check | category | passed | blocks |
|---|---|---|---|
| rust.import | integrity | ✅ |  |
| rust.shapes | integrity | ✅ |  |
| rust.finite | integrity | ✅ |  |
| rust.repeat_determinism | integrity | ✅ |  |
| rust.adapter_vs_direct_core | integrity | ✅ |  |
| rust.channel_order_sensitive | integrity | ✅ |  |
| rust.param_equality | integrity | ✅ |  |
| rust.laplacian_export_matches_replica | integrity | ✅ |  |
| solver.energy_final_is_last_logged_iterate | audit | ✅ |  |
| solver.log_every_does_not_change_z_star | integrity | ✅ |  |
| solver.replica_energy_matches_rust | integrity | ✅ |  |
| solver.autograd_matches_finite_difference | integrity | ✅ |  |
| solver.one_iteration_recovery | integrity | ✅ |  |
| solver.update_direction_matches_energy_gradient | audit | ❌ | interpretation of z* as the energy minimiser |
| solver.stationarity_at_z_star | audit | ❌ | claims that the baseline z* is stationary for E |
| solver.scharr_vs_central_divergence_adjoint | audit | ❌ | treating div as the exact adjoint of the Scharr gradient |
| solver.corrected_variant_reaches_stationarity | integrity | ✅ |  |
| entropy.constant_black.zero | integrity | ✅ |  |
| entropy.constant_white.zero | integrity | ✅ |  |
| entropy.halves.marginal_one_bit_low_conditional | integrity | ✅ |  |
| entropy.checkerboard.marginal_one_bit | integrity | ✅ |  |
| entropy.checkerboard.conditional_zero | integrity | ✅ |  |
| entropy.ramp_otsu.balanced | integrity | ✅ |  |
| entropy.noise.marginal_near_one | integrity | ✅ |  |
| entropy.noise.block_near_four | integrity | ✅ |  |
| entropy.noise.conditional_near_one | integrity | ✅ |  |
| entropy.permutation_preserves_marginal | integrity | ✅ |  |
| entropy.permutation_changes_block | integrity | ✅ |  |
| entropy.scalar_sample_permutation_invariant | integrity | ✅ |  |
| entropy.single_valued_descriptor_zero | integrity | ✅ |  |
| entropy.bin_mass_sums_to_one | integrity | ✅ |  |
| entropy.bits_nats_conversion | integrity | ✅ |  |
| entropy.local_map_nan_outside_support | integrity | ✅ |  |
| entropy.local_map_matches_skimage_rank_entropy | integrity | ✅ |  |
| entropy.miller_madow_not_clamped | integrity | ✅ |  |
| geometry.plane_constant.n65 | integrity | ✅ |  |
| geometry.plane_tilted.n65 | integrity | ✅ |  |
| geometry.paraboloid.n65 | integrity | ✅ |  |
| geometry.saddle.n65 | integrity | ✅ |  |
| geometry.cylinder.n65 | integrity | ✅ |  |
| geometry.cylinder_needs_mean_curvature.n65 | integrity | ✅ |  |
| geometry.vertical_translation_invariant.n65 | integrity | ✅ |  |
| geometry.vertical_scale_changes_geometry.n65 | integrity | ✅ |  |
| geometry.plane_constant.n129 | integrity | ✅ |  |
| geometry.plane_tilted.n129 | integrity | ✅ |  |
| geometry.paraboloid.n129 | integrity | ✅ |  |
| geometry.saddle.n129 | integrity | ✅ |  |
| geometry.cylinder.n129 | integrity | ✅ |  |
| geometry.cylinder_needs_mean_curvature.n129 | integrity | ✅ |  |
| geometry.vertical_translation_invariant.n129 | integrity | ✅ |  |
| geometry.vertical_scale_changes_geometry.n129 | integrity | ✅ |  |
| geometry.plane_constant.n257 | integrity | ✅ |  |
| geometry.plane_tilted.n257 | integrity | ✅ |  |
| geometry.paraboloid.n257 | integrity | ✅ |  |
| geometry.saddle.n257 | integrity | ✅ |  |
| geometry.cylinder.n257 | integrity | ✅ |  |
| geometry.cylinder_needs_mean_curvature.n257 | integrity | ✅ |  |
| geometry.vertical_translation_invariant.n257 | integrity | ✅ |  |
| geometry.vertical_scale_changes_geometry.n257 | integrity | ✅ |  |
| geometry.refinement_converges | integrity | ✅ |  |
| geometry.smoothing_preserves_quadratic_curvature | integrity | ✅ |  |
| coords.round_trip_subpixel | integrity | ✅ |  |
| coords.targets_hit | integrity | ✅ |  |
| coords.marker_lands_on_landmark_no_transpose | integrity | ✅ |  |
| coords.similarity_exact_fit | integrity | ✅ |  |
| nonlocal.measured | integrity | ✅ |  |
| selection.roi_translation_not_forced_central | integrity | ✅ | any claim that the ROI layout cannot by itself produce a central advantage |

Non-local settlement influence (synthetic): cheek_left->eye_inner_left: max |Δz*| in the untouched patch = 0.00243; eye_inner_left->cheek_left: max |Δz*| in the untouched patch = 0.00319.
ROI-translation control (features drawn at offset from the ROI layout): offset (-40, 0): D_K = 0.042, D_bin = -0.084; offset (0, -48): D_K = 0.700, D_bin = -0.038; offset (0, 0): D_K = -0.612, D_bin = -0.007; offset (0, 48): D_K = 0.471, D_bin = -0.011.

## 4. Stages not run / limitations

- prepare: done
- exp0: done (0 integrity failures, 3 audit findings)
- exp1: done (binary endpoints)
- exp2: done (surface endpoints + maps + atlas)
- exp3: done
- exp4: done
- exp5: done
- exp6: not run: fewer than 15 subjects with >= 4 photographs; verification not run
- report: done (11 figures)

- The settled field is an algorithm-constructed proxy surface; nothing here validates it as a metric face reconstruction.
- Regional surface entropy does not localise the source of information: settlement is global (see non-local control).
- The landmark-polygon mask is an approximation, not a parsed skin mask; forehead extension may include hair for some subjects (see `figures/qa_overlays.png`).
- FEI provides one acquisition session per subject with a white background and controlled lighting; conclusions are population- and acquisition-dependent.
- Entropy is not salience, semantic relevance, identity, physical truth or authenticity; Experiment 6 measures utility for one lightweight decoder only.
- The exported Rust `energy_*` terms belong to the last *logged* iterate (every `pde_log_every` iterations), not necessarily to the returned `z*`.
- **This run used synthetic pseudo-faces only. No empirical claim about faces follows from it.**

## 5. Figures

- `figures/ablations.png`
- `figures/atlas.png`
- `figures/example_q000_synth007.png`
- `figures/example_q025_synth004.png`
- `figures/example_q050_synth001.png`
- `figures/example_q075_synth000.png`
- `figures/example_q100_synth002.png`
- `figures/paired_differences.png`
- `figures/policies.png`
- `figures/qa_overlays.png`
- `figures/robustness.png`

## 6. Decision rules

See `protocol.md` §18 of the implementation prompt: a positive binary endpoint stable to thresholds supports a central advantage for the binary estimator; a positive surface endpoint alone supports a central advantage in this constructed surface's quantised curvature distribution; luminance/Forge agreement means a simpler explanation suffices; trust-dependent contrast implicates the trust prior; weak verification utility means entropy did not identify useful identity information for the tested decoder; non-positive or uncertain contrasts leave the hypothesis unsupported for this protocol.
