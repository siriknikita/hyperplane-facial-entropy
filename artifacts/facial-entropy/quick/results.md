# Regional facial entropy on the Hyperplane-Forge surface — results (quick profile)

**Run type:** REAL DATA: FEI  
**Seed:** 20261008  **Status:** {'exp0': 'done (0 integrity failures, 3 audit findings)', 'exp1': 'done (binary endpoints)', 'exp2': 'done (surface endpoints + maps + atlas)', 'exp3': 'done', 'exp4': 'done', 'exp5': 'done', 'exp6': 'done', 'prepare': 'done', 'report': 'done (17 figures)'}  **Output:** `/Users/mykytasiryk/Development/personal-projects/hyperplane-facial-entropy/artifacts/facial-entropy/quick`

## 1. Findings

**Experiment 0 (controls):** 67 checks, 0 integrity failures, 3 audit findings.  
Audit findings (solver discrepancies, reported not corrected in the baseline): ['solver.update_direction_matches_energy_gradient', 'solver.stationarity_at_z_star', 'solver.scharr_vs_central_divergence_adjoint']  

**Solver audit.** The implemented update uses the divergence term with sign `-beta*div (paper/code)`; the true gradient of the implemented discrete energy carries `+beta*div`. At the Rust `z*` the energy-gradient norm is 0.380 × its value at `z_forged` (iterations 24, converged flag True). The baseline `z*` is therefore a damped partial descent, not a stationary point of E. The baseline is preserved unchanged; `corrected_gradient_torch` (float64 L-BFGS on the validated replica energy) is reported as a separately labelled diagnostic variant (gradient ratio 0.000002).

**Co-primary endpoints** (test split only; subjects = independent unit; 40 test subjects, 40 with all eight patches retained; 10 calibration subjects never used for inference).

| endpoint | n subj | central mean ± SD (bits) | peripheral mean ± SD | mean D | median D | 95% boot CI | d_z | P(D>0) | sign-flip p | Holm p | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_conditional (`binary_cond_otsu`) | 40 | 0.109 ± 0.025 | 0.054 ± 0.030 | 0.055 | 0.051 | [0.042, 0.069] | 1.259 | 0.90 | 0.0001 | 0.0002 | 0.0000 |
| gaussian_curvature (`gaussian_curvature`) | 40 | 2.452 ± 0.159 | 3.232 ± 0.228 | -0.780 | -0.790 | [-0.848, -0.712] | -3.453 | 0.00 | 0.0001 | 0.0002 | 0.0000 |

- **binary_conditional:** estimated contrast is positive (central > peripheral); Holm-adjusted p < 0.05. Interpretation: see decision rules (§6).
- **gaussian_curvature:** estimated contrast is non-positive; Holm-adjusted p < 0.05. Interpretation: see decision rules (§6).

**Secondary metrics (exploratory family, own Holm correction):**

| metric | n subj | central ± SD | peripheral ± SD | mean D | median D | 95% CI | d_z | P(D>0) | p | Holm p (secondary) | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_block4_otsu | 40 | 1.182 ± 0.122 | 0.596 ± 0.270 | 0.586 | 0.552 | [0.496, 0.679] | 1.952 | 1.00 | 0.0001 | 0.0017 | 0.0000 |
| binary_cond_otsu_rot1 | 40 | 0.102 ± 0.027 | 0.052 ± 0.032 | 0.050 | 0.048 | [0.035, 0.065] | 0.996 | 0.88 | 0.0001 | 0.0017 | 0.0000 |
| binary_cond_otsu_rot2 | 40 | 0.106 ± 0.026 | 0.054 ± 0.030 | 0.052 | 0.044 | [0.038, 0.067] | 1.134 | 0.80 | 0.0001 | 0.0017 | 0.0002 |
| binary_cond_otsu_rot3 | 40 | 0.107 ± 0.026 | 0.055 ± 0.032 | 0.052 | 0.051 | [0.037, 0.067] | 1.083 | 0.85 | 0.0001 | 0.0017 | 0.0000 |
| binary_cond_t0.35 | 40 | 0.098 ± 0.025 | 0.064 ± 0.036 | 0.033 | 0.031 | [0.019, 0.048] | 0.691 | 0.75 | 0.0001 | 0.0017 | 0.0022 |
| binary_cond_t0.50 | 40 | 0.040 ± 0.032 | 0.028 ± 0.032 | 0.012 | 0.009 | [0.003, 0.021] | 0.403 | 0.70 | 0.0143 | 0.0572 | 0.0012 |
| binary_cond_t0.65 | 40 | 0.000 ± 0.001 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.001] | 0.158 | 0.03 | 1.0000 | 1.0000 | 1.0000 |
| binary_marginal_otsu | 40 | 0.736 ± 0.068 | 0.388 ± 0.174 | 0.348 | 0.312 | [0.298, 0.399] | 2.119 | 1.00 | 0.0001 | 0.0017 | 0.0000 |
| codebook_16 | 40 | 3.402 ± 0.117 | 3.328 ± 0.136 | 0.074 | 0.086 | [0.027, 0.119] | 0.500 | 0.72 | 0.0044 | 0.0220 | 0.0064 |
| codebook_32 | 40 | 4.314 ± 0.119 | 4.126 ± 0.145 | 0.187 | 0.200 | [0.137, 0.236] | 1.157 | 0.90 | 0.0001 | 0.0017 | 0.0000 |
| grad_mag | 40 | 3.504 ± 0.106 | 2.842 ± 0.190 | 0.662 | 0.664 | [0.605, 0.718] | 3.550 | 1.00 | 0.0001 | 0.0017 | 0.0000 |
| joint_KG_HM_R_27 | 40 | 4.068 ± 0.087 | 4.050 ± 0.086 | 0.017 | -0.007 | [-0.026, 0.060] | 0.125 | 0.47 | 0.4284 | 1.0000 | 0.8746 |
| laplacian_analysis | 40 | 3.675 ± 0.071 | 3.070 ± 0.190 | 0.604 | 0.609 | [0.546, 0.663] | 3.184 | 1.00 | 0.0001 | 0.0017 | 0.0000 |
| mean_curvature | 40 | 3.530 ± 0.061 | 3.538 ± 0.085 | -0.008 | -0.005 | [-0.037, 0.021] | -0.081 | 0.50 | 0.6064 | 1.0000 | 1.0000 |
| normal_dir | 40 | 3.529 ± 0.066 | 3.300 ± 0.159 | 0.229 | 0.261 | [0.177, 0.278] | 1.375 | 0.90 | 0.0001 | 0.0017 | 0.0000 |
| residual | 40 | 3.655 ± 0.071 | 3.017 ± 0.192 | 0.638 | 0.628 | [0.579, 0.696] | 3.342 | 1.00 | 0.0001 | 0.0017 | 0.0000 |
| trust | 40 | 2.530 ± 0.226 | 1.523 ± 0.230 | 1.008 | 0.993 | [0.912, 1.106] | 3.179 | 1.00 | 0.0001 | 0.0017 | 0.0000 |

**Estimator / discretisation sensitivity of the surface endpoint:**

| variant | n | mean D | 95% CI | p |
|---|---|---|---|---|
| binary_cond_otsu__all | 40 | 0.056 | [0.042, 0.069] | 0.0001 |
| gaussian_curvature__all | 40 | -0.781 | [-0.849, -0.714] | 0.0001 |
| gaussian_curvature__bins16 | 40 | -0.780 | [-0.848, -0.712] | 0.0001 |
| gaussian_curvature__bins32 | 40 | -0.961 | [-1.045, -0.878] | 0.0001 |
| gaussian_curvature__bins8 | 40 | -0.579 | [-0.631, -0.527] | 0.0001 |
| gaussian_curvature__miller_madow | 40 | -0.780 | [-0.848, -0.712] | 0.0001 |
| gaussian_curvature__sigma0.5 | 40 | -0.801 | [-0.868, -0.735] | 0.0001 |
| gaussian_curvature__sigma1.0 | 40 | -0.780 | [-0.848, -0.712] | 0.0001 |
| gaussian_curvature__sigma2.0 | 33 | -0.751 | [-0.860, -0.645] | 0.0001 |
| gaussian_curvature__surface_weighted | 40 | -0.813 | [-0.890, -0.736] | 0.0001 |

**Maximum-location claim (descriptive atlas, not a test):** fraction of test subjects whose highest supported local-entropy cell lies inside the prespecified features mask — K_G: 0.00, binary conditional: 0.35; features-mask area fraction of valid support: 0.21; top-decile enrichment (mean) K_G 0.075, binary 1.187. Ties, overlapping windows and non-uniform support complicate any uniform-area null; no naive significance test is attached.

### Experiment 3 — what the Forge construction adds

| variant | Δ_K common calib. (mean [CI]) | p | Δ_K within-variant calib. | paired Δ vs baseline [CI], p | map ρ vs baseline | top-decile overlap | patch-rank ρ | converged | runtime s |
|---|---|---|---|---|---|---|---|---|---|
| alpha_x0.5 | -0.788 [-0.855, -0.721] | 0.000 | -0.711 [-0.773, -0.650] | -0.007 [-0.013, -0.002], p=0.010 | 0.999 | 0.888 | 0.989 | 1.00 | 0.075 |
| alpha_x2 | -0.767 [-0.839, -0.696] | 0.000 | -0.776 [-0.847, -0.704] | 0.013 [0.005, 0.022], p=0.004 | 0.997 | 0.878 | 0.985 | 1.00 | 0.104 |
| baseline | -0.780 [-0.848, -0.712] | 0.000 | -0.780 [-0.848, -0.712] | — | n/a | n/a | n/a | 1.00 | 0.085 |
| beta_0 | -0.778 [-0.846, -0.710] | 0.000 | -0.666 [-0.728, -0.605] | 0.003 [-0.015, 0.022], p=0.780 | 0.989 | 0.775 | 0.962 | 1.00 | 0.123 |
| constant_trust | -0.784 [-0.850, -0.718] | 0.000 | -0.766 [-0.831, -0.702] | -0.004 [-0.010, 0.001], p=0.163 | 0.998 | 0.868 | 0.988 | 1.00 | 0.070 |
| corrected_gradient_torch | -0.833 [-0.906, -0.760] | 0.000 | -0.770 [-0.837, -0.704] | -0.053 [-0.084, -0.019], p=0.002 | 0.964 | 0.674 | 0.932 | 1.00 | 0.535 |
| forged_only | -0.786 [-0.849, -0.723] | 0.000 | -0.789 [-0.852, -0.727] | -0.005 [-0.017, 0.005], p=0.372 | 0.993 | 0.741 | 0.976 | 1.00 | 0.086 |
| luminance | -0.887 [-0.962, -0.811] | 0.000 | -0.985 [-1.071, -0.897] | -0.107 [-0.169, -0.047], p=0.001 | 0.818 | 0.415 | 0.769 | 1.00 | 0.000 |
| smoothed_luminance | -0.477 [-0.620, -0.328] | 0.000 | -0.441 [-0.572, -0.305] | 0.303 [0.176, 0.435], p=0.000 | 0.620 | 0.342 | 0.606 | 1.00 | 0.001 |

Matched-smoothed luminance sigma chosen on calibration: 3.0 px. Common calibration = baseline quantizers applied to every variant; within = quantizers refitted per variant on calibration images (sensitivity).

### Experiment 4 — robustness (n = 10 prespecified images)

| perturbation | n | mean ΔD_K (bits) | SD | sign agreement | mean ΔD_binary | K_G map ρ | top-decile overlap | binary map ρ | K_G clip frac | retained |
|---|---|---|---|---|---|---|---|---|---|---|
| bins:b16 | 10 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| bins:b32 | 10 | -0.191 | 0.064 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| bins:b8 | 10 | 0.221 | 0.069 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| landmark_jitter:sd1.0 | 200 | -0.012 | 0.033 | 1.00 | -0.000 | nan | nan | nan | 0.010 | 1.00 |
| landmark_jitter:sd2.0 | 200 | -0.003 | 0.054 | 1.00 | -0.002 | nan | nan | nan | 0.010 | 0.99 |
| photometric:blur_s0.5 | 10 | 0.021 | 0.024 | 1.00 | -0.001 | 0.997 | 0.852 | 0.983 | 0.009 | 1.00 |
| photometric:blur_s1.0 | 10 | 0.107 | 0.116 | 1.00 | 0.002 | 0.971 | 0.714 | 0.955 | 0.009 | 1.00 |
| photometric:brightness_x0.8 | 10 | 0.017 | 0.048 | 1.00 | -0.000 | 0.992 | 0.810 | 0.997 | 0.010 | 1.00 |
| photometric:brightness_x1.2 | 10 | -0.003 | 0.051 | 1.00 | -0.001 | 0.993 | 0.863 | 0.994 | 0.009 | 1.00 |
| photometric:gains_0.95_1.05 | 10 | -0.002 | 0.006 | 1.00 | 0.001 | 0.999 | 0.919 | 0.992 | 0.010 | 1.00 |
| photometric:gains_1.05_0.95 | 10 | 0.005 | 0.012 | 1.00 | -0.002 | 0.999 | 0.916 | 0.990 | 0.010 | 1.00 |
| photometric:gamma_0.8 | 10 | 0.037 | 0.076 | 1.00 | 0.017 | 0.985 | 0.770 | 0.947 | 0.009 | 1.00 |
| photometric:gamma_1.2 | 10 | -0.044 | 0.066 | 1.00 | -0.002 | 0.990 | 0.775 | 0.928 | 0.009 | 1.00 |
| photometric:jpeg_q50 | 10 | 0.025 | 0.085 | 1.00 | 0.012 | 0.974 | 0.592 | 0.967 | 0.009 | 1.00 |
| photometric:jpeg_q75 | 10 | -0.001 | 0.062 | 1.00 | 0.005 | 0.985 | 0.658 | 0.981 | 0.010 | 1.00 |
| photometric:jpeg_q95 | 10 | 0.006 | 0.024 | 1.00 | 0.005 | 0.994 | 0.770 | 0.989 | 0.010 | 1.00 |
| photometric:noise_sd0.01 | 10 | 0.026 | 0.063 | 1.00 | -0.010 | 0.987 | 0.660 | 0.964 | 0.010 | 1.00 |
| photometric:noise_sd0.03 | 10 | 0.060 | 0.085 | 1.00 | -0.043 | 0.959 | 0.477 | 0.863 | 0.010 | 1.00 |
| resolution:res128 | 10 | 0.159 | 0.201 | 1.00 | 0.040 | nan | nan | nan | 0.003 | 0.20 |
| resolution:res512 | 10 | 0.212 | 0.156 | 1.00 | -0.014 | nan | nan | nan | 0.026 | 1.00 |
| smoothing:sigma0.5 | 10 | 0.001 | 0.074 | 1.00 | -0.000 | 0.985 | 0.738 | 0.997 | 0.012 | 1.00 |
| smoothing:sigma1.0 | 10 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| smoothing:sigma2.0 | 10 | 0.011 | 0.194 | 1.00 | 0.002 | 0.924 | 0.627 | 0.971 | 0.004 | 0.90 |
| window:w11 | 10 | 0.000 | 0.000 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| window:w21 | 10 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| window:w31 | 10 | 0.000 | 0.000 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |

Pose was not numerically filtered: FEI pose index 11 (frontal neutral) is the inclusion rule; a landmark yaw proxy is recorded descriptively in `alignment_qa.json`. Glasses/facial hair are not annotated in FEI; no such strata could be formed.

### Experiment 5 — sampling policies (full-image-guided; usefulness untested here)

| policy@q | n | achieved q | frac in central patches | frac in features mask | Jaccard stability |
|---|---|---|---|---|---|
| binary_cond_entropy@q0.1 | 10 | 0.079 | 0.11 | 0.23 | 0.69 |
| binary_cond_entropy@q0.2 | 10 | 0.162 | 0.15 | 0.27 | 0.76 |
| binary_cond_entropy@q0.4 | 10 | 0.327 | 0.17 | 0.32 | 0.86 |
| central_first@q0.1 | 10 | 0.079 | 0.91 | 0.93 | n/a |
| central_first@q0.2 | 10 | 0.162 | 0.68 | 0.88 | n/a |
| central_first@q0.4 | 10 | 0.327 | 0.35 | 0.63 | n/a |
| gradient_magnitude@q0.1 | 10 | 0.079 | 0.18 | 0.41 | n/a |
| gradient_magnitude@q0.2 | 10 | 0.162 | 0.24 | 0.43 | n/a |
| gradient_magnitude@q0.4 | 10 | 0.327 | 0.25 | 0.42 | n/a |
| kg_entropy@q0.1 | 10 | 0.079 | 0.00 | 0.02 | 0.77 |
| kg_entropy@q0.2 | 10 | 0.162 | 0.00 | 0.03 | 0.88 |
| kg_entropy@q0.4 | 10 | 0.327 | 0.03 | 0.09 | 0.93 |
| random@q0.1 | 200 | 0.079 | 0.14 | 0.26 | n/a |
| random@q0.2 | 200 | 0.162 | 0.14 | 0.26 | n/a |
| random@q0.4 | 200 | 0.327 | 0.14 | 0.26 | n/a |

### Experiment 6 — task-relevant information (face verification, unseen identities)

Identity split train/calibration/test = 30/10/10 subjects; test pairs 270 (135 positive) from 10 subjects — pairs are dependent, subjects are the unit.

| representation | dim | C | test AUROC [subject-boot CI] | balanced acc. | log loss | pred. info score (bits) | TPR@cal-FPR0.1 (achieved FPR) |
|---|---|---|---|---|---|---|---|
| whole_face:combined | 768 | 0.01 | 0.980 [0.951, 0.998] | 0.911 | 0.213 | 0.693 | 0.911 (0.081) |
| whole_face:gray16 | 256 | 0.01 | 0.917 [0.819, 0.986] | 0.874 | 0.395 | 0.430 | 0.770 (0.089) |
| whole_face:hog | 1764 | 0.1 | 0.985 [0.974, 0.998] | 0.922 | 0.226 | 0.674 | 0.993 (0.274) |
| whole_face:surface_KG16 | 256 | 0.01 | 0.809 [0.704, 0.895] | 0.681 | 0.534 | 0.229 | 0.444 (0.104) |
| whole_face:surface_z16 | 256 | 0.01 | 0.954 [0.914, 0.990] | 0.867 | 0.321 | 0.537 | 0.874 (0.133) |
| policy_only:binary_cond_entropy | 512 | 0.01 | 0.762 [0.674, 0.873] | 0.685 | 0.583 | 0.159 | 0.415 (0.059) |
| policy_only:central_first | 512 | 0.01 | 0.715 [0.576, 0.858] | 0.644 | 0.618 | 0.109 | 0.274 (0.081) |
| policy_only:gradient_magnitude | 512 | 0.01 | 0.709 [0.555, 0.838] | 0.615 | 0.656 | 0.054 | 0.274 (0.096) |
| policy_only:kg_entropy | 512 | 0.01 | 0.916 [0.847, 0.975] | 0.844 | 0.381 | 0.450 | 0.837 (0.156) |
| policy_only:random | 512 | 0.01 | 0.476 [0.402, 0.560] | 0.456 | 0.741 | -0.069 | 0.037 (0.000) |

**Region-only verifiers** (same protocol, 32×32 gray patch → 16×16):

| region | class | AUROC [CI] |
|---|---|---|
| cheek_left | peripheral | 0.525 [0.393, 0.663] |
| cheek_right | peripheral | 0.575 [0.443, 0.663] |
| eye_inner_left | central | 0.692 [0.576, 0.822] |
| eye_inner_right | central | 0.660 [0.553, 0.783] |
| forehead_left | peripheral | 0.723 [0.534, 0.908] |
| forehead_right | peripheral | 0.733 [0.551, 0.884] |
| nose_bridge | central | 0.616 [0.490, 0.738] |
| nose_tip | central | 0.608 [0.502, 0.743] |

**Occlusion of the frozen whole-face verifier** (AUROC drop; two fill rules; surface variant re-runs the Rust construction on occluded RGB):

| region | fill | AUROC | drop |
|---|---|---|---|
| cheek_left | face_median | 0.916 | 0.001 |
| cheek_left | mean_gray_0.5 | 0.916 | 0.001 |
| cheek_left | surface_rgb_fill_0.5 | 0.954 | -0.000 |
| cheek_right | face_median | 0.918 | -0.000 |
| cheek_right | mean_gray_0.5 | 0.916 | 0.001 |
| cheek_right | surface_rgb_fill_0.5 | 0.932 | 0.022 |
| eye_inner_left | face_median | 0.912 | 0.005 |
| eye_inner_left | mean_gray_0.5 | 0.914 | 0.003 |
| eye_inner_left | surface_rgb_fill_0.5 | 0.871 | 0.082 |
| eye_inner_right | face_median | 0.910 | 0.007 |
| eye_inner_right | mean_gray_0.5 | 0.911 | 0.006 |
| eye_inner_right | surface_rgb_fill_0.5 | 0.931 | 0.023 |
| forehead_left | face_median | 0.913 | 0.004 |
| forehead_left | mean_gray_0.5 | 0.914 | 0.003 |
| forehead_left | surface_rgb_fill_0.5 | 0.910 | 0.044 |
| forehead_right | face_median | 0.906 | 0.011 |
| forehead_right | mean_gray_0.5 | 0.911 | 0.006 |
| forehead_right | surface_rgb_fill_0.5 | 0.900 | 0.053 |
| nose_bridge | face_median | 0.916 | 0.001 |
| nose_bridge | mean_gray_0.5 | 0.914 | 0.003 |
| nose_bridge | surface_rgb_fill_0.5 | 0.923 | 0.030 |
| nose_tip | face_median | 0.917 | -0.000 |
| nose_tip | mean_gray_0.5 | 0.918 | -0.001 |
| nose_tip | surface_rgb_fill_0.5 | 0.895 | 0.058 |

- Entropy-vs-utility rank agreement over 8 regions for `binary_cond_otsu`: Spearman ρ = 0.095 (eight dependent anatomical regions give very few ranking points; descriptive only).
- Entropy-vs-utility rank agreement over 8 regions for `gaussian_curvature`: Spearman ρ = -0.405 (eight dependent anatomical regions give very few ranking points; descriptive only).

Deepfake task: not applicable: FEI has no manipulated images / original-manipulated links.

## 2. How it was measured

- Surface: existing Rust pipeline `forge_detect._core.solve_and_extract` via `CpuBackend` with unchanged defaults ({'albedo_window_radius': 2, 'c_z': 1.0, 'epsilon': 0.001, 'k_ratio': 1.4142135623730951, 'n_scales': 4, 'neigh_radius': 6, 'p_edge': 0.7, 'p_flat': 0.3, 'pde_alpha': 0.5, 'pde_beta': 5.0, 'pde_lambda': 1.0, 'pde_log_every': 10, 'pde_max_iter': 500, 'pde_tol': 1e-05, 'rayon_threads': 4, 'sigma_base': 1.0, 'sigma_ref': 4.0, 'tensor_window_radius': 2, 'trust_mode': 'heuristic', 'w_b': 0.114, 'w_g': 0.587, 'w_r': 0.299}); backend recorded per image in the cache metadata (CPU Rust; never MPS).
- Random variable: `X_{F,A} = Q(F(U))`, `U` uniform over valid pixels of a 32×32 patch; plug-in entropy in bits; 900 fixed seeded samples per patch (800 block anchors for binary blocks).
- Quantizer: calibration-frozen asinh/log1p transform, 16 equal-width bins between calibration quantiles 0.005/0.995 (`calibration.json`); 8/32 bins as sensitivity.
- Geometry: `z̃ = (z* − median)/s_z`, Gaussian σ=1 px analysis smoothing, central differences with h = 1/255, `K_G`, `H_M`, normals as in protocol §4.3.
- Patches: protocol §5.4 rules on iBUG-68 landmarks (face-alignment 2DFAN-4 + S3FD), similarity alignment to (84,100)/(172,100)/(128,184).
- Inference: subjects are the unit; 10,000 subject bootstrap resamples; 10,000 seeded sign-flips; Holm over the two confirmatory endpoints; sign test as a different estimand.
- Cohort: requested 50 subjects, available 200, primary images 50 (calibration 10, test 40), secondary images 250. Notes: []
- Exclusions: 6 rows (`exclusions.csv`); reasons: {'no_face_detected': 6}
- Power (planning only, pilot SD from calibration subjects): binary_conditional: pilot SD 0.038 (n=10), n for d=0.3 at 80% power ≈ 106; gaussian_curvature: pilot SD 0.346 (n=10), n for d=0.3 at 80% power ≈ 106
- Resource use: timings {'align_primary': 17.598047874867916, 'align_secondary': 75.61916795838624, 'calibration': 2.1885726670734584, 'exp1_exp2': 46.902691250201315, 'exp3': 206.41956487484276, 'exp4': 116.68410191591829, 'exp5': 4.765036040917039, 'exp6': 412.15018954174593, 'manifest': 1.148098083678633, 'primary_metrics': 45.58833658415824, 'qa': 94.16966512519866}; output directory size 4076 MB; threads RAYON_NUM_THREADS=4.
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

Non-local settlement influence (synthetic): cheek_left->eye_inner_left: max |Δz*| in the untouched patch = 0.00243; eye_inner_left->cheek_left: max |Δz*| in the untouched patch = 0.00320.
ROI-translation control (features drawn at offset from the ROI layout): offset (-40, 0): D_K = 0.042, D_bin = -0.084; offset (0, -48): D_K = 0.700, D_bin = -0.038; offset (0, 0): D_K = -0.612, D_bin = -0.007; offset (0, 48): D_K = 0.471, D_bin = -0.011.

## 4. Stages not run / limitations

- prepare: done
- exp0: done (0 integrity failures, 3 audit findings)
- exp1: done (binary endpoints)
- exp2: done (surface endpoints + maps + atlas)
- exp3: done
- exp4: done
- exp5: done
- exp6: done
- report: done (17 figures)

- The settled field is an algorithm-constructed proxy surface; nothing here validates it as a metric face reconstruction.
- Regional surface entropy does not localise the source of information: settlement is global (see non-local control).
- The landmark-polygon mask is an approximation, not a parsed skin mask; forehead extension may include hair for some subjects (see `figures/qa_overlays.png`).
- FEI provides one acquisition session per subject with a white background and controlled lighting; conclusions are population- and acquisition-dependent.
- Entropy is not salience, semantic relevance, identity, physical truth or authenticity; Experiment 6 measures utility for one lightweight decoder only.
- The exported Rust `energy_*` terms belong to the last *logged* iterate (every `pde_log_every` iterations), not necessarily to the returned `z*`.

## 5. Figures

- `figures/ablations.png`
- `figures/atlas.png`
- `figures/example_q000_fei192-11.png`
- `figures/example_q025_fei041-11.png`
- `figures/example_q050_fei009-11.png`
- `figures/example_q075_fei159-11.png`
- `figures/example_q100_fei189-11.png`
- `figures/paired_differences.png`
- `figures/policies.png`
- `figures/projected_fei009-11.png`
- `figures/projected_fei041-11.png`
- `figures/projected_fei159-11.png`
- `figures/projected_fei189-11.png`
- `figures/projected_fei192-11.png`
- `figures/qa_overlays.png`
- `figures/robustness.png`
- `figures/verification.png`

## 6. Decision rules

See `protocol.md` §18 of the implementation prompt: a positive binary endpoint stable to thresholds supports a central advantage for the binary estimator; a positive surface endpoint alone supports a central advantage in this constructed surface's quantised curvature distribution; luminance/Forge agreement means a simpler explanation suffices; trust-dependent contrast implicates the trust prior; weak verification utility means entropy did not identify useful identity information for the tested decoder; non-positive or uncertain contrasts leave the hypothesis unsupported for this protocol.
