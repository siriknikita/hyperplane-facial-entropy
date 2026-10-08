# Regional facial entropy on the Hyperplane-Forge surface — results (main profile)

**Run type:** REAL DATA: FEI  
**Seed:** 20261008  **Status:** {'exp0': 'done (0 integrity failures, 3 audit findings; controls re-validated after the determinism-check update)', 'exp1': 'done (binary endpoints)', 'exp2': 'done (surface endpoints + maps + atlas)', 'exp3': 'done', 'exp4': 'done', 'exp5': 'done', 'exp6': 'done', 'prepare': 'done', 'report': 'done (17 figures)'}  **Output:** `/Users/mykytasiryk/Development/personal-projects/hyperplane-facial-entropy/artifacts/facial-entropy/main`

## 1. Findings

**Experiment 0 (controls):** 68 checks, 0 integrity failures, 3 audit findings.  
Audit findings (solver discrepancies, reported not corrected in the baseline): ['solver.update_direction_matches_energy_gradient', 'solver.stationarity_at_z_star', 'solver.scharr_vs_central_divergence_adjoint']  

**Solver audit.** The implemented update uses the divergence term with sign `-beta*div (paper/code)`; the true gradient of the implemented discrete energy carries `+beta*div`. At the Rust `z*` the energy-gradient norm is 0.348 × its value at `z_forged` (iterations 41, converged flag True). The baseline `z*` is therefore a damped partial descent, not a stationary point of E. The baseline is preserved unchanged; `corrected_gradient_torch` (float64 L-BFGS on the validated replica energy) is reported as a separately labelled diagnostic variant (gradient ratio 0.000004).

**Co-primary endpoints** (test split only; subjects = independent unit; 159 test subjects, 159 with all eight patches retained; 40 calibration subjects never used for inference).

| endpoint | n subj | central mean ± SD (bits) | peripheral mean ± SD | mean D | median D | 95% boot CI | d_z | P(D>0) | sign-flip p | Holm p | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_conditional (`binary_cond_otsu`) | 159 | 0.107 ± 0.025 | 0.057 ± 0.031 | 0.050 | 0.047 | [0.044, 0.057] | 1.201 | 0.89 | 0.0001 | 0.0002 | 0.0000 |
| gaussian_curvature (`gaussian_curvature`) | 159 | 2.689 ± 0.151 | 3.382 ± 0.235 | -0.693 | -0.669 | [-0.734, -0.652] | -2.611 | 0.00 | 0.0001 | 0.0002 | 0.0000 |

- **binary_conditional:** estimated contrast is positive (central > peripheral); Holm-adjusted p < 0.05. Interpretation: see decision rules (§6).
- **gaussian_curvature:** estimated contrast is non-positive; Holm-adjusted p < 0.05. Interpretation: see decision rules (§6).

**Secondary metrics (exploratory family, own Holm correction):**

| metric | n subj | central ± SD | peripheral ± SD | mean D | median D | 95% CI | d_z | P(D>0) | p | Holm p (secondary) | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_block4_otsu | 159 | 1.176 ± 0.134 | 0.628 ± 0.283 | 0.549 | 0.535 | [0.504, 0.594] | 1.900 | 0.98 | 0.0001 | 0.0017 | 0.0000 |
| binary_cond_otsu_rot1 | 159 | 0.100 ± 0.026 | 0.056 ± 0.033 | 0.044 | 0.044 | [0.037, 0.051] | 0.954 | 0.86 | 0.0001 | 0.0017 | 0.0000 |
| binary_cond_otsu_rot2 | 159 | 0.103 ± 0.025 | 0.057 ± 0.032 | 0.047 | 0.043 | [0.040, 0.053] | 1.076 | 0.86 | 0.0001 | 0.0017 | 0.0000 |
| binary_cond_otsu_rot3 | 159 | 0.105 ± 0.026 | 0.058 ± 0.031 | 0.046 | 0.046 | [0.039, 0.053] | 1.056 | 0.84 | 0.0001 | 0.0017 | 0.0000 |
| binary_cond_t0.35 | 159 | 0.096 ± 0.031 | 0.071 ± 0.038 | 0.025 | 0.028 | [0.017, 0.034] | 0.466 | 0.65 | 0.0001 | 0.0017 | 0.0002 |
| binary_cond_t0.50 | 159 | 0.039 ± 0.030 | 0.027 ± 0.035 | 0.012 | 0.006 | [0.007, 0.016] | 0.406 | 0.59 | 0.0001 | 0.0017 | 0.0000 |
| binary_cond_t0.65 | 159 | 0.001 ± 0.004 | 0.000 ± 0.004 | 0.000 | 0.000 | [-0.000, 0.001] | 0.077 | 0.04 | 0.4202 | 0.8175 | 0.0703 |
| binary_marginal_otsu | 159 | 0.741 ± 0.074 | 0.406 ± 0.184 | 0.334 | 0.313 | [0.308, 0.360] | 2.000 | 0.99 | 0.0001 | 0.0017 | 0.0000 |
| codebook_16 | 159 | 3.480 ± 0.092 | 3.403 ± 0.120 | 0.077 | 0.081 | [0.056, 0.099] | 0.561 | 0.71 | 0.0001 | 0.0017 | 0.0000 |
| codebook_32 | 159 | 4.436 ± 0.098 | 4.239 ± 0.124 | 0.197 | 0.215 | [0.173, 0.221] | 1.291 | 0.89 | 0.0001 | 0.0017 | 0.0000 |
| grad_mag | 159 | 3.533 ± 0.094 | 2.792 ± 0.271 | 0.741 | 0.716 | [0.699, 0.783] | 2.747 | 1.00 | 0.0001 | 0.0017 | 0.0000 |
| joint_KG_HM_R_27 | 159 | 4.106 ± 0.063 | 3.977 ± 0.128 | 0.128 | 0.115 | [0.107, 0.151] | 0.895 | 0.83 | 0.0001 | 0.0017 | 0.0000 |
| laplacian_analysis | 159 | 3.694 ± 0.066 | 3.043 ± 0.243 | 0.651 | 0.619 | [0.613, 0.689] | 2.659 | 1.00 | 0.0001 | 0.0017 | 0.0000 |
| mean_curvature | 159 | 3.548 ± 0.055 | 3.554 ± 0.095 | -0.007 | -0.012 | [-0.022, 0.009] | -0.066 | 0.47 | 0.4088 | 0.8175 | 0.5259 |
| normal_dir | 159 | 3.581 ± 0.070 | 3.228 ± 0.161 | 0.353 | 0.353 | [0.327, 0.379] | 2.099 | 0.99 | 0.0001 | 0.0017 | 0.0000 |
| residual | 159 | 3.680 ± 0.065 | 3.007 ± 0.245 | 0.673 | 0.638 | [0.634, 0.712] | 2.702 | 1.00 | 0.0001 | 0.0017 | 0.0000 |
| trust | 159 | 2.485 ± 0.189 | 1.535 ± 0.235 | 0.950 | 0.964 | [0.913, 0.987] | 3.867 | 1.00 | 0.0001 | 0.0017 | 0.0000 |

**Estimator / discretisation sensitivity of the surface endpoint:**

| variant | n | mean D | 95% CI | p |
|---|---|---|---|---|
| binary_cond_otsu__all | 159 | 0.051 | [0.045, 0.058] | 0.0001 |
| gaussian_curvature__all | 159 | -0.693 | [-0.734, -0.652] | 0.0001 |
| gaussian_curvature__bins16 | 159 | -0.693 | [-0.734, -0.652] | 0.0001 |
| gaussian_curvature__bins32 | 159 | -0.861 | [-0.911, -0.811] | 0.0001 |
| gaussian_curvature__bins8 | 159 | -0.501 | [-0.532, -0.470] | 0.0001 |
| gaussian_curvature__miller_madow | 159 | -0.693 | [-0.734, -0.652] | 0.0001 |
| gaussian_curvature__sigma0.5 | 159 | -0.700 | [-0.739, -0.661] | 0.0001 |
| gaussian_curvature__sigma1.0 | 159 | -0.693 | [-0.734, -0.652] | 0.0001 |
| gaussian_curvature__sigma2.0 | 139 | -0.688 | [-0.742, -0.636] | 0.0001 |
| gaussian_curvature__surface_weighted | 159 | -0.819 | [-0.871, -0.768] | 0.0001 |

**Maximum-location claim (descriptive atlas, not a test):** fraction of test subjects whose highest supported local-entropy cell lies inside the prespecified features mask — K_G: 0.00, binary conditional: 0.28; features-mask area fraction of valid support: 0.21; top-decile enrichment (mean) K_G 0.063, binary 1.156. Ties, overlapping windows and non-uniform support complicate any uniform-area null; no naive significance test is attached.

### Experiment 3 — what the Forge construction adds

| variant | Δ_K common calib. (mean [CI]) | p | Δ_K within-variant calib. | paired Δ vs baseline [CI], p | map ρ vs baseline | top-decile overlap | patch-rank ρ | converged | runtime s |
|---|---|---|---|---|---|---|---|---|---|
| alpha_x0.5 | -0.698 [-0.739, -0.658] | 0.000 | -0.709 [-0.750, -0.668] | -0.006 [-0.008, -0.003], p=0.000 | 0.999 | 0.873 | 0.990 | 1.00 | 0.075 |
| alpha_x2 | -0.687 [-0.730, -0.645] | 0.000 | -0.697 [-0.740, -0.654] | 0.005 [0.002, 0.009], p=0.006 | 0.997 | 0.849 | 0.985 | 1.00 | 0.107 |
| baseline | -0.693 [-0.734, -0.652] | 0.000 | -0.693 [-0.734, -0.652] | — | n/a | n/a | n/a | 1.00 | 0.086 |
| beta_0 | -0.698 [-0.740, -0.657] | 0.000 | -0.702 [-0.744, -0.662] | -0.006 [-0.012, 0.001], p=0.118 | 0.989 | 0.751 | 0.972 | 1.00 | 0.126 |
| constant_trust | -0.696 [-0.737, -0.656] | 0.000 | -0.704 [-0.745, -0.663] | -0.004 [-0.006, -0.001], p=0.006 | 0.998 | 0.848 | 0.987 | 1.00 | 0.071 |
| corrected_gradient_torch | -0.763 [-0.808, -0.720] | 0.000 | -0.705 [-0.746, -0.665] | -0.071 [-0.084, -0.057], p=0.000 | 0.966 | 0.633 | 0.943 | 1.00 | 0.538 |
| forged_only | -0.698 [-0.738, -0.659] | 0.000 | -0.669 [-0.708, -0.631] | -0.005 [-0.011, 0.000], p=0.062 | 0.993 | 0.718 | 0.977 | 1.00 | 0.087 |
| luminance | -0.793 [-0.832, -0.752] | 0.000 | -0.931 [-0.977, -0.883] | -0.100 [-0.129, -0.072], p=0.000 | 0.827 | 0.347 | 0.754 | 1.00 | 0.000 |
| smoothed_luminance | -0.444 [-0.511, -0.375] | 0.000 | -0.478 [-0.546, -0.409] | 0.249 [0.193, 0.305], p=0.000 | 0.644 | 0.315 | 0.623 | 1.00 | 0.001 |

Matched-smoothed luminance sigma chosen on calibration: 3.0 px. Common calibration = baseline quantizers applied to every variant; within = quantizers refitted per variant on calibration images (sensitivity).

### Experiment 4 — robustness (n = 50 prespecified images)

| perturbation | n | mean ΔD_K (bits) | SD | sign agreement | mean ΔD_binary | K_G map ρ | top-decile overlap | binary map ρ | K_G clip frac | retained |
|---|---|---|---|---|---|---|---|---|---|---|
| bins:b16 | 50 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| bins:b32 | 50 | -0.179 | 0.054 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| bins:b8 | 50 | 0.208 | 0.064 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| landmark_jitter:sd1.0 | 1000 | -0.007 | 0.031 | 1.00 | -0.001 | nan | nan | nan | 0.010 | 1.00 |
| landmark_jitter:sd2.0 | 1000 | -0.009 | 0.051 | 1.00 | -0.002 | nan | nan | nan | 0.010 | 0.99 |
| photometric:blur_s0.5 | 50 | 0.015 | 0.019 | 1.00 | -0.001 | 0.997 | 0.827 | 0.987 | 0.010 | 1.00 |
| photometric:blur_s1.0 | 50 | 0.085 | 0.065 | 1.00 | -0.002 | 0.968 | 0.647 | 0.961 | 0.009 | 1.00 |
| photometric:brightness_x0.8 | 50 | 0.044 | 0.039 | 1.00 | -0.000 | 0.991 | 0.765 | 0.993 | 0.010 | 1.00 |
| photometric:brightness_x1.2 | 50 | -0.034 | 0.030 | 1.00 | -0.001 | 0.993 | 0.825 | 0.993 | 0.009 | 1.00 |
| photometric:gains_0.95_1.05 | 50 | 0.001 | 0.009 | 1.00 | -0.001 | 0.999 | 0.912 | 0.993 | 0.010 | 1.00 |
| photometric:gains_1.05_0.95 | 50 | -0.000 | 0.011 | 1.00 | -0.001 | 0.999 | 0.918 | 0.992 | 0.010 | 1.00 |
| photometric:gamma_0.8 | 50 | 0.023 | 0.055 | 1.00 | 0.016 | 0.984 | 0.724 | 0.923 | 0.010 | 1.00 |
| photometric:gamma_1.2 | 50 | -0.035 | 0.057 | 1.00 | -0.011 | 0.988 | 0.726 | 0.932 | 0.010 | 1.00 |
| photometric:jpeg_q50 | 50 | 0.025 | 0.053 | 1.00 | 0.014 | 0.973 | 0.565 | 0.968 | 0.010 | 1.00 |
| photometric:jpeg_q75 | 50 | -0.009 | 0.047 | 1.00 | 0.007 | 0.984 | 0.630 | 0.977 | 0.010 | 1.00 |
| photometric:jpeg_q95 | 50 | -0.005 | 0.027 | 1.00 | 0.004 | 0.994 | 0.729 | 0.986 | 0.010 | 1.00 |
| photometric:noise_sd0.01 | 50 | 0.012 | 0.047 | 1.00 | -0.007 | 0.986 | 0.637 | 0.963 | 0.010 | 1.00 |
| photometric:noise_sd0.03 | 50 | 0.061 | 0.105 | 1.00 | -0.004 | 0.956 | 0.449 | 0.885 | 0.011 | 1.00 |
| resolution:res128 | 50 | 0.128 | 0.283 | 0.98 | 0.016 | nan | nan | nan | 0.003 | 0.34 |
| resolution:res512 | 50 | 0.254 | 0.149 | 1.00 | -0.018 | nan | nan | nan | 0.030 | 1.00 |
| smoothing:sigma0.5 | 50 | 0.007 | 0.055 | 1.00 | 0.002 | 0.985 | 0.714 | 0.989 | 0.014 | 1.00 |
| smoothing:sigma1.0 | 50 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| smoothing:sigma2.0 | 50 | -0.023 | 0.132 | 1.00 | -0.005 | 0.922 | 0.576 | 0.975 | 0.004 | 0.92 |
| window:w11 | 50 | 0.000 | 0.000 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |
| window:w21 | 50 | 0.000 | 0.000 | 1.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.010 | 1.00 |
| window:w31 | 50 | 0.000 | 0.000 | 1.00 | 0.000 | nan | nan | nan | 0.010 | 1.00 |

Pose was not numerically filtered: FEI pose index 11 (frontal neutral) is the inclusion rule; a landmark yaw proxy is recorded descriptively in `alignment_qa.json`. Glasses/facial hair are not annotated in FEI; no such strata could be formed.

### Experiment 5 — sampling policies (full-image-guided; usefulness untested here)

| policy@q | n | achieved q | frac in central patches | frac in features mask | Jaccard stability |
|---|---|---|---|---|---|
| binary_cond_entropy@q0.1 | 50 | 0.079 | 0.13 | 0.25 | 0.69 |
| binary_cond_entropy@q0.2 | 50 | 0.162 | 0.14 | 0.26 | 0.77 |
| binary_cond_entropy@q0.4 | 50 | 0.328 | 0.16 | 0.29 | 0.86 |
| central_first@q0.1 | 50 | 0.079 | 0.91 | 0.93 | n/a |
| central_first@q0.2 | 50 | 0.162 | 0.67 | 0.89 | n/a |
| central_first@q0.4 | 50 | 0.328 | 0.34 | 0.62 | n/a |
| gradient_magnitude@q0.1 | 50 | 0.079 | 0.20 | 0.39 | n/a |
| gradient_magnitude@q0.2 | 50 | 0.162 | 0.23 | 0.40 | n/a |
| gradient_magnitude@q0.4 | 50 | 0.328 | 0.24 | 0.41 | n/a |
| kg_entropy@q0.1 | 50 | 0.079 | 0.01 | 0.02 | 0.76 |
| kg_entropy@q0.2 | 50 | 0.162 | 0.01 | 0.03 | 0.85 |
| kg_entropy@q0.4 | 50 | 0.328 | 0.03 | 0.08 | 0.92 |
| random@q0.1 | 1000 | 0.079 | 0.14 | 0.25 | n/a |
| random@q0.2 | 1000 | 0.162 | 0.14 | 0.25 | n/a |
| random@q0.4 | 1000 | 0.328 | 0.14 | 0.25 | n/a |

### Experiment 6 — task-relevant information (face verification, unseen identities)

Identity split train/calibration/test = 120/40/40 subjects; test pairs 1180 (590 positive) from 40 subjects — pairs are dependent, subjects are the unit.

| representation | dim | C | test AUROC [subject-boot CI] | balanced acc. | log loss | pred. info score (bits) | TPR@cal-FPR0.1 (achieved FPR) |
|---|---|---|---|---|---|---|---|
| whole_face:combined | 768 | 0.01 | 0.987 [0.975, 0.994] | 0.942 | 0.153 | 0.779 | 0.968 (0.095) |
| whole_face:gray16 | 256 | 0.01 | 0.889 [0.845, 0.928] | 0.820 | 0.423 | 0.389 | 0.783 (0.146) |
| whole_face:hog | 1764 | 0.01 | 0.995 [0.990, 0.999] | 0.968 | 0.088 | 0.873 | 0.985 (0.054) |
| whole_face:surface_KG16 | 256 | 0.01 | 0.850 [0.820, 0.879] | 0.770 | 0.480 | 0.307 | 0.493 (0.056) |
| whole_face:surface_z16 | 256 | 0.01 | 0.972 [0.950, 0.985] | 0.908 | 0.221 | 0.681 | 0.917 (0.093) |
| policy_only:binary_cond_entropy | 512 | 0.01 | 0.856 [0.809, 0.894] | 0.768 | 0.468 | 0.324 | 0.541 (0.071) |
| policy_only:central_first | 512 | 0.1 | 0.744 [0.680, 0.810] | 0.691 | 0.586 | 0.155 | 0.214 (0.086) |
| policy_only:gradient_magnitude | 512 | 0.01 | 0.804 [0.742, 0.860] | 0.734 | 0.570 | 0.178 | 0.415 (0.085) |
| policy_only:kg_entropy | 512 | 0.01 | 0.891 [0.851, 0.927] | 0.797 | 0.414 | 0.402 | 0.669 (0.102) |
| policy_only:random | 512 | 0.01 | 0.476 [0.435, 0.517] | 0.503 | 0.721 | -0.040 | 0.125 (0.168) |

**Region-only verifiers** (same protocol, 32×32 gray patch → 16×16):

| region | class | AUROC [CI] |
|---|---|---|
| cheek_left | peripheral | 0.592 [0.538, 0.645] |
| cheek_right | peripheral | 0.647 [0.587, 0.702] |
| eye_inner_left | central | 0.738 [0.669, 0.806] |
| eye_inner_right | central | 0.754 [0.692, 0.817] |
| forehead_left | peripheral | 0.744 [0.695, 0.797] |
| forehead_right | peripheral | 0.740 [0.679, 0.795] |
| nose_bridge | central | 0.639 [0.563, 0.716] |
| nose_tip | central | 0.636 [0.571, 0.700] |

**Occlusion of the frozen whole-face verifier** (AUROC drop; two fill rules; surface variant re-runs the Rust construction on occluded RGB):

| region | fill | AUROC | drop |
|---|---|---|---|
| cheek_left | face_median | 0.891 | -0.002 |
| cheek_left | mean_gray_0.5 | 0.890 | -0.001 |
| cheek_left | surface_rgb_fill_0.5 | 0.969 | 0.003 |
| cheek_right | face_median | 0.889 | 0.000 |
| cheek_right | mean_gray_0.5 | 0.888 | 0.001 |
| cheek_right | surface_rgb_fill_0.5 | 0.955 | 0.017 |
| eye_inner_left | face_median | 0.883 | 0.006 |
| eye_inner_left | mean_gray_0.5 | 0.877 | 0.012 |
| eye_inner_left | surface_rgb_fill_0.5 | 0.886 | 0.086 |
| eye_inner_right | face_median | 0.878 | 0.011 |
| eye_inner_right | mean_gray_0.5 | 0.873 | 0.016 |
| eye_inner_right | surface_rgb_fill_0.5 | 0.919 | 0.053 |
| forehead_left | face_median | 0.886 | 0.003 |
| forehead_left | mean_gray_0.5 | 0.887 | 0.002 |
| forehead_left | surface_rgb_fill_0.5 | 0.912 | 0.060 |
| forehead_right | face_median | 0.888 | 0.001 |
| forehead_right | mean_gray_0.5 | 0.891 | -0.002 |
| forehead_right | surface_rgb_fill_0.5 | 0.935 | 0.037 |
| nose_bridge | face_median | 0.891 | -0.002 |
| nose_bridge | mean_gray_0.5 | 0.885 | 0.004 |
| nose_bridge | surface_rgb_fill_0.5 | 0.959 | 0.013 |
| nose_tip | face_median | 0.888 | 0.001 |
| nose_tip | mean_gray_0.5 | 0.886 | 0.003 |
| nose_tip | surface_rgb_fill_0.5 | 0.897 | 0.075 |

- Entropy-vs-utility rank agreement over 8 regions for `binary_cond_otsu`: Spearman ρ = 0.143 (eight dependent anatomical regions give very few ranking points; descriptive only).
- Entropy-vs-utility rank agreement over 8 regions for `gaussian_curvature`: Spearman ρ = -0.357 (eight dependent anatomical regions give very few ranking points; descriptive only).

Deepfake task: not applicable: FEI has no manipulated images / original-manipulated links.

## 2. How it was measured

- Surface: existing Rust pipeline `forge_detect._core.solve_and_extract` via `CpuBackend` with unchanged defaults ({'albedo_window_radius': 2, 'c_z': 1.0, 'epsilon': 0.001, 'k_ratio': 1.4142135623730951, 'n_scales': 4, 'neigh_radius': 6, 'p_edge': 0.7, 'p_flat': 0.3, 'pde_alpha': 0.5, 'pde_beta': 5.0, 'pde_lambda': 1.0, 'pde_log_every': 10, 'pde_max_iter': 500, 'pde_tol': 1e-05, 'rayon_threads': 4, 'sigma_base': 1.0, 'sigma_ref': 4.0, 'tensor_window_radius': 2, 'trust_mode': 'heuristic', 'w_b': 0.114, 'w_g': 0.587, 'w_r': 0.299}); backend recorded per image in the cache metadata (CPU Rust; never MPS).
- Random variable: `X_{F,A} = Q(F(U))`, `U` uniform over valid pixels of a 32×32 patch; plug-in entropy in bits; 900 fixed seeded samples per patch (800 block anchors for binary blocks).
- Quantizer: calibration-frozen asinh/log1p transform, 16 equal-width bins between calibration quantiles 0.005/0.995 (`calibration.json`); 8/32 bins as sensitivity.
- Geometry: `z̃ = (z* − median)/s_z`, Gaussian σ=1 px analysis smoothing, central differences with h = 1/255, `K_G`, `H_M`, normals as in protocol §4.3.
- Patches: protocol §5.4 rules on iBUG-68 landmarks (face-alignment 2DFAN-4 + S3FD), similarity alignment to (84,100)/(172,100)/(128,184).
- Inference: subjects are the unit; 10,000 subject bootstrap resamples; 10,000 seeded sign-flips; Holm over the two confirmatory endpoints; sign test as a different estimand.
- Cohort: requested 500 subjects, available 200, primary images 200 (calibration 40, test 160), secondary images 1000. Notes: ['requested 500 subjects but only 200 available; using all']
- Exclusions: 21 rows (`exclusions.csv`); reasons: {'no_face_detected': 20, 'cheek_right_support_0.86': 1}
- Power (planning only, pilot SD from calibration subjects): binary_conditional: pilot SD 0.042 (n=40), n for d=0.3 at 80% power ≈ 106; gaussian_curvature: pilot SD 0.228 (n=40), n for d=0.3 at 80% power ≈ 106
- Resource use: timings {'align_primary': 1.955256207846105, 'align_secondary': 196.5418488746509, 'calibration': 6.629826999735087, 'exp1_exp2': 177.11669037491083, 'exp3': 762.8830912499689, 'exp4': 570.9866614579223, 'exp5': 25.137316250242293, 'exp6': 1159.0763397910632, 'manifest': 1.1245878329500556, 'primary_metrics': 173.11875987472013, 'qa': 199.75252366717905}; output directory size 1686 MB; threads RAYON_NUM_THREADS=4.
- Source: forge repo commit d2881b09d3388cb46c93790f0f6bdeb969f769bb (dirty=True); source digest 871de52f768b2ab6.

## 3. Experiment 0 — control details

| check | category | passed | blocks |
|---|---|---|---|
| rust.import | integrity | ✅ |  |
| rust.shapes | integrity | ✅ |  |
| rust.finite | integrity | ✅ |  |
| rust.repeat_determinism_48px | integrity | ✅ |  |
| rust.repeat_determinism_256px | integrity | ✅ |  |
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

Non-local settlement influence (synthetic): cheek_left->eye_inner_left: max |Δz*| in the untouched patch = 0.00244; eye_inner_left->cheek_left: max |Δz*| in the untouched patch = 0.00318.
ROI-translation control (features drawn at offset from the ROI layout): offset (-40, 0): D_K = 0.042, D_bin = -0.084; offset (0, -48): D_K = 0.700, D_bin = -0.038; offset (0, 0): D_K = -0.612, D_bin = -0.007; offset (0, 48): D_K = 0.471, D_bin = -0.011.

## 4. Stages not run / limitations

- prepare: done
- exp0: done (0 integrity failures, 3 audit findings; controls re-validated after the determinism-check update)
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
- `figures/example_q000_fei020-11.png`
- `figures/example_q025_fei193-11.png`
- `figures/example_q050_fei057-11.png`
- `figures/example_q075_fei139-11.png`
- `figures/example_q100_fei013-11.png`
- `figures/paired_differences.png`
- `figures/policies.png`
- `figures/projected_fei013-11.png`
- `figures/projected_fei020-11.png`
- `figures/projected_fei057-11.png`
- `figures/projected_fei139-11.png`
- `figures/projected_fei193-11.png`
- `figures/qa_overlays.png`
- `figures/robustness.png`
- `figures/verification.png`

## 6. Decision rules

See `protocol.md` §18 of the implementation prompt: a positive binary endpoint stable to thresholds supports a central advantage for the binary estimator; a positive surface endpoint alone supports a central advantage in this constructed surface's quantised curvature distribution; luminance/Forge agreement means a simpler explanation suffices; trust-dependent contrast implicates the trust prior; weak verification utility means entropy did not identify useful identity information for the tested decoder; non-positive or uncertain contrasts leave the hypothesis unsupported for this protocol.
