# Protocol 1.0 (2026-10-08) — regional facial entropy on the Hyperplane-Forge surface

This file fixes every rule the code implements. `protocol.lock.json` written by
`forge-entropy prepare` hashes the concrete cohort, splits, ROIs, quantizers and
solver settings; any change after calibration is a new protocol version.

## Hypotheses and estimands
For subject `s`, metric `m`: `C` = mean entropy over the four central patches, `P` =
mean over the four peripheral patches, `D_s = C − P` (averaged within subject when
several images exist). Primary estimand `Δ_m = mean_s D_s`. Two confirmatory
endpoints, Holm-adjusted together:
1. `binary_cond_otsu`: `H(B11 | B00,B01,B10)` of the whole-face-Otsu binarised luminance.
2. `gaussian_curvature`: 16-bin entropy of signed Gaussian curvature on the baseline settled surface, projected-area sampling.
Direction of interest `Δ > 0`; two-sided inference. All other metrics form an exploratory family.

## Random variable
`X_{F,A} = Q(F(U))`, `U` uniform over valid pixels of region `A`; plug-in entropy in bits.
Within-image spatial feature-distribution entropy; NOT across-person entropy (that is the
separate "between-subject" map), not differential entropy, not identity information.

## Data and grouping
FEI Face Database (200 subjects × 14 poses). Subject = independent unit. Primary image = pose 11
(frontal neutral), chosen by rule, never by entropy. Secondary strata: poses 12/13/14 (expression,
illumination) and 05/06 (small rotation) feed Experiment 6 only. Deterministic seeded subject
selection (`seed 20261008`), 20 % calibration / 80 % test split by subject before any image is
processed. The test split is never used to choose regions, bins, smoothing, trust or solver settings.
Exact duplicates removed by SHA-256; an 8×8 hash near-duplicate screen is reported.

## Alignment and support
68-point iBUG landmarks (face-alignment 2DFAN-4 + S3FD; checkpoints are opt-in assets). Similarity
(rotation + uniform scale + translation, no reflection) fitted by least squares from mean eye
centres (36–41, 42–47) and mouth centre (48–67) to (84,100), (172,100), (128,184) in a 256×256
frame; fit residual saved. Bilinear resampling in pixel-index coordinates. Visible-face mask =
polygon jaw 0..16 + brows 26..17 displaced by −0.8·d·v (d = inter-eye distance, v = unit vector eye
midpoint → mouth), intersected with the in-bounds mask; labelled an approximation. Analysis support
= mask eroded by `ceil(3σ)+2` = 5 px (σ = 1) and the same image border.

## Patches (§5.4 of the prompt)
32×32, anchor = `floor(c + 0.5) − 16` in each axis. Centres: `l39`, `l42`, `(l27+l28)/2`, `l30`
(central); `(l2+l3+l31+l48)/4`, `(l13+l14+l35+l54)/4`, `l19 − 0.45 d v`, `l24 − 0.45 d v`
(peripheral). Reject the image if any patch has < 90 % valid support or a central patch overlaps a
peripheral one; overlaps within a class are recorded. 900 fixed seeded valid positions per patch
(seed = SHA-256 of seed:image_id:patch), 800 seeded 2×2 block anchors; all-valid and
surface-area-weighted estimates are sensitivity analyses.

## Surface and geometry
Baseline = existing Rust pipeline with unchanged defaults and heuristic trust. `z̃ = (z* −
median_face z*)/s_z`, `s_z` = median over calibration images of `(Q75−Q25)/1.349` (floor 1e-6).
Gaussian σ = 1 px analysis smoothing (0.5, 2 as sensitivity), central differences with
`h = 1/255`, `K_G`, `H_M`, normals, gradient magnitude, analysis Laplacian; raw `R`, `W`.

## Quantization
Signed descriptors: `asinh(f/s_F)`, magnitudes `log1p(f/s_F)`, `s_F` = median |f| on
subject-balanced calibration samples (900 central-union + 900 peripheral-union positions per
calibration image). Limits = calibration quantiles 0.005/0.995, 16 equal-width bins (8, 32
sensitivity), out-of-range to end bins with clipping fractions logged; degenerate limits ⇒ one state,
flagged. Joint 27-state `(K_G, H_M, R)` terciles; 16/32-centroid k-means codebooks (exploratory).

## Local maps
Odd window side 21 (11, 31 sensitivity), stride 4, ≥ 90 % valid support else NaN; stored at native
grid positions; interpolation for display only. Overlay on original image through the inverse
similarity. No summation of overlapping windows into "total bits".

## Experiments
0 controls (`controls.json`), 1 binary, 2 surface, 3 ablations (common and within-variant
calibration), 4 robustness (prespecified subset: 10 quick / 50 main), 5 sampling policies
(16×16 tiles, q = 0.1/0.2/0.4, 20 random seeds), 6 face verification (identity-disjoint 60/20/20,
fixed pair manifest, logistic verifiers on |descriptor differences|, region-only / policy-only /
occlusion, subject bootstrap). Deepfake task: not applicable to FEI.

## Inference
10 000 subject bootstrap resamples; 10 000 seeded sign flips (exact enumeration when 2^n ≤ 10 000);
two-sided; Holm over the two primary endpoints; sign test as a separate estimand; Miller–Madow as a
sensitivity estimate (not clamped). Power/CI-width scenarios from calibration-split SD are planning
guidance only.

## Resources
Seed 20261008; float32 Rust outputs, float64 analysis; one surface cache entry per (image SHA,
alignment version, variant, solver params, source digest, RGB override digest); maps cached
separately; Rayon threads 4; one image worker.
