# MUSA+ Research Plan: Small-OAR- and Difficulty-Aware Anatomical Registration

## Current Context

This project has reproduced DIR-MUSA on SegRap2023-style head-and-neck CT data.

Completed pipeline:

- Convert SegRap per-case folders into prepared `.npy` training data.
- Train `M01 + MUSA loss3` two-stage model.
- Train `M05 + MUSA loss3` two-stage model.
- Run prepared-pair inference.
- Run validation-pair evaluation.
- Run M01 vs M05 comparison.
- Generate pair-level visual diagnostics.

Current key scripts:

```text
scripts/preprocess/prepare_segrap_case.py
scripts/preprocess/prepare_segrap_batch.py
scripts/preprocess/validate_prepared_data.py
scripts/infer/infer_prepared_pair.py
scripts/infer/eval_prepared_pairs.py
scripts/infer/visualize_prepared_pair.py
scripts/infer/compare_eval_summaries.py
```

Observed results:

- M05 outperforms M01 on the current validation pairs.
- M05 wins all 5 validation pairs for `seg_o_after`, `seg_o_delta`, `seg_b_after`, and `seg_b_delta`.
- M05 improves both OAR and bone Dice.
- Large anatomical structures and bone are generally aligned well.
- Remaining problems are mainly small-OAR residual mismatch, hard-pair local distortion, and flow plausibility.

Important positioning:

- Diffeomorphic deformation should **not** be the main innovation.
- It may be used only as a safety regularizer, optional ablation, or flow plausibility analysis.
- The main research direction should focus on head-and-neck-specific anatomical challenges.

## Proposed Paper Direction

Possible title:

```text
Small-Organ- and Difficulty-Aware Anatomically Conditioned Registration for Head-and-Neck CT
```

Alternative title:

```text
Difficulty-Aware Anatomical Local Refinement for Head-and-Neck CT Registration
```

Core claim:

> Existing MUSA-style two-stage registration can align large structures and bone reasonably well, but small organs-at-risk and hard registration pairs remain challenging. We propose a small-OAR-aware, pair-difficulty-aware, anatomy-conditioned local refinement framework for head-and-neck CT deformable registration.

Base model:

```text
M05 + MUSA two-stage registration
```

Proposed model:

```text
Stage 1: MUSA coarse bone-aware registration
Stage 2: full-resolution residual registration
Stage 3: difficulty-aware anatomy-conditioned small-OAR local refinement
```

## Innovation 1: Small-OAR-Aware Local Refinement

### Motivation

M05+MUSA already improves global and bone alignment, but small OARs can remain mismatched after Stage 2.

Typical small OARs:

```text
OpticNerve_L/R
Cochlea_L/R
Lens_L/R
Pituitary
Chiasm
IAC_L/R
MiddleEar_L/R
TympanicCavity_L/R
VestibulSemi_L/R
```

### Module

Add a lightweight Stage-3 local residual deformation network.

Inputs:

```text
fixed CT
stage2 warped moving CT
fixed small-OAR mask or distance map
warped moving small-OAR mask or distance map
stage2 DVF magnitude
optional fixed/moving bone mask or distance map
```

Output:

```text
local residual DVF
```

Final deformation:

```text
dvf_final = dvf_stage2 + roi_gate * dvf_local
```

Where:

```text
roi_gate = dilated small-OAR ROI with smooth boundary
```

### Constraints

- Local residual should act only near small-OAR regions.
- Residual DVF outside the ROI should be near zero.
- ROI boundary should use smooth blending.
- Local residual magnitude should be controlled.
- Refinement should not degrade large-OAR or bone alignment.

### Loss Terms

```text
L_small_dice
L_surface or L_boundary
L_local_image
L_residual_smooth
L_residual_magnitude
L_large_structure_preservation
```

Contribution statement:

> We introduce a small-OAR-aware local refinement module that selectively corrects residual misalignment around small organs without perturbing already-aligned large anatomical structures.

## Innovation 2: Pair-Difficulty-Aware Adaptation

### Motivation

Different moving/fixed pairs have very different initial mismatch. A fixed regularization schedule and fixed refinement strength are suboptimal.

Easy pairs:

- Should receive conservative refinement.
- Avoid over-warping.

Hard pairs:

- Need larger local correction.
- Need stronger safety regularization to avoid local tearing.

### Difficulty Score

Estimate pair difficulty:

```text
d_pair in [0, 1]
```

Candidate features:

```text
initial CT MSE / NCC
initial bone Dice
initial seg_o Dice
bone centroid distance
body / neck posture difference
stage2 residual image difference
stage2 residual small-OAR Dice
stage2 flow magnitude p95
```

Initial low-risk version:

```text
d_pair = normalized(
  w1 * (1 - initial_bone_dice)
+ w2 * (1 - initial_seg_o_dice)
+ w3 * image_difference
)
```

Later version:

```text
difficulty_net(features) -> d_pair
```

### How Difficulty Controls the Model

Use `d_pair` to modulate:

```text
ROI dilation radius
small-OAR loss weight
local residual scale
regularization strength
refinement strength
optional failure warning / uncertainty score
```

Examples:

```text
roi_radius = r_min + d_pair * (r_max - r_min)
lambda_small = lambda_min + d_pair * (lambda_max - lambda_min)
residual_scale = scale_min + d_pair * (scale_max - scale_min)
lambda_smooth = lambda_base + d_pair * lambda_extra
```

Contribution statement:

> We propose a pair-difficulty-aware adaptation strategy that modulates local refinement strength and anatomical regularization according to estimated registration difficulty.

## Innovation 3: Anatomy-Conditioned Regularization Map

### Motivation

Different anatomical regions should have different deformation behavior. Global smoothness or fixed hand-designed tissue weights are too coarse for local small-OAR refinement.

### Regularization Maps

Construct anatomy-conditioned maps:

```text
R_smooth(x)
R_mag(x)
R_refine(x)
```

Inputs:

```text
bone mask
small-OAR ROI
soft tissue mask
distance-to-bone
distance-to-small-OAR
difficulty score
```

General form:

```text
R(x) = f_anat(
  bone mask,
  small-OAR ROI,
  soft tissue mask,
  distance maps,
  d_pair
)
```

Regularization:

```text
L_reg = sum_x R_smooth(x) * ||grad dvf_local(x)||^2
      + sum_x R_mag(x) * ||dvf_local(x)||^2
```

### Region Behavior

```text
bone region:
  high rigidity / strong residual suppression

small-OAR boundary:
  allow fine local correction, but enforce smooth boundary

soft tissue:
  moderate smoothness

background:
  ignore or suppress deformation
```

### Implementation Levels

First version:

```text
rule-based anatomy-conditioned maps
```

Example:

```text
R_smooth(x) = high near bone
R_mag(x) = high outside small-OAR ROI
R_refine(x) = high inside small-OAR ROI
```

Later version:

```text
anatomy encoder -> R_smooth, R_mag, R_refine
```

Contribution statement:

> We design anatomy-conditioned spatial regularization maps to constrain local residual deformation according to bone, soft tissue, small-OAR proximity, and pair difficulty.

## Full Proposed Pipeline

Input:

```text
moving CT
fixed CT
moving/fixed OAR masks
moving/fixed bone masks
```

Step 1:

```text
Use trained M05 + MUSA Stage 1/2 to obtain:
  warped moving CT
  warped moving seg_o
  warped moving seg_b
  dvf_stage2
```

Step 2:

```text
Compute pair difficulty:
  d_pair = difficulty_estimator(moving, fixed, bone, OAR, dvf_stage2)
```

Step 3:

```text
Build small-OAR ROI:
  small_oar_mask = selected small-OAR labels
  roi_gate = dilate(small_oar_mask, radius=d_pair-dependent)
  smooth roi_gate boundary
```

Step 4:

```text
Build anatomy-conditioned regularization maps:
  R_smooth, R_mag, R_refine = anatomy_map(
    bone mask,
    small-OAR ROI,
    distance maps,
    d_pair
  )
```

Step 5:

```text
Stage-3 refinement network predicts:
  dvf_local
```

Step 6:

```text
Compose final deformation:
  dvf_final = dvf_stage2 + roi_gate * dvf_local
```

Step 7:

```text
Compute final warped image and segmentations
```

## Training Objective

Overall loss:

```text
L = L_img_local
  + lambda_small(d_pair) * L_small_oar_dice
  + lambda_surface(d_pair) * L_surface
  + lambda_smooth * sum_x R_smooth(x) ||grad dvf_local(x)||^2
  + lambda_mag * sum_x R_mag(x) ||dvf_local(x)||^2
  + lambda_preserve * L_large_structure_preservation
```

Possible terms:

```text
L_img_local:
  image similarity inside small-OAR ROI

L_small_oar_dice:
  weighted Dice for selected small OAR labels

L_surface:
  surface / boundary distance loss for small OARs

L_large_structure_preservation:
  prevents refinement from degrading large-OAR and bone alignment

L_residual_magnitude:
  discourages unnecessarily large local residual flow

optional L_jacobian:
  safety penalty only, not main innovation
```

## Experimental Design

### Baselines

```text
M01 + MUSA
M05 + MUSA
M05 + MUSA + Stage3 small-OAR refinement
M05 + MUSA + Stage3 + difficulty-aware adaptation
M05 + MUSA + Stage3 + difficulty-aware + anatomy-conditioned regularization
```

### Ablations

```text
A1: without small-OAR refinement
A2: without difficulty-aware adaptation
A3: without anatomy-conditioned regularization
A4: fixed ROI radius vs difficulty-adaptive ROI
A5: global residual refinement vs local ROI residual refinement
A6: rule-based anatomy map vs learnable anatomy map
```

### Metrics

```text
Overall OAR Dice
Small-OAR Dice
Large-OAR Dice
Bone Dice
HD95 / ASSD for small OAR
Jacobian <= 0 ratio
DVF magnitude p95 / max
Performance by difficulty group
```

### Difficulty Group Analysis

Split pairs into:

```text
easy
medium
hard
```

Based on:

```text
initial bone Dice
initial OAR Dice
initial image similarity
stage2 residual mismatch
```

Expected result:

```text
hard pairs:
  largest improvement

easy pairs:
  no over-warping and no degradation

small OARs:
  clear improvement in Dice and surface distance

bone / large OAR:
  preserved or minimally changed

flow plausibility:
  controlled Jacobian and DVF magnitude
```

## Implementation Plan

### Phase 1: Low-Risk Prototype

Do not retrain the entire M05 model.

Use:

```text
frozen trained M05 + MUSA Stage 1/2
```

Train:

```text
lightweight Stage-3 ROI residual U-Net
```

Use:

```text
rule-based difficulty score
rule-based anatomy-conditioned maps
```

Goal:

```text
verify whether local small-OAR refinement improves small-OAR Dice / HD95
without damaging bone or large-OAR alignment
```

### Phase 2: Stronger Version

Add:

```text
learnable difficulty predictor
learnable anatomy map encoder
```

Compare:

```text
rule-based vs learnable maps
fixed vs difficulty-adaptive refinement
```

### Phase 3: Paper-Ready Experiments

Run:

```text
full validation set
hard-pair subgroup analysis
small-OAR-specific analysis
flow/Jacobian diagnostics
qualitative visualizations
```

## What Not To Do for Current Paper

Do not use these as main innovations:

```text
diffeomorphic deformation
generic backbone replacement
structure-wise mixture-of-experts flow
```

Reasons:

- Diffeomorphic deformation is common in registration papers.
- M05 replacing M01 is useful but not enough as a new paper contribution.
- Mixture-of-experts flow is more complex than needed and may distract from the current head-and-neck small-OAR story.

## One-Sentence Summary

The proposed paper should build on the reproduced M05+MUSA two-stage model and introduce a Stage-3 small-OAR local residual refinement framework whose strength and regularization are adapted by pair difficulty and anatomy-conditioned spatial maps, improving small-organ registration in hard head-and-neck CT pairs while preserving large-structure alignment and flow plausibility.
