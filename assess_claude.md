# MS-PIGNO Codebase Assessment: Publication-Grade Scientific Inspection

**Reviewer Role:** Senior Research Scientist (SciML) & Principal HPC Systems Engineer (CFD Surrogates)  
**Date:** 2026-10-09  
**Codebase:** `MS-PIGNO` (Multi-Scale Physics-Informed Graph Neural Operator with Slicing Attention)  
**Target Benchmark:** AirfRANS (~180,000 nodes/mesh, 1,000 RANS simulations, 4 benchmark tasks)

---

## Section 1: Scientific & Architectural Critique

### 1.1 Model Mechanics & Mathematical Soundness

#### 1.1.1 MS-PIGNO Architecture Overview

The architecture implements a U-Net-style multi-scale graph hierarchy ($L_0 \to L_1 \to L_2$, with $\sim$180k $\to$ 16,384 $\to$ 1,024 nodes) coupled with a Physics-Aware Slicing Transformer (Transolver engine) operating on the coarsest level. The design is architecturally sound in principle and follows established patterns from MeshGraphNet and Transolver (Wu et al., ICML 2024). However, several fundamental issues warrant scrutiny.

**Strengths:**
- The multi-scale encoder-decoder with skip connections is well-motivated for capturing both local boundary layer gradients and global pressure field variations.
- FiLM conditioning on $(Re, \alpha, U_\infty)$ is an appropriate mechanism for parameter-dependent operator learning, with correct identity initialization ($\gamma \to 1, \beta \to 0$).
- Hard physical constraints (no-slip wall masking via exponential attenuation, $\nu_t \ge 0$ via `softplus`) are correctly implemented and differentiable.

**Critique — Slicing Attention Normalization Inconsistency:**

In `slicing_transformer.py:74`, the slice logits are computed as:
$$\text{logits} = \frac{W_s \cdot h}{\tau \cdot \sqrt{D}}$$

where $\tau = 1.0$ is the temperature and $D = 256$ is the hidden dimension. The softmax is then taken over the $M=64$ slice dimension. This double-scaling (both temperature $\tau$ and $\sqrt{D}$) is non-standard. In the original Transolver formulation, the $\sqrt{D}$ scaling is applied to the attention logits *within* the slice space, not to the slicing projection itself. The current implementation effectively over-dampens the slice assignment logits by a factor of $\sqrt{256} = 16$, pushing the softmax distribution toward uniformity. This means the model may struggle to learn sharply localized physical slices (e.g., wake vs. boundary layer vs. freestream), instead defaulting to diffuse, non-discriminative slice assignments.

**Recommendation:** Remove the `D ** 0.5` scaling from line 74 and rely solely on the learned temperature or a fixed $\tau$.

**Critique — Slicing Attention is Not Normalized Correctly (De-Slicing Step):**

The de-slicing operation (line 90) computes $H_\text{out} = S \cdot Z$, where $S \in \mathbb{R}^{N_c \times M}$ are the softmax slice weights and $Z \in \mathbb{R}^{M \times D}$ is the attended slice representation. However, $Z$ was constructed via $Z = S^\top H$ (line 78), which means each slice $Z_m = \sum_i S_{im} \cdot H_i$ is a *weighted sum* of all node tokens, with weights summing to approximately $N_c / M$ (since $\sum_i S_{im} \approx N_c / M$ for uniform assignment). This means $\|Z_m\|$ scales with $N_c$, and the de-sliced output $H_\text{out} = S \cdot Z$ will have magnitude scaling as $O(N_c)$. The residual connection $h + h_\text{out}$ then creates a scale mismatch that worsens for larger graphs.

**Recommendation:** Normalize the slice aggregation: $Z_m = \frac{\sum_i S_{im} H_i}{\sum_i S_{im} + \epsilon}$, making each slice a proper weighted average rather than a weighted sum.

#### 1.1.2 Anisotropic 7-Channel Edge Features

The edge feature vector (`preprocessor.py:47-58`) is constructed as:
$$\mathbf{e}_{ij} = [\Delta x, \Delta y, \|\Delta \mathbf{x}\|, \hat{\Delta x}, \hat{\Delta y}, \bar{\phi}, |\Delta\phi| / \|\Delta \mathbf{x}\|]$$

**Issue — Only 7 Channels Constructed, `norm_dot` Computed but Dropped:**

At `preprocessor.py:56-58`, the normal dot product $\mathbf{n}_i \cdot \mathbf{n}_j$ is computed but *never included* in the concatenation on line 58. The cat yields 7 channels: `[delta_pos(2), dist(1), dir_unit(2), avg_phi(1), d_phi_dr(1)]` = 7. The `norm_dot` is dead code. This is a significant missed opportunity: the normal dot product encodes local surface curvature and convexity information that would be especially valuable for distinguishing leading edge, trailing edge, and flat mid-chord regions.

**Recommendation:** Include `norm_dot` as the 8th edge channel and update `edge_in_channels` to 8, or replace one of the redundant features (e.g., `dir_unit` is derivable from `delta_pos / dist`).

#### 1.1.3 LS-FV Sparse Gradient Operator — Numerical Diffusion & Metric Singularities

The LS-FV gradient reconstruction (`preprocessor.py:65-126`) solves a local 2$\times$2 least-squares system per node:

$$A_i = \sum_j w_{ij} \, \Delta\mathbf{r}_{ij} \, \Delta\mathbf{r}_{ij}^\top + \lambda_\text{reg} \, I_{2\times 2}$$

where $w_{ij} = 1 / \|\Delta\mathbf{r}_{ij}\|$ is an inverse-distance weighting.

**Critical Issue — 1,000:1 Aspect Ratio Boundary Layer Cells:**

In AirfRANS boundary layer meshes, the first cell height near the wall is $O(10^{-5}c)$ while the streamwise spacing is $O(10^{-2}c)$, producing cell aspect ratios of $\sim 1000:1$. The 2$\times$2 normal equation matrix $A_i$ becomes extremely ill-conditioned:

$$\text{cond}(A_i) \approx \frac{\max(\Delta x_\text{stream}^2, \Delta y_\text{wall}^2)}{\min(\Delta x_\text{stream}^2, \Delta y_\text{wall}^2)} \sim 10^6$$

The regularization $\lambda_\text{reg} = 10^{-6}$ is *six orders of magnitude below* the dominant eigenvalue of $A_i$ for wall-adjacent nodes, rendering it effectively useless for conditioning purposes. The determinant computation at line 96:

```python
det = (A_xx * A_yy - A_xy * A_xy) + 1e-12
```

will suffer from catastrophic cancellation in FP32 when $A_{xx} \gg A_{yy}$ and $A_{xy}$ is of intermediate magnitude. This produces numerically garbage gradient weights for the first $\sim$5 rows of boundary layer cells, precisely where accurate wall-normal velocity gradients are critical for skin friction prediction.

**Quantitative Impact:** For a node with 6 neighbors at $\Delta y \sim 10^{-5}$ and $\Delta x \sim 10^{-2}$:
- $A_{xx} \sim 6 \times 10^{-2} / 10^{-2} \times (10^{-2})^2 / 10^{-2} \sim O(10^{-2})$
- $A_{yy} \sim 6 \times 10^{-5} / 10^{-5} \times (10^{-5})^2 / 10^{-5} \sim O(10^{-5})$
- $\text{det} \sim 10^{-7}$ in FP32 — dangerously close to machine epsilon products.

**Recommendation:**
1. Increase $\lambda_\text{reg}$ to at least $10^{-3}$ or make it adaptive: $\lambda_\text{reg} = \epsilon_\text{rel} \cdot \text{trace}(A_i)$.
2. Compute LS-FV weights in FP64 (`float64`) during preprocessing (this is offline, so performance is not critical).
3. Alternatively, use a QR factorization instead of the explicit $2\times 2$ inverse for better numerical stability.

**Numerical Diffusion Assessment:** Even with correct conditioning, the LS-FV operator with inverse-distance weighting introduces $O(h)$ numerical diffusion, where $h$ is the local mesh spacing. On boundary layer cells with $h_n \sim 10^{-5}$, this is negligible for wall-normal gradients. However, the streamwise gradient $\partial u / \partial x$ will be smoothed over the $O(10^{-2})$ streamwise spacing, which is acceptable for RANS but would be problematic for LES-scale predictions.

#### 1.1.4 Boundary Layer Refinement Head — $\log(y^+)$ Fourier Embeddings

The BL head (`bl_refinement_head.py:14-40`) computes:

$$\gamma(\phi) = \left[\sin\left(2^k \pi \frac{\log(1 + \phi/\delta)}{\ln 2}\right), \cos(\cdot)\right]_{k=0}^{K-1}$$

with $\delta_\text{BL} = 0.05$ and $K = 16$ Fourier modes.

**Issue — Mathematical Insufficiency for Viscous Sublayer Resolution:**

The viscous sublayer extends to $y^+ < 5$, which for $Re_c \sim 3 \times 10^6$ corresponds to:

$$y_\text{sublayer} = \frac{5 \nu}{u_\tau} \approx \frac{5 \times 1.5 \times 10^{-5}}{1.5} \approx 5 \times 10^{-5} \, \text{m}$$

The logarithmic coordinate $\xi = \log(1 + \phi/\delta) / \ln 2$ maps this to $\xi \approx \log_2(1 + 10^{-3}) \approx 1.4 \times 10^{-3}$. The highest Fourier frequency is $2^{15} \pi \approx 10^5$, so the argument is $2^{15} \pi \times 1.4 \times 10^{-3} \approx 144$ radians, which gives $\sim 23$ full oscillations within the sublayer. This is *sufficient* spectral resolution.

**However**, the BL head applies a smooth decay weight:

$$\alpha(\phi) = \left(1 - \frac{\phi}{\delta_\text{BL}}\right)^2$$

This quadratic decay means the refinement correction is attenuated by a factor of $(1 - 10^{-3})^2 \approx 0.998$ at the sublayer edge — nearly full strength, which is correct. The design is mathematically adequate for sublayer resolution.

**Real Concern — BL Head Sees Raw Decoder Features, Not Physical Coordinates:**

The BL head input concatenates `[h_bl, fourier_feat, y_tilde, normals]` where `h_bl` are the Level-0 decoder features. These features have already been processed through the multi-scale hierarchy and may have lost fine-scale wall-normal structure due to the pooling/unpooling operations (scatter-mean at ~11:1 compression ratio from L0 to L1). The BL head is thus attempting to refine a signal that has already been spatially homogenized.

**Recommendation:** Feed the BL head the *original lifted features* `h0` (pre-encoder) concatenated with the decoded features `h0_dec`, providing a skip connection that preserves the raw wall-distance signal.

#### 1.1.5 RANS Residual Evaluator — Missing Reynolds Stress Closure

The RANS momentum residual (`rans_residuals.py:59-83`) computes:

$$\mathbf{R}_\text{mom} = (\mathbf{u} \cdot \nabla)\mathbf{u} + \frac{1}{\rho}\nabla p - \nabla \cdot \left[\nu_\text{eff}(\nabla \mathbf{u} + \nabla \mathbf{u}^\top)\right]$$

where $\nu_\text{eff} = \nu_\text{lam} + \nu_t$.

**Issue — Incomplete Boussinesq Approximation:**

The standard incompressible RANS momentum equation with the Boussinesq eddy-viscosity hypothesis is:

$$(\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p^* + \nabla \cdot \left[\nu_\text{eff}(\nabla \mathbf{u} + \nabla \mathbf{u}^\top)\right]$$

where $p^* = p + \frac{2}{3}\rho k$ is the modified pressure incorporating the turbulent kinetic energy trace. The codebase treats the predicted pressure $p$ as if it were $p^*$, which is only valid if the AirfRANS dataset stores the modified pressure (which it does for incompressible solvers like OpenFOAM's `simpleFoam`). This is a subtle but important assumption that should be documented.

**More critically**, the viscous stress divergence requires *second derivatives* of velocity, computed by applying the LS-FV operator twice:

$$\nabla \cdot (\nu_\text{eff} \nabla \mathbf{u}) \approx \text{LS-FV}\left[\nu_\text{eff} \cdot \text{LS-FV}[\mathbf{u}]\right]$$

This double application of the first-order LS-FV stencil is *not* equivalent to a proper second-order Laplacian stencil. It introduces $O(h)$ truncation error in the viscous terms, which is larger than the $O(h^2)$ truncation error of a proper compact Laplacian stencil. For RANS with thin boundary layers, this artificial viscous dissipation may mask genuine PDE residual signals during training.

#### 1.1.6 Surface Integral Computation — Skin Friction Sign Error

In `surface_integrals.py:82`:

```python
tau_w = self.mu_laminar * torch.abs(du_t_dn)
```

**Issue:** The `torch.abs()` destroys the sign of the wall shear stress. In separated flow regions (which are the primary challenge for AoA extrapolation, Task 4), the skin friction changes sign at separation and reattachment points. By taking the absolute value, the integrator will:
1. Over-predict friction drag in separated regions (where $\tau_w$ should be negative/reversed).
2. Miss the critical separation point location entirely.
3. Produce a $C_{D,\text{fric}}$ that is always positive, even in massive separation where the recirculation zone should *reduce* net friction drag.

**Furthermore**, only $\mu_\text{laminar}$ is used for wall shear stress, not $\mu_\text{eff} = \mu_\text{lam} + \mu_t$. At the wall, the turbulent viscosity $\nu_t$ should vanish (enforced by the no-slip + turbulence model boundary conditions), so this is technically correct for a well-resolved wall treatment. However, since the LS-FV gradient is evaluated at the wall *node* (not the cell center), the $\nu_t$ at that node may not be exactly zero despite the softplus constraint, introducing a small but systematic bias.

**Recommendation:** Remove `torch.abs()` and use the signed $\tau_w$. For separated flow, the friction drag contribution should be computed with the correct sign convention.

#### 1.1.7 Lift and Drag Integration — Hardcoded Freestream Direction

In `surface_integrals.py:95-104`, the lift is computed as the force component in the $-y$ direction and drag in the $+x$ direction:

```python
lift_force = (-f_p_y + f_v_y).sum()
drag_press_force = f_p_x.sum()
```

**Issue:** This assumes the freestream is aligned with the $x$-axis. For non-zero angle of attack $\alpha$, the correct decomposition requires rotation:

$$C_L = \frac{1}{q_\infty c} \oint (-p \, n_y + \tau_w \, t_y) \cos\alpha - (-p \, n_x + \tau_w \, t_x) \sin\alpha \, ds$$
$$C_D = \frac{1}{q_\infty c} \oint (-p \, n_x + \tau_w \, t_x) \cos\alpha + (-p \, n_y + \tau_w \, t_y) \sin\alpha \, ds$$

For $\alpha = 15°$ (Task 4 upper bound), this introduces a $\sim 3.5\%$ systematic error in $C_L$ and a much larger relative error in $C_D$ (since $C_D \ll C_L$, the $\sin\alpha$ cross-contamination from the lift component dominates).

**Recommendation:** Pass $\alpha$ (available in `batch.cond`) and apply the rotation matrix to the integrated forces.

### 1.2 Hidden Failure Modes & Edge-Case Vulnerabilities

#### 1.2.1 Post-Stall Flow Separation at $\alpha > 12°$

The AoA Extrapolation task (Task 4) tests on $10° < \alpha < 15°$, which for many airfoil geometries in AirfRANS approaches or exceeds the stall angle. At post-stall conditions:

- The wake becomes massively separated with a recirculation zone spanning $> 30\%$ of the chord.
- RANS solutions themselves become unreliable (steady-state RANS often fails to converge for separated flows).
- The smooth Boussinesq eddy-viscosity model cannot represent the counter-gradient transport in the separation bubble.

**The model has no mechanism to detect or flag non-physical predictions in these regimes.** The `softplus` constraint on $\nu_t$ prevents negative values but does not prevent non-physically *large* values ($\nu_t / \nu_\text{lam} > 10^4$) that RANS solvers sometimes produce in separated regions.

**Risk Level: HIGH** for Task 4. The model will likely produce smooth, plausible-looking but physically incorrect flow fields in post-stall conditions, with no built-in diagnostic to flag the failure.

#### 1.2.2 Training Collapse During Phase 2/3 Curriculum Transitions

The curriculum transitions are *hard boundaries*: at epoch 51, the PDE loss suddenly activates (with a 20-epoch linear warmup), and at epoch 151, the surface force coefficient loss activates at full weight ($\lambda_\text{coeff} = 1.0$).

**Stiffness Risk:** The PDE residual loss and data loss operate at vastly different scales. The RANS momentum residual for a well-predicted velocity field is $O(10^{-2})$ to $O(1)$ in physical units, while the normalized data loss is $O(10^{-3})$ to $O(10^{-2})$. Even with ReLoBRaLo balancing, the sudden introduction of stiff physics losses can cause:

1. **Gradient conflict:** The PDE loss gradient may point in the opposite direction from the data loss gradient, especially for the near-wall velocity field where the LS-FV operator is least accurate.
2. **Learning rate mismatch:** The Phase 2 learning rate ($5 \times 10^{-4}$) is half the Phase 1 rate, but the curriculum manager does not actually *change* the learning rate at phase transitions — it only sets the initial LR. The `CosineAnnealingWarmRestarts` scheduler with $T_0 = 50$ and $T_\text{mult} = 2$ will be at a different point in its cosine cycle when Phase 2 begins at epoch 51, creating an uncontrolled interaction between the curriculum and the LR schedule.

**Risk Level: MEDIUM.** Monitor the training loss curve for spikes at epoch 51 and 151. Consider using a smooth sigmoid ramp for all phase transitions.

#### 1.2.3 Non-Physical Turbulent Viscosity Predictions

The `softplus` activation ensures $\hat{\nu}_t > 0$, but imposes no upper bound. In the far-field ($\phi > 1.0$), the ground truth $\nu_t / \nu_\text{lam}$ approaches zero (the freestream is laminar), but the model can predict arbitrarily large values. The log-transform $\tilde{\nu}_t = \log_{10}(\nu_t / \nu_\text{lam} + 1)$ compresses the target range to $[0, 3.5]$, but the *inverse* transform at inference time can amplify small absolute errors in log-space to large errors in physical space:

$$\delta \nu_t / \nu_\text{lam} = 10^{\tilde{\nu}_t} \cdot \ln(10) \cdot \delta\tilde{\nu}_t$$

For $\tilde{\nu}_t = 3.0$ (highly turbulent wake), a 10% relative error in normalized space produces a $\sim$230% error in physical $\nu_t$.

#### 1.2.4 Coarsening Graph Quality — SDF-Stratified Subsampling

The multi-level clustering (`preprocessor.py:128-206`) uses stride-based subsampling within SDF-stratified strata, not proper graph coarsening algorithms (e.g., METIS, Graclus, or voxel pooling). This means:

1. **No connectivity guarantees:** The L1 and L2 graphs are built via $r$-ball or KNN queries on the subsampled point cloud, not inherited from the original mesh topology. This can create artificial edges that cross solid boundaries or skip important topological features (trailing edge, leading edge stagnation point).
2. **Strided subsampling is spatially biased:** The `idx[::step]` pattern selects every $k$-th node in the sorted index order, which is not spatially uniform — it depends on the mesh numbering convention (which is often front-to-back for structured regions, random for unstructured).

**Recommendation:** Use `torch_geometric.nn.pool.voxel_grid` or METIS-based coarsening for Level 1, which preserves connectivity and provides area-weighted representatives.

---

## Section 2: Comprehensive GPU Hardware, Cost & Time Breakdown

### 2.1 Model Parameter & Memory Budget Analysis

**Trainable Parameter Count (Default Configuration):**

| Component | Parameters |
|:---|---:|
| Node Lifting MLP | $(7 \times 64 + 64) + (64 \times 64 + 64) = 4,608 + 4,160 = 8,768$ |
| AnisotropicMeshConvBlock $\times 6$ (Enc+Dec, 3 levels) | $\sim 6 \times [2 \cdot (2C+7) \cdot H + H \cdot C + (2C+7) \cdot H/2 + H/2 + 2C \cdot H + H \cdot C] \approx 1.8\text{M}$ |
| FiLM Generators $\times 4$ | $\sim 4 \times (3 \times 128 + 128 \times 128 + 128 \times D_\text{target} \times 2) \approx 0.5\text{M}$ |
| SlicingAttentionLayer $\times 6$ | $\sim 6 \times (256 \times 64 + 3 \times 256^2 + 4 \times 256^2 + 256 \times 1024 + 1024 \times 256) \approx 5.4\text{M}$ |
| BL Refinement Head | $\sim (64+32+1+2) \times 128 + 128 \times 128 + 128 \times 4 \approx 30\text{K}$ |
| Output Head | $(64+5) \times 64 + 64 \times 4 \approx 4.7\text{K}$ |
| Level Projections (6 linear layers) | $\sim 150\text{K}$ |
| **Total Estimated** | **$\sim 8.0\text{M}$** |

This is a moderately sized model — well within the capacity of all target GPUs.

### 2.2 Per-Sample VRAM Footprint

For a single AirfRANS graph with $N \approx 180,889$ nodes, $E \approx 1.2\text{M}$ edges (typical for body-fitted Delaunay mesh), $N_1 = 16,384$ and $N_2 = 1,024$:

| Tensor | Shape | BF16 Size | FP32 Size |
|:---|:---|---:|---:|
| Node features $x$ | $[N, 5]$ | 1.7 MB | 3.4 MB |
| Positions $\text{pos}$ | $[N, 2]$ | 0.7 MB | 1.4 MB |
| Edge index | $[2, E]$ | 9.6 MB (int64) | 9.6 MB |
| Edge attributes | $[E, 7]$ | 16.8 MB | 33.6 MB |
| LS-FV sparse data | $[2, \text{nnz}] + 2 \times [\text{nnz}]$ | $\sim$40 MB | $\sim$40 MB |
| Cluster indices | $[N] + [N_1]$ | 1.4 MB (int64) | 1.4 MB |
| L0 activations $h_0$ | $[N, 64]$ | 22.1 MB | 44.2 MB |
| L0 encoder output | $[N, 64]$ | 22.1 MB | 44.2 MB |
| L1 activations | $[N_1, 128]$ | 4.0 MB | 8.0 MB |
| L2 activations | $[N_2, 256]$ | 0.5 MB | 1.0 MB |
| Transformer intermediate | $[N_2, 256] \times 6$ layers | 3.0 MB | 6.0 MB |
| Decoder activations (mirrored) | $\sim$ Encoder | $\sim$49 MB | $\sim$98 MB |
| **Forward Pass Total** | | **$\sim$170 MB** | **$\sim$290 MB** |
| **Backward Pass Overhead** ($\sim 2\times$ activations) | | **$\sim$340 MB** | **$\sim$580 MB** |
| Optimizer States (AdamW: 2 moments) | 8M params $\times 2$ | — | 64 MB |
| Model Weights | 8M params | — | 32 MB |
| **Total Training VRAM** | | **$\sim$2.0 GB** | **$\sim$3.5 GB** |

**Note:** The dominant memory consumers are the Level-0 fine-mesh activations ($N \times d_0$) and their gradients. The LS-FV sparse data adds a fixed ~40 MB overhead. The Slicing Transformer on Level 2 ($N_2 = 1,024$) is negligible in comparison.

**Critical caveat:** The above estimates assume `batch_size=1`. The LS-FV PDE loss in Phases 2-3 requires storing intermediate gradient tensors ($\nabla u$, $\nabla v$, $\nabla p$, $\nabla \nu_t$, stress tensor components, and their divergences), approximately doubling the backward pass memory to **$\sim$4.5 GB** total during Phase 2/3.

### 2.3 GPU Performance & Cost Comparison Matrix

**Assumptions:**
- 1 graph/step, gradient accumulation = 4 steps per optimizer update.
- 800 training samples (Full task), 250 epochs, AMP BF16 enabled.
- Epoch = 800 forward + backward passes (batch_size=1).
- Validation every epoch: 200 inference passes (no grad).

| Metric | RTX 4090 (24 GB) | L40S (48 GB) | A100 80 GB SXM4 | H100 80 GB SXM5 |
|:---|:---|:---|:---|:---|
| **Architecture** | Ada Lovelace | Ada Lovelace | Ampere | Hopper |
| **BF16 TFLOPS** | 165 (sparse) / 82.6 (dense) | 181 (w/ sparsity) / 90.5 | 312 (dense) | 989 (sparse) / 495 (dense) |
| **Memory Bandwidth** | 1,008 GB/s | 864 GB/s | 2,039 GB/s | 3,350 GB/s |
| **Fits in VRAM?** | Yes (Phase 1-3: ~4.5 GB peak) | Yes (ample headroom) | Yes (ample headroom) | Yes (ample headroom) |
| **Est. Forward Pass (BF16)** | ~85 ms | ~90 ms | ~45 ms | ~25 ms |
| **Est. Backward Pass (BF16)** | ~170 ms | ~180 ms | ~90 ms | ~50 ms |
| **Est. Time per Graph (Fwd+Bwd)** | ~255 ms | ~270 ms | ~135 ms | ~75 ms |
| **Est. Train Epoch (800 graphs)** | ~204 s (~3.4 min) | ~216 s (~3.6 min) | ~108 s (~1.8 min) | ~60 s (1.0 min) |
| **Est. Val Epoch (200 graphs)** | ~17 s | ~18 s | ~9 s | ~5 s |
| **Total Epoch Wall-Clock** | ~221 s | ~234 s | ~117 s | ~65 s |
| **250-Epoch Training (1 task)** | **~15.4 hrs** | **~16.3 hrs** | **~8.1 hrs** | **~4.5 hrs** |
| **Full 4-Task Suite** | **~61.5 hrs** | **~65.0 hrs** | **~32.5 hrs** | **~18.0 hrs** |

**Notes on Performance Estimates:**
- The bottleneck is *not* FLOPS but **memory bandwidth** — the scatter-gather operations in message passing and LS-FV SpMM are memory-bound. The A100's 2$\times$ higher bandwidth vs. RTX 4090 yields nearly 2$\times$ speedup despite only 3.8$\times$ higher peak FLOPS.
- The H100's HBM3 at 3,350 GB/s provides the single largest performance lever for this workload.
- Gradient accumulation (4 steps) does not change per-step latency but reduces optimizer overhead by 4$\times$.

### 2.4 Cloud Compute Financial Cost Matrix

| Provider & GPU | Spot $/hr | On-Demand $/hr | 1-Task (250 ep) | 4-Task Suite | Notes |
|:---|---:|---:|---:|---:|:---|
| **Lambda Labs RTX 4090** | — | $0.69 | $10.63 | $42.50 | Limited availability |
| **RunPod RTX 4090** | $0.39 | $0.69 | $6.01–$10.63 | $24.0–$42.5 | Community cloud |
| **GCP L4 (proxy for L40S)** | $0.56 | $0.98 | $9.13–$15.97 | $36.5–$63.9 | L40S not in GCP; L4 is closest |
| **Lambda Labs A100 80 GB** | — | $1.29 | $10.45 | $41.9 | Best $/performance |
| **GCP A100 80 GB** | $1.65 | $3.67 | $13.4–$29.7 | $53.6–$119 | a2-ultragpu-1g |
| **RunPod A100 80 GB** | $0.89 | $1.64 | $7.21–$13.3 | $28.8–$53.2 | Good spot rates |
| **Lambda Labs H100 SXM** | — | $2.49 | $11.21 | $44.8 | Premium tier |
| **GCP H100 80 GB** | $3.33 | $7.19 | $15.0–$32.4 | $60.0–$129.5 | a3-highgpu-1g |
| **RunPod H100 SXM** | $1.89 | $3.29 | $8.51–$14.8 | $34.0–$59.2 | Best H100 value |

**Optimal Cost Strategy:**
- **Budget-constrained:** RunPod RTX 4090 spot at $0.39/hr: **$24 total for the full 4-task benchmark suite** (~62 hrs). Hard to beat.
- **Time-constrained:** RunPod H100 SXM spot at $1.89/hr: **$34 total for 4 tasks in ~18 hours.** Best time-to-completion per dollar.
- **Sweet spot:** Lambda Labs A100 at $1.29/hr: **$42 total for 4 tasks in ~33 hours.** Reliable availability, good price/performance.

---

## Section 3: Google TPU Builder Program & TPU Infrastructure Evaluation

### 3.1 TPU Builder Program & TRC Allocation Breakdown

The Google TPU Research Cloud (TRC) and TPU Builder Program provide free access to TPU accelerators for qualifying research projects. Typical allocations:

| TPU Configuration | Chips | HBM per Chip | Total HBM | BF16 TFLOPS/chip | Total BF16 TFLOPS | Interconnect |
|:---|---:|---:|---:|---:|---:|:---|
| **TPU v4-8** | 4 | 32 GB HBM2e | 128 GB | 275 | 1,100 | ICI 4$\times$48 GB/s |
| **TPU v5e-8** | 8 | 16 GB HBM2e | 128 GB | 197 | 1,576 | ICI 8$\times$28 GB/s |
| **TPU v5p-8** | 8 | 96 GB HBM3 | 768 GB | 459 | 3,672 | ICI 8$\times$56 GB/s |

**Key Observation:** Even a single TPU v4 chip (275 BF16 TFLOPS) exceeds the H100's peak FLOPS for dense operations. The inter-chip interconnect (ICI) enables data parallelism across the 4-8 chip pod slice. However, the *realized* throughput for this workload depends critically on XLA compilation efficiency, not raw FLOPS.

### 3.2 PyTorch/XLA vs. PyTorch Geometric Compatibility Audit

#### 3.2.1 The Fundamental Incompatibility

PyTorch Geometric (PyG) is built on **dynamic graph execution**: each forward pass constructs scatter/gather operations whose dimensions depend on the input graph's node count $N$, edge count $E$, and cluster sizes. Google's XLA (Accelerated Linear Algebra) compiler requires **statically known tensor shapes** at graph compilation time to generate optimized TPU machine code.

**The conflict is architectural, not superficial.** Every PyG operation — `MessagePassing.propagate()`, `scatter_add_()`, `torch.zeros(num_nodes, ...)` — emits XLA HLO operations with concrete dimension sizes. When $N$ changes from 158,319 to 204,063 between consecutive training samples, XLA must:

1. Invalidate the previously compiled HLO graph.
2. Re-trace the entire Python forward pass.
3. Re-compile the HLO graph to TPU instructions (typically 30-120 seconds).
4. Execute the newly compiled graph.

**For AirfRANS, where every simulation has a different mesh size**, this means XLA recompilation on *every single forward pass*. A 2-minute compilation + 0.05s execution per sample renders the TPU completely unusable — the effective throughput would be ~1000$\times$ worse than a CPU.

#### 3.2.2 Graph Padding Solution — Static XLA Execution

The required refactoring strategy is to **pad all graphs to a fixed maximum size** $N_\text{max}$, creating a static computation graph:

**Step 1: Determine $N_\text{max}$**

Scan all 1,000 AirfRANS simulations:
$$N_\text{max} = \max_{i \in [1,1000]} N_i$$

For AirfRANS, $N_\text{max} \approx 204,063$. Round up to $N_\text{max} = 204,800$ (multiple of 128 for TPU tiling efficiency).

**Step 2: Pad Node Features, Positions, Targets**

```python
def pad_graph_to_static_size(data, n_max=204800, e_max=1400000):
    """Pad a PyG Data object to fixed sizes for XLA static shapes."""
    N = data.x.shape[0]
    E = data.edge_index.shape[1]
    
    # Create valid-node mask
    node_mask = torch.zeros(n_max, dtype=torch.bool)
    node_mask[:N] = True
    
    # Pad node features [N, C] -> [N_max, C]
    x_padded = torch.zeros(n_max, data.x.shape[1])
    x_padded[:N] = data.x
    
    pos_padded = torch.zeros(n_max, 2)
    pos_padded[:N] = data.pos
    
    y_padded = torch.zeros(n_max, data.y.shape[1])
    y_padded[:N] = data.y
    
    # Pad edge index: add self-loops on dummy nodes
    edge_pad_count = e_max - E
    dummy_edges = torch.arange(N, N + edge_pad_count).clamp(max=n_max - 1)
    edge_index_padded = torch.cat([
        data.edge_index,
        torch.stack([dummy_edges, dummy_edges])
    ], dim=1)[:, :e_max]
    
    # Pad edge attributes
    edge_attr_padded = torch.zeros(e_max, data.edge_attr.shape[1])
    edge_attr_padded[:E] = data.edge_attr
    
    data.x = x_padded
    data.pos = pos_padded
    data.y = y_padded
    data.edge_index = edge_index_padded
    data.edge_attr = edge_attr_padded
    data.node_mask = node_mask
    
    return data
```

**Step 3: Mask Losses to Ignore Padded Nodes**

All loss computations must be masked:
```python
loss = (node_mask.float() * per_node_loss).sum() / node_mask.sum()
```

**VRAM Impact of Padding:** Padding from $N=158,319$ to $N_\text{max}=204,800$ increases the effective graph size by $\sim 30\%$ on average, wasting $\sim 30\%$ of compute on dummy nodes. This is acceptable for TPU throughput gains but represents a pure overhead tax.

#### 3.2.3 Additional XLA Compatibility Issues

1. **`scatter_add_()` is not XLA-friendly.** The in-place `scatter_add_` used extensively in the LS-FV operator and `_pool()` must be replaced with `torch.scatter_add()` (functional form) or XLA-compatible `index_put_` operations.

2. **Dynamic Python control flow.** The `if not bl_mask.any(): return` branch in the BL head (`bl_refinement_head.py:104`) creates a data-dependent control flow branch that triggers XLA recompilation. Must be replaced with a masked multiply: always compute the full BL head output and zero out non-BL nodes via multiplication.

3. **`torch.clamp()`, `F.softplus()`, `torch.exp()`** are all XLA-compatible — no issues.

4. **`scipy.spatial.cKDTree`** in the preprocessor is CPU-only and runs offline — no TPU impact.

### 3.3 TPU Performance, Cost & Speed Advantage

**Estimated TPU Training Performance (after static graph refactoring):**

| Metric | TPU v4-8 (4 chips) | TPU v5e-8 (8 chips) | TPU v5p-8 (8 chips) | H100 GPU (baseline) |
|:---|:---|:---|:---|:---|
| **Effective BF16 TFLOPS** | ~400 (memory-bound) | ~350 (cost-optimized) | ~900 (memory-bound) | ~200 (memory-bound) |
| **HBM Bandwidth** | 4$\times$1,200 = 4,800 GB/s | 8$\times$820 = 6,560 GB/s | 8$\times$2,765 = 22,120 GB/s | 3,350 GB/s |
| **XLA Compilation (one-time)** | ~90 s | ~60 s | ~90 s | N/A |
| **Est. Time per Graph (Fwd+Bwd)** | ~40 ms | ~50 ms | ~18 ms | ~75 ms |
| **Est. Train Epoch** | ~32 s | ~40 s | ~14 s | ~60 s |
| **250-Epoch Training** | **~2.5 hrs** | **~3.1 hrs** | **~1.1 hrs** | **~4.5 hrs** |
| **Speedup vs. H100** | **1.8$\times$** | **1.5$\times$** | **4.1$\times$** | **1.0$\times$** |
| **Full 4-Task Suite** | **~10 hrs** | **~12.5 hrs** | **~4.4 hrs** | **~18 hrs** |

**Cost Comparison:**

| Platform | Hourly Cost | 4-Task Suite Cost | Notes |
|:---|---:|---:|:---|
| TPU v4-8 (TRC free) | **$0.00** | **$0.00** | 90-day TRC allocation |
| TPU v5e-8 (TRC free) | **$0.00** | **$0.00** | If TRC allocation includes v5e |
| TPU v5p-8 (on-demand) | $8.80/hr | $38.7 | ~4.4 hrs compute |
| H100 SXM (RunPod spot) | $1.89/hr | $34.0 | ~18 hrs compute |
| A100 80 GB (Lambda) | $1.29/hr | $41.9 | ~33 hrs compute |
| RTX 4090 (RunPod spot) | $0.39/hr | $24.0 | ~62 hrs compute |

**Verdict on Cost:** If you have a TRC or TPU Builder allocation, the cost is **$0 for the full benchmark suite**, making it the obvious winner. Even the on-demand TPU v5p-8 at $8.80/hr competes well with H100 due to the $4\times$ speedup.

### 3.4 Step-by-Step TPU Migration Roadmap

#### Step 1: Install PyTorch/XLA

```bash
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

#### Step 2: Refactor `trainer.py` for XLA

```python
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

class SurrogateTrainerTPU(SurrogateTrainer):
    def __init__(self, model, train_loader, val_loader, normalizer, config):
        # Get XLA device instead of CUDA
        device = xm.xla_device()
        super().__init__(model, train_loader, val_loader, normalizer, config, device)
        
        # Disable CUDA AMP — use XLA BF16 instead
        self.amp_enabled = False
        self.scaler = None  # Not needed on TPU
        
        # Wrap data loaders with MpDeviceLoader for async prefetch
        self.train_loader_xla = pl.MpDeviceLoader(
            self.train_loader,
            device,
            loader_prefetch_size=2,
            device_prefetch_size=2
        )
        self.val_loader_xla = pl.MpDeviceLoader(
            self.val_loader,
            device,
            loader_prefetch_size=2,
            device_prefetch_size=2
        )
    
    def train_epoch(self, epoch):
        self.model.train()
        phase = self.curriculum.get_phase(epoch)
        phase_config = self.curriculum.get_phase_config(epoch)
        
        epoch_losses = {"loss": 0.0, "loss_data": 0.0}
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(self.train_loader_xla):
            # batch is already on XLA device via MpDeviceLoader
            
            # Forward pass (XLA traces and compiles on first call)
            y_pred_norm = self.model(
                x=batch.x_norm,
                pos=batch.pos,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                cond=batch.cond_norm,
                cluster_l0_l1=batch.cluster_l0_l1,
                cluster_l1_l2=batch.cluster_l1_l2,
                edge_index_l1=batch.edge_index_l1,
                edge_index_l2=batch.edge_index_l2,
                apply_physical_constraints=True
            )
            
            loss_dict = self.loss_fn.compute_loss(
                y_pred_norm=y_pred_norm,
                batch_data=batch,
                phase=phase,
                phase_config=phase_config,
                current_epoch=epoch
            )
            loss = loss_dict["loss"] / self.accumulate_batches
            loss.backward()
            
            if (i + 1) % self.accumulate_batches == 0:
                # XLA-compatible optimizer step
                xm.reduce_gradients(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_norm
                )
                xm.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
            
            # CRITICAL: Mark step boundary for XLA lazy execution
            xm.mark_step()
        
        return epoch_losses
```

#### Step 3: Replace In-Place Operations

```python
# BEFORE (not XLA-safe):
df_dx = torch.zeros(num_nodes, f.shape[-1], device=device, dtype=dtype)
df_dx.scatter_add_(0, row.unsqueeze(-1).expand_as(msg_x), msg_x)

# AFTER (XLA-safe):
df_dx = torch.zeros(num_nodes, f.shape[-1], device=device, dtype=dtype)
df_dx = df_dx.scatter_add(0, row.unsqueeze(-1).expand_as(msg_x), msg_x)
```

#### Step 4: Eliminate Dynamic Control Flow in BL Head

```python
# BEFORE:
if not bl_mask.any():
    return delta_y_full

# AFTER (always execute, mask output):
delta_y_active = self.head(feat)
decay_weight = torch.clamp(1.0 - (phi / self.delta_bl), min=0.0, max=1.0) ** 2
# This naturally zeros out nodes with phi >= delta_bl
delta_y_full = delta_y_active * decay_weight
```

#### Step 5: Graph Padding Collation

```python
from torch_geometric.loader import DataLoader

class StaticSizeCollater:
    """Pads graphs to fixed size for XLA static compilation."""
    def __init__(self, n_max=204800, e_max=1400000):
        self.n_max = n_max
        self.e_max = e_max
    
    def __call__(self, data_list):
        padded = [pad_graph_to_static_size(d, self.n_max, self.e_max) 
                  for d in data_list]
        return padded[0]  # batch_size=1

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    collate_fn=StaticSizeCollater(n_max=204800)
)
```

#### Step 6: Environment Variables for TPU Optimization

```bash
# Enable BF16 matrix multiplication on TPU
export XLA_USE_BF16=1
# Persist XLA compilation cache across runs
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/dev/null"  
export XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache
# Suppress excessive XLA logging
export TPU_LOG_LEVEL=0
```

**Migration Effort Estimate:** 3-5 days for an experienced PyTorch/XLA developer. The primary work is graph padding + elimination of dynamic shapes, not the XLA API integration itself.

---

## Section 4: Summary Verdict & Actionable Recommendations

### 4.1 Overall Architectural Rating

| Dimension | Score (0-10) | Commentary |
|:---|:---:|:---|
| **Scientific Rigor** | **7.0 / 10** | Solid RANS physics integration with LS-FV gradients and differentiable surface integrals. Deducted for sign error in $\tau_w$, missing AoA rotation in force decomposition, and incomplete Transolver slice normalization. |
| **CFD Fidelity** | **6.5 / 10** | Good near-wall treatment with log($y^+$) Fourier embeddings and hard no-slip constraints. Deducted for LS-FV numerical stability on high-aspect-ratio cells, dead `norm_dot` edge feature, and double-application Laplacian approximation in PDE residual. |
| **HPC Scalability** | **7.5 / 10** | Clean micro-batched AMP pipeline with gradient accumulation. Correct use of `scatter_add_` for multi-GPU readiness. Deducted for lack of DDP/FSDP support, no mixed-precision gradient scaling for BF16 (not needed but should be explicit), and no profiling hooks. |
| **Code Quality** | **8.0 / 10** | Well-structured, modular, Hydra-configured, comprehensive test suite including analytical gradient validation. Clear separation of concerns. |
| **Reproducibility** | **7.5 / 10** | Pinned dependencies, seeded splits, deterministic pooling. Deducted for metadata extraction hack in `splits.py` (uses filename hashing instead of actually loading flow parameters), and missing `torch.use_deterministic_algorithms(True)`. |
| **Overall** | **7.3 / 10** | A competent, above-average SciML codebase with several non-trivial physics bugs that must be fixed before publication. |

### 4.2 Hardware Recommendation

| Scenario | Recommendation | Rationale |
|:---|:---|:---|
| **Free compute available (TRC/Builder)** | **TPU v4-8 or v5e-8** | $0 cost, ~2.5x faster than A100 after padding refactor. 3-5 day migration investment pays for itself immediately. |
| **Budget < $50** | **RunPod RTX 4090 spot** | $24 for full 4-task suite. Acceptable 62-hour wall-clock. No code changes needed. |
| **Budget < $100, time-sensitive** | **RunPod H100 SXM spot** | $34 for 4 tasks in 18 hours. Best $/wall-clock ratio. No code changes needed. |
| **Production/reproducibility** | **Lambda Labs A100 80 GB** | $42 for 4 tasks. Stable on-demand pricing, reliable availability, no preemption risk. |

**Primary Recommendation:** If you have access to the TPU Builder Program, **use TPU v4-8 with the graph padding refactor.** The $0 cost and competitive performance make it the clear winner. If the 3-5 day migration effort is prohibitive, **RunPod H100 SXM spot at $1.89/hr** is the GPU-native optimum.

### 4.3 Prioritized Action Plan

**P0 — Must Fix Before Training (Physics Correctness):**

1. **Fix `tau_w` sign error** in `surface_integrals.py:82`. Remove `torch.abs()` and use signed wall shear stress. Without this fix, the $C_{D,\text{fric}}$ loss signal in Phase 3 is physically wrong for any separated flow case.

2. **Add AoA rotation to lift/drag decomposition** in `surface_integrals.py:95-104`. Pass $\alpha$ from `batch.cond` and rotate the integrated force vector. Current implementation introduces ~3.5% systematic error in $C_L$ and potentially catastrophic error in $C_D$ at high AoA.

3. **Fix Slicing Attention normalization** in `slicing_transformer.py:74`. Remove the `D ** 0.5` divisor from the slice logits. Additionally, normalize the slice aggregation $Z_m = S^\top H / (\mathbf{1}^\top S + \epsilon)$ to prevent output magnitude scaling with graph size.

4. **Include `norm_dot` in edge features** (`preprocessor.py:58`) or remove the dead computation. If included, update `edge_in_channels` to 8 in all configs.

**P1 — Should Fix Before Training Campaign (Numerical Stability):**

5. **Compute LS-FV weights in FP64** in `preprocessor.py:65-126`. The $2\times 2$ inverse is numerically unstable in FP32 for aspect ratios > 100:1. This is offline preprocessing — FP64 has zero runtime cost.

6. **Increase LS-FV regularization** to $\lambda_\text{reg} = \epsilon_\text{rel} \cdot \text{trace}(A_i)$ with $\epsilon_\text{rel} \sim 10^{-4}$, adapting to the local metric tensor scale.

7. **Fix metadata extraction** in `splits.py:36-78`. The current implementation fabricates flow parameters from filename hashing (`sim_id * 17 % 21 - 5.0`) instead of loading actual conditions from the `.pth` files. This means the Reynolds and AoA extrapolation splits (Tasks 3 & 4) are **completely wrong** — they partition on fake metadata, not actual flow conditions. This is a critical data engineering bug.

**P2 — Recommended Optimizations:**

8. **Add BL head skip connection** from pre-encoder features $h_0$ to provide unsmoothed wall-distance information.

9. **Smooth curriculum transitions** using sigmoid warmup instead of hard epoch boundaries.

10. **Add `torch.use_deterministic_algorithms(True)`** for full reproducibility.

11. **Profile the training loop** with `torch.profiler` to identify the actual memory/compute bottleneck (likely the Level-0 message passing, not the Transformer).

12. **Add gradient norm logging per loss component** to diagnose gradient conflicts between data and physics losses during Phase 2 transition.

**P3 — Nice-to-Have for Publication:**

13. Replace stride-based graph coarsening with METIS or voxel-grid pooling for topologically consistent multi-level graphs.

14. Add wandb/TensorBoard integration for experiment tracking.

15. Implement DDP wrapper for multi-GPU training (2$\times$A100 would halve wall-clock time for ~$84 total cost).

---

*Assessment prepared by automated scientific review pipeline. All findings are based on static code analysis and theoretical reasoning — empirical validation against the AirfRANS dataset is required to confirm quantitative impact estimates.*
