# Age-Decoupled SurrealGAN: April 2026 Technical Overview

## Executive Summary

This project is a SurrealGAN-inspired ROI-space model that attempts to **separate normative aging from additional non-age brain processes**. The current implementation replaces the original single latent space with:

- one explicit **age latent** \(a\)
- \(K\) non-age **process latents** \(r_1, ..., r_K\)

The scientific motivation is straightforward:

1. The original SurrealGAN learns latent dimensions that transform a reference brain into a target brain.
2. In practice, several of those learned dimensions are strongly correlated with each other and with age.
3. That is plausible biologically, but it makes interpretation difficult if the goal is to identify **non-age mechanisms** such as disease-related or atypical structural changes.
4. The extension therefore tries to force age into one branch of the latent space and leave the remaining latent axes available for additional processes.

At the current stage, the implementation is successful in one sense and unsuccessful in another:

- **Successful:** the age latent is consistently strongly correlated with chronological age.
- **Unsuccessful:** the direct generator is still almost flat with respect to age/process slider manipulations.

In other words, the current model is already a reasonably good **age regressor with low residual age leakage**, but it is not yet a convincing **controllable SurrealGAN-style generator**.

That is the most important result so far.

---

## Data And Problem Setup

The model is trained on ROI volumetrics from `cleaned_istaging.csv`, restricted to **atomic** MUSE ROIs:

\[
\text{ROI ID} \le 299
\]

Composite MUSE regions are excluded to avoid duplicating information from constituent ROIs.

The current cohort logic is:

- **Reference (REF):** cognitively normal, ages \(20\) to \(49\)
- **Target (TAR):** all subjects, ages \(50\) to \(97\)
- **OOD/application holdout:** entire `HANDLS` study

So the model is not trained as a diagnosis classifier. Instead, it is trained as a structured transformation model in ROI space, with diagnosis retained for downstream interpretation.

---

## Model Definition

Let:

- \(x \in \mathbb{R}^d\): ROI vector from the reference cohort
- \(y \in \mathbb{R}^d\): ROI vector from the target cohort
- \(a \in [0,1]\): age latent
- \(r = (r_1,\dots,r_K) \in [0,1]^K\): process latents

### 1. Encoder

The encoder maps a target ROI vector to the latent space:

\[
E(y) = (a, r)
\]

where \(a\) is intended to capture chronological aging and \(r\) is intended to capture additional non-age processes.

### 2. Generator

The generator takes a reference ROI vector and latent controls:

\[
G(x, a, r) = \Delta
\]

and produces a synthetic target:

\[
\hat{y} = x + \Delta
\]

This is the SurrealGAN-like transformation step.

### 3. Decomposer

The total synthetic change is decomposed into one age component and \(K\) process-specific components:

\[
Q(\Delta) = \left(\Delta^{age}, \Delta^{proc}_1, \dots, \Delta^{proc}_K\right)
\]

with the additivity constraint:

\[
\Delta \approx \Delta^{age} + \sum_{i=1}^{K} \Delta^{proc}_i
\]

### 4. Latent Reconstruction

The decomposed components are mapped back to latent values:

\[
R_a(\Delta^{age}) \approx a
\]

\[
R_i(\Delta^{proc}_i) \approx r_i
\]

### 5. Discriminator

The discriminator tries to distinguish real target ROI vectors from synthetic ones:

\[
D(y) \to 1, \qquad D(\hat{y}) \to 0
\]

### 6. Age Adversary

An adversarial head tries to predict age from the process latents:

\[
A(r) \approx a
\]

but through gradient reversal, the encoder is trained to make this prediction difficult. This is one of the main mechanisms used to discourage age leakage into \(r\).

---

## Core Objective

The full generator-side objective is a weighted sum:

\[
L_G =
\lambda_{adv} L_{adv}
+ \lambda_{age} L_{age}
+ \lambda_{ref-age} L_{ref-age}
+ \lambda_{age-adv} L_{age-adv}
+ \lambda_{latent} L_{latent}
+ \lambda_{decomp} L_{decomp}
+ \lambda_{id} L_{id}
+ \lambda_{mono} L_{mono}
+ \lambda_{orth} L_{orth}
+ \lambda_{cov} L_{cov}
+ \lambda_{ref-proc} L_{ref-proc}
+ \lambda_{sens-age} L_{sens-age}
+ \lambda_{sens-proc} L_{sens-proc}
+ \lambda_{shrink-age} L_{shrink-age}
+ \lambda_{shrink-proc} L_{shrink-proc}
+ \text{optional ablation terms}
\]

The main terms are:

### Adversarial realism

\[
L_{adv} = \mathrm{BCE}(D(\hat{y}), 1)
\]

This pressures the synthetic target to resemble the real target cohort.

### Supervised age latent

\[
L_{age} = \|a - \tilde{age}\|_2^2
\]

where \(\tilde{age}\) is normalized chronological age.

### Reference age supervision

\[
L_{ref-age} = \|E_a(x) - \tilde{age}_{ref}\|_2^2
\]

This stabilizes the age branch on REF examples as well.

### Age adversary

\[
L_{age-adv} = \|A(\mathrm{GRL}(r)) - \tilde{age}\|_2^2
\]

The adversary is trained to predict age from \(r\), while the encoder is trained to hide age from \(r\).

### Latent reconstruction

\[
L_{latent} = \|R_a(\Delta^{age}) - a\|_2^2 + \sum_i \|R_i(\Delta_i^{proc}) - r_i\|_2^2
\]

### Decomposition consistency

\[
L_{decomp} = \left\|\Delta - \Delta^{age} - \sum_i \Delta_i^{proc}\right\|_2^2
\]

### Identity / low-activation regularization

\[
L_{id} = \|G(x, 0, 0) + x - x\|_1
\]

and optional low-activation identity terms constrain the model near the origin of latent space.

### Monotonicity

The current monotonicity regularizer encourages larger latent values to produce larger-magnitude changes:

\[
L_{mono} = \max\left(\|\Delta(x,a,0.5r)\| - \|\Delta(x,a,r)\|, 0\right)
\]

This is weaker than the original SurrealGAN dimension-wise monotonicity construction, but it serves the same intuition: increasing latent activation should not reduce the magnitude of the modeled effect.

### Orthogonality

\[
L_{orth} = \|PP^\top - I\|_F^2
\]

where \(P\) is a normalized matrix of process-delta vectors. This attempts to make process axes distinct in ROI space.

### Age-process covariance penalty

\[
L_{cov} = \|\mathrm{Cov}(a, r)\|_F^2
\]

This is a direct disentanglement penalty.

### Sensitivity margins

These terms are critically important for this project because the current failure mode is a **flat generator**.

Age sensitivity:

\[
S_a =
\frac{100}{nd}\sum_{j=1}^{n}\sum_{k=1}^{d}
\frac{|G(x_j, a_{max}, 0)_k - G(x_j, a_{min}, 0)_k|}{|x_{jk}| + \epsilon}
\]

Process sensitivity:

\[
S_{r_i} =
\frac{100}{nd}\sum_{j=1}^{n}\sum_{k=1}^{d}
\frac{|G(x_j, a_{anchor}, e_i)_k - G(x_j, a_{anchor}, 0)_k|}{|x_{jk}| + \epsilon}
\]

The corresponding losses penalize the model when these sensitivities fall below configured targets.

### Shrinkage-biased losses

These are optional directional priors, motivated by your desire to bias the model toward atrophy-like effects:

\[
L_{shrink-age} = \mathbb{E}\left[\max\left(G(x,a_{max},0)-G(x,a_{min},0), 0\right)\right]
\]

\[
L_{shrink-proc} = \mathbb{E}_i\left[\max\left(G(x,a_{anchor},e_i)-G(x,a_{anchor},0), 0\right)\right]
\]

These are scientifically defensible as **priors**, but not as universal truths, because some structures genuinely enlarge with age or disease.

---

## What Has Been Run So Far

The runs currently present on disk are nine large scenario-training experiments:

1. `scenario-train-balanced-4`
2. `scenario-train-balanced-5`
3. `scenario-train-faithful-surreal-5`
4. `scenario-train-sensitivity-push-4`
5. `scenario-train-shrinkage-directional-4`
6. `scenario-train-shrinkage-directional-5`
7. `scenario-train-sparse-disentangled-4`
8. `scenario-train-generator-freedom-5`
9. `scenario-train-change-regularized-4`

These differ primarily along four conceptual axes:

- \(K=4\) versus \(K=5\)
- stronger versus weaker reconstruction/identity constraints
- stronger versus weaker generator-sensitivity pressure
- stronger versus weaker disentanglement and shrinkage-direction priors

The exact config values are stored in each run’s `resolved_config.json`.

---

## What The Results Suggest

### 1. Age modeling is working well

Across all nine completed scenario runs, the validation correlation between age latent and chronological age is consistently high:

\[
\rho(a, age) \approx 0.896 \text{ to } 0.902
\]

This is the strongest positive result so far. The age branch is learning a stable signal.

### 2. Four-process models currently dominate on decoupling metrics

The strongest validation composite score currently belongs to:

- `scenario-train-sparse-disentangled-4`

with:

\[
\text{composite}_{val} \approx 0.888
\]

This run also has the smallest mean process-age correlation:

\[
\text{mean}|\rho(r_i, age)| \approx 0.0094
\]

This is a very strong sign that the process branch can be made nearly age-invariant under the current metrics.

### 3. The direct generator is still essentially collapsed

This is the central current limitation.

For most runs:

- age sensitivity is effectively zero
- process sensitivity is effectively zero
- process-separation in generated outputs is effectively zero

Numerically, these sensitivity metrics are often on the order of:

\[
10^{-10} \text{ to } 10^{-6}
\]

This means:

- the encoder can estimate \(a\)
- the process branch can be made age-invariant
- but changing \(a\) or \(r_i\) does **not** yet produce meaningful direct generator output

So the system behaves more like a **structured latent regressor** than a **controllable generative transformation model**.

### 4. Five-process runs look more SurrealGAN-like in latent amplitude

Several five-process runs, especially:

- `balanced-5`
- `shrinkage-directional-5`
- `generator-freedom-5`

show much larger raw latent amplitudes for some \(r_i\), with some dimensions approaching:

\[
r_i \approx 1
\]

For example, in `generator-freedom-5`, two process dimensions saturate near \(1\) for some subjects. This indicates that some configurations do allow larger process activation than the sparse four-process solutions.

However, this has **not** yet translated into strong direct generator sensitivity.

### 5. OOD generalization remains limited

For all runs, OOD/application performance on `HANDLS` is much worse than ID validation:

\[
\text{composite}_{ood} \approx 0.36 \text{ to } 0.48
\]

and

\[
\rho(a, age)_{ood} \approx 0.57 \text{ to } 0.60
\]

So the age branch still generalizes somewhat OOD, but substantially worse than in-distribution.

### 6. Agreement is moderate, not yet excellent

The strongest repetition agreement among current runs is again `sparse-disentangled-4`, with:

- dimension correlation \(\approx 0.417\)
- difference correlation \(\approx 0.196\)

This is the best reproducibility in the current set, but still not strong enough to claim rock-solid stable latent factors.

---

## Current Interpretation

The model is currently best described as:

> a promising age-decoupled ROI latent model whose encoder branch works substantially better than its generator branch.

That is already scientifically meaningful, because it suggests that:

- age can be isolated into a dedicated latent
- non-age process latents can be regularized to reduce age contamination

But it also means that the strongest SurrealGAN-like claim is **not yet supported**:

> that the learned latent sliders directly generate strong, interpretable, and controllable ROI transformations.

The web GUI has been useful precisely because it exposed this weakness visually.

---

## Immediate Next Questions

1. Can the generator be made genuinely sensitive without collapsing disentanglement?
2. Is the current decomposition branch learning something useful even when the direct generator is weak?
3. Are five-process models better for interpretability even if four-process models win the composite score?
4. Should the original SurrealGAN change-regularization and monotonicity structure be restored more faithfully?
5. Should ROI normalization be reintroduced, both for numerical stability and to restore SurrealGAN-like geometry?

---

## Bottom Line

If asked today what the project has shown, the honest answer is:

- The age-decoupling idea is viable at the encoder level.
- A dedicated age latent can be learned robustly.
- Process latents can be pushed toward lower age leakage.
- The current training system still fails to produce a strongly controllable generator.
- Therefore the next phase should focus on **generator identifiability, sensitivity, normalization, and objective design**, not merely running more of the same baseline.

That is a defensible and scientifically useful status update.
