# Objectives And Losses

This document describes the **current redesigned model**, not the earlier concatenation-generator extension. The redesign keeps the SurrealGAN idea of learning ROI-space transformations, but changes three core ingredients:

1. ROI features are normalized against the training `ref` cohort.
2. The generator is **factorized additive**, so age and each process own explicit basis patterns.
3. Training is again **sampled-latent**, closer in spirit to the original SurrealGAN, instead of relying only on direct encoder-style inference.

## Notation

- \(x \in \mathbb{R}^d\): normalized ROI vector from a `ref` subject.
- \(y \in \mathbb{R}^d\): normalized ROI vector from a `tar` subject.
- \(a \in [0,1]\): normalized age latent.
- \(r = (r_1,\dots,r_K) \in [0,1]^K\): process latents.
- \(H_{\text{age}}(x) \in \mathbb{R}^d\): age basis pattern.
- \(H_k(x) \in \mathbb{R}^d\): process-\(k\) basis pattern.
- \(Q(\cdot)\): decomposer.
- \(R_{\text{age}}(\cdot)\), \(R_k(\cdot)\): latent reconstruction heads.
- \(D(\cdot)\): discriminator.

## ROI Normalization

Training no longer uses raw ROI magnitudes directly. For each ROI \(j\), normalization statistics are fit on the training `ref` cohort:

$$
\mu_j^{ref} = \mathbb{E}[x_j \mid \text{train ref}], \qquad
\sigma_j^{ref} = \mathrm{Std}[x_j \mid \text{train ref}].
$$

The normalized ROI value is:

$$
x'_j = \mathrm{clip}\left(\frac{x_j - \mu_j^{ref}}{\sigma_j^{ref} + \varepsilon}, -c, c\right).
$$

In the default redesign:

- `roi_normalization = "zscore"`
- \(\varepsilon = 10^{-6}\)
- \(c = 6\)

This is the single most important numerical change relative to the earlier failed runs on raw ROI scale.

## Model

### Generator

The generator is factorized additive:

$$
\Delta_{\text{age}}(x,a) = a \, H_{\text{age}}(x)
$$

$$
\Delta_{\text{proc}}(x,r) = \sum_{k=1}^{K} r_k \, H_k(x)
$$

$$
\hat y = x + \Delta(x,a,r)
      = x + \Delta_{\text{age}}(x,a) + \Delta_{\text{proc}}(x,r).
$$

This is more interpretable than the earlier concatenation MLP because each latent dimension now owns an explicit basis field.

### Decomposer and Reconstruction

The decomposer predicts one age component and \(K\) process components from a synthetic or real target:

$$
Q(z) = \left(\tilde\Delta_{\text{age}}(z), \tilde\Delta_1(z), \dots, \tilde\Delta_K(z)\right).
$$

The latent reconstruction heads invert those components:

$$
\hat a = R_{\text{age}}(\tilde\Delta_{\text{age}}), \qquad
\hat r_k = R_k(\tilde\Delta_k).
$$

On real `tar` subjects, the latent inference path is:

$$
(\hat a(y), \hat r(y)) = \left(R_{\text{age}}(Q_{\text{age}}(y)), R_1(Q_1(y)), \dots, R_K(Q_K(y))\right).
$$

## Training Strategy

For each mini-batch:

1. draw a `ref` sample \(x\),
2. draw a real `tar` sample \(y\),
3. set the sampled age latent to the target subject’s normalized age,
4. sample a process vector \(r\).

By default the redesigned model uses **one-hot process sampling**:

$$
r = (0,\dots,0,u,0,\dots,0), \qquad u \sim \mathrm{Uniform}(0,1),
$$

and a “later” version for monotonicity:

$$
r^{later}_k \ge r_k.
$$

This sampled-latent supervision is much closer to the original SurrealGAN philosophy than the older extension was.

## Losses

The generator-side objective is:

$$
\begin{aligned}
L_G =\;&
\lambda_{adv} L_{adv}
 + \lambda_{age} L_{age}
 + \lambda_{ref\_age} L_{ref\_age}
 + \lambda_{age\_adv} L_{age\_adv} \\
&+ \lambda_{latent} L_{latent}
 + \lambda_{decomp} L_{decomp}
 + \lambda_{id} L_{id}
 + \lambda_{mono} L_{mono} \\
&+ \lambda_{orth} L_{orth}
 + \lambda_{cov} L_{cov}
 + \lambda_{ref\_sparse} L_{ref\_sparse}
 + \lambda_{change} L_{change} \\
&+ \lambda_{low\_id} L_{low\_id}
 + \lambda_{proc\_age} L_{proc\_age}
 + \lambda_{proc\_sparse} L_{proc\_sparse}
 + \lambda_{age\_sens} L_{age\_sens} \\
&+ \lambda_{proc\_sens} L_{proc\_sens}
 + \lambda_{age\_shrink} L_{age\_shrink}
 + \lambda_{proc\_shrink} L_{proc\_shrink} \\
&+ \lambda_{gen\_sep} L_{gen\_sep}
 + \lambda_{gen\_red} L_{gen\_red}
 + \lambda_{pair} L_{pair}.
\end{aligned}
$$

The discriminator objective is:

$$
L_D = \frac{1}{2}\left(
\mathrm{BCE}(D(y), 1) + \mathrm{BCE}(D(\hat y), 0)
\right).
$$

### Core losses

| Loss | Formula | Intuition |
| --- | --- | --- |
| Adversarial | \(L_{adv} = \mathrm{BCE}(D(\hat y), 1)\) | Generated targets should look like real `tar` subjects. |
| Age supervision | \(L_{age} = \lVert \hat a(y) - a(y) \rVert_2^2\) | Real targets should encode their own age. |
| Reference age supervision | \(L_{ref\_age} = \lVert \hat a(x) - a(x) \rVert_2^2\) | Age branch should also behave correctly on `ref`. |
| Age adversary | \(L_{age\_adv} = \lVert A(\hat r(y)) - a(y) \rVert_2^2\) with GRL | Remove age information from process latents. |
| Latent reconstruction | \(L_{latent} = \lVert \hat a(\hat y) - a \rVert_2^2 + \sum_k \lVert \hat r_k(\hat y) - r_k \rVert_2^2\) | If the generator used a latent, the decomposer should recover it. |
| Decomposition | \(L_{decomp} = \lVert \tilde\Delta_{\text{age}}(\hat y) - \Delta_{\text{age}} \rVert_2^2 + \sum_k \lVert \tilde\Delta_k(\hat y) - \Delta_k \rVert_2^2\) | Explicit supervision of the synthetic components. |
| Identity | \(L_{id} = \lVert G(x,0,0) - x \rVert_1\) | No latent activation should mean no change. |
| Low-activation identity | \(L_{low\_id} = \lVert G(x,a_{small},r_{small}) - x \rVert_1\) | Tiny latent activations should stay near the reference manifold. |
| Change magnitude | \(L_{change} = \lVert \Delta(x,a,r) \rVert_1\) | Prevent gratuitously large deformations. |

### Monotonicity and disentanglement

| Loss | Formula | Intuition |
| --- | --- | --- |
| Monotonicity | \(L_{mono} = \max\big(\lvert \Delta(r) \rvert - \lvert \Delta(r^{later}) \rvert, 0\big)\) | Larger process activation should not produce a weaker effect. |
| Orthogonality | \(L_{orth} = \lVert \bar H \bar H^\top - I \rVert_2^2\) | Mean process bases should not align too strongly. |
| Age-process covariance | \(L_{cov} = \lVert \mathrm{Cov}(\hat a(y), \hat r(y)) \rVert_F^2\) | Keep age and processes statistically separate. |
| Process-age correlation | \(L_{proc\_age} = \sum_k \rho(\hat r_k(y), a(y))^2\) | Extra direct penalty against age leakage. |
| Process latent sparsity | \(L_{proc\_sparse} = \lVert \hat r(y) \rVert_1\) | Encourage compact process codes. |
| Reference process sparsity | \(L_{ref\_sparse} = \lVert \hat r(x) \rVert_1\) | `ref` subjects should have low process burden. |
| Pairwise latent correlation | \(L_{pair} = \mathrm{mean}_{i \ne j}\,\rho(\hat r_i(y), \hat r_j(y))^2\) | Prevent all process latents from tracking the same scalar. |

### Generator-collapse controls

These are the most important redesign additions because the earlier model family was failing by making all process axes either silent or redundant.

| Loss | Formula | Intuition |
| --- | --- | --- |
| Age sensitivity | \(L_{age\_sens} = \max(\tau_{age} - S_{age}, 0)\) | The generator must respond when age changes. |
| Process sensitivity | \(L_{proc\_sens} = \max(\tau_{proc} - S_{proc}, 0)\) | The generator must respond when a process is activated. |
| Generator separation | \(L_{gen\_sep} = \max(m - \mathrm{dist}(H_i, H_j), 0)\) | Different process bases should not collapse together. |
| Generator redundancy | \(L_{gen\_red} = \mathrm{mean}_{i \ne j} |\mathrm{corr}(H_i, H_j)|\) | Penalize near-duplicate process bases directly. |

Here \(S_{age}\) and \(S_{proc}\) are the percent-change sensitivities measured on a held-out set of `ref` subjects.

### Optional directional priors

| Loss | Formula | Intuition |
| --- | --- | --- |
| Age shrinkage | \(L_{age\_shrink} = \mathrm{mean}\,\max(\Delta_{age}^{raw}, 0)\) | Penalize positive age-driven ROI growth. |
| Process shrinkage | \(L_{proc\_shrink} = \mathrm{mean}\,\max(\Delta_{proc}^{raw}, 0)\) | Penalize positive process-driven ROI growth. |

These are useful as **priors or ablations**, but they are not biologically universal. Ventricles and CSF spaces often enlarge with age or disease, so shrinkage losses should not be treated as a default truth constraint.

## What We Are Optimizing For

The current method is trying to satisfy all of the following simultaneously:

1. \(a\) should capture chronological age.
2. \(r_k\) should carry little age leakage.
3. Each \(r_k\) should drive a nontrivial generator response.
4. Different \(r_k\) should drive different generator responses.
5. The synthetic target should stay on a realistic `tar` manifold.

That combination is the key difference between the redesigned model and the earlier failed extension, which often optimized age separation successfully while letting the generator remain silent or collapsed.

## Why Hemisphere Symmetry Is Not a Default Loss

A bilateral symmetry prior is tempting, but it is not currently enabled by default. The reason is scientific rather than technical:

- age-related global atrophy is often roughly bilateral,
- disease effects can be genuinely asymmetric,
- forcing symmetry too early could erase the very lateralized structure we may want to discover.

If symmetry is tested later, it should be a mild ablation, ideally restricted to the **age basis** before being imposed on process bases.
