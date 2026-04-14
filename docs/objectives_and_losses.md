# Objectives And Losses

This document describes the **current redesigned model** and explains exactly what each loss is doing, where it is applied, and why it exists.

## 1. The Big Picture

The redesigned model differs from the earlier extension in five important ways:

1. ROI inputs are normalized relative to the training `ref` cohort.
2. The generator is **factorized additive** rather than latent-concatenative.
3. Training is **sampled-latent**, closer to the original SurrealGAN idea.
4. The decomposer is supervised against known synthetic components.
5. There are explicit losses against **generator silence** and **process collapse**.

So the current method is trying to solve two problems at once:

- disentangle age from non-age processes,
- ensure those non-age processes still correspond to distinct generator-side anatomical effects.

## 2. Notation

- \(x \in \mathbb{R}^d\): normalized ROI vector from a `ref` subject.
- \(y \in \mathbb{R}^d\): normalized ROI vector from a `tar` subject.
- \(a \in [0,1]\): sampled or inferred age latent.
- \(r = (r_1,\dots,r_K) \in [0,1]^K\): sampled or inferred process latent vector.
- \(H_{age}(x)\): age basis pattern predicted by the generator.
- \(H_k(x)\): process-\(k\) basis pattern predicted by the generator.
- \(\Delta_{age}(x,a) = a\,H_{age}(x)\)
- \(\Delta_k(x,r_k) = r_k\,H_k(x)\)
- \(\hat y = x + \Delta_{age}(x,a) + \sum_k \Delta_k(x,r_k)\)

## 3. Normalization

The model does not train on raw ROI magnitudes.

For each ROI \(j\), fit REF statistics on the **training ref split only**:

$$
\mu_j^{ref} = \mathbb{E}[x_j \mid \text{train ref}],
\qquad
\sigma_j^{ref} = \mathrm{Std}[x_j \mid \text{train ref}].
$$

Then normalize:

$$
x_j' = \mathrm{clip}\left(\frac{x_j - \mu_j^{ref}}{\sigma_j^{ref} + \varepsilon}, -c, c\right).
$$

This matters because the older raw-ROI training space made large ROIs dominate the objective and made generator sensitivity hard to calibrate.

## 4. Generator

The generator first forms a context representation:

$$
h(x) = T(x)
$$

then predicts:

$$
H_{age}(x) = W_{age} h(x)
$$

$$
\left(H_1(x),\dots,H_K(x)\right) = W_{proc} h(x).
$$

The synthetic target is:

$$
\hat y = x + a\,H_{age}(x) + \sum_{k=1}^{K} r_k H_k(x).
$$

This makes each latent dimension own an explicit basis pattern, which is much easier to regularize than a generic concatenation MLP.

## 5. Training Inputs and Sampling

Each training step uses:

- a `ref` batch \(x\),
- a `tar` batch \(y\),
- a sampled age latent equal to the normalized chronological age of the target batch,
- sampled process latents.

By default the redesign uses **one-hot process sampling**:

$$
r = (0,\dots,0,u,0,\dots,0), \qquad u \sim \mathrm{Uniform}(0,1),
$$

and also constructs a later version:

$$
r^{later}_k \ge r_k.
$$

This is close in spirit to the original SurrealGAN logic: sample process severities, generate a synthetic target, then ask whether the model can recover what it just used.

## 6. Decomposer and Latent Reconstruction

The decomposer predicts:

$$
Q(z) =
\left(\tilde\Delta_{age}(z), \tilde\Delta_1(z), \dots, \tilde\Delta_K(z)\right).
$$

The latent reconstruction heads then estimate:

$$
\hat a = R_{age}(\tilde\Delta_{age}),
\qquad
\hat r_k = R_k(\tilde\Delta_k).
$$

This is used in two ways:

1. synthetic-target supervision:
   the model knows the sampled \(a\) and \(r\), so it can check whether those are recovered;
2. real-target inference:
   for actual subjects, the inferred \(\hat a\) and \(\hat r\) become the age latent and `r`-indices.

## 7. Loss Table

### A. Adversarial realism

| Loss | Formula | Why it exists |
| --- | --- | --- |
| Discriminator | \(L_D = \frac{1}{2}\big(\mathrm{BCE}(D(y),1) + \mathrm{BCE}(D(\hat y),0)\big)\) | Push synthetic targets toward the target manifold. |
| Generator adversarial | \(L_{adv} = \mathrm{BCE}(D(\hat y),1)\) | Make generated targets look target-like. |

### B. Age supervision

| Loss | Formula | Why it exists |
| --- | --- | --- |
| Age supervision | \(L_{age} = \lVert \hat a(y) - a(y)\rVert_2^2\) | Real targets should encode their actual age. |
| Reference age supervision | \(L_{ref-age} = \lVert \hat a(x) - a(x)\rVert_2^2\) | The age branch should work on the reference cohort too. |
| Age adversary | \(L_{age-adv} = \lVert A(\hat r(y)) - a(y)\rVert_2^2\) with GRL | Process latents should not be an alternate age axis. |

### C. Synthetic invertibility

| Loss | Formula | Why it exists |
| --- | --- | --- |
| Latent reconstruction | \(L_{latent} = \lVert \hat a(\hat y)-a\rVert_2^2 + \sum_k \lVert \hat r_k(\hat y)-r_k\rVert_2^2\) | If the generator used a latent, the decomposer should recover it. |
| Decomposition | \(L_{decomp} = \lVert \tilde\Delta_{age}(\hat y)-\Delta_{age}\rVert_2^2 + \sum_k \lVert \tilde\Delta_k(\hat y)-\Delta_k\rVert_2^2\) | The decomposer should recover the actual synthetic components. |

This is one of the biggest conceptual corrections relative to the older extension: decomposition is now supervised against known synthetic components, not only against their sum.

### D. Identity / magnitude regularization

| Loss | Formula | Why it exists |
| --- | --- | --- |
| Identity | \(L_{id} = \lVert G(x,0,0)-x\rVert_1\) | No activation should imply no change. |
| Low-activation identity | \(L_{low-id} = \lVert G(x,a_{small},r_{small})-x\rVert_1\) | Tiny activations should stay close to the baseline. |
| Change magnitude | \(L_{change} = \lVert \Delta(x,a,r)\rVert_1\) | Prevent gratuitously large deformations. |

### E. Monotonicity and disentanglement

| Loss | Formula | Why it exists |
| --- | --- | --- |
| Monotonicity | \(L_{mono} = \max\big(|\Delta(r)| - |\Delta(r^{later})|, 0\big)\) | A stronger process activation should not give a weaker effect. |
| Orthogonality | \(L_{orth} = \lVert \bar H \bar H^\top - I\rVert_2^2\) | Mean process bases should not all point in the same direction. |
| Age-process covariance | \(L_{cov} = \lVert \mathrm{Cov}(\hat a(y), \hat r(y))\rVert_F^2\) | Keep age and process branches statistically separate. |
| Process-age correlation | \(L_{proc-age} = \mathrm{mean}_k \rho(\hat r_k(y), a(y))^2\) | Directly penalize age leakage in process latents. |
| Reference process sparsity | \(L_{ref-sparse} = \lVert \hat r(x)\rVert_1\) | Reference subjects should have low process burden. |
| Process latent sparsity | \(L_{proc-sparse} = \lVert \hat r(y)\rVert_1\) | Encourage compact process codes. |
| Pairwise latent correlation | \(L_{pair} = \mathrm{mean}_{i\neq j}\rho(\hat r_i(y), \hat r_j(y))^2\) | Stop all process latents from becoming the same scalar. |

### F. Generator-collapse controls

These were the key missing ingredients before the redesign.

Let

$$
g_k(x) = G(x, a_0, e_k) - G(x, a_0, 0)
$$

and

$$
g_{age}(x) = G(x, a_{max}, 0) - G(x, a_{min}, 0).
$$

| Loss | Formula | Why it exists |
| --- | --- | --- |
| Age sensitivity | \(L_{age-sens} = \max(\tau_{age} - S_{age}, 0)\) | Prevent an age-silent generator. |
| Process sensitivity | \(L_{proc-sens} = \max(\tau_{proc} - \overline{S}_{proc}, 0)\) | Prevent a process-silent generator. |
| Generator separation | \(L_{gen-sep} = \max(m - \mathrm{dist}(H_i,H_j), 0)\) | Force different process bases apart. |
| Generator redundancy | \(L_{gen-red} = \mathrm{mean}_{i<j} |\rho(H_i,H_j)|\) | Penalize near-duplicate process bases directly. |

This is the main reason the redesign behaves better qualitatively than the older model family.

### G. Optional directional priors

| Loss | Formula | Why it exists |
| --- | --- | --- |
| Age shrinkage | \(L_{age-shrink} = \mathrm{mean}\,\max(g_{age}^{raw},0)\) | Penalize positive age-driven raw-volume changes. |
| Process shrinkage | \(L_{proc-shrink} = \mathrm{mean}_k \max(g_k^{raw},0)\) | Penalize positive process-driven raw-volume changes. |

These are **optional priors**, not biological truths. Ventricles and CSF-like structures can enlarge.

## 8. Full Generator Objective

The full generator-side objective is:

$$
\begin{aligned}
L_G =\;&
\lambda_{adv}L_{adv}
 + \lambda_{age}L_{age}
 + \lambda_{ref-age}L_{ref-age}
 + \lambda_{age-adv}L_{age-adv} \\
&+ \lambda_{latent}L_{latent}
 + \lambda_{decomp}L_{decomp}
 + \lambda_{id}L_{id}
 + \lambda_{mono}L_{mono} \\
&+ \lambda_{orth}L_{orth}
 + \lambda_{cov}L_{cov}
 + \lambda_{ref-sparse}L_{ref-sparse}
 + \lambda_{change}L_{change} \\
&+ \lambda_{low-id}L_{low-id}
 + \lambda_{proc-age}L_{proc-age}
 + \lambda_{proc-sparse}L_{proc-sparse}
 + \lambda_{age-sens}L_{age-sens} \\
&+ \lambda_{proc-sens}L_{proc-sens}
 + \lambda_{age-shrink}L_{age-shrink}
 + \lambda_{proc-shrink}L_{proc-shrink} \\
&+ \lambda_{gen-sep}L_{gen-sep}
 + \lambda_{gen-red}L_{gen-red}
 + \lambda_{pair}L_{pair}.
\end{aligned}
$$

Not every scenario uses all losses strongly. The scenario matrix varies the **weights**, not the presence of the code paths.

## 9. What the Losses Are Trying to Achieve Together

The redesigned model is optimizing for all of the following at once:

1. \(a\) should encode chronological age.
2. \(r_k\) should not simply be age proxies.
3. synthetic targets should remain target-like.
4. sampled latent factors should be recoverable from the synthetic target.
5. each process should have a real, nontrivial anatomical effect.
6. different processes should have different anatomical effects.
7. process latents should not collapse onto each other.

That is the actual scientific intent of the redesign.

## 10. Why Not Use Agreement as a Loss?

Agreement is computed **after training**, across repetitions. It is a model-selection criterion, not a differentiable within-epoch loss.

So the workflow is:

- losses shape one training run,
- selection metrics choose the best epoch inside each repetition,
- agreement chooses the representative repetition for that run.

Agreement answers “is this stable across reruns?”, not “what should the gradients be?”.

## 11. Hemisphere Symmetry

A symmetry prior is not currently active by default.

Reason:

- age/global atrophy is often approximately bilateral,
- disease effects can be genuinely asymmetric,
- adding a symmetry loss too early could make factors look neat while removing biologically real lateralization.

If symmetry is ever added, it should be a **mild ablation**, probably starting with the age basis rather than all process bases.
