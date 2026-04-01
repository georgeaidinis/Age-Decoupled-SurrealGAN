# Objectives And Losses

This document compares the current age-decoupled extension against the original SurrealGAN code in this repository and explains the rationale for each objective.

## Loss Table

| Category | Network(s) | Original SurrealGAN | Age-Decoupled Extension (default) | Optional Extension Ablation |
| --- | --- | --- | --- | --- |
| Adversarial discrimination | Discriminator | $L_D = \frac{1}{2}\left(\mathrm{CE}(D(\hat y), 0)\,p_\phi(z) + \mathrm{CE}(D(y), 1)\,p_\phi(z)\right)$ | $L_D = \frac{1}{2}\left(\mathrm{BCE}(D(\hat y), 0) + \mathrm{BCE}(D(y), 1)\right)$ |  |
| Adversarial generation | Generator | $\mathrm{CE}(D(\hat y), 1)\,p_\phi(z)$ | $\mathrm{BCE}(D(\hat y), 1)$ |  |
| Latent correlation / copula | $\phi$ only | $L_\phi = \mathrm{CE}(D(\hat y), 1)\,p_\phi(z) + \alpha\,\mathrm{KL}(p_0 \| p_\phi)$ |  |  |
| Latent reconstruction | Reconstruction heads | $\sum_i \|R(Q_i(\hat y)) - z_i\|_2^2$ | $\|\hat a - a\|_2^2 + \sum_i \|\hat r_i - r_i\|_2^2$ |  |
| Decomposition consistency | Decomposer | $\sum_i \|Q_i(\hat y) - \Delta_i\|_2^2$ | $\left\|\Delta^{age} + \sum_i \Delta_i^{proc} - \Delta\right\|_2^2$ |  |
| Change magnitude | Generator | $\|\hat y - x\|_1$ |  | $\|\Delta\|_1$ |
| Monotonicity | Generator | $\left\|\max(|\Delta^{pre}| - |\Delta^{later}|, 0)\right\|_2^2$ | $\max(\|\Delta(x,a,0.5r)\|_1 - \|\Delta(x,a,r)\|_1, 0)$ |  |
| Pattern orthogonality | Generator | $\|CC^\top - I\|_2^2$ on pattern changes | $\|\bar P \bar P^\top - I\|_2^2$ on process deltas |  |
| CN neighborhood identity | Generator | $\|G(x, z_{small}) - x\|_1$ for small random $z$ |  | $\|G(x, a_{small}, r_{small}) - x\|_1$ |
| Exact zero identity | Generator |  | $\|G(x, 0, 0) - x\|_1$ |  |
| Age supervision | Encoder |  | $\|a - \tilde{\mathrm{age}}\|_2^2$ |  |
| Reference age supervision | Encoder |  | $\|E_a(x_{ref}) - \tilde{\mathrm{age}}_{ref}\|_2^2$ |  |
| Age adversary | Encoder + age adversary |  | $\|A(\mathrm{GRL}(r)) - \tilde{\mathrm{age}}\|_2^2$ |  |
| Age-process covariance | Encoder |  | $\|\mathrm{Cov}(a, r)\|_F^2$ |  |
| Direct process-age decorrelation | Encoder |  |  | $\sum_i \rho(r_i, \tilde{\mathrm{age}})^2$ |
| Reference process sparsity | Encoder |  | $\|E_r(x_{ref})\|_1$ |  |
| Process latent sparsity | Encoder |  |  | $\|r\|_1$ |

## Original Objective

The original generator-side objective is:

$$
L_G =
\frac{1}{\mathbb{E}[p_\phi(z)]}L_{\text{mapping}}
 + \lambda L_{\text{orth}}
 + \zeta L_{\text{recons}}
 + \kappa L_{\text{decompose}}
 + \gamma L_{\text{change}}
 + \mu L_{\text{mono}}
 + \eta L_{\text{cn}}
$$

with the discriminator objective:

$$
L_D = \frac{1}{2}\left(\mathrm{CE}(D(\hat y), 0)\,p_\phi(z) + \mathrm{CE}(D(y), 1)\,p_\phi(z)\right)
$$

and the copula objective:

$$
L_\phi = \mathrm{CE}(D(\hat y), 1)\,p_\phi(z) + \alpha \,\mathrm{KL}(p_0 \| p_\phi)
$$

## Current Extension Objective

The current default extension objective is:

$$
\begin{aligned}
L_G =\;&
w_{adv}L_{adv}
+ w_{age}L_{age}
+ w_{ref-age}L_{ref-age}
+ w_{age-adv}L_{age-adv} \\
&+ w_{latent}L_{latent}
+ w_{decomp}L_{decomp}
+ w_{id}L_{id}
+ w_{mono}L_{mono} \\
&+ w_{orth}L_{orth}
+ w_{cov}L_{cov}
+ w_{ref-proc}L_{ref-proc}
\end{aligned}
$$

with:

$$
L_D = \frac{1}{2}\left(\mathrm{BCE}(D(\hat y), 0) + \mathrm{BCE}(D(y), 1)\right)
$$

The optional ablation losses are implemented but default to zero weight:

$$
L_G^{abl} =
w_{change}L_{change}
+ w_{low-id}L_{low-id}
+ w_{corr}L_{corr}
+ w_{sparse}L_{sparse}
$$

## Why Keep Or Drop Each Loss

### Adversarial loss

For:
- It preserves the SurrealGAN idea that transformed samples should match the target distribution rather than only reconstruct labels or ages.
- It is still the main mechanism forcing realistic target-side ROI structure.

Against:
- It is the least stable part of the optimization and interacts with every regularizer.
- In the extension it is no longer weighted by a learned copula density, so it is a blunter signal than in the original implementation.

### Copula / latent-correlation objective

For:
- The original code uses it to model correlated latent severities without directly supervising them.
- It is theoretically attractive if we want the latent prior itself to encode co-occurring biological processes.

Against:
- It is tightly tied to the original sampled-latent formulation. The extension now infers latents from real data rather than sampling them first.
- Reintroducing it cleanly would likely require a variational or prior-matching reformulation, not just copying `phi`.

### Latent reconstruction

For:
- It keeps the decomposition interpretable by making the latent code recoverable from decomposed changes.
- This is the clearest bridge to the original SurrealGAN philosophy.

Against:
- In the extension, the latent is produced by an encoder on real targets, so it can collapse into trivial encodings if reconstruction is over-weighted.
- It can make the encoder and decomposer co-adapt in a way that looks consistent internally but is not biologically meaningful.

### Decomposition consistency

For:
- Some decomposition objective is essential. Otherwise the process branches are only labels on a tensor slice.
- The additive reconstruction of the total delta is the minimum consistency condition.

Against:
- The original decomposition loss is stronger because each component has a synthetic target. The current version only enforces summation, so multiple decompositions can satisfy it.
- This is one of the scientifically weakest parts of the current extension.

### Change magnitude

For:
- The original implementation explicitly penalizes total change size. This discourages gratuitous deformation and often improves identifiability.
- It is a natural regularizer if we believe only a subset of ROIs should move strongly.

Against:
- If weighted too highly, it will suppress real age- or disease-related signal and encourage underfitting.
- It can conflict with adversarial realism if the target cohort truly differs strongly from the reference cohort.

Recommendation:
- Keep it implemented as an ablation term and test small weights first.

### Monotonicity

For:
- The original SurrealGAN uses it to encode the idea that higher latent severity should produce larger pattern-specific changes.
- This is one of the most interpretable regularizers in the original method.

Against:
- The extension’s current monotonicity loss is weaker than the original and only compares a half-scaled process state to the full state.
- Without restoring the original sampled-prev/sampled-later construction, this loss is only an approximation.

### Orthogonality

For:
- Prevents all processes from collapsing into one shared direction.
- Helps the process sliders remain visually and statistically distinct.

Against:
- Real biology is not orthogonal. Over-enforcing orthogonality can carve one process into artificial fragments.
- It can directly conflict with correlated disease mechanisms.

Recommendation:
- Keep it, but do ablations on lower weights.

### CN neighborhood identity and exact-zero identity

For:
- Both stabilize the generator near the reference manifold.
- Exact-zero identity is the cleanest invariance: if no latent is active, nothing should happen.
- Low-activation identity is closer to the original `cn_loss` and protects a neighborhood around the origin, not just one point.

Against:
- Too much identity pressure can make the generator reluctant to move at all.
- If the age latent is meant to capture normal aging, some low-latent movement may be biologically reasonable.

Recommendation:
- Keep exact-zero identity as default.
- Use low-activation identity as an ablation if the model exhibits unstable or overly large small-latent deformations.

### Age supervision

For:
- Necessary if we truly want a designated age branch.
- Makes the age latent scientifically interpretable and testable.

Against:
- It hard-codes chronological age as the privileged notion of aging, which may not match biological age.
- If over-weighted, it can absorb non-age structure simply because age correlates with many processes.

### Age adversary

For:
- Directly expresses the scientific target: process latents should not be predictive of age.
- More flexible than direct correlation penalties because it can learn nonlinear age leakage.

Against:
- Adversarial terms are harder to optimize and can be noisy.
- If the age adversary is too weak, age still leaks. If too strong, it can erase legitimate signal.

### Age-process covariance and direct process-age decorrelation

For:
- These are simpler, more stable complements to the adversary.
- They provide linear leakage control even if the adversary is under-trained.

Against:
- Covariance and correlation penalties only constrain linear dependence.
- They can be redundant if the age adversary is already strong.

Recommendation:
- Use covariance as a mild default stabilizer.
- Use direct correlation penalty as an ablation alternative or complement.

### Reference process sparsity and process latent sparsity

For:
- These encourage process latents to stay quiet on clean reference subjects.
- Good for interpretability if the reference cohort is meant to be minimally pathologic.

Against:
- Too much sparsity can force all pathology into the age branch or a single process branch.
- In mixed target cohorts, sparsity can work against realism.

## Alternatives Worth Testing

### Architecture alternatives

1. Encode from target only, as now.
   - Advantage: simple and direct.
   - Risk: less faithful to original SurrealGAN because the latent is inferred rather than sampled-and-recovered.

2. Hybrid original-style sampled latent plus age branch.
   - Advantage: closer to original decomposition supervision.
   - Risk: substantially more engineering and more difficult optimization.

3. Decompose synthetic target $\hat y$ instead of delta $\Delta$.
   - Advantage: closer to original code.
   - Risk: harder to interpret the decomposition as pure change components.

### Objective alternatives

1. Add change magnitude back.
   - Likely effect: smaller, cleaner transformations; risk of underfitting.

2. Add low-activation identity.
   - Likely effect: smoother local generator geometry; risk of suppressing weak real effects.

3. Add direct process-age decorrelation.
   - Likely effect: easier optimization than pure adversarial disentanglement; risk of only removing linear leakage.

4. Lower orthogonality and increase decomposition.
   - Likely effect: more biologically correlated processes but potentially more entanglement.

5. Restore stronger original-style monotonicity with sampled low/high latent pairs for each process.
   - Likely effect: better slider semantics; requires more code changes than the current extension.

## Practical Recommendation

For publishable experiments, the most defensible near-term path is:

1. Keep the current default objective as the baseline.
2. Run ablations for:
   - `change_magnitude`
   - `low_activation_identity`
   - `process_age_correlation`
   - lower `process_orthogonality`
3. Treat the current decomposition formulation as a documented limitation, because it is weaker than the original SurrealGAN decomposition supervision.
