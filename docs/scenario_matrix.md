# Scenario Matrix

This document explains **what scenarios we are running, why they exist, and how they fit into model selection**.

## 1. What a Scenario Is

A **scenario** is a deliberately designed region of model space:

- one architecture family,
- one range of loss emphases,
- one choice of selection metric,
- sometimes one particular scientific prior.

It is not “the final answer”. A scenario is a hypothesis about what type of model behavior should be encouraged.

## 2. The Current Strategy

Right now we are **not** doing a full \(K\)-sweep over all possible numbers of process factors.

Instead, the current redesign strategy is:

1. first identify a good **model family**,
2. then tune hyperparameters inside that family,
3. then, once the family is trustworthy, run a broader **\(K\)-sweep** if needed.

That is why the current redesign scenarios focus on:

$$
K \in \{4, 5\}.
$$

This is a pragmatic choice, not a scientific claim that only 4 or 5 can ever work.

### Why only 4 or 5 right now?

- the original SurrealGAN paper settled on 5,
- earlier experiments in this project also suggested that 4 and 5 were the most plausible ranges,
- a broader \(K\)-sweep is expensive and not useful if the generator family itself is still unstable.

So the current logic is:

> fix the method first, then sweep \(K\) more broadly.

## 3. Center of Gravity

The redesign scenario family is centered on the strong indicator run:

- `20260409_221636_age-decoupled-surrealgan-redesign-train`

That run succeeded because it combined:

- REF-based ROI z-scoring,
- sampled process supervision,
- additive factorized generator,
- direct generator-separation penalties,
- collapse-aware selection.

This means the current scenario design is no longer centered on the older pre-redesign “balanced / sensitivity-push / faithful” family, but on the redesigned model that actually produces interpretable population patterns.

## 4. Train Scenarios

### `redesign_balanced_5`

This is the primary reference scenario.

Use it when you want the closest thing to the current best general-purpose redesign.

Main idea:

- keep the successful redesign settings,
- use \(K=5\),
- keep age separation, generator responsiveness, and collapse penalties balanced.

### `redesign_balanced_4`

Same regime, but with:

$$
K=4.
$$

Purpose:

- test whether fewer process factors improve stability or interpretability,
- without changing the rest of the method too much.

### `redesign_generator_push_5`

Purpose:

- test whether the generator is still somewhat underexpressive.

Main change pattern:

- increase adversarial and process-sensitivity pressure,
- increase generator-basis separation pressure,
- reduce identity and low-activation identity pressure.

This is the scenario to inspect if the process maps still look too weak or muted.

### `redesign_disentangle_5`

Purpose:

- test whether cleaner process axes help interpretability.

Main change pattern:

- stronger age adversary,
- stronger age-process covariance penalty,
- stronger process-age correlation penalty,
- stronger pairwise latent decorrelation,
- stronger basis orthogonality.

This can help prevent “all process factors mean the same thing”, but if pushed too hard it can also make the generator quieter.

### `redesign_stability_5`

Purpose:

- favor reproducibility.

Main change pattern:

- more repetitions,
- more epochs,
- slightly stronger reconstruction / decomposition / orthogonality.

This scenario is useful when a family looks promising but unstable across reruns.

### `redesign_wide_5`

Purpose:

- test whether current capacity is the limiting factor.

Main change pattern:

- wider encoder / generator / decomposer.

This is not about changing the scientific prior; it is about testing whether the model is underparameterized.

### `redesign_mixed_latent_5`

Purpose:

- test whether one-hot process sampling is too restrictive.

Main change:

```toml
sampled_process_one_hot_only = false
```

Meaning:

- instead of activating only one process at a time during sampled-latent training, allow mixed sampled process vectors.

Scientific tradeoff:

- may learn more realistic combined targets,
- may also blur factor identity.

### `redesign_directional_5`

Purpose:

- test a mild biological sign prior.

Main change pattern:

- small age/process shrinkage weights,
- selection by `directional_quality_score`.

This is an ablation, not the default scientific assumption.

## 5. Tune Scenarios

Tune scenarios do not define a single model. They define a **search space** for Optuna.

### `redesign_balanced_4_5`

Purpose:

- broad search around the successful redesign regime.

It searches:

- \(K \in \{4,5\}\),
- widths,
- dropout,
- learning rates,
- moderate ranges around the successful indicator run.

### `redesign_generator_push_4_5`

Purpose:

- search a more aggressive generator-responsive regime.

This scenario is useful if the generator is still too quiet even after the redesign.

### `redesign_disentangle_4_5`

Purpose:

- search stronger age/process separation regimes.

This is for testing whether cleaner factor separation helps downstream interpretability more than it hurts effect size.

### `redesign_stability_4_5`

Purpose:

- search a more conservative, reproducibility-oriented regime.

Useful when effects look plausible but move around too much across repetitions.

### `redesign_mixed_latent_4_5`

Purpose:

- test whether mixed sampled processes outperform one-hot process supervision.

### `redesign_directional_5`

Purpose:

- search only within mild shrinkage-biased models.

This is for testing a directional prior, not for defining the universal default.

## 6. Selection Metrics Used by Scenarios

The redesign scenarios use one of:

- `collapse_aware_quality_score`
- `directional_quality_score`

### Collapse-aware quality

$$
\mathrm{collapse\_aware\_quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{collapse\_aware\_latent\_sensitivity\_score}
 - 0.75 \cdot C_{latent}.
$$

Use this when you want:

- strong age disentanglement,
- a responsive generator,
- low process collapse.

### Directional quality

$$
\mathrm{directional\_quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{directional\_latent\_sensitivity\_score}
 - 0.5 \cdot C_{latent}.
$$

Use this when you explicitly want to favor shrinkage-biased models.

## 7. What the Overall Selection Algorithm Is

This is the part that is easiest to lose track of, so here it is explicitly.

### Within one training run

Suppose you run one config.

That config trains:

$$
R
$$

repetitions.

For each repetition:

1. train for \(E\) epochs,
2. evaluate on validation after each epoch,
3. choose the best epoch by `training.monitor_metric`.

Then:

4. collect the validation predictions from each repetition’s best epoch,
5. compute agreement across repetitions,
6. choose the repetition with the strongest mean agreement to the others,
7. declare that repetition’s best checkpoint the run winner.

So the selected checkpoint is:

- best epoch inside the selected repetition,
- where the repetition was chosen by agreement.

### Across many runs

Across many configs or Optuna trials, there is **not** one single universal scalar that decides everything scientifically.

In practice, the comparison should use:

1. validation selection metric,
2. agreement,
3. qualitative population patterns,
4. process distinctness,
5. downstream post-hoc interpretability.

So yes, agreement matters, but it is **not** the only criterion.

## 8. Are We Going To Run a Broader \(K\)-Sweep?

Probably yes, but **after** we settle the family and loss regime.

The scientifically defensible sequence is:

1. fix the architecture and loss family,
2. tune hyperparameters inside that family,
3. then run a broader \(K\)-sweep, for example:

$$
K \in \{3,4,5,6,7\}
$$

and compare:

- agreement,
- collapse-aware validation score,
- interpretability of isolated factor patterns,
- downstream post-hoc usefulness.

That is much cleaner than sweeping \(K\) while the whole model family is still changing.

## 9. Symmetry Prior

A hemisphere symmetry prior is **not** active in the default redesign scenarios.

Reason:

- age/global atrophy is often roughly bilateral,
- disease effects can be genuinely asymmetric,
- a strong symmetry prior could produce visually neat but biologically misleading factors.

If tested later, it should be a separate ablation, probably starting with the age basis only.

## 10. Which Scenario Should Be Treated as the Current Default?

Right now:

- `redesign_balanced_5` is the main reference train scenario,
- `redesign_balanced_4_5` is the main broad tune scenario.

The others are targeted ablations around specific failure modes:

- quiet generator,
- latent collapse,
- over-regularization,
- mixed-process supervision,
- directional sign priors.

That is the scientifically intended interpretation of the scenario matrix.
