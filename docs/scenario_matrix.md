# Scenario Matrix

This matrix describes the **redesign-era** train and tuning sweeps.

## Center of Gravity

The scenario family is centered on the strong indicator run:

- `20260409_221636_age-decoupled-surrealgan-redesign-train`

That run succeeded because it combined:

- REF-based ROI z-scoring
- sampled process supervision
- additive factorized generator
- direct generator-separation penalties
- collapse-aware model selection

## Selection Metrics

The redesign sweeps use one of:

- `collapse_aware_quality_score`
- `directional_quality_score`

with

$$
\mathrm{collapse\_aware\_quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{collapse\_aware\_latent\_sensitivity\_score}
 - 0.75 \cdot C_{latent}.
$$

This is deliberate: the earlier sweeps were overrewarding age decoupling while ignoring generator silence and process collapse.

## Train Scenarios

| Scenario | Core idea | Main changes |
| --- | --- | --- |
| `redesign_balanced_5` | Primary reference scenario | Matches the successful redesign run closely: \(K=5\), balanced age supervision, decomposition, generator responsiveness, and collapse penalties. |
| `redesign_balanced_4` | Same regime with fewer processes | Tests whether \(K=4\) improves stability or process distinctness without sacrificing signal. |
| `redesign_generator_push_5` | Louder generator | \(\uparrow \lambda_{adv}\), \(\uparrow \lambda_{proc\_sens}\), \(\uparrow \lambda_{gen\_sep}\), \(\downarrow \lambda_{id}\), \(\downarrow \lambda_{low\_id}\). |
| `redesign_disentangle_5` | Stronger age/process separation | \(\uparrow \lambda_{age\_adv}\), \(\uparrow \lambda_{cov}\), \(\uparrow \lambda_{proc\_age}\), \(\uparrow \lambda_{pair}\), \(\uparrow \lambda_{orth}\). |
| `redesign_stability_5` | Higher reproducibility pressure | More repetitions/epochs and slightly stronger reconstruction + decomposition + orthogonality. |
| `redesign_wide_5` | Higher capacity | Wider encoder/generator/decomposer trunks to test whether the current design is capacity-limited. |
| `redesign_mixed_latent_5` | Mixed process supervision | Sets `sampled_process_one_hot_only = false` to test whether multi-factor sampled targets improve process realism or just blur factors. |
| `redesign_directional_5` | Mild sign prior | Adds small age/process shrinkage terms and selects by `directional_quality_score`. |

## Tune Scenarios

| Scenario | Search hypothesis | Search emphasis |
| --- | --- | --- |
| `redesign_balanced_4_5` | Broad search around the successful redesign regime | Searches \(K \in \{4,5\}\), widths, dropout, and moderate loss ranges around the indicator run. |
| `redesign_generator_push_4_5` | The generator may still be underexpressive | Higher process sensitivity, stronger basis separation, lower identity/change constraints. |
| `redesign_disentangle_4_5` | Cleaner factors may help interpretability | Stronger age-adversary, covariance, orthogonality, and pairwise latent decorrelation. |
| `redesign_stability_4_5` | Process axes may need stronger structural regularization | Larger reconstruction/decomposition/orthogonality ranges and slightly longer training. |
| `redesign_mixed_latent_4_5` | One-hot latent sampling may be too restrictive | Allows mixed sampled processes and tunes the same collapse-aware objective. |
| `redesign_directional_5` | Mild biological sign prior may help realism | Keeps \(K=5\), tunes shrinkage losses, and selects by `directional_quality_score`. |

## Why These Families Exist

The redesign run already suggests the new architecture is directionally correct, but there are still scientific uncertainties:

1. Do we want four or five process factors?
2. Is the generator still too quiet?
3. Are process factors distinct enough?
4. Does one-hot process sampling help or hurt?
5. Is a mild directional shrinkage prior useful or overconstraining?

Each scenario isolates one of these questions instead of searching one giant undifferentiated space.

## What Not To Do

The current evidence argues against spending large compute on the old pre-redesign scenario families because they were dominated by one of two failure modes:

- silent generator
- correlated process collapse

Those older scenario TOMLs remain in the repo for reproducibility, but the cluster sweep files now point to the redesign families above.

## Symmetry Prior

A hemisphere symmetry prior is **not** included in the redesign matrix yet. That is a deliberate decision.

Reason:

- global aging effects are often bilateral,
- disease effects can be meaningfully asymmetric,
- adding a symmetry loss too early risks removing real signal before the process factors are stable.

If symmetry is tested later, it should be a targeted ablation, probably starting with the age basis only:

$$
L_{sym,age} =
\mathrm{mean}_{(i,j) \in \mathcal{P}}
\left|H_{\text{age},i}(x) - H_{\text{age},j}(x)\right|
$$

where \(\mathcal{P}\) is the set of left/right ROI pairs.

## Sweep Files

The active SLURM lists are:

- [`scripts/slurm/train_scenarios.txt`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/train_scenarios.txt)
- [`scripts/slurm/tune_scenarios.txt`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/tune_scenarios.txt)

They now point to the redesign-era configs rather than the older pre-redesign families.
