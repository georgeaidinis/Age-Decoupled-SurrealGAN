# Model Architecture

This document describes the **current redesigned Age-Decoupled SurrealGAN architecture**.

## Goal

Given ROI volumetrics, the model should learn:

1. an explicit age axis,
2. several non-age process axes,
3. generator-side basis patterns that let each latent axis produce an interpretable ROI-space change.

## Inputs

The model trains on normalized ROI vectors:

$$
x' = \mathcal{N}(x), \qquad y' = \mathcal{N}(y)
$$

where \(\mathcal{N}\) is the REF-fitted z-score normalization saved during preprocessing.

At training time:

- \(x'\): a `ref` ROI vector
- \(y'\): a `tar` ROI vector
- \(a\): normalized target age
- \(r\): sampled process vector

## Generator

The generator first builds a context representation:

$$
h(x') = T(x')
$$

then predicts:

$$
H_{\text{age}}(x') = W_{\text{age}} h(x')
$$

$$
H_{\text{proc}}(x') =
\left(H_1(x'), \dots, H_K(x')\right)
= W_{\text{proc}} h(x').
$$

The final synthetic delta is:

$$
\Delta(x', a, r)
= a\,H_{\text{age}}(x') + \sum_{k=1}^{K} r_k H_k(x').
$$

and the synthetic target is:

$$
\hat y' = x' + \Delta(x', a, r).
$$

## Decomposer

The decomposer maps a target vector back into one age component and \(K\) process components:

$$
Q(z') =
\left(\tilde\Delta_{\text{age}}(z'),
\tilde\Delta_1(z'),
\dots,
\tilde\Delta_K(z')\right).
$$

This is applied both to synthetic targets and to real target subjects.

## Latent Reconstruction Heads

Each decomposed component is inverted back to a latent:

$$
\hat a = R_{\text{age}}(\tilde\Delta_{\text{age}})
$$

$$
\hat r_k = R_k(\tilde\Delta_k).
$$

The synthetic path uses this for sampled-latent supervision; the real-target path uses it for latent inference.

## Real-Target Inference Path

For a real target subject \(y'\):

$$
(\hat a(y'), \hat r(y')) =
\left(
R_{\text{age}}(Q_{\text{age}}(y')),
R_1(Q_1(y')),
\dots,
R_K(Q_K(y'))
\right).
$$

This produces the age latent and the `r`-indices saved in prediction CSVs.

## Discriminator

The discriminator is a standard ROI-space adversary:

$$
D: \mathbb{R}^d \rightarrow \mathbb{R}
$$

and is trained to classify real `tar` vectors as real and synthetic targets as fake.

## Age Adversary

To reduce age leakage into process latents, the model uses an age adversary on \(\hat r(y')\):

$$
A(\hat r(y')) \rightarrow \hat a_{adv}
$$

with gradient reversal so that:

- the adversary tries to predict age from process latents,
- the main model tries to make that prediction fail.

## Why This Is Better Than the Earlier Extension

The earlier extension had three structural problems:

1. it trained on raw ROI scale,
2. it used a concatenation generator,
3. it let the decomposer look organized even when the generator was silent.

The redesign addresses all three:

- normalized inputs,
- additive bases,
- direct generator-separation and collapse penalties.

## What the Sliders Mean Now

In the GUI:

- age slider changes \(a\),
- each process slider changes \(r_k\),
- subject mode uses the selected subject as the baseline,
- population mode uses precomputed isolated factor patterns.

The process sliders are now much closer to the intended semantics:

$$
r_k \uparrow \quad \Rightarrow \quad \text{increase contribution of } H_k(x').
$$

That was not reliably true in the older model family.
