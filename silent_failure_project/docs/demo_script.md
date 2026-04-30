# Demo Narration Script
## Silent Failure Detection for Clinical Machine Learning
### Estimated duration: 4–5 minutes (~620 words at normal speaking pace)

---

## SLIDE 1 — Title  *(~20 seconds)*

"Welcome. This project tackles a deceptively dangerous problem in clinical AI: **silent model failure** — when a model's accuracy degrades without triggering any alert.

We built an end-to-end pipeline comparing four uncertainty quantification methods across two clinical datasets and three failure modes."

---

## SLIDE 2 — The Problem  *(~40 seconds)*

"Imagine a diabetes-risk model scoring 87% accuracy at deployment. Six months later, the hospital's data pipeline shifts — a sensor recalibrates, a lab vendor changes its encoding. The model keeps producing predictions. Clinicians keep trusting them. Accuracy has quietly fallen to 72%. No alert fired.

Our question: can the model's own *uncertainty* sense something is wrong — and warn us *before* the damage is done?

We test three scenarios: covariate shift, label noise, and feature missingness."

---

## SLIDE 3 — Our Approach  *(~50 seconds)*

"We compare four methods, each producing a per-sample uncertainty score.

**Conformal prediction** via MAPIE gives assumption-free coverage guarantees using maximum softmax confidence.

**NGBoost** is a gradient-boosted probabilistic model outputting Bernoulli entropy natively.

**MC Dropout MLP** runs 100 stochastic forward passes and measures prediction variance.

**TabTransformer** applies self-attention to tabular features. Key design decision: we use *entropy of the mean prediction* rather than MC variance — because residual connections and LayerNorm structurally suppress dropout noise in transformer architectures.

All four streams feed a single alarm: three consecutive KS-test rejections at p less than 0.05."

---

## SLIDE 4 — The Experiment  *(~30 seconds)*

"We sweep severity alpha from 0.0 — clean data — to 0.9 in steps of 0.1. That's 2 datasets times 3 failure modes times 10 levels: 240 measurements per method.

Covariate shift is **detectable**: the feature distribution really shifts and the KS test catches it. Label noise and feature missingness are **not detectable** by uncertainty alone — understanding *why* is one of this project's key contributions."

---

## SLIDE 5 — Key Results  *(~45 seconds)*

"The numbers: on Pima Diabetes, XGBoost achieves 70.1% accuracy and 79% AUC. Cleveland Heart Disease reaches 86.7% and 89.4%.

For Pima covariate shift, all four methods fire an alarm — but all fire *late*. NGBoost and TabTransformer alarm at severity 0.7, two steps after the accuracy drop at 0.5 — a delay of plus 0.2. Conformal and MC Dropout alarm at 0.8, delay plus 0.3.

This is the honest finding: the KS-test alarm detects covariate shift but does not give early warning. For label noise and feature missingness, no method fires — and we document the root cause for each."

---

## SLIDE 6 — Interface Demo  *(~40 seconds)*

"The Streamlit dashboard at localhost 8501 ties this together. The sidebar selects dataset, failure mode, alarm threshold, and methods.

Panel 1 is an interactive Plotly chart — one trace per method — with an alarm threshold line, accuracy-drop marker, and shaded detection window.

Panel 2 is a colour-coded delay table: green for early warning, yellow for slight delay, red for missed.

Panel 3 shows live status badges — at covariate shift severity 0.5, all four methods correctly show alarm.

The Rerun Evaluation button re-executes the full sweep without touching a terminal."

---

## SLIDE 7 — Responsible AI  *(~35 seconds)*

"We are explicit about limitations. Label noise is *invisible* to feature-derived uncertainty — you need calibration-error monitoring instead. Feature missingness after StandardScaler equals mean imputation, so no shift is visible. Cleveland's 90-sample test set leaves the KS test underpowered.

Recommendations: use sentinel imputation, lower the consecutive-rejection threshold for small datasets, and trigger human review at the *first* warning — not the third."

---

## SLIDE 8 — Conclusion  *(~25 seconds)*

"In summary: a fully reproducible, config-driven pipeline with four UQ methods, three failure modes, and one honest answer — uncertainty can detect covariate shift, but arrives late and is blind to label corruption.

Four next steps form the roadmap: sentinel imputation, calibration monitoring, subgroup fairness analysis, and a prospective clinical trial. Thank you."
