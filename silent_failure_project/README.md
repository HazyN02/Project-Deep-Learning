# When Do Clinical Tabular Models Fail Silently?
**A Temporal Deployment Benchmark with Controlled Failure Injection**

Nigamanth Rajagopalan · M.S. AI Systems, University of Florida · Spring 2026

---

## Overview

Clinical ML models are trained once and deployed into a world that keeps changing. This project builds a benchmark to detect *silent failure* — the period where a model's uncertainty estimates degrade before accuracy visibly drops.

Four uncertainty methods are compared under three controlled failure modes at ten severity levels (α = 0.0 – 0.9):

| Uncertainty Method | Type | Uncertainty Signal |
|--------------------|------|--------------------|
| Conformal Prediction (MAPIE) | Classical | 1 − max softmax probability |
| NGBoost Entropy | Classical | Bernoulli predictive entropy |
| MC Dropout MLP | Deep Learning | Variance across 100 stochastic forward passes |
| **TabTransformer** | **Deep Learning** | Entropy of mean probability across 100 stochastic passes |

| Failure Mode | Description |
|--------------|-------------|
| Covariate Shift | Skews feature distribution via importance resampling |
| Label Noise | Randomly flips a fraction of labels |
| Feature Missingness | Progressively masks a feature column |

---

## Results

### Detection Delay (α_alarm − α_drop)

Negative = method warned *before* accuracy dropped (early warning). Positive = too late. None = alarm never fired.

**Pima Diabetes** — Baseline ACC=0.7013, AUC=0.7900

| Failure Mode | Conformal | NGBoost | MC Dropout | TabTransformer |
|---|---|---|---|---|
| Covariate Shift | +0.3 | **+0.2** | +0.3 | **+0.2** |
| Label Noise | None | None | None | None |
| Feature Missingness | None | None | None | None |

**Cleveland Heart Disease** — Baseline ACC=0.8667, AUC=0.8938

| Failure Mode | Conformal | NGBoost | MC Dropout | TabTransformer |
|---|---|---|---|---|
| Covariate Shift | None | None | None | None |
| Label Noise | None | None | None | None |
| Feature Missingness | None | None | None | None |

> **Key finding:** On Pima covariate shift, NGBoost and TabTransformer detect the failure at α=0.7 (one severity step earlier than Conformal and MC Dropout at α=0.8), both achieving detection delay +0.2 vs +0.3.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the full experiment sweep
```bash
python run_experiments.py
```
Results saved to `results/detection_delay_table.csv`

### Generate plots
```bash
python plot_results.py
```
Plots saved to `results/` — includes detection delay heatmaps, alarm vs. drop bar charts, and per-dataset training curves.

### Launch the interactive interface
```bash
streamlit run ui/app.py
```

---

## Repository Structure

```
├── src/
│   ├── data_loader.py      # UCI dataset loading + train/cal/test splits
│   ├── inject.py           # Failure injection pipeline
│   ├── models.py           # XGBoost + MC Dropout MLP + TabTransformer (PyTorch)
│   ├── uncertainty.py      # Conformal, NGBoost, MC Dropout, TabTransformer uncertainty
│   └── alarm.py            # KS-test alarm logic + detection delay metric
├── ui/
│   └── app.py              # Streamlit interface
├── docs/
│   └── interface_screenshot.png  # Streamlit UI evidence
├── results/                # Output plots and CSV tables
│   ├── detection_delay_table.csv
│   ├── heatmap_pima.png
│   ├── heatmap_cleveland.png
│   ├── alarm_vs_drop_pima.png
│   ├── alarm_vs_drop_cleveland.png
│   ├── training_curves.png
│   ├── training_curves_pima.png
│   └── training_curves_cleveland.png
├── run_experiments.py      # Full experimental sweep
├── run_qc.py               # Quality control checks
├── plot_results.py         # Figure generation
└── requirements.txt
```

---

## Key Metric

**Detection Delay** = α_alarm − α_drop

- **Negative** → method warned *before* accuracy dropped (early warning)
- **Positive** → method alerted *after* accuracy already dropped (too late)
- **None** → method never raised an alarm

---

## Known Limitations

**Label noise is undetectable by uncertainty-based monitoring.** All four methods return None for label noise on both datasets. This is expected: label flipping does not alter the input feature distribution, so feature-derived uncertainty estimates cannot capture it. Label noise monitoring requires ground-truth label access (e.g., spot-checking or delayed feedback loops).

**Feature missingness produces no detectable shift.** Zeroing out a feature column after StandardScaler normalization is equivalent to imputing the training-set mean, so the model sees no covariate shift and accuracy does not drop either. More realistic missing-data patterns (MNAR, block missingness) would be needed to trigger a detectable alarm.

**Cleveland KS test is underpowered.** The Cleveland test split contains approximately 90 samples. The three-consecutive-rejection KS alarm requires sufficient statistical power to detect distribution shifts at each severity level. With n~90, the alarm never fires even when accuracy drops at α=0.3. Larger held-out sets or a lower consecutive-rejection threshold (e.g., 2 instead of 3) would improve sensitivity.

**Alarm latency vs. consecutive rejection.** The KS alarm uses 3 consecutive p<0.05 rejections to suppress false positives, which adds systematic latency. This trades precision for recall — methods that alarm earlier may do so at the cost of more false positives on clean data.

---

## Author

Nigamanth Rajagopalan · n.rajagopalan@u.edu  
University of Florida · EGN 6217 Applied Deep Learning
