# When Do Clinical Tabular Models Fail Silently?
**A Temporal Deployment Benchmark with Controlled Failure Injection**

Nigamanth Rajagopalan · M.S. AI Systems, University of Florida · Spring 2026

---

## Overview

Clinical ML models are trained once and deployed into a world that keeps changing. This project builds a benchmark to detect *silent failure* — the period where a model's uncertainty estimates degrade before accuracy visibly drops.

Three uncertainty methods are compared under three controlled failure modes at ten severity levels:

| Uncertainty Method | Type |
|--------------------|------|
| Conformal Prediction (MAPIE) | Classical |
| NGBoost Entropy | Classical |
| MC Dropout MLP | **Deep Learning** |

| Failure Mode | Description |
|--------------|-------------|
| Covariate Shift | Skews feature distribution via importance resampling |
| Label Noise | Randomly flips a fraction of labels |
| Feature Missingness | Progressively masks a feature column |

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
Plots saved to `results/`

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
│   ├── models.py           # XGBoost baseline + MC Dropout MLP (PyTorch)
│   ├── uncertainty.py      # Conformal, NGBoost, MC Dropout uncertainty methods
│   └── alarm.py            # KS-test alarm logic + detection delay metric
├── ui/
│   └── app.py              # Streamlit interface
├── results/                # Output plots and CSV tables
├── notebooks/              # EDA and training walkthrough
├── run_experiments.py      # Full experimental sweep
├── plot_results.py         # Figure generation
└── requirements.txt
```

---

## Key Metric

**Detection Delay** = α_alarm − α_drop

- **Negative** → method warned *before* accuracy dropped (early warning ✓)
- **Positive** → method alerted *after* accuracy already dropped (too late ✗)
- **None** → method never raised an alarm

---

## Author

Nigamanth Rajagopalan · n.rajagopalan@u.edu  
University of Florida · EGN 6217 Applied Deep Learning
