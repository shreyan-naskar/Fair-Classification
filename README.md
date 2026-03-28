# Reproducibility Guide

This repository contains four standalone scripts for dataset preparation and fair-classification experiments. Run them in the order shown below to recreate the outputs for Tasks A, B, C, and D.

## 1) Create a virtual environment

### Windows (PowerShell)

```powershell
python -m venv env
```

### Linux / macOS

```bash
python3 -m venv env
```

## 2) Activate the virtual environment

### Windows (PowerShell)

```powershell
.\env\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source env/bin/activate
```

## 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r req.txt
```

## 4) Run the scripts in order

```bash
python generate_datasets.py
python covariance_constrained_classifier.py
python loss_constrained_classifier.py
python noisy_label_comparison.py
```

## 5) What each script does

- `generate_task_a_datasets.py`: generates the synthetic datasets, downloads the Adult dataset, preprocesses it, and writes the train/test splits used by later tasks.
- `run_task_b_covariance_constrained_classifier.py`: trains and evaluates the Task B covariance-constrained fair classifier on all prepared datasets.
- `run_task_c_loss_constrained_classifier.py`: trains and evaluates the Task C loss-constrained fair classifier on all prepared datasets.
- `run_task_d_noisy_label_comparison.py`: compares the Task B and Task C approaches on synthetic data with noisy labels.

## 6) Output locations

- Task A generated data: `TaskA/data/raw/...` and `TaskA/data/splits/...`
- Task B results: `TaskB/results/taskB_all_datasets.csv`
- Task C results: `TaskC/results/taskC_all_datasets.csv`
- Task D results: `TaskD/results/taskD_results.csv`

## 7) Deactivate the environment (optional)

```bash
deactivate
```
