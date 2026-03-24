# Roll - 25CS60R41
# NAME - Shreyan Naskar

# Reproducibility Guide

This README explains how to recreate the results for Task A, B, C, and D.

## 1) Create virtual environment

### Windows (PowerShell)

```powershell
python -m venv env
```

### Linux / macOS

```bash
python3 -m venv env
```

## 2) Activate virtual environment

### Windows (PowerShell)

```powershell
.\env\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source env/bin/activate
```

## 3) Install dependencies from `req.txt`

```bash
pip install --upgrade pip
pip install -r req.txt
```

## 4) Run files in order

```bash
python Assignment1_25CS60R41_data_generator_taskA.py
python Assignment1_25CS60R41_classifier_taskB.py
python Assignment1_25CS60R41_classifier_taskC.py
python Assignment1_25CS60R41_classifier_taskD.py
```

## 5) Output locations

- Task A generated data:
  - `TaskA/data/raw/...`
  - `TaskA/data/splits/...`
- Task B results:
  - `TaskB/results/taskB_all_datasets.csv`
- Task C results:
  - `TaskC/results/taskC_all_datasets.csv`
- Task D results:
  - `TaskD/results/taskD_results.csv`

## 6) Deactivate environment (optional)

```bash
deactivate
```
