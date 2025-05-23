# LinearDAE NeurIPS Supplementary Code

This repository contains code to reproduce the experiments from our NeurIPS submission. Please follow the instructions below to set up the environment and run the experiments.

---

## 1. Environment Setup

### 1.1. Install Python

Make sure you have **Python 3.8 or later** installed.

### 1.2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 1.3. Install Required Packages

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 2. Running the Code

To reproduce specific figures from the paper, run:

```bash
python3 linear_dae.py -f 1  # Generates Figure 1
python3 linear_dae.py -f 2  # Generates Figure 2
python3 linear_dae.py -f 3  # Generates Figure 3
```

Each flag `-f` corresponds to a specific experiment setup.

You can adjust experiment parameters (e.g., rank, bottleneck dimension, stride) by modifying them directly in the function definitions within `linear_dae.py`.

---

## 3. Reproducibility Notes

- All experiments were conducted on a **T4 GPU using Google Colab**.
- Default parameters match those reported in the paper:
  - Rank: `r = 100`
  - Bottleneck dimension: `k = 50`
  - Test size: `4500`
  - Dataset: **CIFAR-10**
- For further details, refer to the **supplementary PDF** included in the submission.

---