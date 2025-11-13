# Experiment Runners: LGP, Sudoku, and Real User Training

This repository contains scripts to run experiments for learning preferences in two domains: LGP (Logic Grid Puzzles) and Sudoku. The main scripts for running these experiments are:

- `lgp_exp_runner.py` (for oracles)  
- `sudoku_exp_runner.py` (for oracles)  
- `real_user_sudoku_training.ipynb` (for real users)

⚠️ **Important:** Running these experiments requires a valid **Gurobi license**.

---

## Purpose

The experiments for the oracle can be configured to use different users, normalization strategies, step selection strategies, diversification strategies, and a random seed.

---

## `lgp_exp_runner.py` and `sudoku_exp_runner.py`

These scripts are used to perform preference elicitation experiments.

### Command-Line Arguments

Both scripts accept the following command-line arguments:

- `-u`, `--user`  
  **Type:** `int`  
  **Description:** User index or identifier used to select which user's weights to use.  
  **Default:** `0`

- `-n`, `--normalization`  
  **Type:** `int`  
  **Description:** Type of normalization to apply.  
  **Choices:**  
    - `0`: none  
    - `1`: nadir  
    - `2`: local  
    - `3`: global  
  **Default:** `1`

- `-s`, `--steps`  
  **Type:** `str`  
  **Description:** Step selection strategy.  
  **Choices:**  
    - `SMUS` — SES selection strategy  
    - `random`  
    - `standard` — Online selection strategy  
  **Default:** `standard`

- `-t`, `--type`  
  **Type:** `str`  
  **Description:** Diversification strategy for learning.  
  **Choices:**  
    - `baseline`  
    - `disjunction`  
    - `w_disjunction`  
    - `MACHOP`  
  **Default:** `baseline`

- `-f`, `--seed`  
  **Type:** `int`  
  **Description:** Random seed for reproducibility.  
  **Default:** `42`
### Example Usage

```bash
python lgp_exp_runner.py -u 0 -n 2 -s standard -t MACHOP -f 42
```
```bash
python sudoku_exp_runner.py -u 0 -n 2 -s standard -t MACHOP -f 0
```
---

## `real_user_sudoku_training.ipynb`

This Jupyter notebook allows for interactive training and evaluation of the preference learning system using real user data. By running all the cells sequentially, you can:

- Train the model using pairwise comparisons provided by a real user.
- Evaluate the learned preferences as described in the paper.

This workflow enables reproducible experiments and analysis of the preference elicitation process with real user feedback.

---

## Results and Graph Generation

- All experiments will be stored in the `results_AAAI` folder.
- The `graph_generation.py` script allows you to generate the graphs shown in the paper, or to get the information presented in the tables.

---
