# BEP: Causal Counterfactual Robustness Experiments

Maria Magdalini Kantara
---
This repository contains the code for my Bachelor End Project (BEP), which investigates the relationship between **causal consistency** and **stability/robustness** in counterfactual explanations.

The main goal of the project is to compare **standard DiCE counterfactuals** with **causally constrained counterfactuals**. The experiments evaluate whether adding causal constraints makes counterfactual explanations more causally realistic, and whether this also improves their stability after local perturbations and model retraining.

---

## Project Structure

```text
bep-causal-counterfactuals/
│
├── counterfactual_algorithms_dice/
│   ├── CAUSAL_DICE.py
│   └── DICE.py
│
├── DATASET/
│   └── german.csv
│
├── GERMAN_CREDIT_EXPERIMENT/
│   ├── run_german_credit_causal.py
│   └── VISUALIZATIONS.ipynb
│
├── SIMPLE_BN_EXPERIMENT/
│   ├── run_simple_bn_causal.py
│   └── VISUALIZATIONS.ipynb
│
├── requirements.txt
└── README.md
```

---

## Folder Description

### `counterfactual_algorithms_dice/`

This folder contains the counterfactual generation algorithms used in the experiments.

* `DICE.py`
  Contains the standard DiCE genetic counterfactual generation method.

* `CAUSAL_DICE.py`
  Contains the modified causal version of DiCE. This version adds a causal penalty to the counterfactual generation objective, encouraging generated counterfactuals to follow the assumed structural causal model.

---

### `DATASET/`

This folder stores the datasets used in the experiments.

For the German Credit experiment, the main dataset is:

```text
german.csv
```

The German Credit dataset is used to train binary classification models and generate counterfactual explanations for credit decisions.

---

### `GERMAN_CREDIT_EXPERIMENT/`

This folder contains the full German Credit experiment.

The German Credit experiment uses a simple causal relation:

```text
Credit Amount -> Duration of Credit
```

The experiment trains two XGBoost models:

* **Model 1**: trained on the initial training set.
* **Model 2**: trained on a larger retraining set.

This setup is used to evaluate whether counterfactual explanations remain stable after the predictive model is updated.

The experiment compares:

* **Normal DiCE**
* **CausalDiCE** with different causal weights


The notebook `VISUALIZATIONS.ipynb` is used to analyze and visualize the generated CSV result files.

---

### `SIMPLE_BN_EXPERIMENT/`

This folder contains the synthetic Simple-BN experiment.

The Simple-BN dataset is generated from a known causal structure. This allows the experiment to evaluate causal consistency in a controlled setting where the true causal relation is known.

The experiment compares standard and causal counterfactuals across different causal weights and random seeds.



The notebook `VISUALIZATIONS.ipynb` is used to inspect and visualize the Simple-BN results.

---

## Experiment Overview

Each experiment follows the same general workflow:

1. Load or generate the dataset.
2. Split the data into training, retraining, and test sets.
3. Train two predictive models:

   * Model 1 on the initial training data.
   * Model 2 on the expanded retraining data.
4. Fit or define the structural causal model.
5. Generate standard DiCE counterfactuals.
6. Generate CausalDiCE counterfactuals for several causal weights.
7. Evaluate the generated counterfactuals.
8. Save detailed and aggregated results as CSV files.
9. Use the visualization notebooks to analyze the results.

---

## Main Evaluation Metrics

The experiments evaluate counterfactual explanations using several metrics.

### Validity

Validity checks whether the generated counterfactual successfully changes the model prediction to the desired class.

A counterfactual is considered valid if:

```text
model(counterfactual) = desired class
```

---

### Stability

Stability measures how reliable a counterfactual remains under small perturbations around the counterfactual instance.

In this project, local stability is measured as:

```text
mean predicted probability in the neighbourhood - standard deviation
```

A higher stability score means that nearby points around the counterfactual are also likely to receive the desired prediction.

---

### Causal Consistency

Causal consistency is measured using the SCM residual.

The SCM residual compares the generated counterfactual value of an endogenous feature with the value implied by the structural causal model.

A lower SCM residual means that the counterfactual follows the assumed causal relation more closely.

---

### Feature Changes

The experiments also measure how much the counterfactual differs from the original instance.

These metrics include:

* number of changed features,
* total numeric change,
* normalized numeric change,
* number of categorical changes.

These metrics help evaluate how large and interpretable the suggested counterfactual changes are.

---

### Robustness After Retraining

Robustness is evaluated by comparing the counterfactuals generated for Model 1 and Model 2.

If the counterfactuals change a lot after retraining, this suggests that the explanation is sensitive to model updates.

The comparison includes:

* number of features that differ between Model 1 and Model 2 counterfactuals,
* numeric magnitude of the differences,
* normalized numeric magnitude,
* categorical differences.

---

## Causal Weights

CausalDiCE is evaluated using multiple causal weights.

A causal weight controls how strongly the counterfactual generation process is penalized for violating the structural causal model.

A low causal weight gives more freedom to DiCE, while a high causal weight forces the generated counterfactuals to follow the causal relation more closely.

The German Credit experiment uses causal weights such as:

```python
[0.05, 0.1, 0.5, 0.7, 0.85, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
```

---


## How to Run the Experiments

First, install the required packages:

```bash
pip install -r requirements.txt
```

Then update the dataset and output paths inside the experiment script if needed.

For the German Credit experiment, run:

```bash
python GERMAN_CREDIT_EXPERIMENT/run_german_credit_causal.py
```

For the Simple-BN experiment, run:

```bash
python SIMPLE_BN_EXPERIMENT/run_simple_bn_causal.py
```

After running the scripts, open the corresponding `VISUALIZATIONS.ipynb` notebook to inspect the results.

---

## Requirements

All required packages should be listed in:

```text
requirements.txt
```

---

## Notes

The experiment scripts can take a long time to run because counterfactual generation is repeated for multiple seeds, causal weights, models, and test instances.

To avoid losing progress, the scripts save partial result files after each seed. This makes it possible to inspect intermediate outputs even if the full experiment is interrupted.

---

## Project Purpose

This project studies whether adding causal constraints to counterfactual explanations improves their robustness.

The results are used to analyze the trade-off between:

* causal consistency,
* local stability,
* validity,
* and robustness after model retraining.

The central question is whether more causally consistent counterfactual explanations are also more stable and robust, or whether improving one property can sometimes come at the cost of another.
