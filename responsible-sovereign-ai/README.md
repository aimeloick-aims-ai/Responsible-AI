# Responsible & Sovereign AI — Fairness in Healthcare

> **Can we reduce group disparities without giving up useful predictive performance?**

This notebook studies that question on **16,000 healthcare records**. I use `Sex` only for fairness auditing and compare three modelling strategies:

**Logistic Regression → Fairlearn ThresholdOptimizer → Fairness-aware Neural Network**

---

## What the baseline hides

The baseline Logistic Regression looks strong at first glance:

| Metric | Score |
|---|---:|
| Accuracy | **0.848** |
| Precision | 0.837 |
| Recall | 0.786 |
| F1-score | 0.811 |

But similar overall accuracy masks very different error profiles.

<p align="center">
  <img src="assets/baseline_group_rates.png" width="760">
</p>

| Rate | Female | Male |
|---|---:|---:|
| Selection rate | 0.318 | **0.471** |
| True positive rate | 0.706 | **0.873** |
| False positive rate | 0.051 | **0.176** |

The model detects positives much more often for males, but it also produces substantially more false positives for them.

Baseline fairness gaps:

- **DPD = 0.153**
- **EOD = 0.167**

---

## Post-processing: a surprisingly strong result

I apply Fairlearn's `ThresholdOptimizer` with an **Equalized Odds** constraint.

Only **204 of 3,200 test predictions** change.

Yet:

| Metric | Before | After |
|---|---:|---:|
| Accuracy | 0.848 | **0.863** |
| Precision | 0.837 | **0.869** |
| Recall | 0.786 | 0.787 |
| F1-score | 0.811 | **0.826** |
| DPD ↓ | 0.153 | **0.023** |
| EOD ↓ | 0.167 | **0.016** |

<p align="center">
  <img src="assets/before_after_performance.png" width="680">
</p>

After mitigation, the group rates become much closer:

- TPR: **0.783 vs 0.791**
- FPR: **0.077 vs 0.092**
- Selection rate: **0.365 vs 0.388**

Here, fairness improves **without a predictive-performance penalty** on this split.

---

## Fairness during training

I also train a small neural network with a fairness penalty added directly to the loss:

\[
L = L_{\text{classification}} + \lambda L_{\text{fairness}}
\]

This produces an even smaller demographic-parity gap:

| Model | Accuracy | Recall | DPD ↓ | EOD ↓ |
|---|---:|---:|---:|---:|
| Standard NN | **0.848** | **0.784** | 0.157 | 0.171 |
| Fairness-aware NN | 0.822 | 0.749 | **0.002** | **0.019** |

<p align="center">
  <img src="assets/fairness_aware_nn.png" width="680">
</p>

Unlike ThresholdOptimizer, the in-processing approach pays a visible price in accuracy and recall.

---

## Main finding

The interesting result is not simply that *fairness can be improved*. It is that **how** fairness is introduced matters.

- **Baseline:** good aggregate performance, but large group-specific differences.
- **ThresholdOptimizer:** sharply reduces DPD/EOD while slightly improving predictive performance on this split.
- **Fairness-aware NN:** achieves near-zero demographic-parity difference, but sacrifices accuracy and recall.

So the practical question is not *accuracy or fairness?*  
It is **which intervention gives the trade-off appropriate for the application?**

---

## Stack

`Python` · `scikit-learn` · `Fairlearn` · `PyTorch` · `pandas` · `Matplotlib`

## Notebook

[`Responsible_and_Sovereign_AI_Tutorial_refined.ipynb`](Responsible_and_Sovereign_AI_Tutorial_refined.ipynb)

> **Note:** This is a technical fairness experiment on the protected groups recorded in the dataset. It is not a clinical validation or deployment recommendation.
