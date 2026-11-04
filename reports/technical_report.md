from pathlib import Path

markdown = r"""# Helicopter Drag Prediction – Technical Report

## 1. Overview

This project develops a **Multi-Layer Perceptron (MLP)-based surrogate model** to predict helicopter drag from a set of input parameters.

The objective is to learn an approximation of the input–output relationship represented by the underlying [physical model / data source], enabling rapid drag prediction without repeatedly evaluating the original model.

**Method:**
1. Generate the dataset
2. Preprocess the input features, split the data and normalize data.
3. Train an MLP regression model.
4. Evaluate the trained model on the test set.

## 2. Dataset

### 2.1 Input and Output

The task is formulated as a supervised regression problem.

- **Input (\(X\)):** Air density (\(\rho\), kg/m³), flight velocity (\(V\), m/s), parasite drag factor (\(f\), m²), induced drag correction factor (\(k\), dimensionless), helicopter weight (\(W\), N), and main rotor radius (\(R\), m).

- **Output (\(y\)):** Total helicopter drag (\(D\), N), calculated as the sum of parasite drag (\(D_p\)) and induced drag (\(D_i\)).
- **Number of samples:** 6000
- **Number of input features:** 6

| Variable | Description | Unit | Range |
|---|---|---|---|
| `rho` | Air density | kg/m³ | 1.1–1.125 |
| `f_factor` | Equivalent flat plate area (parasite drag factor) | m² | 0.2–1.5 |
| `weight` | Helicopter weight | N | 5,000–50,000 |
| `k_factor` | Induced drag correction factor | Dimensionless | 1.1–1.5 |
| `rotor_radius` | Main rotor radius | m | 3.5–10 |
| `velocity` | Flight velocity | m/s | 20–200 |
| `D` | Total helicopter drag (\(D_p + D_i\)) | N | To be predicted |

### 2.2 Preprocessing

- **Data splitting:** Training [70%], validation [15%], and test [15%].
- **Normalization:** [Standardization].
- The normalization parameters are fitted on the training set only, then applied to the validation and test sets to prevent data leakage.
- **Random seed:** [42].

## 3. MLP Architecture and Training

An MLP is used to approximate the nonlinear mapping between the input parameters and the drag target.

### 3.1 Architecture

| Layer | Output dimension | Activation |
|---|---:|---|
| Input | 6 | — |
| Hidden layer 1 | 128 | ReLU |
| Hidden layer 2 | 64 | ReLU |
| Output | 1 | Linear |

### 3.2 Training Configuration

| Setting | Value |
|---|---|
| Loss function | [MSE] |
| Optimizer | [AdamW] |
| Learning rate | [1e-3] |
| Batch size | [32] |
| Epochs | [100] |
| Best-model selection | [lowest validation loss] |
| Compute device | [CPU] |

## 4. Results

The final model is evaluated on the held-out test set using regression metrics.

| Metric | Test result |
|---|---:|
| MAE | [798] |
| MSE | [2292258.5] |
| RMSE | [1514] |
| R² | [0.945] |

### 4.1 Visual Evaluation

![Training and validation loss and accurancy](figures/plot.jpeg)


### 4.2 Summary

### 4.2 Summary

The MLP model achieves an R² score of **0.945** and an RMSE of **1,514** on the test set, compared with an R² score of **0.900** and an RMSE of **2,037** on the training set. The comparable performance across the two datasets suggests that the model generalizes well to unseen data, with no clear evidence of overfitting.

These results demonstrate the MLP's ability to effectively approximate the input–output relationship represented by the underlying dataset.

---


