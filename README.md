# helicopter-drag-prediction

## 1. Overview

This project develops a **Multi-Layer Perceptron (MLP)-based surrogate model** to predict helicopter drag from a set of input parameters.

The objective is to learn an approximation of the input–output relationship defined by a physics-based drag model, enabling rapid drag prediction without repeatedly evaluating the original equations.

**Method:**
1. Generate the dataset by Latin Hypercube method.
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
| MAE | [720.93] |
| MSE | [1339636.9] |
| RMSE | [1157.4] |
| R² | [0.967] |

The model achieves an R² score of 0.967 and an RMSE of 1157.4 on the test set. Its performance is comparable to that on the training set (R² = 0.900, RMSE = 2,037), suggesting good generalization with no clear evidence of overfitting.

## 5. Running Experiments

Run the complete pipeline, including dataset generation, preprocessing, model training, and test evaluation:

```bash
python main.py
```

To evaluate a previously trained model without retraining:

```bash
python -m src.test
```

**Experiment outputs:**
- `results/checkpoints/`: Saved model checkpoints.
- `results/metrics/`: Training history and test metrics.
- `results/figures/`: Training and evaluation plots.
- `results/logs/`: Experiment logs.


---


