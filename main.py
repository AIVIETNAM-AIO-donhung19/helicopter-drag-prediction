
import json
import torch
import pandas as pd
import joblib

from pathlib import Path

from src.data_preprocessing import (
    convert_to_torch_tensors,
    split_features_target,
    split_data,
    normalize_data
)
from src.data_generate import generate_dataset
from src.data_loader import create_dataset
from src.data_loader import create_data_loaders 
from src.model import MLP
from src.train import train_model
from src.test import test_model
from src.visualization import plot_training_history


# ==========================================
# 1. Configuration
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data/generated/helicopter_drag.csv"

CHECKPOINT_DIR = ROOT_DIR / "results/checkpoints"
METRICS_DIR = ROOT_DIR / "results/metrics"
FIGURES_DIR = ROOT_DIR / "results/figures"

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

N_SAMPLES = 6000
TARGET_COLUMN = "Total_drag"

design_space = {
    "rho": [1.1, 1.125],
    "f_factor": [0.2, 1.5],
    "weight": [5000, 50000],
    "k_factor": [1.1, 1.5],
    "rotor_radius": [3.5, 10],
    "velocity": [20, 200]
}


# ==========================================
# 2. Main pipeline
# ==========================================

def main():

    # Create output directories
    for directory in [CHECKPOINT_DIR, METRICS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------
    # Step 0: Generate dataset
    # --------------------------------------

    print("[0/7] Generating dataset...")

    df = generate_dataset(
        design_space=design_space,
        n_samples=N_SAMPLES
    )

    # Save generated dataset
    df.to_csv(DATA_PATH, index=False)

    # --------------------------------------
    # Step 1: Load dataset
    # --------------------------------------

    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # --------------------------------------
    # Step 2: Split features and target
    # --------------------------------------

    print("\n[2/7] Splitting features and target...")

    X, y = split_features_target(
        df,
        target_column=TARGET_COLUMN
    )

    # --------------------------------------
    # Step 3: Split train, validation, test
    # --------------------------------------

    print("\n[3/7] Splitting dataset...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=0.3,
        random_state=RANDOM_STATE
    )

    # --------------------------------------
    # Step 4: Normalize features
    # --------------------------------------

    print("\n[4/7] Normalizing data...")

    X_train, X_val, X_test = normalize_data(
        X_train, X_val, X_test)

    X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = convert_to_torch_tensors(X_train, X_val, X_test, y_train, y_val, y_test)

    # --------------------------------------
    # Step 5: Create DataLoaders
    # --------------------------------------

    print("\n[5/7] Creating DataLoaders...")

    dataset_train, dataset_val, dataset_test = create_dataset(
        X_train_tensor, y_train_tensor,
        X_val_tensor, y_val_tensor,
        X_test_tensor, y_test_tensor
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_train,
        dataset_val,
        dataset_test,
        batch_size=BATCH_SIZE)

    # --------------------------------------
    # Step 6: Initialize and train model
    # --------------------------------------

    print("\n[6/7] Initializing model and training...")

    input_dims = X_train.shape[1]

    model = MLP(input_dims=input_dims).to(device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        checkpoint_path=checkpoint_path
    )

    # Save training history
    pd.DataFrame(history).to_csv(
        METRICS_DIR / "training_history.csv",
        index=False
    )

    # Plot training history
    plot_training_history(history)

    # --------------------------------------
    # Step 7: Test best model
    # --------------------------------------

    print("\n[7/7] Evaluating best model on test set...")

    test_metrics = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path
    )

    # Save test metrics
    test_metrics = {
    key: float(value)
    for key, value in test_metrics.items()
    }

    with open(METRICS_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
