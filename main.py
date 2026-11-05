
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
from src.data_pipeline import prepare_data


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

    print("[0/3] Generating dataset...")

    df = generate_dataset(
        design_space=design_space,
        n_samples=N_SAMPLES
    )

    # Save generated dataset
    df.to_csv(DATA_PATH, index=False)

    # --------------------------------------
    # Step 1: Processing dataset
    # --------------------------------------

    print("\n[1/3] Processing dataset...")

    train_loader, val_loader, test_loader, input_dims = prepare_data()

    # --------------------------------------
    # Step 2: Initialize and train model
    # --------------------------------------

    print("\n[2/3] Initializing model and training...")


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
    # Step 3: Test best model
    # --------------------------------------

    print("\n[3/3] Evaluating best model on test set...")

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
