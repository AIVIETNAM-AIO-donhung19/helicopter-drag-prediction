
import torch
import torch.nn as nn
from src.evaluate import evaluate
from src.model import MLP
from src.utils import save_checkpoint


def initialize_training(
    input_dims,
    device,
    hidden1_dims=1024,
    hidden2_dims=512,
    learning_rate=1e-3
):
    """
    Initialize the MLP model, loss function, and optimizer.
    """

    # Initialize model
    model = MLP(
        input_dims=input_dims,
        hidden1_dims=hidden1_dims,
        hidden2_dims=hidden2_dims,
        output_dims=1
    ).to(device)

    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    return model, criterion, optimizer


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=100,
    checkpoint_path=None
):
    """
    Train the model and evaluate its performance
    on training and validation sets.
    """

    best_val_loss = float("inf")

    # Store training history
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_maes": [],
        "val_maes": [],
        "train_rmses": [],
        "val_rmses": [],
        "train_r2s": [],
        "val_r2s": []
    }

    for epoch in range(epochs):

        # =====================================
        # 1. TRAINING
        # =====================================

        model.train()

        for X_samples, y_samples in train_loader:

            X_samples = X_samples.to(device)
            y_samples = y_samples.to(device).float()

            optimizer.zero_grad()

            outputs = model(X_samples)

            # Ensure target and prediction have the same shape
            y_samples = y_samples.reshape_as(outputs)

            loss = criterion(outputs, y_samples)

            loss.backward()

            optimizer.step()

        # =====================================
        # 2. EVALUATION
        # =====================================

        train_metrics = evaluate(
        model, train_loader, criterion, device
        )

        train_loss = train_metrics["MSE"]
        train_mae = train_metrics["MAE"]
        train_rmse = train_metrics["RMSE"]
        train_r2 = train_metrics["R2"]

        val_metrics = evaluate(
        model, val_loader, criterion, device
        )

        val_loss = val_metrics["MSE"]
        val_mae = val_metrics["MAE"]
        val_rmse = val_metrics["RMSE"]
        val_r2 = val_metrics["R2"]

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                "results/checkpoints/best_model.pth"
        )

        # =====================================
        # 3. SAVE METRICS
        # =====================================

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)

        history["train_maes"].append(train_mae)
        history["val_maes"].append(val_mae)

        history["train_rmses"].append(train_rmse)
        history["val_rmses"].append(val_rmse)

        history["train_r2s"].append(train_r2)
        history["val_r2s"].append(val_r2)

        # =====================================
        # 4. PRINT RESULTS
        # =====================================

        print(
            f"\nEPOCH {epoch + 1}/{epochs}:"
            f"\n\tTrain Loss: {train_loss:.4f}"
            f"\tVal Loss: {val_loss:.4f}"
            f"\n\tTrain MAE: {train_mae:.4f}"
            f"\tVal MAE: {val_mae:.4f}"
            f"\n\tTrain RMSE: {train_rmse:.4f}"
            f"\tVal RMSE: {val_rmse:.4f}"
            f"\n\tTrain R2: {train_r2:.4f}"
            f"\tVal R2: {val_r2:.4f}"
        )

    return history