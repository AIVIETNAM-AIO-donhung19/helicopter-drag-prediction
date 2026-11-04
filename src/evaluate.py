import torch
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score


def evaluate(model, loader, criterion, device):
    """
    Evaluate model performance on a given DataLoader.

    Returns:
        avg_loss, mae, rmse, r2
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for X_samples, y_samples in loader:

            X_samples = X_samples.to(device)
            y_samples = y_samples.to(device).float()

            outputs = model(X_samples)

            # Ensure target and prediction have the same shape
            y_samples = y_samples.reshape_as(outputs)

            loss = criterion(outputs, y_samples)

            batch_size = X_samples.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.extend(
                outputs.cpu().numpy().reshape(-1)
            )

            all_targets.extend(
                y_samples.cpu().numpy().reshape(-1)
            )

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")

    avg_loss = total_loss / total_samples

    mae = mean_absolute_error(all_targets, all_predictions)

    rmse = np.sqrt(
        np.mean(
            (np.array(all_targets) - np.array(all_predictions)) ** 2
        )
    )

    r2 = r2_score(all_targets, all_predictions)

    return {
    "MSE": float(avg_loss),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
    }