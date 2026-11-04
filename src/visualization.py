
import matplotlib.pyplot as plt


def plot_training_history(history):
    """
    Plot training and validation metrics over epochs.

    Metrics:
        - Loss
        - R2 Score
        - MAE
        - RMSE
    """

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ("losses", "Loss"),
        ("r2s", "R² Score"),
        ("maes", "MAE"),
        ("rmses", "RMSE")
    ]

    for i, (metric, title) in enumerate(metrics):

        row = i // 2
        col = i % 2

        ax[row, col].plot(
            history[f"train_{metric}"],
            label="Training",
            color="green"
        )

        ax[row, col].plot(
            history[f"val_{metric}"],
            label="Validation",
            color="orange"
        )

        ax[row, col].set_xlabel("Epoch")
        ax[row, col].set_ylabel(title)
        ax[row, col].set_title(title)
        ax[row, col].legend()
        ax[row, col].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
