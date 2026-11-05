import torch
import torch.nn as nn

from pathlib import Path

from data_pipeline import prepare_data
from model import MLP
from evaluate import evaluate
from utils import load_checkpoint


ROOT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = ROOT_DIR / "results/checkpoints/best_model.pth"


def test_model(model, test_loader, criterion, device, checkpoint_path):

    model, checkpoint = load_checkpoint(
        model, checkpoint_path, device
    )

    test_metrics = evaluate(
        model, test_loader, criterion, device
    )

    return test_metrics


def main():

    # 1. Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Testing on: {device}")

    # 2. Prepare test data
    _, _, test_loader, input_dims = prepare_data()

    # 3. Initialize model
    model = MLP(input_dims=input_dims).to(device)

    # 4. Define criterion
    criterion = nn.MSELoss()

    # 5. Evaluate model
    test_metrics = test_model(
        model,
        test_loader,
        criterion,
        device,
        CHECKPOINT_PATH
    )

    # 6. Print results
    print("\nTest Results:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")


if __name__ == "__main__":
    main()