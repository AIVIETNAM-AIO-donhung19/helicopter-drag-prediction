import torch
from src.evaluate import evaluate
from src.utils import load_checkpoint

def test_model(model, test_loader, criterion, device, checkpoint_path):
    model, checkpoint = load_checkpoint(
        model, checkpoint_path, device
    )

    test_metrics = evaluate(
        model, test_loader, criterion, device
    )

    return test_metrics