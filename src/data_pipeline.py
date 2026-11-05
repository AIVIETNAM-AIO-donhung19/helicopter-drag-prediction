import pandas as pd

from pathlib import Path

from data_preprocessing import (
    convert_to_torch_tensors,
    split_features_target,
    split_data,
    normalize_data
)

from data_loader import create_dataset, create_data_loaders


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data/generated/helicopter_drag.csv"

BATCH_SIZE = 32
RANDOM_STATE = 42
TARGET_COLUMN = "Total_drag"


def prepare_data():

    # 1. Load existing dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Please generate the dataset first."
        )

    df = pd.read_csv(DATA_PATH)

    # 2. Split features and target
    X, y = split_features_target(
        df,
        target_column=TARGET_COLUMN
    )

    # 3. Split train, validation, test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=0.3,
        random_state=RANDOM_STATE
    )

    # 4. Normalize data
    X_train, X_val, X_test = normalize_data(
        X_train, X_val, X_test
    )

    # 5. Convert to PyTorch tensors
    (
        X_train_tensor,
        X_val_tensor,
        X_test_tensor,
        y_train_tensor,
        y_val_tensor,
        y_test_tensor
    ) = convert_to_torch_tensors(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

    # 6. Create datasets
    dataset_train, dataset_val, dataset_test = create_dataset(
        X_train_tensor, y_train_tensor,
        X_val_tensor, y_val_tensor,
        X_test_tensor, y_test_tensor
    )

    # 7. Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_train,
        dataset_val,
        dataset_test,
        batch_size=BATCH_SIZE
    )

    input_dims = X_train.shape[1]

    return train_loader, val_loader, test_loader, input_dims