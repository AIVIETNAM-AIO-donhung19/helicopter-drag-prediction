
import torch
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 32


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for helicopter drag prediction.
    """

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)


def create_dataset(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor):
    dataset_train = CustomDataset(X_train_tensor, y_train_tensor)
    dataset_val = CustomDataset(X_val_tensor, y_val_tensor)
    dataset_test = CustomDataset(X_test_tensor, y_test_tensor)

    return dataset_train, dataset_val, dataset_test


def create_data_loaders(
    dataset_train,
    dataset_val,
    dataset_test,
    batch_size=BATCH_SIZE,
):
    """
    Create DataLoaders for training, validation, and testing.
    """

    # Create DataLoaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
