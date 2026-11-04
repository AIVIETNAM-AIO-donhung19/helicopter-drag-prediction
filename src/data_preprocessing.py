
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


RANDOM_STATE = 42

def split_features_target(df, target_column="Total_drag"):
    """
    Separate input features (X) and target variable (y).
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def split_data(X, y, test_size=0.3, random_state=RANDOM_STATE):
    """
    Split dataset into Train, Validation, and Test sets.

    Default split ratio: 70% / 15% / 15%.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(X_train, X_val, X_test):
    """
    Normalize features using StandardScaler.

    The scaler is fitted on training data only.
    """

    scaler_X = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)  

    return X_train, X_val, X_test


def convert_to_torch_tensors(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Convert numpy arrays to PyTorch tensors.
    """

    X_train_tensor = torch.from_numpy(X_train).float()                      # [4200, 6]
    y_train_tensor = torch.from_numpy(y_train.values).float().view(-1, 1)   # [4200, 1]

    X_val_tensor = torch.from_numpy(X_val).float()                      # [900, 6]
    y_val_tensor = torch.from_numpy(y_val.values).float().view(-1, 1)   # [900, 1]

    X_test_tensor = torch.from_numpy(X_test).float()                      # [900, 6
    y_test_tensor = torch.from_numpy(y_test.values).float().view(-1, 1)   # [900, 1]

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor