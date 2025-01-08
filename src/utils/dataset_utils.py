from typing import Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    A PyTorch dataset for time series data.

    Attributes
    ----------
    data : np.ndarray
        The normalized time series data.
    input_length : int
        Number of timesteps in the input sequence.
    target_length : int
        Number of timesteps in the target sequence.

    Methods
    -------
    __len__()
        Returns the total number of samples in the dataset.
    __getitem__(idx)
        Returns the input and target sequences for a given index.
    """

    def __init__(self, data: np.ndarray, input_length: int, target_length: int):
        self.data = data
        self.input_length = input_length
        self.target_length = target_length

    def __len__(self):
        return len(self.data) - self.input_length - self.target_length + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_length]
        y = self.data[
            idx + self.input_length : idx + self.input_length + self.target_length
        ]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


class DatasetUtils:
    """
    A utility class for splitting datasets and creating PyTorch data loaders.

    Methods
    -------
    split_and_create_loaders(dataset, batch_size, train_ratio, shuffle_train, shuffle_test, device)
        Splits the dataset into training and testing sets and returns DataLoaders.
    """

    def train_test_split(
        self,
        dataset: Any,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        shuffle_train: bool = True,
        shuffle_test: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Split dataset into training and testing sets and create data loaders.

        Parameters
        ----------
        dataset : Dataset
            The complete dataset to split. Must be a PyTorch Dataset.
        batch_size : int, optional
            Number of samples per batch (default is 32).
        train_ratio : float, optional
            Ratio of training samples (default is 0.8).
        shuffle_train : bool, optional
            Whether to shuffle the training dataset (default is True).
        shuffle_test : bool, optional
            Whether to shuffle the testing dataset (default is True).
        device : str, optional
            The device to load data on (default is "cpu").

        Returns
        -------
        train_loader : DataLoader
            DataLoader for the training set.
        test_loader : DataLoader
            DataLoader for the testing set.
        """
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        # Split the dataset
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle_test
        )

        print(
            f"Train size: {train_size}, Test size: {test_size}, Batch size: {batch_size}"
        )
        return train_loader, test_loader

    def dataset_to_numpy(self, dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        dataset: A PyTorch-style dataset yielding (x, y).
        Each (x, y) = ( [50, 3], [1, 3] ) for Lorenz.

        We'll flatten x into shape [50*3] or just use x[-1,:] if we want
        a pure single-step approach. However, typically for an ESN with single-step,
        we feed the final state or some condensed representation.

        For demonstration, let's feed the final row of x:
        - X[i] = x[i][ -1 ]  => shape (3,)
        - y[i] = y[i][  0 ]  => shape (3,)

        Then we get X, y as (n_samples, 3).
        """
        X_list = []
        y_list = []
        for x_tensor, y_tensor in dataset:
            # x_tensor shape = (50, 3), y_tensor shape = (1, 3)
            # We'll extract the last step of x, and the single step of y
            x_np = x_tensor[-1].cpu().numpy()  # shape (3,)
            y_np = y_tensor[0].cpu().numpy()  # shape (3,)

            X_list.append(x_np)
            y_list.append(y_np)

        X_arr = np.array(X_list)  # shape (n_samples, 3)
        Y_arr = np.array(y_list)  # shape (n_samples, 3)
        return X_arr, Y_arr
