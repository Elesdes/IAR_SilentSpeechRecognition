import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils.dataset_utils import TimeSeriesDataset
from tqdm import tqdm


class EMGRawDataset:
    """
    A class to load, preprocess, extract features, and create datasets from raw EMG data stored in .npy files.

    Methods
    -------
    load_and_preprocess_data(max_files)
        Loads EMG data and applies specified preprocessing steps.
    create_dataset(input_length, target_length)
        Creates a PyTorch TimeSeriesDataset from the preprocessed data.
    """

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: Path to the root directory containing EMG .npy files.
        """
        self.root_dir = root_dir
        self.data = None

    def load_and_preprocess_data(
        self,
        max_files: int = None,
    ) -> np.ndarray:
        """
        Loads EMG data from .npy files.

        Args:
            max_files: Maximum number of files to load. If None, load all files.

        Returns:
            Loaded EMG data as a numpy array.
        """
        file_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".npy") and "_emg" in file:
                    file_paths.append(os.path.join(root, file))

        if not file_paths:
            raise FileNotFoundError(
                f"No .npy files with '_emg' found in {self.root_dir} or its subdirectories."
            )

        # Limit number of files if max_files is specified
        if max_files is not None:
            file_paths = file_paths[:max_files]

        data_list = []
        for path in tqdm(file_paths, desc="Loading EMG files"):
            arr = np.load(path)
            data_list.append(arr)

        self.data = np.concatenate(data_list, axis=0)

        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)

        return self.data

    def create_dataset(
        self, input_length: int = 50, target_length: int = 50
    ) -> TimeSeriesDataset:
        """
        Creates a PyTorch TimeSeriesDataset using the loaded EMG data.

        Args:
            input_length: Number of timesteps per input segment.
            target_length: Number of timesteps per target segment.

        Returns:
            A TimeSeriesDataset ready for training.
        """
        if self.data is None:
            raise ValueError(
                "Data not loaded. Please run load_and_preprocess_data first."
            )

        return TimeSeriesDataset(self.data, input_length, target_length)
