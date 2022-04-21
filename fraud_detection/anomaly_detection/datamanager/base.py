import numpy as np
import scipy.io
import torch
from abc import abstractmethod
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple
import pandas as pd


class AbstractDataset(Dataset):
    def __init__(self, path: str, **kwargs):
        self.name = None
        X = self._load_data(path)
        anomaly_label = self.get_anomaly_label()

        self.X = X[:, :-1]
        self.y = X[:, -1].astype(np.uint8)

        self.anomaly_ratio = (X[:, -1] == anomaly_label).sum() / len(X)
        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def _load_data(self, path: str):
        if path.endswith(".npz"):
            if self.npz_key() is not None:
                X = np.load(path)[self.npz_key()]
            else:
                X = np.load(path)
        elif path.endswith(".mat"):
            data = scipy.io.loadmat(path)
            X = np.concatenate((data['X'], data['y']), axis=1)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            X = df.to_numpy()
        elif path.endswith(".npy"):
            X = np.load(path)
        else:
            raise RuntimeError(f"Could not open {path}. Dataset can only read .npz and .mat files.")
        assert np.isnan(X).sum() == 0, "Found NaN values in data. Aborting"
        return X

    @staticmethod
    def get_anomaly_label():
        return 1

    @staticmethod
    def get_normal_label():
        return 0

    @abstractmethod
    def npz_key(self):
        return None

    def n_features(self):
        return self.X.shape[1]

    def shape(self):
        return self.X.shape

    def get_data_index_by_label(self, label):
        return np.where(self.y == label)[0]

    def loaders(self,
                test_pct: float = 0.5,
                label: int = 0,
                batch_size: int = 128,
                num_workers: int = 0,
                seed: int = None) -> (DataLoader, DataLoader):
        train_set, test_set = self.split_train_test(test_pct, label, seed)
        train_ldr = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers)
        test_ldr = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers)
        return train_ldr, test_ldr

    def split_train_test(self, test_pct: float = .5, label: int = 0, seed=None) -> Tuple[Subset, Subset]:
        assert (label == 0 or label == 1)

        if seed:
            torch.manual_seed(seed)

        # Fetch and shuffle indices of a single class
        label_data_idx = np.where(self.y == label)[0]
        shuffled_idx = torch.randperm(len(label_data_idx)).long()

        # Generate training set
        num_test_sample = int(len(label_data_idx) * test_pct)
        num_train_sample = int(len(label_data_idx) * (1. - test_pct))
        train_set = Subset(self, label_data_idx[shuffled_idx[num_train_sample:]])

        # Generate test set based on the remaining data and the previously filtered out labels
        remaining_idx = np.concatenate([
            label_data_idx[shuffled_idx[:num_test_sample]],
            np.where(self.y == int(not label))[0]
        ])
        test_set = Subset(self, remaining_idx)

        return train_set, test_set
