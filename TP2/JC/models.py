import numpy as np
import sklearn.model_selection
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
from sklearn import metrics as sk_metrics
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler, Subset

from torch import nn, optim
from tqdm import tqdm
import sklearn.metrics as sk_metrics

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple
from sklearn.model_selection import train_test_split


class InstaCartDataset(Dataset):
    def __init__(self, data: np.array, val_ratio: float = 0.1, test_ratio: float = 0.33, batch_size: int = 32,
                 class_ratio=None):
        self.name = self.__class__.__name__
        # self.X = data[:, :-1]
        # self.y = data[:, -1]
        self.X = data
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size

        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def loaders(self, num_workers: int = 4, seed: int = None) -> Tuple[DataLoader, DataLoader]:
        X_train, X_test = train_test_split(self.X, test_size=self.test_ratio)
        train_ldr = DataLoader(dataset=X_train, batch_size=self.batch_size, num_workers=num_workers)
        test_ldr = DataLoader(dataset=X_test, batch_size=self.batch_size, num_workers=num_workers)
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


class MLP5(nn.Module):
    name = "MLP5"

    def __init__(self, in_features: int, n_class: int = 2, p_dropout=0.1):
        super(MLP5, self).__init__()

        self.in_layer = nn.Linear(in_features, 1024)
        self.h1 = nn.Linear(1024, 1024)
        self.h2 = nn.Linear(1024, 512)
        self.h3 = nn.Linear(512, 256)
        self.h4 = nn.Linear(256, 128)
        self.h5 = nn.Linear(128, 64)
        self.out_layer = nn.Linear(64, n_class - 1)
        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor):
        out = self.relu(self.in_layer(X))
        out = self.relu(self.h1(out))
        out = self.relu(self.h2(out))
        out = self.relu(self.h3(out))
        out = self.relu(self.h4(out))
        out = self.relu(self.h5(out))
        out = self.dropout(out)
        out = self.out_layer(out)
        return out


class MLP3(nn.Module):
    name = "MLP3"

    def __init__(self, in_features: int, n_class: int = 2, p_dropout=0.1):
        super(MLP3, self).__init__()

        self.in_layer = nn.Linear(in_features, 256)
        self.h1 = nn.Linear(256, 256)
        self.h2 = nn.Linear(256, 128)
        self.h3 = nn.Linear(128, 64)
        self.out_layer = nn.Linear(64, n_class - 1)
        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor):
        out = self.relu(self.in_layer(X))
        out = self.relu(self.h1(out))
        out = self.relu(self.h2(out))
        out = self.relu(self.h3(out))
        out = self.dropout(out)
        out = self.out_layer(out)
        return out


class MLP2(nn.Module):
    name = "MLP2"

    def __init__(self, in_features: int, n_class: int = 2, p_dropout=0.1):
        super(MLP2, self).__init__()

        self.in_layer = nn.Linear(in_features, in_features // 2)
        self.h1 = nn.Linear(in_features // 2, in_features // 2)
        self.h2 = nn.Linear(in_features // 2, in_features // 4)
        self.out_layer = nn.Linear(in_features // 4, n_class - 1)
        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor):
        out = self.relu(self.in_layer(X))
        out = self.relu(self.h1(out))
        out = self.relu(self.h2(out))
        out = self.dropout(out)
        out = self.out_layer(out)
        return out


class MLPTrainer:
    def __init__(self, model, optimizer, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = nn.BCEWithLogitsLoss()

    def test(self, dataset):
        self.model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for row in dataset:
                # X, y = row
                # X, y = X.to(self.device).float(), y.float()
                X = row[:, :-1].to(self.device).float()
                y = row[:, -1].to(self.device).float()

                logits = self.model(X)
                scores = torch.sigmoid(logits)
                # logits / (logits + torch.abs(logits))
                scores[scores >= 0.7] = 1
                scores[scores < 0.3] = 0
                y_true.extend(y.tolist())
                y_pred.extend(scores.cpu().squeeze(1).tolist())
        return np.array(y_true, dtype=np.int8), np.array(y_pred, dtype=np.int8)

    def evaluate(self, y_true, y_pred, pos_label=0) -> dict:
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1}

        res["Precision"], res["Recall"], res["F1-Score"], _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label
        )

        return res

    def train(self, dataset_ldr, num_epochs):
        """
        Train the model for num_epochs times
        Args:
            num_epochs: number times to train the model
        """

        # Create pytorch's train data_loader
        train_loader = dataset_ldr

        # train num_epochs times
        for epoch in range(num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0
            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs = data[:, :-1].to(self.device).float()
                    train_labels = data[:, -1].to(self.device).float().unsqueeze(1)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)
                    # compute loss
                    loss = self.loss_fn(train_outputs, train_labels)

                    # Use autograd to compute the backward pass.
                    loss.backward()
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

            # evaluate the model on validation data after each epoch
            # self.metric_values['train_loss'].append(np.mean(train_losses))


def main():
    available_models = [MLP5, MLP3]
    df = pd.read_csv("./data/train_orders.csv")
    X = df.to_numpy()
    cls_0_ratio = (X[:, -1] == 0).sum() / len(X)
    cls_1_ratio = (X[:, -1] == 1).sum() / len(X)
    class_ratio = {0: cls_0_ratio, 1: cls_1_ratio}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 200
    n_runs = 10

    available_metrics = ["Precision", "Recall", "F1-Score"]
    results = {model.name: {metric_name: [] for metric_name in available_metrics} for model in available_models}

    for model_cls in available_models:
        for run in range(n_runs):
            # Partition des données aléatoires à chaque itération
            dataset = InstaCartDataset(X, test_ratio=0.5, class_ratio=class_ratio)
            train_ldr, test_ldr = dataset.loaders()
            # Entraînement
            model = model_cls(in_features=dataset.in_features-1)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            trainer = MLPTrainer(model=model, optimizer=optimizer, device=device)
            trainer.train(train_ldr, n_epochs)
            # Évaluation
            y_true, y_pred = trainer.test(test_ldr)
            run_results = trainer.evaluate(y_true, y_pred)
            print("Results for %s" % model_cls.name)
            print(run_results)
            for metric_name in available_metrics:
                results[model_cls.name][metric_name].append(run_results[metric_name])

    full_results = []
    for model_results in results.values():
        full_results.append(["{:2.4f} ({:2.4f})".format(np.mean(res), np.std(res)) for res in model_results.values()])

    summary_df = pd.DataFrame(
        full_results,
        columns=available_metrics,
        index=[model.name for model in available_models]
    )
    summary_df.to_csv("results/instacart_results.csv")
    print(summary_df)


if __name__ == "__main__":
    main()
