import numpy as np
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
    def __init__(self, model, optimizer, device: str = "cuda", verbose: bool = True):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.metric_values = dict(train_loss=[])
        self.verbose = verbose

    def evaluate(self, y_true, y_pred, pos_label=0) -> dict:
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1}

        res["Precision"], res["Recall"], res["F1-Score"], _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label
        )

        return res

    def test(self, dataset):
        self.model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row
                X, y = X.to(self.device).float(), y.float()

                logits = self.model(X)
                scores = torch.sigmoid(logits)
                # logits / (logits + torch.abs(logits))
                scores[scores >= 0.5] = 1
                scores[scores < 0.5] = 0
                y_true.extend(y.tolist())
                y_pred.extend(scores.cpu().tolist())
        return y_true, y_pred

    def train(self, dataset_ldr, num_epochs):
        # Create pytorch's train data_loader
        train_loader = dataset_ldr

        # train num_epochs times
        for epoch in range(num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0

            with tqdm(range(len(train_loader)), disable=not self.verbose) as t:
                train_losses = []
                for i, row in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs = row[0].to(self.device, dtype=torch.float).float()
                    train_labels = row[1].to(self.device, dtype=torch.long).float()
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)
                    # compute loss
                    loss = self.loss_fn(train_outputs, train_labels.unsqueeze(1))

                    # Use autograd to compute the backward pass.
                    loss.backward()
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss="{:05.3f}".format(train_loss / (i + 1)))
                    t.update()

            # evaluate the model on validation data after each epoch
            self.metric_values["train_loss"].append(np.mean(train_losses))


class InstaCartDataset(Dataset):
    def __init__(self, data: np.array, test_ratio: float = 0.30, batch_size: int = 32, class_ratio: dict = None):
        self.name = self.__class__.__name__
        self.X = data[:, :-1]
        self.y = data[:, -1]
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.class_ratio = class_ratio
        self.n_samples = self.X.shape[0]
        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def loaders(self, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        train_set, test_set = self.train_test_split(self.test_ratio)
        train_ldr = DataLoader(dataset=train_set, batch_size=self.batch_size, num_workers=num_workers)
        test_ldr = DataLoader(dataset=test_set, batch_size=self.batch_size, num_workers=num_workers)
        return train_ldr, test_ldr

    def train_test_split(self, test_ratio: float):
        train_ratio = 1 - test_ratio
        train_idx, test_idx = [], []

        if self.class_ratio:
            for label, ratio in self.class_ratio.items():
                # Fetch and shuffle indices of a single class
                label_data_idx = np.where(self.y == label)[0]
                shuffled_idx = torch.randperm(len(label_data_idx)).long()

                # Generate training and test sets
                n_train_sample = int(len(label_data_idx) * train_ratio * ratio)
                train_idx.extend(label_data_idx[shuffled_idx[:n_train_sample]])
                test_idx.extend(label_data_idx[shuffled_idx[n_train_sample:]])
        else:
            shuffled_idx = torch.randperm(len(self.X)).long()
            n_train_sample = int(len(self.X) * train_ratio)
            train_idx.extend(shuffled_idx[:n_train_sample])
            test_idx.extend(shuffled_idx[n_train_sample:])

        train_set = Subset(self, train_idx)
        test_set = Subset(self, test_idx)

        return train_set, test_set


def main():
    available_models = [MLP5, MLP3, MLP2]
    df = pd.read_csv("./data/train_orders.csv")
    X = df.to_numpy()
    cls_0_ratio = (X[:, -1] == 0).sum() / len(X)
    cls_1_ratio = (X[:, -1] == 1).sum() / len(X)
    class_ratio = None # {0: cls_0_ratio, 1: cls_1_ratio}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 1
    n_runs = 1

    available_metrics = ["Precision", "Recall", "F1-Score"]
    results = {model.name: {metric_name: [] for metric_name in available_metrics} for model in available_models}

    for model_cls in available_models:
        for run in range(n_runs):
            # Partition des données aléatoires à chaque itération
            dataset = InstaCartDataset(X, class_ratio=class_ratio)
            train_ldr, test_ldr = dataset.loaders()
            # Entraînement
            model = model_cls(in_features=dataset.in_features)
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
        results,
        columns=available_metrics,
        index=[model.name for model in available_models]
    )
    summary_df.to_csv("results/instacart_results.csv")
    print(summary_df)


if __name__ == "__main__":
    main()
