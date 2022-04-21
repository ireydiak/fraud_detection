import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Union
from sklearn import metrics as sk_metrics
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import trange


class BaseTrainer(ABC):

    def __init__(self, model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda",
                 verbose=False):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = self.set_optimizer()
        self.verbose = verbose

    @abstractmethod
    def train_iter(self, sample: torch.Tensor):
        pass

    @abstractmethod
    def score(self, sample: torch.Tensor):
        pass

    def after_training(self):
        """
        Perform any action after training is done
        """
        pass

    def before_training(self, dataset: DataLoader):
        """
        Optionally perform pre-training or other operations.
        """
        pass

    def set_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, dataset: DataLoader):
        self.model.train()

        self.before_training(dataset)

        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for sample in dataset:
                X, _ = sample
                X = X.to(self.device).float()

                # Reset gradient
                self.optimizer.zero_grad()

                loss = self.train_iter(X)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                if self.verbose:
                    print("Epoch {}: loss: {:05.3f}".format(epoch + 1, epoch_loss))
        self.after_training()

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())

        return np.array(y_true), np.array(scores)

    def get_params(self) -> dict:
        return {
            "learning_rate": self.lr,
            "epochs": self.n_epochs,
            "batch_size": self.batch_size,
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

    def evaluate(self, scores, y_true, pos_label=1, nq=100):
        ratio = 100 * sum(y_true == 0) / len(y_true)
        q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
        thresholds = np.percentile(scores, q)

        result_search = []
        confusion_matrices = []
        f1 = np.zeros(shape=nq)
        r = np.zeros(shape=nq)
        p = np.zeros(shape=nq)
        auc = np.zeros(shape=nq)
        aupr = np.zeros(shape=nq)

        for i, (thresh, qi) in enumerate(zip(thresholds, q)):
            # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
            # Prediction using the threshold value
            y_pred = (scores >= thresh).astype(int)
            y_true = y_true.astype(int)

            accuracy = sk_metrics.accuracy_score(y_true, y_pred)
            precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
                y_true, y_pred, average="binary", pos_label=pos_label
            )
            avgpr = sk_metrics.average_precision_score(y_true, scores)
            roc = sk_metrics.roc_auc_score(y_true, scores)
            cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])
            confusion_matrices.append(cm)
            result_search.append([accuracy, precision, recall, f_score])
            f1[i] = f_score
            r[i] = recall
            p[i] = precision
            auc[i] = roc
            aupr[i] = avgpr

        arm = np.argmax(f1)

        return {
            "Precision": p[arm],
            "Recall": r[arm],
            "F1-Score": f1[arm],
            "AUPR": aupr[arm],
            "AUROC": auc[arm],
            "Thresh_star": thresholds[arm]
        }


class BaseShallowTrainer(ABC):

    def __init__(self, model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda"):
        """
        Parameters are mostly ignored but kept for better code consistency

        Parameters
        ----------
        model
        batch_size
        lr
        n_epochs
        n_jobs_dataloader
        device
        """
        self.device = None
        self.model = model
        self.batch_size = None
        self.n_jobs_dataloader = None
        self.n_epochs = None
        self.lr = None

    def train(self, dataset: DataLoader):
        self.model.clf.fit(dataset.dataset.dataset.X)

    def score(self, sample: torch.Tensor):
        return self.model.predict(sample.numpy())

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        y_true, scores = [], []
        for row in dataset:
            X, y = row
            score = self.score(X)
            y_true.extend(y.cpu().tolist())
            scores.extend(score)

        return np.array(y_true), np.array(scores)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

    def evaluate(self, scores, y_true, pos_label=1, nq=100):
        ratio = 100 * sum(y_true == 0) / len(y_true)
        q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
        thresholds = np.percentile(scores, q)

        result_search = []
        confusion_matrices = []
        f1 = np.zeros(shape=nq)
        r = np.zeros(shape=nq)
        p = np.zeros(shape=nq)
        auc = np.zeros(shape=nq)
        aupr = np.zeros(shape=nq)

        for i, (thresh, qi) in enumerate(zip(thresholds, q)):
            # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
            # Prediction using the threshold value
            y_pred = (scores >= thresh).astype(int)
            y_true = y_true.astype(int)

            accuracy = sk_metrics.accuracy_score(y_true, y_pred)
            precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
                y_true, y_pred, average="binary", pos_label=pos_label
            )
            avgpr = sk_metrics.average_precision_score(y_true, scores)
            roc = sk_metrics.roc_auc_score(y_true, scores)
            cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])
            confusion_matrices.append(cm)
            result_search.append([accuracy, precision, recall, f_score])
            f1[i] = f_score
            r[i] = recall
            p[i] = precision
            auc[i] = roc
            aupr[i] = avgpr

        arm = np.argmax(f1)

        return {
            "Precision": p[arm],
            "Recall": r[arm],
            "F1-Score": f1[arm],
            "AUPR": aupr[arm],
            "AUROC": auc[arm],
            "Thresh_star": thresholds[arm]
        }
