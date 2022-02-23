
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


def euclidean_dist(X: np.array, Y: np.array):
    ax = X.ndim - 1 if X.ndim > 1 else 0
    return np.linalg.norm(X - Y, axis=ax)

def manhattan_dist(X: np.array, Y: np.array):
    dist = np.abs(X - Y)
    ax = X.ndim - 1 if X.ndim > 1 else 0
    return dist.sum(axis=ax)

def mahalanobis_dist(X: np.array, Y: np.array, Sigma: np.array):
    pass

available_distances = {
    "euclidean": euclidean_dist,
    "manhattan": manhattan_dist,
}
#"mahalanobis": mahalanobis_dist

class NearestCentroid:
    def __init__(self, label_encoder: LabelEncoder, metric="euclidean"):
        m = available_distances.get(metric, None)
        assert m, "Implémentation de la métrique {} non fournie".format(metric)
        self.metric = m
        self.centroids = None
        self.le = label_encoder

    def fit(self, X: np.array, y: np.array) -> None:
        c = len(np.unique(y))
        y = self.le.transform(y)
        self.centroids = np.zeros((c, X.shape[1]))
        # calculer la moyenne de chaque attribut pour chaque classe
        for i in range(c):
            for j in range(X.shape[1]):
                self.centroids[i, j] = X[y == i][:, j].mean()

    def predict(self, X: np.array) -> np.array:
        assert self.centroids is not None, "Le modèle n'a pas encore été entraîné"
        dist = self.metric(np.expand_dims(X, 1), self.centroids)
        y_pred = np.argmin(dist, axis=1)
        return self.le.inverse_transform(y_pred)

    def evaluate(self, y_true: np.array, y_pred: np.array) -> pd.DataFrame:
        d = metrics.classification_report(y_true, y_pred, output_dict=True)
        return pd.DataFrame.from_dict(d)


class KNN:
    def __init__(self, n_neighbors: int, label_encoder: LabelEncoder, weights: np.array, metric="euclidean") -> None:
        m = available_distances.get(metric, None)
        assert m, "Implémentation de la métrique {} non fournie".format(metric)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.le = label_encoder
        self.metric = m
        self.data = None

    def fit(self, X: np.array, y: np.array) -> None:
        self.data = X
        self.targets = self.le.transform(y)

    def predict(self, X: np.array) -> np.array:
        assert X.ndim > 1, "Le paramètre X doit être de format (n_observations, n_variables); vous avez {}".format(X.shape)
        dist = self.metric(np.expand_dims(X, 1), self.data)
        nn = np.argsort(dist, axis=1)[:, 0:self.n_neighbors]
        scores = np.zeros((dist.shape[0], self.n_neighbors))
        for i, row in enumerate(self.targets[nn]):
            for el in row:
                scores[i, el] += self.weights[el]
        y_pred = np.argmax(scores, axis=1)
        return self.le.inverse_transform(y_pred)

    def evaluate(self, y_true: np.array, y_pred: np.array) -> pd.DataFrame:
        d = metrics.classification_report(y_true, y_pred, output_dict=True)
        return pd.DataFrame.from_dict(d)