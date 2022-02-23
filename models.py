
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


def euclidean_dist(X: np.array, Y: np.array):
    dist = np.sqrt((X - Y)**2)
    ax = X.ndim - 1 if X.ndim > 1 else 0
    return dist.sum(axis=ax)

def manhattan_dist(X: np.array, Y: np.array):
    dist = np.abs(X - Y)
    ax = X.ndim - 1 if X.ndim > 1 else 0
    return dist.sum(axis=ax)

def mahalanobis_dist(X: np.array, Y: np.array, Sigma: np.array):
    pass

available_distances = {
    "euclidean": euclidean_dist,
    "manhattan": manhattan_dist,
    "mahalanobis": mahalanobis_dist
}

class NearestCentroid:
    def __init__(self, label_encoder: LabelEncoder, metric="euclidean"):
        m = available_distances.get(metric, None)
        assert m, "Implémentation de la métrique {} non fournie".format(metric)
        self.metric = m
        self.centroids = None
        self.le = label_encoder

    def fit(self, X: np.array, y: np.array):
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


# def nearest_centroid_classifier(data: np.array, centers: np.array, dist_measure="euclidean"):
#     centroids_map = {0: "Class -1", 1: "Class -2", 2: "Class 0", 3: "Class 1", 4: "Class 2"}
#     # Calcul de la distance euclidienne
#     # On augmente la matrice X pour obtenir un tensor
#     # Avec le broadcasting, numpy nous fournit un tensor de dimension 
#     # (n_rows, n_classes)
#     # qui contient la distance euclidienne de chaque observation par rapport aux centroids 
#     # de toutes les classes
#     if dist_measure == "euclidean":
#         dist = (np.sqrt((np.expand_dims(data, 1) - centers)**2)).sum(axis=2)
#     elif dist_measure == "manhattan":
#         dist = (np.abs(np.expand_dims(data, 1) - centers)).sum(axis=2)
#     elif dist_measure == "mahalanobis":
#         S = np.cov(data.T)
#         dist = mahalanobis(data, centers, S)
#     else:
#         raise Exception("Implémentation de {} non fournie".format(dist_measure))
#     y_pred = np.array([centroids_map[p] for p in np.argmin(dist, axis=1)])
#     d = metrics.classification_report(y_true, y_pred, output_dict=True)
#     df_metrics = pd.DataFrame.from_dict(d)
#     return df_metrics

class KNN:
    def __init__(self, n_neighbors: int, weights: np.array, metric="euclidean") -> None:
        m = d.get(metric, None)
        assert m, "Implémentation de la métrique {} non fournie".format(metric)
        self.n_neighbors = n_neighbors
        self.weights = weights
    
    def fit(self, X: np.array, y: np.array):
        c = len(np.unique(y))

    def predict(self, X: np.array) -> np.array:
        pass

    def evaluate(self, y_true: np.array, y_pred: np.array) -> pd.DataFrame:
        d = metrics.classification_report(y_true, y_pred, output_dict=True)
        return pd.DataFrame.from_dict(d)