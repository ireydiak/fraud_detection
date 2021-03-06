import torch.nn as nn
from torch import Tensor
from anomaly_detection.model.base import BaseModel


class DeepSVDD(BaseModel):
    """
    Follows SKLearn's API
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM.decision_function)
    """

    def resolve_params(self, dataset_name: str):
        pass

    def __init__(self, **kwargs):
        super(DeepSVDD, self).__init__(**kwargs)
        self.rep_dim = self.in_features // 4
        self.name = "DeepSVDD"
        self.net = self._build_network()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.in_features, self.in_features // 2),
            nn.ReLU(),
            nn.Linear(self.in_features // 2, self.rep_dim)
        ).to(self.device)

    def forward(self, X: Tensor):
        return self.net(X)

    def get_params(self) -> dict:
        return {
            "in_features": self.in_features,
            "rep_dim": self.rep_dim
        }
