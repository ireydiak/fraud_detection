from sklearn.svm import OneClassSVM
from anomaly_detection.model.base import BaseShallowModel


class OCSVM(BaseShallowModel):
    def __init__(self, kernel="rbf", gamma="scale", shrinking=False, verbose=True, nu=0.5, **kwargs):
        super(OCSVM, self).__init__(**kwargs)
        self.clf = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            shrinking=shrinking,
            verbose=verbose,
            nu=nu
        )
        self.name = "OC-SVM"

    def get_params(self) -> dict:
        return {
            "kernel": self.clf.kernel,
            "gamma": self.clf.gamma,
            "shrinking": self.clf.shrinking,
            "nu": self.clf.nu
        }
