from anomaly_detection.datamanager.base import AbstractDataset


class IEEEFraudDetection(AbstractDataset):
    def __init__(self, path, **kwargs):
        super(IEEEFraudDetection, self).__init__(path=path, **kwargs)
        self.name = "IEEEFraudDetection"

    def npz_key(self):
        return "fraud_detection"
