from datamanager.base import AbstractDataset


class IEEEFraudDetection(AbstractDataset):
    def __init__(self, **kwargs):
        super(IEEEFraudDetection, self).__init__(**kwargs)
        self.name = "IEEEFraudDetection"

    def npz_key(self):
        return "fraud_detection"
