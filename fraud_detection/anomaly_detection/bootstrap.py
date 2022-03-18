import numpy as np
import torch
import os
from collections import defaultdict
from datetime import datetime as dt

from torch.utils.data import DataLoader
from anomaly_detection.model.base import BaseModel
from anomaly_detection.model.one_class import DeepSVDD
from anomaly_detection.model.reconstruction import AutoEncoder as AE, DAGMM, MemAutoEncoder as MemAE
from anomaly_detection.model.shallow import RecForest, OCSVM, LOF
from anomaly_detection.trainer.one_class import DeepSVDDTrainer
from anomaly_detection.trainer.reconstruction import AutoEncoderTrainer as AETrainer, DAGMMTrainer, MemAETrainer
from anomaly_detection.trainer.shallow import OCSVMTrainer, RecForestTrainer, LOFTrainer
from anomaly_detection.utils import metrics
from anomaly_detection.utils.utils import average_results
from anomaly_detection.datamanager.dataset import AbstractDataset
from anomaly_detection.datamanager.dataset import IEEEFraudDetection

available_models = [
    "AE",
    "DAGMM",
    "DeepSVDD",
    "LOF",
    "MemAE",
    "NeuTraLAD"
    "OC-SVM",
    "RecForest",
]
available_datasets = [
    "IEEEFraudDetection",
]


def store_results(results: dict, params: dict, model_name: str, dataset: str, dataset_path: str, results_path: str = None):
    output_dir = results_path or f"../results/{dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fname = output_dir + '/' + f'{model_name}_results.txt'
    with open(fname, 'a') as f:
        hdr = "Experiments on {}\n".format(dt.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write(hdr)
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
        f.write(f'{dataset} ({dataset_path.split("/")[-1].split(".")[0]})\n')
        f.write(", ".join([f"{param_name}={param_val}" for param_name, param_val in params.items()]) + "\n")
        f.write("\n".join([f"{met_name}: {res}" for met_name, res in results.items()]) + "\n")
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
    return fname


def store_model(model, model_name: str, dataset: str, models_path: str = None):
    output_dir = models_path or f'../models/{dataset}/{model_name}/{dt.now().strftime("%d_%m_%Y_%H_%M_%S")}"'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(f"{output_dir}/model")


model_trainer_map = {
    # Deep Models
    "AE": (AE, AETrainer),
    "DAGMM": (DAGMM, DAGMMTrainer),
    "MemAE": (MemAE, MemAETrainer),
    "DeepSVDD": (DeepSVDD, DeepSVDDTrainer),
    # Shallow Models
    "OC-SVM": (OCSVM, OCSVMTrainer),
    "LOF": (LOF, LOFTrainer),
    "RecForest": (RecForest, RecForestTrainer)
}


def resolve_model_trainer(
        model_name: str,
        dataset: AbstractDataset,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str
):
    model_trainer_tuple = model_trainer_map.get(model_name, None)
    assert model_trainer_tuple, "Model %s not found" % model_name
    model, trainer = model_trainer_tuple
    model = model(
        dataset_name=dataset.name,
        in_features=dataset.in_features,
        n_instances=dataset.n_instances,
        device=device
    )
    trainer = trainer(
        model=model,
        lr=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device
    )

    return model, trainer


def train_model(
        model: BaseModel,
        model_trainer,
        train_ldr: DataLoader,
        test_ldr: DataLoader,
        dataset_name: str,
        n_runs: int,
        thresh: float,
        device: str,
        model_path: str,
        test_mode: bool
):
    # Training and evaluation on different runs
    all_results = defaultdict(list)

    if test_mode:
        for model_file_name in os.listdir(model_path):
            model = BaseModel.load(f"{model_path}/{model_file_name}")
            model = model.to(device)
            model_trainer.model = model
            print("Evaluating the model on test set")
            # We test with the minority samples as the positive class
            y_train_true, train_scores = model_trainer.test(train_ldr)
            y_test_true, test_scores = model_trainer.test(test_ldr)
            y_true = np.concatenate((y_train_true, y_test_true), axis=0)
            scores = np.concatenate((train_scores, test_scores), axis=0)
            print("Evaluating model")
            results = metrics.evaluate(scores, test_scores, y_true, thresh)
            for k, v in results.items():
                all_results[k].append(v)
    else:
        for i in range(n_runs):
            print(f"Run {i + 1} of {n_runs}")
            _ = model_trainer.train(train_ldr)
            print("Finished learning process")
            print("Evaluating model on test set")
            # We test with the minority samples as the positive class
            y_train_true, train_scores = model_trainer.test(train_ldr)
            y_test_true, test_scores = model_trainer.test(test_ldr)
            y_true = np.concatenate((y_train_true, y_test_true), axis=0)
            scores = np.concatenate((train_scores, test_scores), axis=0)
            results = metrics.estimate_optimal_threshold(scores, test_scores, y_true, thresh)
            print(results)
            for k, v in results.items():
                all_results[k].append(v)
            store_model(model, model.name, dataset_name, model_path)
            model.reset()

    # Compute mean and standard deviation of the performance metrics
    print("Averaging results ...")
    return average_results(all_results)


def train(
        model_name: str,
        dataset_name: str,
        dataset_path: str,
        batch_size: int,
        pct: float,
        corruption_ratio: float,
        n_runs: int,
        n_epochs: int,
        learning_rate: float,
        results_path: str,
        models_path: str,
        test_mode: bool
):
    # Dynamically load the Dataset instance
    clsname = globals()[f'{dataset_name}Dataset']
    dataset = clsname(path=dataset_path, pct=pct)
    anomaly_thresh = 1 - dataset.anomaly_ratio
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # split data in train and test sets
    # we train only on the majority class
    train_ldr, test_ldr = dataset.loaders(batch_size=batch_size, seed=42)

    # check path
    for p in [results_path, models_path]:
        if p:
            assert os.path.exists(p), "Path %s does not exist" % p

    model, model_trainer = resolve_model_trainer(
        model_name=model_name,
        dataset=dataset,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device,
    )
    res = train_model(
        model=model,
        model_trainer=model_trainer,
        train_ldr=train_ldr,
        test_ldr=test_ldr,
        dataset_name=dataset_name,
        n_runs=n_runs,
        device=device,
        thresh=anomaly_thresh,
        model_path=models_path,
        test_mode=test_mode
    )
    print(res)
    params = dict(
        {"BatchSize": batch_size, "Epochs": n_epochs, "CorruptionRatio": corruption_ratio,
         "Threshold": anomaly_thresh},
        **model.get_params()
    )
    # Store the average of results
    fname = store_results(res, params, model_name, dataset.name, dataset_path, results_path)
    print(f"Results stored in {fname}")
