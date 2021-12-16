import copy
from datetime import datetime
from pathlib import Path

import mlflow
from omegaconf import DictConfig
from utils import log_params_from_omegaconf_dict


def fill_config_with_num_classes(cfg_layer: DictConfig, num_classes: int) -> DictConfig:
    updated_config = copy.deepcopy(cfg_layer)
    for n, (key, value) in enumerate(list(cfg_layer.items())):
        if n == len(list(cfg_layer.values())) - 1:
            updated_config[key]["n_out"] = num_classes
    return updated_config


def set_run_training(func):
    def run(*args):
        a, data_loader = args[0], args[1]
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.set_experiment(a.cfg.name)
        with mlflow.start_run():
            log_params_from_omegaconf_dict(a.cfg)
            func(*args)
    return run


def set_run_testing(func):
    def run(*args):
        with mlflow.start_run():
            func(*args)
    return run


def log_metrics(func):
    def run(*args, **kwargs):
        acc, loss, path_to_plot = func(*args, **kwargs)
        # log metric
        mlflow.log_metric("loss", loss, step=1)
        mode = "train" if "train" in path_to_plot.split("/")[-1] else "test"
        mlflow.log_metric(f"{mode}_acc", acc, step=1)
        mlflow.log_artifact(path_to_plot)
        return acc, loss, path_to_plot
    return run


def get_data_time() -> str:
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string
