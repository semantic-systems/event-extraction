import copy

import mlflow
from omegaconf import DictConfig
from utils import log_params_from_omegaconf_dict


def fill_config_with_num_classes(cfg_layer: DictConfig, num_classes: int) -> DictConfig:
    updated_config = copy.deepcopy(cfg_layer)
    for n, (key, value) in enumerate(list(cfg_layer.items())):
        if n == len(list(cfg_layer.values())) - 1:
            updated_config[key]["n_out"] = num_classes
    return updated_config


def set_run(func):
    def run(*args):
        a, data_loader = args[0], args[1]
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.set_experiment(a.cfg.name)
        with mlflow.start_run():
            log_params_from_omegaconf_dict(a.cfg)
            # print(f"{args} train model")
            func(*args)
    return run


def log_metrics(func):
    def run(*args):
        # print(f"{args} train per epoch")
        loss, acc = func(*args)
        # log metric
        mlflow.log_metric("loss", loss.item(), step=1)
        mlflow.log_metric("train_acc", acc, step=1)
        mlflow.log_artifact(f"./outputs/confusion_matrix_train_epoch_{args[2]}.png")
        return loss, acc
    return run
