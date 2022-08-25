import random
import logging
import mlflow
import numpy as np
import torch
from hydra import initialize, compose
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def instantiate_config(config_path: str):
    config_dir = config_path.rsplit("/", 1)[0]
    config_file = config_path.rsplit("/", 1)[-1]
    with initialize(config_path=config_dir, job_name=config_path):
        cfg = compose(config_name=config_file)
    return cfg


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
    elif isinstance(element, int) or isinstance(element, str):
        mlflow.log_param(parent_name, element)
    else:
        logger.warning(f"Configuration field {parent_name} with value {element} not logged in mlflow.")

