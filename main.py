import logging
import warnings
from models import SingleLabelSequenceClassification, PrototypicalNetworks
from data_generators import DataGenerator, DataGeneratorSubSample
from data_generators.samplers import EpisodicBatchSampler, CategoricalSampler
from hydra import initialize, compose
from helper import fill_config_with_num_classes
from validate import ConfigValidator
import numpy as np
import torch
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def instantiate_config(config_path: str, job_name: str):
    config_dir = config_path.rsplit("/", 1)[0]
    config_file = config_path.rsplit("/", 1)[-1]
    with initialize(config_path=config_dir, job_name=job_name):
        cfg = compose(config_name=config_file)
    return cfg


def run_many_shot_training(config_path: str, job_name: str = "many_shot"):
    cfg = instantiate_config(config_path, job_name)
    set_seed(cfg.seed)
    validator = ConfigValidator(cfg)
    validator()
    generator = DataGenerator(cfg)
    data_loader_train = generator("train")
    data_loader_test = generator("test", batch_size=1)
    cfg.model.layers = fill_config_with_num_classes(cfg.model.layers, generator.num_labels)
    model = SingleLabelSequenceClassification(cfg)
    model.train_model(data_loader_train)
    # model = torch.load(f"./outputs/test_model_{cfg.name}.pt")
    model.test_model(data_loader_test)


def run_episodic_training(config_path: str, job_name: str = "few_shot"):
    cfg = instantiate_config(config_path, job_name)
    set_seed(cfg.seed)
    validator = ConfigValidator(cfg)
    validator()
    generator = DataGenerator(cfg)
    sampler_train = CategoricalSampler(data_source=generator.training_dataset,
                                       n_way=cfg.episode.n_way,
                                       k_shot=cfg.episode.k_shot,
                                       iterations=cfg.episode.iteration,
                                       n_query=cfg.episode.n_query,
                                       replacement=cfg.episode.replacement)
    sampler_test = CategoricalSampler(data_source=generator.testing_dataset,
                                      n_way=cfg.episode.n_way,
                                      k_shot=cfg.episode.k_shot,
                                      iterations=cfg.episode.iteration,
                                      n_query=cfg.episode.n_query,
                                      replacement=cfg.episode.replacement)
    data_loader_train = generator("train", sampler=sampler_train)
    data_loader_test = generator("test", sampler=sampler_test)
    cfg.model.layers = fill_config_with_num_classes(cfg.model.layers, generator.num_labels)
    model = PrototypicalNetworks(cfg)
    model.train_model(data_loader_train)
    # model = torch.load("./outputs/test_model.pt")
    # model.test_model(data_loader_test)


if __name__ == "__main__":
    run_many_shot_training("./configs/event_detection/many_shot/trec_is.yaml")
    # run_episodic_training("./configs/intent_classification/banking77_few_shot.yaml")
    # run_many_shot_training("./configs/intent_classification/banking77_many_shot.yaml")

