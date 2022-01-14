import logging
import warnings
from models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
from data_generators import DataGenerator, DataGeneratorSubSample
from data_generators.samplers import EpisodicBatchSampler
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


def run_many_shot_training(config_name: str):
    with initialize(config_path="./configs", job_name="test"):
        cfg = compose(config_name=config_name+".yaml")
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


def run_episodic_training(config_name: str):
    with initialize(config_path="./configs", job_name="test"):
        cfg = compose(config_name=config_name+".yaml")
    set_seed(cfg.seed)
    validator = ConfigValidator(cfg)
    validator()
    generator = DataGeneratorSubSample(cfg)
    sampler_train = EpisodicBatchSampler(data_source=generator.training_dataset,
                                         n_way=cfg.episode.n_way,
                                         k_shot=cfg.episode.k_shot,
                                         iterations=cfg.episode.iteration)
    sampler_test = EpisodicBatchSampler(data_source=generator.testing_dataset,
                                        n_way=cfg.episode.n_way,
                                        k_shot=cfg.episode.k_shot,
                                        iterations=cfg.episode.iteration)
    data_loader_train = generator("train")#, sampler=sampler_train)
    data_loader_test = generator("test", batch_size=1)#, sampler=sampler_test)
    cfg.model.layers = fill_config_with_num_classes(cfg.model.layers, generator.num_labels)
    model = SingleLabelSequenceClassification(cfg)
    model.train_model(data_loader_train)
    # model = torch.load("./outputs/test_model.pt")
    model.test_model(data_loader_test)


if __name__ == "__main__":
    #run_many_shot_training("trec_is")
    run_episodic_training("banking77")

