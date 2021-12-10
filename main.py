import logging
import warnings
import torch
from models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
from data_generators import DataGenerator, DataGeneratorSubSample
from data_generators.samplers import EpisodicBatchSampler
from hydra import initialize, compose

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


def run_many_shot_training(config_name: str):
    with initialize(config_path="./configs", job_name="test"):
        cfg = compose(config_name=config_name+".yaml")
    generator = DataGenerator(cfg)
    data_loader_train = generator("train")
    data_loader_test = generator("test", batch_size=1)
    model = SingleLabelSequenceClassification(cfg, num_classes=generator.num_labels)
    model.train_model(data_loader_train)
    model = torch.load("./outputs/test_model.pt")
    model.test_model(data_loader_test)


def run_episodic_training(config_name: str):
    with initialize(config_path="./configs", job_name="test"):
        cfg = compose(config_name=config_name+".yaml")
    generator = DataGeneratorSubSample(cfg)
    sampler_train = EpisodicBatchSampler(data_source=generator.training_dataset,
                                         n_way=cfg.episode.n_way,
                                         k_shot=cfg.episode.k_shot,
                                         iterations=cfg.episode.iteration)
    sampler_test = EpisodicBatchSampler(data_source=generator.testing_dataset,
                                        n_way=cfg.episode.n_way,
                                        k_shot=cfg.episode.k_shot,
                                        iterations=cfg.episode.iteration)
    data_loader_train = generator("train", sampler=sampler_train)
    data_loader_test = generator("test", batch_size=1, sampler=sampler_test)
    model = SingleLabelSequenceClassification(cfg, num_classes=generator.num_labels)
    model.train_model(data_loader_train)
    model = torch.load("./outputs/test_model.pt")
    model.test_model(data_loader_test)


if __name__ == "__main__":
    run_many_shot_training("example_config")
    # run_episodic_training("banking77")

