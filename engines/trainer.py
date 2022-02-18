import logging
from pathlib import Path
from typing import List
from abc import abstractmethod

import torch
from tqdm import tqdm
from omegaconf import DictConfig
from engines.agent import Agent, BatchLearningAgent, MetaLearningAgent
from engines.environment import Environment, StaticEnvironment
from helper import fill_config_with_num_classes, get_data_time, set_run_training, set_run_testing
from utils import set_seed
from validate import ConfigValidator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup()
        self.environment = self.instantiate_environment()
        # update config according to envrionment instance - num_labels
        self.config.model.layers = fill_config_with_num_classes(self.config.model.layers,
                                                                self.environment.num_labels)
        self.agent = self.instantiate_agent()

    def run(self):
        self.train()
        self.test()

    def setup(self):
        set_seed(self.config.seed)
        validator = ConfigValidator(self.config)
        validator()

    @property
    def training_type(self):
        if "episode" in self.config:
            return "episodic_training"
        else:
            return "batch_training"

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def instantiate_environment(self):
        raise NotImplementedError

    @abstractmethod
    def instantiate_agent(self):
        raise NotImplementedError


class SingleAgentTrainer(Trainer):
    def __init__(self, config: DictConfig):
        super(SingleAgentTrainer, self).__init__(config)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def instantiate_environment(self) -> Environment:
        return Environment(self.config)

    def instantiate_agent(self) -> Agent:
        raise NotImplementedError

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")


class MultiAgentTrainer(Trainer):
    def __init__(self, config: DictConfig):
        super(MultiAgentTrainer, self).__init__(config)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def instantiate_environment(self) -> Environment:
        return Environment(self.config)

    def instantiate_agent(self) -> List[Agent]:
        raise NotImplementedError


class BatchLearningTrainer(SingleAgentTrainer):
    def __init__(self, config: DictConfig):
        super(BatchLearningTrainer, self).__init__(config)

    def instantiate_environment(self) -> Environment:
        return StaticEnvironment(self.config)

    def instantiate_agent(self) -> Agent:
        return BatchLearningAgent(self.config, self.device)

    @set_run_training
    def train(self):
        # run per batch
        data_loader = self.environment.load_environment("train", self.training_type)

        self.agent.policy.to(self.agent.policy.device)
        self.agent.policy.train()
        self.agent.policy.optimizer.zero_grad()
        # start new run
        for n in tqdm(range(self.config.model.epochs)):
            y_predict, y_true, loss = self.agent.act(data_loader)
            acc, _, _ = self.environment.evaluate(y_predict, y_true, loss, num_epoch=n)
            logger.warning(f"Epoch: {n}, Average loss: {loss}, Average acc: {acc}")
        torch.save(self, Path(self.config.model.output_path, self.config.name, "pretrained_models",
                              f"{self.config.name}_{get_data_time()}.pt").absolute())

    @set_run_testing
    def test(self):
        data_loader = self.environment.load_environment("test", self.training_type)
        self.agent.policy.eval()
        with torch.no_grad():
            y_predict, y_true, loss = self.agent.act(data_loader, test=True)
            acc, _, _ = self.environment.evaluate(y_predict, y_true, loss)
            logger.warning(f"Testing Accuracy: {acc}")


class MetaLearningTrainer(SingleAgentTrainer):
    def __init__(self, config: DictConfig):
        super(MetaLearningTrainer, self).__init__(config)

    def instantiate_environment(self) -> Environment:
        return StaticEnvironment(self.config)

    def instantiate_agent(self) -> Agent:
        return MetaLearningAgent(self.config, self.device)

    @set_run_training
    def train(self):
        data_loader = self.environment.load_environment("train", self.training_type)
        self.agent.policy.to(self.agent.policy.device)
        self.agent.policy.train()
        self.agent.policy.optimizer.zero_grad()
        # start new run
        for n in tqdm(range(self.config.model.epochs)):
            y_predict, y_true, loss = self.agent.act(data_loader)
            acc, _, _ = self.environment.evaluate(y_predict, y_true, loss, num_epoch=n)
            logger.warning(f"Epoch: {n}, Average loss: {loss}, Average acc: {acc}")
        torch.save(self, Path(self.config.model.output_path, self.config.name, "pretrained_models",
                              f"{self.config.name}_{get_data_time()}.pt").absolute())

    @set_run_testing
    def test(self):
        data_loader = self.environment.load_environment("test", self.training_type)
        self.agent.policy.eval()
        with torch.no_grad():
            y_predict, y_true, loss = self.agent.act(data_loader, test=True)
            acc, _, _ = self.environment.evaluate(y_predict, y_true, loss)
            logger.warning(f"Testing Accuracy: {acc}")


