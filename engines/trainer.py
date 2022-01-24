from typing import List
from abc import abstractmethod
from omegaconf import DictConfig
from torch.nn import Module
from engines.agent import Agent
from engines.environment import Environment


class Trainer(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.environment = self.instantiate_environment()
        self.agent = self.instantiate_agent()

    @abstractmethod
    def train(self, model: Module):
        raise NotImplementedError

    @abstractmethod
    def test(self, model: Module):
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

    def train(self, model: Module):
        raise NotImplementedError

    def test(self, model: Module):
        raise NotImplementedError

    def instantiate_environment(self) -> Environment:
        return Environment(self.config)

    def instantiate_agent(self) -> Agent:
        return Agent(self.config)


class MultiAgentTrainer(Trainer):
    def __init__(self, config: DictConfig):
        super(MultiAgentTrainer, self).__init__(config)

    def train(self, model: Module):
        raise NotImplementedError

    def test(self, model: Module):
        raise NotImplementedError

    def instantiate_environment(self) -> Environment:
        return Environment(self.config)

    def instantiate_agent(self) -> List[Agent]:
        raise NotImplementedError




