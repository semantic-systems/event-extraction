from typing import Dict
from abc import abstractmethod
from omegaconf import DictConfig
from dataclasses import dataclass, asdict


@dataclass
class EnvironmentState(object):
    task: str = 'classification'
    num_agent: int = 1


class Environment(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.state = EnvironmentState()

    @abstractmethod
    def load_environment(self):
        # in single sequence classification task, the environment is where the dataloader is located
        # in RL/multiple sequence classification task, the environment is where the simulator is located
        # an environment in a classification task is a simplied version of the RL environment,
        # a static environment is expecting to receive an action from the agent(s) at each time step.
        # a dynamic environment is expecting to receive an action after t time step.
        raise NotImplementedError

    @abstractmethod
    def update_state(self, state: Dict):
        raise NotImplementedError

    def return_state_as_dict(self):
        return asdict(self.state)


class StaticEnvironment(Environment):
    def __init__(self, config: DictConfig):
        super(StaticEnvironment, self).__init__(config)

    def load_environment(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        raise NotImplementedError


class DynamicEnvironment(Environment):
    def __init__(self, config: DictConfig):
        super(DynamicEnvironment, self).__init__(config)

    def load_environment(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        raise NotImplementedError

