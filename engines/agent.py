from typing import Dict
from omegaconf import DictConfig
from dataclasses import dataclass


@dataclass
class AgentState(object):
    pause: bool = False
    early_stopping: bool = False


class Agent(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.state = AgentState()

    def log_something(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        self.state = AgentState(**state)

    def travel_back(self):
        raise NotImplementedError

