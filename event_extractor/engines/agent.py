from typing import Dict, Union, Type, Optional, Tuple, List

import torch
from omegaconf import DictConfig
from dataclasses import dataclass

from torch import tensor
from torch.utils.data import DataLoader

from event_extractor.models import SingleLabelSequenceClassification, PrototypicalNetworks
from event_extractor.schema import InputFeature, SingleLabelClassificationForwardOutput, PrototypicalNetworksForwardOutput

PolicyClasses = Union[SingleLabelSequenceClassification, PrototypicalNetworks]


@dataclass
class AgentState(object):
    done: bool = False
    early_stopping: bool = False


class Agent(object):
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.state = AgentState()
        self.device = device

    def act(self, **kwargs):
        raise NotImplementedError

    def instantiate_policy(self):
        raise NotImplementedError

    def log_something(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        self.state = AgentState(**state)

    def travel_back(self):
        raise NotImplementedError


class BatchLearningAgent(Agent):
    def __init__(self, config: DictConfig, device: torch.device):
        super(BatchLearningAgent, self).__init__(config, device)
        self.policy = self.instantiate_policy()

    @property
    def policy_class(self) -> Type[PolicyClasses]:
        return SingleLabelSequenceClassification

    def instantiate_policy(self):
        return self.policy_class(self.config)

    def log_something(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        self.state = AgentState(**state)

    def travel_back(self):
        raise NotImplementedError

    def act(self, data_loader: DataLoader, test: Optional[bool] = False) -> Tuple[List, List, int]:
        # action per time step - here it will be a batch
        y_predict, y_true = [], []
        loss = 0
        for i, batch in enumerate(data_loader):
            if not test:
                self.policy.optimizer.zero_grad()
            labels: tensor = batch["label"].to(self.device)
            y_true.extend(labels)
            batch = self.policy.preprocess(batch)
            input_ids: tensor = batch["input_ids"].to(self.device)
            attention_masks: tensor = batch["attention_mask"].to(self.device)
            # convert labels to None if in testing mode.
            labels = None if test else labels
            input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            outputs: SingleLabelClassificationForwardOutput = self.policy(input_feature)
            prediction = outputs.prediction_logits.argmax(1)
            y_predict.extend(prediction)
            if not test:
                loss = (loss + outputs.loss.item())/(i+1)
        return y_predict, y_true, loss


class MetaLearningAgent(BatchLearningAgent):
    def __init__(self, config: DictConfig, device: torch.device):
        super(MetaLearningAgent, self).__init__(config, device)

    @property
    def policy_class(self) -> Type[PolicyClasses]:
        return PrototypicalNetworks

    def instantiate_policy(self):
        return self.policy_class(self.config)

    def act(self, data_loader: DataLoader, test: Optional[bool] = False) -> Tuple[List, List, int]:
        # action per time step - here it will be an episode
        y_predict, y_true = [], []
        loss = 0
        n_way = self.config.episode.n_way
        k_shot = self.config.episode.k_shot
        for i, episode in enumerate(data_loader):
            if not test:
                self.policy.optimizer.zero_grad()
            labels: tensor = torch.as_tensor(episode["label"]).to(self.device)
            episode = self.policy.preprocess(episode)
            input_ids: tensor = episode["input_ids"].to(self.device)
            attention_masks: tensor = episode["attention_mask"].to(self.device)
            support_feature: InputFeature = InputFeature(input_ids=input_ids[:n_way * k_shot],
                                                         attention_mask=attention_masks[:n_way * k_shot],
                                                         labels=labels[:n_way * k_shot])
            query_feature: InputFeature = InputFeature(input_ids=input_ids[n_way * k_shot:],
                                                       attention_mask=attention_masks[n_way * k_shot:],
                                                       labels=labels[n_way * k_shot:] if not test else None)
            y_true.extend(labels[n_way * k_shot:])
            label_map = {i_episode: i_whole for i_episode, i_whole in
                         enumerate(torch.unique(labels[:n_way * k_shot]).tolist())}

            outputs: PrototypicalNetworksForwardOutput = self.policy(support_feature, query_feature)
            prediction_per_episode = outputs.distance.argmin(1).tolist()
            prediction = [*map(label_map.get, prediction_per_episode)]
            y_predict.extend(prediction)
            if not test:
                loss = (loss + outputs.loss.item()) / (i + 1)
        return y_predict, y_true, loss

    def log_something(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        self.state = AgentState(**state)

    def travel_back(self):
        raise NotImplementedError