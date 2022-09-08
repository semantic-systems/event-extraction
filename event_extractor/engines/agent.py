from typing import Dict, Union, Type

import torch
from omegaconf import DictConfig
from dataclasses import dataclass
from copy import deepcopy
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_augmenters.data_augmenter import FSMTBackTranslationAugmenter, RandomAugmenter, DropoutAugmenter
from event_extractor.models import SingleLabelSequenceClassification, PrototypicalNetworks, SingleLabelContrastiveSequenceClassification
from event_extractor.schema import InputFeature, SingleLabelClassificationForwardOutput, \
    PrototypicalNetworksForwardOutput, AgentPolicyOutput, TSNEFeature

PolicyClasses = Union[SingleLabelSequenceClassification, PrototypicalNetworks, SingleLabelContrastiveSequenceClassification]


@dataclass
class AgentState(object):
    done: bool = False
    early_stopping: bool = False


class Agent(object):
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.state = AgentState()
        self.device = device

    def act(self, **kwargs) -> AgentPolicyOutput:
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
        self.Augmenter = self.instantiate_augmenter(device) if self.is_contrastive else None

    @property
    def policy_class(self) -> Type[PolicyClasses]:
        if self.is_contrastive:
            return SingleLabelContrastiveSequenceClassification
        else:
            return SingleLabelSequenceClassification

    @property
    def is_contrastive(self) -> bool:
        return self.config.model.contrastive.contrastive_loss_ratio > 0

    def instantiate_policy(self):
        return self.policy_class(self.config)

    def log_something(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        self.state = AgentState(**state)

    def travel_back(self):
        raise NotImplementedError

    def act(self, data_loader: DataLoader, mode: str) -> AgentPolicyOutput:
        # action per time step - here it will be a batch
        y_predict, y_true = [], []
        loss = 0
        tsne_features: Dict = {"final_hidden_states": [], "encoded_features": [], "labels": []}
        for i, batch in enumerate(tqdm(data_loader)):
            if mode == "train":
                self.policy.optimizer.zero_grad()
                if self.Augmenter is not None:
                    batch = self.augment(batch)
            labels: tensor = batch["label"].to(self.device)
            y_true.extend(labels)
            batch = self.policy.preprocess(batch)
            input_ids: tensor = batch["input_ids"].to(self.device)
            attention_masks: tensor = batch["attention_mask"].to(self.device)
            # convert labels to None if in testing mode.
            labels = None if mode == "test" else labels
            input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            outputs: SingleLabelClassificationForwardOutput = self.policy(input_feature, mode=mode)
            prediction = outputs.prediction_logits.argmax(1)
            y_predict.extend(prediction)
            if mode in ["train", "validation"]:
                loss = (loss + outputs.loss.item())/(i+1)
            if "tsne" in self.config.visualizer and mode in ["validation", "test"]:
                tsne_features["encoded_features"].extend(outputs.encoded_features.tolist())
                tsne_features["final_hidden_states"].extend(outputs.prediction_logits.tolist())

        return AgentPolicyOutput(**{"y_predict": y_predict, "y_true": y_true, "loss": loss,
                                    "tsne_feature": TSNEFeature(**tsne_features)})

    @staticmethod
    def instantiate_augmenter(device):
        return DropoutAugmenter()

    def augment(self, batch: Dict) -> Dict:
        augmented_text_a = self.Augmenter.augment(batch["text"], num_return_sequences=1)
        augmented_batch = deepcopy(batch)
        augmented_batch["text"].extend(augmented_text_a)
        augmented_batch["label"] = torch.cat((batch["label"], batch["label"]), dim=0)
        return augmented_batch


class MetaLearningAgent(BatchLearningAgent):
    def __init__(self, config: DictConfig, device: torch.device):
        super(MetaLearningAgent, self).__init__(config, device)

    @property
    def policy_class(self) -> Type[PolicyClasses]:
        return PrototypicalNetworks

    def instantiate_policy(self):
        return self.policy_class(self.config)

    def act(self, data_loader: DataLoader, mode: str) -> AgentPolicyOutput:
        # action per time step - here it will be an episode
        y_predict, y_true = [], []
        loss = 0
        n_way = self.config.episode.n_way
        k_shot = self.config.episode.k_shot
        for i, episode in enumerate(tqdm(data_loader)):
            if mode == "train":
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
                                                       labels=labels[n_way * k_shot:] if mode in ["train", "validation"]
                                                       else None)
            y_true.extend(labels[n_way * k_shot:])
            label_map = {i_episode: i_whole for i_episode, i_whole in
                         enumerate(torch.unique(labels[:n_way * k_shot]).tolist())}

            outputs: PrototypicalNetworksForwardOutput = self.policy(support_feature, query_feature, mode=mode)
            prediction_per_episode = outputs.distance.argmin(1).tolist()
            prediction = [*map(label_map.get, prediction_per_episode)]
            y_predict.extend(prediction)
            if mode in ["train", "validation"]:
                loss = (loss + outputs.loss.item()) / (i + 1)
        return AgentPolicyOutput(**{"y_predict": y_predict, "y_true": y_true, "loss": loss})

    def log_something(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        self.state = AgentState(**state)

    def travel_back(self):
        raise NotImplementedError
