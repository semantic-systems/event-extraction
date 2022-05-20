from pathlib import Path
from typing import Dict, Union, List, Optional, Tuple
from abc import abstractmethod

import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from dataclasses import dataclass, asdict

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.utils.data import DataLoader, Sampler

from event_extractor.data_generators import DataGenerator, DataGeneratorSubSample
from event_extractor.data_generators.samplers import FixedSizeCategoricalSampler
from event_extractor.helper import log_metrics
from event_extractor.schema import ClassificationResult


@dataclass
class EnvironmentState(object):
    task: str = 'classification'
    num_agent: int = 1


class Environment(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.state = EnvironmentState()
        self.environment = self.instantiate_environment()

    @abstractmethod
    def instantiate_environment(self) -> DataGenerator:
        # in single sequence classification task, the environment is where the dataloader is located
        # in RL/multiple sequence classification task, the environment is where the simulator is located
        # an environment in a classification task is a simplied version of the RL environment,
        # a static environment is expecting to receive an action from the agent(s) at each time step.
        # a dynamic environment is expecting to receive an action after t time step.
        raise NotImplementedError

    @abstractmethod
    def load_environment(self, mode: str, training_type: str) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def update_state(self, state: Dict):
        raise NotImplementedError

    def return_state_as_dict(self):
        return asdict(self.state)


class StaticEnvironment(Environment):
    def __init__(self, config: DictConfig):
        super(StaticEnvironment, self).__init__(config)
        self.label_index_map = self.environment.label_index_map

    def instantiate_environment(self) -> DataGenerator:
        if "subset" in self.config.data:
            return DataGeneratorSubSample(self.config)
        else:
            return DataGenerator(self.config)

    def instantiate_sampler(self, mode: str, training_type: str) -> Union[Sampler, None]:
        data_source = self.environment.training_dataset if mode == "train" else self.environment.testing_dataset
        if training_type == "episodic_training":
            return FixedSizeCategoricalSampler(data_source=data_source,
                                               n_way=self.config.episode.n_way,
                                               k_shot=self.config.episode.k_shot,
                                               iterations=self.config.episode.iteration,
                                               n_query=self.config.episode.n_query,
                                               replacement=self.config.episode.replacement)
        elif training_type == "batch_training":
            return None

    def load_environment(self, mode: str, training_type: str) -> DataLoader:
        # mode -> "train" or "test"
        sampler = self.instantiate_sampler(mode, training_type)
        return self.environment(mode=mode, sampler=sampler)

    def update_state(self, state: Dict):
        raise NotImplementedError

    @property
    def num_labels(self):
        return self.environment.num_labels

    @property
    def labels_list(self) -> List[str]:
        return [*self.environment.label_index_map]

    @log_metrics
    def evaluate(self,
                 y_predict: List,
                 y_true: List,
                 loss: int,
                 num_epoch: Optional[int] = None) -> ClassificationResult:
        # y_predict = torch.stack(y_predict)
        # y_true = torch.stack(y_true)
        y_predict = torch.tensor(y_predict)
        y_true = torch.tensor(y_true)
        acc = (y_predict == y_true).sum().item() / y_predict.size(0)
        # y_predict = y_predict.cpu().detach().numpy()
        # y_true = y_true.cpu().detach().numpy()
        f1_micro = f1_score(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels), average='micro')
        f1_macro = f1_score(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels), average='macro')
        f1_per_class = f1_score(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels), average=None)
        cm = confusion_matrix(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels))

        cmd_obj = ConfusionMatrixDisplay(cm, display_labels=self.labels_list)
        fig, ax = plt.subplots(figsize=(10, 10))
        cmd_obj.plot(xticks_rotation=30, ax=ax)
        cmd_obj.ax_.set(
            title='Confusion Matrix',
            xlabel='Predicted Labels',
            ylabel='True Labels')

        plt.axis('scaled')
        if num_epoch is not None:
            path_to_plot: str = str(Path(self.config.model.output_path, self.config.name, "plots",
                                         f'confusion_matrix_train_epoch_{num_epoch}.png').absolute())
        else:
            path_to_plot = str(Path(self.config.model.output_path, self.config.name, "plots",
                                    'confusion_matrix_test.png').absolute())
        plt.savefig(path_to_plot, aspect='auto', dpi=100)
        plt.close()
        results = ClassificationResult(**{"acc": acc,
                                          "loss": loss,
                                          "f1_micro": f1_micro,
                                          "f1_macro": f1_macro,
                                          "f1_per_class": {label: f1_per_class[i] for i, label in enumerate(self.labels_list)},
                                          "path_to_plot": path_to_plot
                                          }
                                       )
        return results


class DynamicEnvironment(Environment):
    def __init__(self, config: DictConfig):
        super(DynamicEnvironment, self).__init__(config)

    def load_environment(self, mode: str, training_type: str) -> DataLoader:
        raise NotImplementedError

    def update_state(self, state: Dict):
        raise NotImplementedError

