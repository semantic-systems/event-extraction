import json
import logging
from pathlib import Path
from typing import List, Optional
from abc import abstractmethod

import torch
from omegaconf import DictConfig
from event_extractor.engines.agent import Agent, BatchLearningAgent, MetaLearningAgent
from event_extractor.engines.environment import Environment, StaticEnvironment
from event_extractor.helper import fill_config_with_num_classes, get_data_time, set_run_training, set_run_testing
from event_extractor.schema import ClassificationResult
from utils import set_seed
from event_extractor.validate import ConfigValidator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EarlyStopping(object):
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.previous_loss = 9999
        self.best_score = None

    def __call__(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup()
        self.environment = self.instantiate_environment()
        # update config according to envrionment instance - num_labels
        self.config.model.layers = fill_config_with_num_classes(self.config.model.layers,
                                                                self.environment.num_labels)
        self.agent = self.instantiate_agent()
        self.print_trainer_info()

    def run(self):
        self.train()
        self.test()

    def setup(self):
        set_seed(self.config.seed)
        validator = ConfigValidator(self.config)
        self.config = validator()

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

    def print_trainer_info(self):
        logger.warning(f"Training Info: \n"
                       f"    Trainer: {self.__class__.__name__}\n"
                       f"    Agent: {self.agent.__class__.__name__}\n"
                       f"    Environment: {self.environment.__class__.__name__}\n"
                       f"    Policy: {self.agent.policy.__class__.__name__}")

    @staticmethod
    def log_result(result_per_epoch: ClassificationResult, final_result: List, epoch: Optional[int] = None):
        result_this_epoch = result_per_epoch.__dict__
        if epoch:
            result_this_epoch.update({"epoch": epoch})
        result_this_epoch.pop('path_to_plot', None)
        final_result.append(result_this_epoch)

    def save_best_model(self, best_validation_metric: int, result_per_epoch: ClassificationResult):
        if result_per_epoch.f1_macro > best_validation_metric:
            label_index_map = dict([(str(value), key) for key, value in self.environment.label_index_map.items()])
            self.agent.policy.save_model(
                Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}", "pretrained_models",
                     f"{self.config.name}_best_model.pt").absolute(),
                index_label_map=label_index_map)

    def dump_result(self, result: List, mode: str):
        with open(Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}",
                       f"{mode}_result.json"), "w") as final:
            json.dump(result, final, indent=2)

    def save_final_model(self):
        label_index_map = dict([(str(value), key) for key, value in self.environment.label_index_map.items()])
        self.agent.policy.save_model(
            Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}", "pretrained_models",
                 f"{self.config.name}_{get_data_time()}.pt").absolute(),
            index_label_map=label_index_map)


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
        if self.config.data.validation:
            self.early_stopping = EarlyStopping(tolerance=self.config.early_stopping.tolerance,
                                                min_delta=self.config.early_stopping.delta)

    def instantiate_environment(self) -> Environment:
        return StaticEnvironment(self.config)

    def instantiate_agent(self) -> Agent:
        return BatchLearningAgent(self.config, self.device)

    @set_run_training
    def train(self):
        # run per batch
        data_loader = self.environment.load_environment("train", self.training_type)
        if self.config.data.validation:
            validation_data_loader = self.environment.load_environment("validation", self.training_type)
        self.agent.policy.to(self.agent.policy.device)
        self.agent.policy.train()
        self.agent.policy.optimizer.zero_grad()
        train_result = []
        validation_result = []
        best_validation_metric = 0

        # start new run
        for n in range(self.config.model.epochs):
            # training
            y_predict, y_true, train_loss = self.agent.act(data_loader, mode="train")
            train_result_per_epoch: ClassificationResult = self.environment.evaluate(y_predict,
                                                                                     y_true,
                                                                                     train_loss,
                                                                                     mode="train",
                                                                                     num_epoch=n)
            logger.warning(f"Training results:")
            logger.warning(f"Epoch: {n}, Average loss: {train_loss}, Average acc: {train_result_per_epoch.acc}, "
                           f"F1 macro: {train_result_per_epoch.f1_macro},"
                           f"F1 micro: {train_result_per_epoch.f1_micro}, "
                           f"F1 per class: {train_result_per_epoch.f1_per_class}, "
                           f"Precision macro: {train_result_per_epoch.precision_macro}, "
                           f"Recall macro: {train_result_per_epoch.recall_macro}, "
                           f"Other: {train_result_per_epoch.other}")
            self.log_result(result_per_epoch=train_result_per_epoch, final_result=train_result, epoch=n)

            # validation
            if self.config.data.validation:
                y_predict, y_true, validation_loss = self.agent.act(validation_data_loader, mode="validation")
                validation_result_per_epoch: ClassificationResult = self.environment.evaluate(y_predict,
                                                                                              y_true,
                                                                                              validation_loss,
                                                                                              mode="validation",
                                                                                              num_epoch=n)
                logger.warning(f"Validation results:")
                logger.warning(
                    f"Epoch: {n}, Average loss: {validation_loss}, Average acc: {validation_result_per_epoch.acc}, "
                    f"F1 macro: {validation_result_per_epoch.f1_macro},"
                    f"F1 micro: {validation_result_per_epoch.f1_micro}, "
                    f"F1 per class: {validation_result_per_epoch.f1_per_class}, "
                    f"Precision macro: {validation_result_per_epoch.precision_macro}, "
                    f"Recall macro: {validation_result_per_epoch.recall_macro}, "
                    f"Other: {validation_result_per_epoch.other}")
                self.log_result(result_per_epoch=validation_result_per_epoch, final_result=validation_result, epoch=n)
                self.save_best_model(best_validation_metric, validation_result_per_epoch)
                # early stopping
                self.early_stopping(validation_loss)
                if self.early_stopping.early_stop:
                    logger.warning(f"Early stopping reached at epoch: {n}")
                    break

        self.dump_result(train_result, mode='train')
        if self.config.data.validation:
            self.dump_result(validation_result, mode='validation')
        else:
            # save the final trained model -> not recommended
            self.save_final_model()

    @set_run_testing
    def test(self):
        data_loader = self.environment.load_environment("test", self.training_type)
        self.agent.policy.eval()
        test_result = []
        with torch.no_grad():
            y_predict, y_true, loss = self.agent.act(data_loader, mode="test")
            result = self.environment.evaluate(y_predict, y_true, loss, mode="test")
            logger.warning(f"Testing Accuracy: {result.acc}, F1 micro: {result.f1_micro},"
                           f"F1 macro: {result.f1_macro}, F1 per class: {result.f1_per_class}, "
                           f"Precision macro: {result.precision_macro}, "
                           f"Recall macro: {result.recall_macro}, "
                           f"Other: {result.other}")
            self.log_result(result_per_epoch=result, final_result=test_result)
            self.dump_result(test_result, mode='test')


class MetaLearningTrainer(BatchLearningTrainer):
    def __init__(self, config: DictConfig):
        super(MetaLearningTrainer, self).__init__(config)

    def instantiate_environment(self) -> Environment:
        return StaticEnvironment(self.config)

    def instantiate_agent(self) -> Agent:
        return MetaLearningAgent(self.config, self.device)

