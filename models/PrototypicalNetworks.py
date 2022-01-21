import torch
import logging
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from typing import List, Optional, Tuple
from pathlib import Path
from torch import tensor
from torch.nn import CrossEntropyLoss, Dropout, Identity
from tqdm import tqdm
from transformers import PreTrainedModel, AdamW, PreTrainedTokenizer, BertTokenizer, AutoModel
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay

from models import SequenceClassification
from models.heads.prototypical_head import PrototypicalHead
from schema import InputFeature, PrototypicalNetworksForwardOutput, EncodedFeature
from helper import log_metrics, set_run_testing, set_run_training, get_data_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PrototypicalNetworks(SequenceClassification):
    def __init__(self, cfg: DictConfig):
        super(PrototypicalNetworks, self).__init__(cfg)
        self.optimizer = AdamW(self.parameters(), lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()
        self.dropout = Dropout(p=cfg.model.dropout_rate)
        self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(cfg.model.from_pretrained)

    def forward(self, support_features: InputFeature, query_features: InputFeature) -> PrototypicalNetworksForwardOutput:
        # Prototype i is the mean of all support features vector with label i
        support_encoded_feature: EncodedFeature = EncodedFeature(
            encoded_feature=self.encoder(
                input_ids=support_features.input_ids,
                attention_mask=support_features.attention_mask).pooler_output,
            labels=support_features.labels)
        query_encoded_feature: EncodedFeature = EncodedFeature(
            encoded_feature=self.encoder(
                input_ids=query_features.input_ids,
                attention_mask=query_features.attention_mask).pooler_output,
            labels=query_features.labels)
        head_output = self.classification_head(support_encoded_feature, query_encoded_feature)

        label_map = {i_whole: i_episode for i_episode, i_whole in enumerate(torch.unique(query_encoded_feature.labels).tolist())}
        if query_encoded_feature.labels is not None:
            query_label_episode = torch.tensor([*map(label_map.get, query_encoded_feature.labels.tolist())])
            loss = self.loss(head_output.output, query_label_episode)
            loss.backward()
            self.optimizer.step()
            return PrototypicalNetworksForwardOutput(loss=loss, distance=-head_output.output)
        else:
            return PrototypicalNetworksForwardOutput(distance=-head_output.output)

    def instantiate_classification_head(self):
        return PrototypicalHead(self.cfg)

    def instantiate_encoder(self) -> PreTrainedModel:
        encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.model.from_pretrained)
        encoder = self.trim_encoder_layers(encoder, self.cfg.model.num_transformer_layers)
        return encoder

    def instantiate_feature_transformer(self):
        # this returns an empty Module
        return Identity()

    @set_run_training
    def train_model(self, data_loader: DataLoader):
        self.to(self.device)
        self.train()
        self.optimizer.zero_grad()
        # start new run
        for n in tqdm(range(self.cfg.model.epochs)):
            y_predict, y_true, loss = self.run_per_epoch(data_loader)
            acc, _, _ = self.evaluate(y_predict, y_true, loss, num_epoch=n)
            logger.warning(f"Epoch: {n}, Average loss: {loss}, Average acc: {acc}")
        torch.save(self, Path(self.cfg.model.output_path, self.cfg.name, "pretrained_models",
                              f"{self.cfg.name}_{get_data_time()}.pt").absolute())

    def run_per_epoch(self, data_loader: DataLoader, test: Optional[bool] = False) -> Tuple[List, List, int]:
        y_predict, y_true = [], []
        loss = 0
        n_way = self.cfg.episode.n_way
        k_shot = self.cfg.episode.k_shot
        for i, episode in enumerate(data_loader):
            if not test:
                self.optimizer.zero_grad()
            labels: tensor = torch.as_tensor(episode["label"]).to(self.device)
            episode = self.preprocess(episode)
            input_ids: tensor = episode["input_ids"].to(self.device)
            attention_masks: tensor = episode["attention_mask"].to(self.device)
            support_feature: InputFeature = InputFeature(input_ids=input_ids[:n_way*k_shot],
                                                         attention_mask=attention_masks[:n_way*k_shot],
                                                         labels=labels[:n_way*k_shot])
            query_feature: InputFeature = InputFeature(input_ids=input_ids[n_way*k_shot:],
                                                       attention_mask=attention_masks[n_way*k_shot:],
                                                       labels=labels[n_way*k_shot:] if not test else None)
            y_true.extend(labels[n_way*k_shot:])
            label_map = {i_episode: i_whole for i_episode, i_whole in
                         enumerate(torch.unique(labels[n_way*k_shot:]).tolist())}

            outputs: PrototypicalNetworksForwardOutput = self(support_feature, query_feature)
            prediction_per_episode = outputs.distance.argmin(1).tolist()
            prediction = [*map(label_map.get, prediction_per_episode)]
            y_predict.extend(prediction)
            if not test:
                loss = (loss + outputs.loss.item())/(i+1)
        return y_predict, y_true, loss

    @log_metrics
    def evaluate(self,
                 y_predict: List,
                 y_true: List,
                 loss: int,
                 num_epoch: Optional[int] = None) -> Tuple[float, float, str]:
        y_predict = torch.tensor(y_predict)
        y_true = torch.tensor(y_true)
        acc = (y_predict == y_true).sum().item() / y_predict.size(0)
        ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_predict)
        if num_epoch is not None:
            path_to_plot: str = str(Path(self.cfg.model.output_path, self.cfg.name, "plots",
                                         f'confusion_matrix_train_epoch_{num_epoch}.png').absolute())
        else:
            path_to_plot = str(Path(self.cfg.model.output_path, self.cfg.name, "plots",
                                    'confusion_matrix_test.png').absolute())
        plt.savefig(path_to_plot)
        plt.close()
        return acc, loss, path_to_plot

    @set_run_testing
    def test_model(self, data_loader: DataLoader):
        self.eval()
        with torch.no_grad():
            y_predict, y_true, loss = self.run_per_epoch(data_loader, test=True)
            acc, _, _ = self.evaluate(y_predict, y_true, loss)
            logger.warning(f"Testing Accuracy: {acc}")