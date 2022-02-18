import torch

from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss, Dropout, Identity
from transformers import PreTrainedModel, AdamW, PreTrainedTokenizer, AutoTokenizer, AutoModel

from models import SequenceClassification
from models.heads.prototypical_head import PrototypicalHead
from schema import InputFeature, PrototypicalNetworksForwardOutput, EncodedFeature


class PrototypicalNetworks(SequenceClassification):
    def __init__(self, cfg: DictConfig):
        super(PrototypicalNetworks, self).__init__(cfg)
        self.optimizer = AdamW(self.parameters(), lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()
        self.dropout = Dropout(p=cfg.model.dropout_rate)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model.from_pretrained, normalization=True)

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

        label_map = {i_whole: i_episode for i_episode, i_whole in enumerate(torch.unique(support_encoded_feature.labels).tolist())}
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
