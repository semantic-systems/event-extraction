from itertools import chain
from omegaconf import DictConfig
from torch.nn import Module, CrossEntropyLoss, Identity
from transformers import AutoModel, AdamW, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
from models import SequenceClassification
from models.heads import LinearLayerHead
from schema import SingleLabelClassificationForwardOutput, InputFeature, EncodedFeature


class SingleLabelSequenceClassification(SequenceClassification):
    def __init__(self, cfg: DictConfig):
        super(SingleLabelSequenceClassification, self).__init__(cfg)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model.from_pretrained, normalization=True)
        params = chain(self.encoder.parameters(), self.classification_head.parameters())
        self.optimizer = AdamW(params, lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()

    def forward(self, input_feature: InputFeature) -> SingleLabelClassificationForwardOutput:
        output = self.encoder(input_ids=input_feature.input_ids, attention_mask=input_feature.attention_mask).pooler_output
        encoded_feature: EncodedFeature = EncodedFeature(encoded_feature=output, labels=input_feature.labels)
        head_output = self.classification_head(encoded_feature)

        if input_feature.labels is not None:
            loss = self.loss(head_output.output, input_feature.labels)
            loss.backward()
            self.optimizer.step()
            return SingleLabelClassificationForwardOutput(loss=loss, prediction_logits=head_output.output)
        else:
            return SingleLabelClassificationForwardOutput(prediction_logits=head_output.output)

    def instantiate_encoder(self) -> PreTrainedModel:
        encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.model.from_pretrained)
        encoder = self.trim_encoder_layers(encoder, self.cfg.model.num_transformer_layers)
        encoder = self.freeze_encoder(encoder, self.cfg.model.freeze_transformer_layers)
        return encoder

    def instantiate_feature_transformer(self) -> Module:
        # this returns an empty Module
        return Identity()

    def instantiate_classification_head(self) -> LinearLayerHead:
        return LinearLayerHead(self.cfg)
