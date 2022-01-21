from dataclasses import dataclass
from typing import Optional, Tuple
from torch import tensor


@dataclass
class SingleLabelClassificationForwardOutput:
    loss: Optional[tensor] = None
    prediction_logits: tensor = None
    hidden_states: Optional[Tuple[tensor]] = None
    attentions: Optional[Tuple[tensor]] = None


@dataclass
class PrototypicalNetworksForwardOutput:
    loss: Optional[tensor] = None
    distance: Optional[tensor] = None


@dataclass
class InputFeature:
    input_ids: tensor
    attention_mask: tensor
    labels: Optional[tensor] = None


@dataclass
class EncodedFeature:
    encoded_feature: tensor
    labels: Optional[tensor] = None


@dataclass
class TransformedFeature:
    transformed_feature: tensor
    labels: Optional[tensor] = None


@dataclass
class HeadOutput:
    output: tensor
    labels: Optional[tensor] = None