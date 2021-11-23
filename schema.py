from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class SingleLabelClassificationForwardOutput():
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
