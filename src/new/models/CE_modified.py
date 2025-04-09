from torch import nn
import torch
from src.new.config.cnn_config import TrainingSetup


class ModifiedCELoss(nn.CrossEntropyLoss):
    def __init__(self, config: TrainingSetup):
        self._weights = torch.Tensor([config.chance_of_sampling_empty / (1-config.chance_of_sampling_empty), 1.0])
        super().__init__(weight=self._weights)
