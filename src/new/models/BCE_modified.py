from torch import nn
import torch
from src.new.config.cnn_config import TrainingSetup


class ModifiedBCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, config: TrainingSetup):
        self._weights = torch.Tensor([config.chance_of_sampling_empty / (1-config.chance_of_sampling_empty)])
        super().__init__(pos_weight=self._weights)
