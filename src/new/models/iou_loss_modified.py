import torch
import torchvision
from torch import nn

from src.new.config.cnn_config import TrainingSetup


class IOULossBase(nn.Module):
    def __init__(self, config: TrainingSetup, base_2d_iou_loss, reduction="mean"):
        """

        @param config: config helper
        @param base_2d_iou_loss: for example torchvision.ops.distance_box_iou_loss
        @param reduction: loss reduction
        """
        super().__init__()
        self.config = config
        self.base_loss = base_2d_iou_loss
        self.reduction = reduction
        self._device = None

    def to(self, device):
        self._device = device
        return super().to(device)

    def forward(self, pred: torch.Tensor, target: torch.tensor):
        zeros = torch.zeros(self.config.batch_size).to(self._device)
        ones = torch.ones(self.config.batch_size).to(self._device)
        pred_off = pred[:, 0]
        pred_scale = pred[:, 1]

        pred_x1 = pred_off - pred_scale / 2
        pred_x2 = pred_off + pred_scale / 2

        target_off = target[:, 0]
        target_scale = target[:, 1]

        target_x1 = target_off - target_scale / 2
        target_x2 = target_off + target_scale / 2

        return self.base_loss(
            torch.stack([pred_x1, zeros, pred_x2, ones], -1).to(self._device),
            torch.stack([target_x1, zeros, target_x2, ones], -1).to(self._device),
            reduction=self.reduction
        )


class IOULossBaseWithWeight(nn.Module):
    def __init__(self, config: TrainingSetup, base_2d_iou_loss, reduction="mean", outlier_weight: float = 1.5):
        """
        @param config: config helper
        @param base_2d_iou_loss: for example torchvision.ops.distance_box_iou_loss
        @param reduction: loss reduction
        """
        super().__init__()
        self.config = config
        self.base_loss = base_2d_iou_loss
        self.reduction = reduction
        self._device = None
        self.outlier_weight = outlier_weight

    def to(self, device):
        self._device = device
        return super().to(device)

    def forward(self, pred: torch.Tensor, target: torch.tensor):
        zeros = torch.zeros(self.config.batch_size).to(self._device)
        ones = torch.ones(self.config.batch_size).to(self._device)
        pred_off = pred[:, 0]
        pred_scale = pred[:, 1]

        pred_x1 = pred_off - pred_scale / 2
        pred_x2 = pred_off + pred_scale / 2

        target_off = target[:, 0]
        target_scale = target[:, 1]

        target_x1 = target_off - target_scale / 2
        target_x2 = target_off + target_scale / 2

        ious_unweighted = self.base_loss(
            torch.stack([pred_x1, zeros, pred_x2, ones], -1).to(self._device),
            torch.stack([target_x1, zeros, target_x2, ones], -1).to(self._device),
        )

        a = 4*(1-self.outlier_weight)
        b = 4*(self.outlier_weight-1)
        c = self.outlier_weight

        final_iou = ious_unweighted * (a*target_off**2 + b*target_off + c)

        if self.reduction == "mean":
            return torch.mean(final_iou)
        else:
            return final_iou


class ModifiedDistanceBoxIOULoss(IOULossBase):
    def __init__(self, config: TrainingSetup, reduction="mean"):
        super().__init__(config, torchvision.ops.distance_box_iou_loss, reduction)


class ModifiedCompleteBoxIOULoss(IOULossBase):
    def __init__(self, config: TrainingSetup, reduction="mean"):
        super().__init__(config, torchvision.ops.complete_box_iou_loss, reduction)


class ModifiedCompleteBoxIOULossWeighted(IOULossBaseWithWeight):
    def __init__(self, config: TrainingSetup, reduction="mean"):
        super().__init__(config, torchvision.ops.complete_box_iou_loss, reduction)
