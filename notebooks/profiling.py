import init
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor

import numpy as np

import os
DATA_ROOT = "../data/processed"
assert os.path.exists(DATA_ROOT)

from src.loaders.CNNPatt2Schemas import CNNPatt2NetworkOutput
from src.loaders.CNNPatt2Dataset import CNNPatt2Dataset

hop_len = 0.004
fft = 0.025
max_feq = 2000
sample_len = 0.2

H = max_feq*fft
W = sample_len/hop_len

print(H, W)

config = {
    "audio_mean": -47.913372,
    "audio_std": 17.81253,
    "fft": fft,
    "frequency": 44100,
    "hop_length": hop_len,
    "max_freq": max_feq,
    "overlapping": 2.0,
    "window_type": "hann"
}
# setting up datasets
train_data = CNNPatt2Dataset(os.path.join(DATA_ROOT, "train"), config, samples_per_sound=3, sample_length=sample_len)
valid_data = CNNPatt2Dataset(os.path.join(DATA_ROOT, "valid"), config, samples_per_sound=3, sample_length=sample_len)

# set up data loader
BATCH_SIZE = 8
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)


class CNNPatt21Network(nn.Module):
    OUTPUT_SHAPE = 4

    def __init__(self):
        super().__init__()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.4)
        self.batchNorm128 = nn.BatchNorm1d(128)
        self.conv2d_a = nn.Conv2d(1, 10, 3)
        self.conv2d_b = nn.Conv2d(10, 5, 5)
        self.max_pool = nn.MaxPool2d(5, 5)
        self.flatten = nn.Flatten()
        self.fc_a = nn.Linear(360, 128)
        self.fc_b = nn.Linear(128, self.OUTPUT_SHAPE)

    def forward(self, x: torch.Tensor):
        x = self.conv2d_a(x)
        x = self.conv2d_b(x)
        x = self.max_pool(x)

        x = self.flatten(x)
        x = self.act(x)

        x = self.fc_a(x)
        x = self.batchNorm128(x)
        x = self.dropout(x)
        x = self.act(x)

        x = self.fc_b(x)
        y = self.act(x)
        return y


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_multiplier: float = 10):
        super().__init__()
        self._weight_multiplier = weight_multiplier - 1  # Class imbalance accounted for

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tensor:
        squared_error = (pred - target) ** 2
        weights = self._weight_multiplier * target[:, 0]
        error = torch.matmul(squared_error, weights.transpose(0, -1)) + squared_error
        return torch.mean(squared_error)


def IOU_loss(pred: CNNPatt2NetworkOutput, target: CNNPatt2NetworkOutput):
    pred_s, pred_e = pred.relative_start_end
    target_s, target_e = target.relative_start_end

    intersection = min(pred_e, target_e) - max(pred_s, target_s)
    union = max(pred_e, target_e) - min(pred_s, target_s)

    iou = intersection / union
    return iou * pred.detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Runnning on {device}")
model = CNNPatt21Network().to(device)
optimizer = optim.AdamW(model.parameters())
loss_fn = WeightedMSELoss(weight_multiplier=10).to(device)

n_epochs = 50
for epoch in range(n_epochs):
    print("epoch:", epoch)
    model.train()
    batches = 0
    for X_batch, y_batch in train_loader:
        batches += 1
        print("batch:", batches)
        X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 5 != 0:
        continue
    model.eval()
    with torch.no_grad():
        total_total_iou = 0
        total_total_MSE = 0
        batches = 0
        for X_valid_batch, Y_valid_batch in valid_loader:
            batches += 1
            X_valid_batch, Y_valid_batch = X_valid_batch.to(device).float(), Y_valid_batch.to(device).float()
            pred = model(X_valid_batch)
            total_iou_loss = 0
            for sample_pred, sample_valid_Y in zip(pred, Y_valid_batch):
                pred = CNNPatt2NetworkOutput.from_numpy(pred)
                target = CNNPatt2NetworkOutput.from_numpy(target)
                iou_loss = IOU_loss(pred, target)
                # TODO : ONLY WORKS ON THE SINGULAR SOUND OR NO SOUND (SAME AS IN TRAINING, NOT ACTUAL INFERENCE)
                total_iou_loss += iou_loss
                total_total_MSE += torch.nn.MSELoss().forward(sample_pred, sample_valid_Y).values()
            total_total_iou += total_iou_loss / len(pred)
        total_total_iou /= batches
        total_total_MSE /= batches
        print(f"EPOCH: {epoch:10} IOU LOSS: {total_total_iou:10}, MSE LOSS: {total_total_MSE:10}")

