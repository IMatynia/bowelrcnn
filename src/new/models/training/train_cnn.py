from src.new.config.constants import NUM_WORKERS_DATALOADERS
from src.new.models.validation.validation import ValidationBase
from .train import train
from src.new.config.cnn_config import CNNConfig, LossFunctions
from src.new.config.model_config import DatasetSetup
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.new.models.cnn_model_builder import CNNModelBuilder
import torch.cuda
from src.new.models.iou_loss_modified import ModifiedDistanceBoxIOULoss, ModifiedCompleteBoxIOULoss, ModifiedCompleteBoxIOULossWeighted
from src.new.models.CE_modified import ModifiedCELoss
from src.new.models.BCE_modified import ModifiedBCELoss
from torch import nn
import torch.optim as optim
from wandb.wandb_run import Run


LOSS_FN_ENUM_TO_LOSS_FN = {
    LossFunctions.DistanceBoxIOU: ModifiedDistanceBoxIOULoss,
    LossFunctions.CompleteBoxIOU: ModifiedCompleteBoxIOULoss,
    LossFunctions.CE: ModifiedCELoss,
    LossFunctions.BCE: ModifiedBCELoss,
    LossFunctions.CompleteBoxIOUWeighted: ModifiedCompleteBoxIOULossWeighted,
    LossFunctions.MSE: lambda _: nn.MSELoss()
}


def train_cnn(model_config: CNNConfig, train_dataset: Dataset, valid_dataset: Dataset, validation: ValidationBase, reporter: Run):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, model_config.training.batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS_DATALOADERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, model_config.training.batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS_DATALOADERS, pin_memory=True)

    model = CNNModelBuilder(model_config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_config.training.lr, weight_decay=model_config.training.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    loss_fn = LOSS_FN_ENUM_TO_LOSS_FN[model_config.training.loss_fn](model_config.training)
    loss_fn = loss_fn.to(device)

    return train(
        model_config.training,
        train_loader,
        valid_loader,
        model,
        device,
        loss_fn,
        optimizer,
        scheduler,
        validation,
        reporter
    )
