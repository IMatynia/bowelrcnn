from torch import nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from src.new.config.cnn_config import TrainingSetup
from src.new.models.iou_loss_modified import ModifiedCompleteBoxIOULoss, ModifiedCompleteBoxIOULossWeighted
from src.new.models.BCE_modified import ModifiedBCELoss
from src.new.models.CE_modified import ModifiedCELoss
import wandb
from src.new.models.reporting.training_stat_reporter import TrainingStatReporterBase
from pathlib import Path
import logging


class ValidationBase:
    def __init__(self, model_save_location: Path) -> None:
        self.last_best_loss = 1e9
        self.model_save_location = model_save_location
        self.epochs_without_improvement = 0


class PatternValidation(ValidationBase):
    def __call__(self, config: TrainingSetup, device: str, model: nn.Module, valid_loader, epoch: int, reporter: TrainingStatReporterBase):
        model.eval()
        with torch.no_grad():
            modified_iou_fn = ModifiedCompleteBoxIOULossWeighted(config).to(device)
            raw_iou_fn = ModifiedCompleteBoxIOULoss(config, reduction="none").to(device)

            raw_iou_losses = np.zeros((len(valid_loader), config.batch_size))
            avg_modified_iou_loss = 0.
            for batch, (X_valid_batch, Y_valid_batch) in enumerate(valid_loader):
                X_valid_batch, Y_valid_batch = X_valid_batch.to(device).float(), Y_valid_batch.to(device).float()
                pred_raw = model(X_valid_batch)
                raw_iou_losses[batch] = raw_iou_fn(pred_raw, Y_valid_batch).cpu().numpy()
                avg_modified_iou_loss += float(modified_iou_fn(pred_raw, Y_valid_batch))

            raw_iou_losses = raw_iou_losses.reshape(-1)
            avg_raw_iou_loss = np.average(raw_iou_losses)
            false_positive_fraction = np.sum(raw_iou_losses > 0.9) / (len(valid_loader) * config.batch_size)

            avg_modified_iou_loss /= len(valid_loader)

            progress = f"{epoch + 1}/{config.n_epochs}"
            logging.info(f"VALIDATION: {progress} | Modified IOU: {avg_modified_iou_loss:5.8f}, Raw IOU: {avg_raw_iou_loss:5.8f}")
            reporter.log({
                "val_iou": avg_raw_iou_loss,
                "val_iou_modified": avg_modified_iou_loss,
                "val_iou_histogram": wandb.Histogram(sequence=raw_iou_losses, num_bins=25),
                # "val_scale_histogram":
                # "val_offset_histogram":
                "val_bad_false_positive": false_positive_fraction,
                "epoch": epoch
            })

            if self.last_best_loss > avg_modified_iou_loss and epoch > 25:
                self.last_best_loss = avg_modified_iou_loss
                self.epochs_without_improvement = 0
                torch.save(model.state_dict(), self.model_save_location)
                logging.info("Better model saved")
            else:
                self.epochs_without_improvement += 1
        return raw_iou_losses.mean()


class ClassifierValidation(ValidationBase):
    def __call__(self, config: TrainingSetup, device: str, model: nn.Module, valid_loader, epoch: int, reporter: TrainingStatReporterBase):
        model.eval()
        with (torch.no_grad()):
            BCE_Modified = ModifiedBCELoss(config).to(device)
            BCE_base = nn.BCEWithLogitsLoss().to(device)

            valid_loss_bce_mod = 0.
            valid_loss_bce = 0.
            for X_valid_batch, Y_valid_batch in valid_loader:
                X_valid_batch, Y_valid_batch = X_valid_batch.to(device).float(), Y_valid_batch.to(device).float()
                pred_raw = model(X_valid_batch)
                valid_loss_bce_mod += float(BCE_Modified(pred_raw, Y_valid_batch))
                valid_loss_bce += float(BCE_base(pred_raw, Y_valid_batch))

            valid_loss_bce_mod /= len(valid_loader)
            valid_loss_bce /= len(valid_loader)
            logging.info(f"VALIDATION: {epoch:5}/{config.n_epochs} | Modified BCE LOSS: {valid_loss_bce_mod:5.8f}, Raw BCE: {valid_loss_bce:5.8f}")
            reporter.log({
                "val_loss": valid_loss_bce_mod,
                "val_bce_loss": valid_loss_bce,
                "epoch": epoch
            })

            if self.last_best_loss > valid_loss_bce_mod and epoch > 25:
                self.epochs_without_improvement = 0
                self.last_best_loss = valid_loss_bce_mod
                torch.save(model.state_dict(), self.model_save_location)
                reporter.save(self.model_save_location)
                logging.info("Better model saved")
            else:
                self.epochs_without_improvement += 1
        return valid_loss_bce_mod
