from src.new.config.cnn_config import TrainingSetup
from torch import nn
import logging
from tqdm import tqdm
from wandb.wandb_run import Run

from src.new.models.validation.validation import ValidationBase


def train(config: TrainingSetup, train_loader, valid_loader, model: nn.Module, device: str, loss_fn, optimizer, scheduler, validation: ValidationBase, wandb_run: Run):
    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0
        progress = f"{epoch + 1}/{config.n_epochs}"
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {progress}'):
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()

            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f'Epoch {progress}, Loss: {avg_loss:.4f}')
        wandb_run.log({
            "train_loss": avg_loss,
            "epoch": epoch
        })

        if epoch % config.validation_freq == 0:
            val_loss = validation(config, device, model, valid_loader, epoch, wandb_run)
            scheduler.step(val_loss)

        if validation.epochs_without_improvement * config.validation_freq > 75:
            break
