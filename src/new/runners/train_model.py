from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

import numpy as np
from src.new.config.set_seed import set_seeds
from src.new.models.reporting.training_stat_reporter import PatternModelTrainingReporter, ClassificationModelTrainingReporter
from src.new.config.model_config import RCNNModelConfig
import logging
from enum import Enum
from src.new.models.training.train_cnn import train_cnn
from src.new.datasets.cnn_pattern_bounding_box_dataset import CNNPatternBoundingBoxDataset
from src.new.datasets.cnn_region_classification_dataset import CNNRegionClassificationDataset
from src.new.models.validation.validation import ClassifierValidation, PatternValidation
from src.new.models.training.augments import Augments
from src.new.runners.runner_base import RunnerBase
from src.new.config.logging_config import LOG_LEVEL, LOG_FORMAT


class MODELS(str, Enum):
    pattern_model = "pattern_model"
    classification_model = "classification_model"


class TrainModelCommand(RunnerBase):
    class TrainModelArgs(Namespace):
        config: Path
        data_root: Path
        model_to_train: MODELS
        model_output: Path
        wandb: bool

    def __init__(self, args: TrainModelArgs) -> None:
        super().__init__(args)
        self.config = None

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument("--config", required=True, type=Path, help="Path to config json")
        parser.add_argument("--data-root", required=True, type=Path, help="Dataset folder root")
        parser.add_argument("--model-to-train", required=True, type=MODELS, help=f"Which model to train ({MODELS.pattern_model.value} or {MODELS.classification_model.value})")
        parser.add_argument("--model-output", required=True, type=Path,
                            help="Where to save the resulting model weights")
        parser.add_argument("--wandb", action="store_true", help="Enable wandb integration")

    def handle_pattern_model_training(self):
        augments = [Augments.from_enum(aug_name) for aug_name in
                    self.config.pattern_model.training.augments]

        reporter = PatternModelTrainingReporter(self.config.name, self.config.model_dump(), self.args.wandb)

        try:
            train_cnn(
                self.config.pattern_model,
                train_dataset=CNNPatternBoundingBoxDataset(self.config,
                                                           self.config.pattern_model.training.samples_per_sound,
                                                           self.args.data_root, self.config.dataset.train_sample,
                                                           augments),
                valid_dataset=CNNPatternBoundingBoxDataset(self.config,
                                                           self.config.pattern_model.training.samples_per_sound,
                                                           self.args.data_root, self.config.dataset.valid_sample),
                validation=PatternValidation(
                    model_save_location=self.args.model_output,
                ),
                reporter=reporter
            )
        finally:
            reporter.finish()

    def handle_classification_model_training(self):
        augments = [Augments.from_enum(aug_name) for aug_name in
                    self.config.classification_model.training.augments]

        reporter = ClassificationModelTrainingReporter(self.config.name, self.config.model_dump(), self.args.wandb)

        try:
            train_cnn(
                self.config.classification_model,
                train_dataset=CNNRegionClassificationDataset(self.config,
                                                             self.config.classification_model.training.samples_per_sound,
                                                             self.args.data_root,
                                                             self.config.dataset.train_sample,
                                                             chance_of_sampling_empty=self.config.classification_model.training.chance_of_sampling_empty,
                                                             augments=augments),
                valid_dataset=CNNRegionClassificationDataset(self.config,
                                                             self.config.classification_model.training.samples_per_sound,
                                                             self.args.data_root,
                                                             self.config.dataset.valid_sample,
                                                             chance_of_sampling_empty=self.config.classification_model.training.chance_of_sampling_empty),
                validation=ClassifierValidation(
                    model_save_location=self.args.model_output,
                ),
                reporter=reporter
            )
        finally:
            reporter.finish()

    def run(self):
        self.config = RCNNModelConfig.model_validate_json(self.args.config.read_bytes())
        set_seeds(self.config.seed)

        match self.args.model_to_train:
            case MODELS.pattern_model:
                self.handle_pattern_model_training()
            case MODELS.classification_model:
                self.handle_classification_model_training()


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    parser = ArgumentParser()
    TrainModelCommand.add_arguments(parser)
    command = TrainModelCommand(parser.parse_args())
    command.run()
