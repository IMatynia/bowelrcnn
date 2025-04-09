from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

import numpy as np
from src.new.config.logging_config import LOG_FORMAT, LOG_LEVEL
from src.new.config.model_config import RCNNModelConfig
from src.new.config.set_seed import set_seeds
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from src.new.models.cnn_model_builder import CNNModelBuilder
from src.new.config.model_config import RCNNModelConfig
from src.new.models.testing.inference import CombinedNetworkOutput, RCNNInferenceHandler
from src.new.audio.spectrogram_handler import SpectrogramHandler
from src.new.dataloaders.bowel_sound_csv_file_handler import BowelSoundCSVFileHandler
import torch
import logging
import pickle


class ModelInferenceCommand:
    class ModelInferenceArgs(Namespace):
        config: Path
        wav_file: Path | None
        spectrogram_dump: Path | None
        output: Path
        pattern_model_weights: Path
        classification_model_weights: Path
        detection_treshold: float
        min_vote_fraction: float
        region_overlap: int

    def __init__(self, args) -> None:
        self.args = args
        assert self.args.wav_file or self.args.spectrogram_dump, "Must set either wav or spectorgram file as source"
        self.config = RCNNModelConfig.model_validate_json(self.args.config.read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def parse_args(parser: ArgumentParser):
        parser.add_argument("--config", required=True, type=Path)
        parser.add_argument("--wav-file", type=Path, help="Data source to infer BS from, exclusive with spectrogram dump")
        parser.add_argument("--spectrogram-dump", type=Path, help="Data source to infer BS from, exclusive with wav file")
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--pattern-model-weights", required=True, type=Path)
        parser.add_argument("--classification-model-weights", required=True, type=Path)
        parser.add_argument("--detection-treshold", required=True, type=float)
        parser.add_argument("--min-vote-fraction", required=True, type=float)
        parser.add_argument("--region-overlap", required=True, type=int)
        return parser.parse_args(namespace=ModelInferenceCommand.ModelInferenceArgs())

    def load_models(self):
        # Load pattern model
        self.pattern_model = CNNModelBuilder(self.config.pattern_model)
        self.pattern_model.load_state_dict(torch.load(self.args.pattern_model_weights))
        self.pattern_model.to(self.device)

        # Load classification model
        self.classification_model = CNNModelBuilder(self.config.classification_model)
        self.classification_model.load_state_dict(torch.load(self.args.classification_model_weights))
        self.classification_model.to(self.device)

    def do_inference(self):
        self.handler = RCNNInferenceHandler(
            self.config,
            self.classification_model,
            self.pattern_model,
            self.device,
            inference_overlap=self.args.region_overlap,
            vote_fraction=self.args.min_vote_fraction,
            classification_threshold=self.args.detection_treshold,
        )

        logging.info("Infering")
        spectrogram_handler = SpectrogramHandler(self.config)

        if self.args.wav_file:
            spectrogram_handler.generate_spectrogram_from_wav_signle(self.args.wav_file)
        elif self.args.spectrogram_dump:
            spectrogram_handler.load(self.args.spectrogram_dump)

        # Infer all
        return self.handler.infer_all_sounds_in_spectrogram(spectrogram_handler.get_all(), padding="extend")

    def save_inferences(self, inferences: list[BowelSoundRaw]):
        logging.info("Saving inferences to a file")
        csv_handler = BowelSoundCSVFileHandler(self.args.output)
        csv_handler.save(inferences)

    def save_raw_detections(self, raw_detections: list[CombinedNetworkOutput], confidence_x, confidence_y):
        logging.info("Saving raw statistics")
        stats_path = self.args.output.with_suffix(".bin")
        data = {
            "configuration": {
                "detection_treshold": self.args.detection_treshold,
                "min_vote_fraction": self.args.min_vote_fraction,
                "region_overlap": self.args.region_overlap,
            },
            "detections": raw_detections,
            "confidence": {
                "x": confidence_x,
                "y": confidence_y
            }
        }
        with open(stats_path, mode="wb") as fd:
            pickle.dump(data, fd)

    def run(self):
        self.load_models()
        set_seeds(self.config.seed)

        inferences, confidence_x, confidence_y, raw_detections = self.do_inference()
        self.save_inferences(inferences)
        self.save_raw_detections(raw_detections, confidence_x, confidence_y)


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    parser = ArgumentParser()
    args = ModelInferenceCommand.parse_args(parser)
    command = ModelInferenceCommand(args)
    command.run()
