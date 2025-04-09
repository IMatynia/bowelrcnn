import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from src.new.config.logging_config import LOG_FORMAT, LOG_LEVEL
from src.new.config.model_config import RCNNModelConfig
from src.new.config.set_seed import set_seeds
from src.new.preprocessing.bs_merger import BowelSoundMerger
import logging
from src.new.audio.spectrogram_handler import SpectrogramHandler
from enum import Enum
import random
import numpy as np


class DATA(str, Enum):
    RAW = "raw"
    INTERMEDIATE = "intermediate"
    PROCESSED = "processed"


class ProcessDataCommand:
    class ProcessDataArgs(Namespace):
        config: Path
        data_root: Path

    def __init__(self, args):
        self.args = args
        self.config = None

    @staticmethod
    def parse_args(parser: ArgumentParser):
        parser.add_argument("--config", required=True, type=Path, help="Config json path")
        parser.add_argument("--data-root", required=True, type=Path, help="Dataset root containig a 'raw' folder")
        return parser.parse_args(namespace=ProcessDataCommand.ProcessDataArgs())

    def merging(self):
        merger = BowelSoundMerger(self.args.data_root / DATA.RAW.value, self.args.data_root / DATA.PROCESSED.value, self.config.dataset)
        train_dropped = merger.merge_bowel_sound_samples("train", self.config.dataset.train_files)
        valid_dropped = merger.merge_bowel_sound_samples("valid", self.config.dataset.valid_files)
        test_dropped = merger.merge_bowel_sound_samples("test", self.config.dataset.test_files, merge_wav=True)

    def spectrogram_gen_all(self):
        self.spectrogram_gen("train", self.config.dataset.train_files)
        self.spectrogram_gen("valid", self.config.dataset.valid_files)
        self.spectrogram_gen("test", self.config.dataset.test_files)

    def spectrogram_gen(self, flavour, file_list):
        logging.info(f"Generating spectrogram for {flavour}")
        handler = SpectrogramHandler(self.config)
        handler.generate_spectrogram_from_wav_list(self.args.data_root / DATA.RAW.value, file_list)
        handler.save(self.args.data_root / DATA.PROCESSED.value / f"{flavour}.bin")

    def run(self):
        logging.info("Starting data prep")
        self.config = RCNNModelConfig.model_validate_json(self.args.config.read_bytes())
        (self.args.data_root / DATA.PROCESSED.value).mkdir(exist_ok=True)
        set_seeds(self.config.seed)

        self.merging()
        self.spectrogram_gen_all()
        logging.info("Finished data prep")


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    parser = ArgumentParser()
    args = ProcessDataCommand.parse_args(parser)
    command = ProcessDataCommand(args)
    command.run()
