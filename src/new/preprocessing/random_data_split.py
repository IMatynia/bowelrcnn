from src.new.config.model_config import DatasetSetup, BowelSoundSampleFile, AudioProperties
from pathlib import Path
import random


class RandomDatasetSplit:
    """Assumes all files are equal in length, simply assigns files from a folder to a dataset"""

    def __init__(self, dataset_root: Path, train_split: float, valid_split: float, test_split: float, audio_properties: AudioProperties):
        self.dataset_root = dataset_root
        split_sum = train_split + valid_split + test_split
        self.train_split = train_split / split_sum
        self.valid_split = valid_split / split_sum
        self.test_split = test_split / split_sum
        self.audio_properties = audio_properties

    def get_all_bs_sample_ids(self):
        bs_sample_ids = []
        for file in (self.dataset_root / "raw").iterdir():
            if file.is_file() and file.exists() and file.name.endswith(".wav"):
                bs_sample_ids.append(file.stem)
        return bs_sample_ids

    def get_random_data_setup(self):
        all_ids = self.get_all_bs_sample_ids()
        all_ids = list(map(lambda sample_id: BowelSoundSampleFile(sample_id=sample_id), all_ids))
        random.shuffle(all_ids)

        train_valid_split_idx = round(self.train_split * len(all_ids))
        valid_test_split_idx = round((self.train_split + self.valid_split) * len(all_ids))

        random_data_setup = DatasetSetup(
            audio_properties=self.audio_properties,
            dataset_root=self.dataset_root,
            train_files=all_ids[:train_valid_split_idx],
            valid_files=all_ids[train_valid_split_idx + 1:valid_test_split_idx],
            test_files=all_ids[valid_test_split_idx + 1:]
        )
        return random_data_setup
