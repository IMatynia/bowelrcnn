from pathlib import Path
from tqdm import tqdm
from src.new.config.model_config import DatasetSetup, BowelSoundSampleFile
from src.new.audio.audio_utilities import get_wav_length, get_wav_samples
from src.new.dataloaders.bowel_sound_csv_file_handler import BowelSoundCSVFileHandler
import soundfile
import logging


class BowelSoundMerger:
    def __init__(self, source_folder: Path, destination_path: Path, data_setup: DatasetSetup):
        self.source = source_folder
        self.destination = destination_path

        self.destination.mkdir(exist_ok=True)

        self.data_setup = data_setup

    def merge_bowel_sound_samples(self, destination_name: str, file_list: list[BowelSoundSampleFile], merge_wav: bool = False):
        logging.info(f"Merging {len(file_list)} files into {destination_name}")
        dst_wav_file = self.destination / f"{destination_name}.wav"
        dst_csv_file = self.destination / f"{destination_name}.csv"

        if merge_wav:
            dst_wav_fd = soundfile.SoundFile(dst_wav_file, samplerate=self.data_setup.audio_properties.sample_rate, mode="w", channels=1)
        dst_csv_handler = BowelSoundCSVFileHandler(dst_csv_file)
        resulting_bowel_sounds = []

        total_length = 0
        dropped_bs = []

        for file in file_list:
            src_wav_file = self.source / file.sample_wav_name
            src_csv_file = self.source / file.sample_csv_name

            src_csv_handler = BowelSoundCSVFileHandler(src_csv_file)

            wav_length = get_wav_length(src_wav_file)

            for bs in src_csv_handler.load():
                bs.start = min(max(bs.start, 0), wav_length)
                bs.end = min(max(bs.end, 0), wav_length)
                bs_offset = bs.offset_by(total_length)
                if bs.length > self.data_setup.audio_properties.window_length or bs.length < 0.000001:
                    logging.warning(f"Invalid bowel sound: {bs}")
                    dropped_bs.append(bs)
                    continue
                resulting_bowel_sounds.append(bs_offset)
            total_length += wav_length

            if merge_wav:
                samples, _ = get_wav_samples(src_wav_file, self.data_setup.audio_properties.sample_rate)
                dst_wav_fd.write(samples)
        dst_csv_handler.save(resulting_bowel_sounds)
        logging.info(f"Created {dst_csv_file}")
        logging.info(f"Dropped {len(dropped_bs)} bowel sounds")
        if merge_wav:
            dst_wav_fd.close()
            logging.info(f"Saved merged wav to {dst_wav_file}")

        return dropped_bs
