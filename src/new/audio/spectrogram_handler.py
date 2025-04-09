from pathlib import Path
from src.new.audio.audio_utilities import get_normalized_spectrogram, get_wav_length
from src.new.config.model_config import BowelSoundSampleFile, ModelConfigBase
import numpy as np
from multiprocessing import Pool, cpu_count
import pickle
import logging as lg
logger = lg.getLogger("spec-handler")


class SpectrogramHandler:
    true_length: float
    """Length of the audio file in seconds
    """
    _spectrogram: np.ndarray
    """Normalized spectrogram values
    """

    def __init__(self, config: ModelConfigBase):
        self._config = config

    def time_offset_to_spectrogram_index(self, time_offset: float):
        file_length_fraction = time_offset / self.true_length
        time_series_length = self._spectrogram.shape[1]
        return int(np.floor(time_series_length*file_length_fraction))

    def process_file(self, args):
        file, root_source, audio_properties = args
        src_wav = root_source / file.sample_wav_name
        spectrogram = get_normalized_spectrogram(src_wav, audio_properties)
        length = get_wav_length(src_wav)
        return spectrogram, length

    def compute_spectrograms_parallel(self, file_list, root_source, audio_properties):
        spectrograms = []
        total_length = 0

        logger.info("Computing spectrograms in parallel")
        args = [(file, root_source, audio_properties) for file in file_list]

        with Pool(cpu_count()) as pool:
            results = list(pool.imap(self.process_file, args))

        for spectrogram, length in results:
            spectrograms.append(spectrogram)
            total_length += length

        concatenated_spectrogram = np.concatenate(spectrograms, axis=1) if len(spectrograms) > 0 else []
        return concatenated_spectrogram, total_length

    def generate_spectrogram_from_wav_list(self, root_source: Path, file_list: list[BowelSoundSampleFile]):
        self._spectrogram, self.true_length = self.compute_spectrograms_parallel(file_list, root_source, self._config.dataset.audio_properties)

    def generate_spectrogram_from_wav_signle(self, file_path: Path):
        self._spectrogram = get_normalized_spectrogram(file_path, self._config.dataset.audio_properties)
        self.true_length = get_wav_length(file_path)

    def load(self, dump_file: str):
        with open(dump_file, "rb") as fd:
            self._spectrogram, self.true_length = pickle.load(fd)

    # @log_exec_time
    def get_spectrogram_start_end(self, start: float, end: float) -> np.ndarray:
        start_idx = self.time_offset_to_spectrogram_index(start)
        end_idx = self.time_offset_to_spectrogram_index(end)

        start_idx = max(0, start_idx)
        end_idx = min(self._spectrogram.shape[1], end_idx)

        width = self._config.dataset.audio_properties.window_width
        pad_amount = width - end_idx + start_idx + 1

        spectrogram_slice = self._spectrogram.T[start_idx:end_idx - 1, :]
        spectrogram_slice = np.pad(spectrogram_slice, ((pad_amount, 0), (0, 0)), mode="edge")
        assert spectrogram_slice.shape == (self._config.dataset.audio_properties.window_shape)
        return spectrogram_slice

    def get_spectrogram_offset_length(self, offset: float, length: float):
        return self.get_spectrogram_start_end(offset, offset+length)

    def save(self, dump_file: Path):
        with open(dump_file, "wb") as fd:
            pickle.dump((self._spectrogram, self.true_length), fd)

    def get_all(self):
        return self._spectrogram
