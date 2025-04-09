import bisect
from pydantic import BaseModel
from src.new.audio.spectrogram_handler import SpectrogramHandler
from src.new.config.model_config import ModelConfigBase, BowelSoundSampleFile
from typing import Literal, Sized
from pathlib import Path
from more_itertools import windowed
from itertools import chain
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from src.new.dataloaders.bowel_sound_csv_file_handler import BowelSoundCSVFileHandler
import numpy as np

from src.new.utilities.profiler import log_exec_time


class SampleRegionBase(BaseModel):
    start: float
    end: float

    @property
    def length(self):
        return self.end - self.start


class BowelSoundSampleRegion(SampleRegionBase):
    bowel_sound: BowelSoundRaw | None = None


class EmptySampleRegion(SampleRegionBase):
    pass


class BowelSoundFileHandler:
    """
    Klasa zamieniająca plik wav + csv na spektrogram + interfejst do zdobywania okien. Wzoruj się na CNNPatt2Loader
    """

    def __init__(self, config: ModelConfigBase, data_folder_root: Path, sample_file: BowelSoundSampleFile):
        self.config = config
        self.sample_file = sample_file
        self.data_folder_root = data_folder_root
        self.spectrogram = SpectrogramHandler(self.config)
        self.bowel_sound_sampling_regions: list[BowelSoundSampleRegion] = []
        self.bowel_cumulative_lengths: list[float] = []
        self.bowel_sound_total_length: float | None = None
        self.empty_sample_regions: list[EmptySampleRegion] = []
        self.empty_cumulative_lengths: list[float] = []
        self.empty_total_length: float | None = None

    # @log_exec_time
    def _sample_from_regions(self, fractional_sample: float, total_region_length: float, regions: list[SampleRegionBase], cumulative_lengths: list[float]):
        offset_remaining = fractional_sample * total_region_length

        region_idx = bisect.bisect_left(cumulative_lengths, offset_remaining)

        if region_idx == len(cumulative_lengths):
            raise ValueError("Something went wrong")

        region = regions[region_idx - 1]
        region_start = cumulative_lengths[region_idx - 1]
        true_offset = region.start + (offset_remaining - region_start)

        window_half_length = self.config.dataset.audio_properties.window_length / 2
        samples = self.spectrogram.get_spectrogram_start_end(true_offset-window_half_length, true_offset+window_half_length)
        return samples, (true_offset, region)

    def sample_from_empty(self, fractional_sample: float) -> tuple[np.ndarray, tuple[float, EmptySampleRegion | None]]:
        return self._sample_from_regions(fractional_sample, self.empty_total_length, self.empty_sample_regions, self.empty_cumulative_lengths)

    def sample_from_bowel_sounds(self, fractional_sample: float) -> tuple[np.ndarray, tuple[float, BowelSoundSampleRegion | None]]:
        return self._sample_from_regions(fractional_sample, self.bowel_sound_total_length, self.bowel_sound_sampling_regions, self.bowel_cumulative_lengths)

    def load_spectrogram_data(self, source: Literal["FROM WAV", "FROM SPECTROGRAM NUMPY"]):
        # Load wav file or numpy spectrogram file
        # If wavefile, make a spectrogram
        match source:
            case "FROM WAV":
                self.spectrogram.generate_spectrogram_from_wav(
                    self.data_folder_root / self.sample_file.sample_wav_name
                )
            case "FROM SPECTROGRAM NUMPY":
                self.spectrogram.load(
                    self.data_folder_root / self.sample_file.spectrogram_cache
                )

    def get_associated_bowel_sounds_raw(self):
        return list(
            BowelSoundCSVFileHandler(self.data_folder_root / self.sample_file.sample_csv_name).load()
        )

    @log_exec_time
    def prepare_sampling_regions(self, sample_bowel_sounds_raw: list[BowelSoundRaw]):
        window_length = self.config.dataset.audio_properties.window_length
        window_half_length = window_length/2

        bowel_sound_list = chain(
            [BowelSoundRaw(start=0.0, end=0.0)],
            sorted(sample_bowel_sounds_raw),
            [BowelSoundRaw(start=self.spectrogram.true_length, end=self.spectrogram.true_length)],
        )
        # Fill in bs sampling regions
        for prev_bs, current_bs, next_bs in windowed(bowel_sound_list, 3):
            lower_bound_from_midpoints = (prev_bs.midpoint + current_bs.midpoint)/2
            lower_bound_from_window_size = current_bs.midpoint - window_half_length
            true_lower_bound = max(0, lower_bound_from_midpoints, lower_bound_from_window_size)

            upper_bound_from_midpoints = (next_bs.midpoint + current_bs.midpoint) / 2
            upper_bound_from_window_size = current_bs.midpoint + window_half_length
            true_uppper_bound = min(self.spectrogram.true_length, upper_bound_from_midpoints, upper_bound_from_window_size)

            self.bowel_sound_sampling_regions.append(BowelSoundSampleRegion(start=true_lower_bound, end=true_uppper_bound, bowel_sound=current_bs))

        # Fill in empty regions from spaces between bs regions
        region_list = chain(
            [BowelSoundSampleRegion(start=0, end=0)],
            self.bowel_sound_sampling_regions,
            [BowelSoundSampleRegion(start=self.spectrogram.true_length, end=self.spectrogram.true_length)],
        )
        for current_region, next_region in windowed(region_list, 2):
            candidate_region = EmptySampleRegion(start=current_region.end, end=next_region.start)
            if candidate_region.length > 0:
                self.empty_sample_regions.append(candidate_region)

        # Calculate lengths of all regions
        self.bowel_sound_total_length = sum(region.length for region in self.bowel_sound_sampling_regions)
        self.empty_total_length = sum(region.length for region in self.empty_sample_regions)

        # Calculate cumulative lengths for binary search
        self.empty_cumulative_lengths = self._calc_cumulitive_lengths(self.empty_sample_regions)
        self.bowel_cumulative_lengths = self._calc_cumulitive_lengths(self.bowel_sound_sampling_regions)

    def _calc_cumulitive_lengths(self, regions: list[SampleRegionBase]):
        cumulative_lengths = [0]
        total = 0
        for region in regions:
            total += region.length
            cumulative_lengths.append(total)
        return cumulative_lengths
