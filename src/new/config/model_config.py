from typing_extensions import Self

from pydantic import BaseModel, Field
from typing import Literal, Any
from pathlib import Path
from src.new.config.cnn_config import CNNConfig
from enum import Enum

ModelTypes = Literal["CNN"]


class BowelSoundSampleFile(BaseModel):
    sample_id: str

    @property
    def sample_wav_name(self):
        return f"{self.sample_id}.wav"

    @property
    def sample_csv_name(self):
        return f"{self.sample_id}.csv"

    @property
    def spectrogram_cache(self):
        return f"{self.sample_id}.bin"


class AudioProperties(BaseModel):
    sample_rate: int = 44100
    window_height: int = 64
    """Window frequency resolution"""
    window_width: int = 128
    """Window temporal resolution"""
    window_length: float = 1.0
    """Sliding window length in seconds"""

    max_frequency: int = 2000
    """Frequency cutoff"""
    spectrogram_hop_length: float = -1
    spectrogram_fft: float = -1
    audio_mean: float = -47.913372
    audio_std: float = 17.81253
    spectrogram_window_type: str = "hann"

    def model_post_init(self, *args, **kwargs):
        self.spectrogram_hop_length = self.window_length / self.window_width
        self.spectrogram_fft = self.window_height / self.max_frequency

        assert int(self.sample_rate*self.spectrogram_hop_length) == int(self.sample_rate*self.spectrogram_hop_length)

    @property
    def window_shape(self):
        return (self.window_width, self.window_height)


class DatasetSetup(BaseModel):
    audio_properties: AudioProperties
    train_files: list[BowelSoundSampleFile]
    valid_files: list[BowelSoundSampleFile]
    test_files: list[BowelSoundSampleFile]

    @property
    def train_sample(self):
        return BowelSoundSampleFile(sample_id="train")

    @property
    def valid_sample(self):
        return BowelSoundSampleFile(sample_id="valid")

    @property
    def test_sample(self):
        return BowelSoundSampleFile(sample_id="test")


class ModelConfigBase(BaseModel):
    name: str
    version: str = "1.0"
    type_model: ModelTypes
    description: str = ""
    dataset: DatasetSetup
    seed: int

    @classmethod
    def from_example(cls):
        return cls(
            name="example",
            version="example",
            type_model="CNN",
            description="example",
            seed=123,
            dataset=DatasetSetup(
                audio_properties=AudioProperties(window_height=100, window_width=100),
                dataset_root=Path("."),
                train_files=[],
                valid_files=[],
                test_files=[],
            ),
        )

    def __str__(self):
        return f"{self.name}-{self.version}-{self.type_model}"


class RCNNModelConfig(ModelConfigBase):
    type_model: ModelTypes = "CNN"
    pattern_model: CNNConfig
    classification_model: CNNConfig
