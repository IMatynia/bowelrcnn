import numpy as np

from src.new.audio.spectrogram_handler import SpectrogramHandler
from src.new.config.model_config import ModelConfigBase, BowelSoundSampleFile
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from src.new.dataloaders.bowel_sound_file_handler import BowelSoundFileHandler


def test_simple_region_processing(tmp_path):
    config = ModelConfigBase.from_example()
    config.dataset.audio_properties.window_length = 3.0
    sample_file = BowelSoundSampleFile(
        sample_id="test"
    )
    handler = BowelSoundFileHandler(config, ".", sample_file)
    handler.spectrogram = SpectrogramHandler(config)
    handler.spectrogram.true_length = 20
    handler.prepare_sampling_regions([
        BowelSoundRaw(start=1.0, end=2.0),
        BowelSoundRaw(start=6.0, end=7.0),
        BowelSoundRaw(start=11.0, end=13.0),
        BowelSoundRaw(start=8.0, end=10.0),
    ])
    assert len(handler.empty_sample_regions) == 3
    assert len(handler.bowel_sound_sampling_regions) == 4
    assert handler.bowel_sound_sampling_regions[0].start == 0.75
    assert handler.bowel_sound_sampling_regions[1].end == 7.75
    assert handler.empty_sample_regions[-1].start == 13.5
    assert handler.bowel_sound_total_length == 10.75
    assert handler.empty_total_length == 9.25
    pass
