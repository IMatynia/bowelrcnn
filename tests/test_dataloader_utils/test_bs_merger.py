import pytest
from src.new.preprocessing.bs_merger import BowelSoundMerger
from src.new.dataloaders.bowel_sound_csv_file_handler import BowelSoundCSVFileHandler
from src.new.config.model_config import AudioProperties, DatasetSetup, BowelSoundSampleFile
from src.new.audio.audio_utilities import get_wav_length


def test_bs_merger(subscale_data_folder, tmp_path):
    ds = DatasetSetup(
        audio_properties=AudioProperties(
            window_height=64,
            window_width=128,
            window_length=0.2
        ),
        train_files=[],
        valid_files=[],
        test_files=[]
    )
    files = [
        BowelSoundSampleFile(sample_id="0_a"),
        BowelSoundSampleFile(sample_id="0_d"),
    ]
    merger = BowelSoundMerger(subscale_data_folder, tmp_path, ds)
    dropped = merger.merge_bowel_sound_samples("test", files, merge_wav=True)

    # Check file lengths
    a_len = get_wav_length(subscale_data_folder / files[0].sample_wav_name)
    d_len = get_wav_length(subscale_data_folder / files[1].sample_wav_name)
    test_len = get_wav_length(tmp_path / "test.wav")
    assert a_len + d_len == pytest.approx(test_len)

    # Check CSV
    handler = BowelSoundCSVFileHandler(tmp_path / "test.csv")
    bs_gen = handler.load()
    bs_list = list(bs_gen)
    assert len(bs_list) == 7
