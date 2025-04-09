import pytest
from src.new.dataloaders.bowel_sound_csv_file_handler import BowelSoundCSVFileHandler
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from pydantic import ValidationError
import numpy as np


def test_bowel_sound_invalid_data():
    with pytest.raises(ValidationError):
        bs = BowelSoundRaw(
            start=1.0,
            end=-1.0
        )


def test_csv_handler_loading(subscale_data_folder):
    handler = BowelSoundCSVFileHandler(subscale_data_folder / "0_a.csv")
    loaded_bs = list(handler.load())
    assert loaded_bs[0].start == pytest.approx(0.14, abs=0.01)
    assert loaded_bs[0].end == pytest.approx(0.1556, abs=0.01)
    assert loaded_bs[0].min_frequency == pytest.approx(1833.82, abs=1)
    assert loaded_bs[0].max_frequency == pytest.approx(1842.7, abs=1)
    assert len(loaded_bs) == 4


def test_csv_handler_writing(tmp_path):
    csv_temp_file = tmp_path / "example.csv"
    handler = BowelSoundCSVFileHandler(csv_temp_file)
    bses = [
        BowelSoundRaw(
            start=1.0,
            end=2.0,
        )
    ]
    handler.save(bses)
    assert csv_temp_file.exists()
    bses_check = list(handler.load())
    assert len(bses_check) == 1
    assert bses_check[0].start == pytest.approx(1.0)
