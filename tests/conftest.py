import pytest
from pathlib import Path

from src.new.models.testing.statistics import BasicMetricsCalculator

TESTS_ROOT = Path(__file__).parent


@pytest.fixture()
def subscale_data_folder():
    return TESTS_ROOT / "sub_scale_test_data"
