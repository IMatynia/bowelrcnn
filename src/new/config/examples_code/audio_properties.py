from src.new.config.cnn_config import *
from src.new.config.model_config import *
from src.new.config.constants import PROJECT_ROOT
import numpy as np
import random


def get_audio_properties_315x126_0_2s_f2000() -> AudioProperties:
    window = (126, 315)
    return AudioProperties(
        window_height=window[0],
        window_width=window[1],
        window_length=0.2,
        max_frequency=2000
    )


def get_audio_properties_126x64_0_2s_f2000() -> AudioProperties:
    window = (64, 126)
    return AudioProperties(
        window_height=window[0],
        window_width=window[1],
        window_length=0.2,
        max_frequency=2000
    )
