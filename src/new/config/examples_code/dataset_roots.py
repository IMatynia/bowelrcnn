from pathlib import Path
from src.new.config.constants import PROJECT_ROOT
from src.new.config.examples_code.dataset_presets import random_dataset_7_2_1
from src.new.config.set_seed import set_seeds
from .audio_properties import *

window_315x126 = (126, 315)
window_126x64 = (64, 126)


def dataset_315x126(seed): return PROJECT_ROOT / "datasets" / f"315x126_{seed}"


def dataset_126x64(seed): return PROJECT_ROOT / "datasets" / f"126x64_{seed}"


def get_315x126_dataset(seed):
    set_seeds(seed)
    audio_props = get_audio_properties_315x126_0_2s_f2000()
    return random_dataset_7_2_1(audio_properties=audio_props, dataset_root=dataset_315x126(seed)), dataset_315x126(seed), window_315x126


def get_126x64_dataset(seed):
    set_seeds(seed)
    audio_props = get_audio_properties_126x64_0_2s_f2000()
    return random_dataset_7_2_1(audio_properties=audio_props, dataset_root=dataset_126x64(seed)), dataset_126x64(seed), window_126x64
