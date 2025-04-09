
from src.new.config.model_config import RCNNModelConfig
from src.new.config.set_seed import set_seeds
from .audio_properties import get_audio_properties_126x64_0_2s_f2000
from .classification_models import make_classification_bigger
from .pattern_models import make_pattern_model_bigger
from .dataset_presets import random_dataset_7_2_1
from .dataset_roots import dataset_126x64


def get_smaller_spec_bigger_net_config():
    SEED = 42
    set_seeds(42)
    window = (64, 126)
    audio_props = get_audio_properties_126x64_0_2s_f2000()
    data_setup = random_dataset_7_2_1(audio_properties=audio_props, dataset_root=dataset_126x64)
    return RCNNModelConfig(
        seed=SEED,
        name="RCNN model with 126x64 spectrogramwith bigger nets",
        version="1.0",
        description="Bowel sound detection model based around the use of region based CNN",
        dataset=data_setup,
        pattern_model=make_pattern_model_bigger(SEED, window),
        classification_model=make_classification_bigger(SEED, window),
    )
