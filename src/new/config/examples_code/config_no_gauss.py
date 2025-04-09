
from src.new.config.model_config import RCNNModelConfig
from src.new.config.set_seed import set_seeds
from .audio_properties import get_audio_properties_315x126_0_2s_f2000
from .classification_models import make_classification_model_no_gauss
from .pattern_models import make_pattern_model_no_gauss
from .dataset_presets import random_dataset_7_2_1
from .dataset_roots import dataset_315x126


def get_config_no_gauss():
    SEED = 42
    set_seeds(42)
    window = (126, 315)
    audio_props = get_audio_properties_315x126_0_2s_f2000()
    data_setup = random_dataset_7_2_1(audio_properties=audio_props, dataset_root=dataset_315x126)
    return RCNNModelConfig(
        seed=SEED,
        name="RCNN model without gauss augmentation",
        version="1.0",
        description="Bowel sound detection model based around the use of region based CNN",
        dataset=data_setup,
        pattern_model=make_pattern_model_no_gauss(SEED, window),
        classification_model=make_classification_model_no_gauss(SEED, window),
    )
