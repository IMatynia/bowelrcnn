from src.new.config.constants import PROJECT_ROOT
from src.new.config.model_config import RCNNModelConfig
from src.new.runners.train_model import MODELS


from pathlib import Path


class PATHS:
    EXPERIMENTS_PATH = PROJECT_ROOT / "experiments" / "auto"
    CONFIGS_PATH = EXPERIMENTS_PATH / "configs"
    MODELS_PATH = EXPERIMENTS_PATH / "models"
    PREDICTIONS_PATH = EXPERIMENTS_PATH / "predictions"
    PRED_PARAM_EXPERIMENT_PREDS_PATH = PREDICTIONS_PATH / "pred_param_experiment"

    RESULTS_PATH = EXPERIMENTS_PATH / "results"
    PRED_PARAM_EXPERIMENT_RESULTS = RESULTS_PATH / "pred_param_experiment"
    BEST_MODEL_POI_RESULTS = RESULTS_PATH / "best_model_poi"
    BEST_VS_PREV_COMPARISON_RESULTS = RESULTS_PATH / "best_vs_previous"
    BASIC_STATS_RESULTS = RESULTS_PATH / "basic_stats"

    PREVIOUS_MODEL = EXPERIMENTS_PATH / "previous_model"


class ConfigEntry:
    id: str
    config: RCNNModelConfig
    dataset: Path

    def __init__(self, config: RCNNModelConfig, dataset: Path, id: str | None = None):
        self.id = config.name if id is None else id
        self.config = config
        self.dataset = dataset

        self.get_config_path().write_text(config.model_dump_json(indent=4))

    def get_config_path(self):
        return (PATHS.CONFIGS_PATH / self.id).with_suffix(".config.json")

    def get_model_output_location(self, model: MODELS):
        if model == MODELS.classification_model:
            return PATHS.MODELS_PATH / f"{self.config.classification_model.get_hash()}.{model.value}.h5"
        elif model == MODELS.pattern_model:
            return PATHS.MODELS_PATH / f"{self.config.pattern_model.get_hash()}.{model.value}.h5"
        else:
            raise Exception("bad model")

    def get_predictions_path_default(self):
        return PATHS.PREDICTIONS_PATH / f"{self.id}.predictions.csv"

    def get_basic_stats_path(self):
        return PATHS.BASIC_STATS_RESULTS / self.id
