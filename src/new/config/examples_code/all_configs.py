from src.new.config.examples_code.dataset_presets import random_dataset_7_2_1
from src.new.config.set_seed import set_seeds
from .audio_properties import *
from .classification_models import *
from .pattern_models import *
from .dataset_roots import *
from .config_entry import ConfigEntry

_ALL_GENERATORS = []


def mark_config(func):
    _ALL_GENERATORS.append(func)
    return func


@mark_config
def baseline_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"baseline_small_spectrogram_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_baseline(seed, window),
        classification_model=make_classification_model_baseline(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def baseline_big_spectrogram(seed):
    dataset, dataset_root, window = get_315x126_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"baseline_big_spectrogram_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_baseline(seed, window),
        classification_model=make_classification_model_baseline(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def bigger_net_big_spectrogram(seed):
    dataset, dataset_root, window = get_315x126_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"bigger_net_big_spectrogram_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_bigger(seed, window),
        classification_model=make_classification_bigger(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def bigger_net_small_spectrogram(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"bigger_net_small_spectrogram_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_bigger(seed, window),
        classification_model=make_classification_bigger(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def smaller_net_big_spectrogram(seed):
    dataset, dataset_root, window = get_315x126_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"smaller_net_big_spectrogram_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_smaller(seed, window),
        classification_model=make_classification_model_smaller(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def smaller_net_small_spectrogram(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"smaller_net_small_spectrogram_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_smaller(seed, window),
        classification_model=make_classification_model_smaller(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def mse_in_pattern_big_spec(seed):
    dataset, dataset_root, window = get_315x126_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"mse_in_pattern_big_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_MSE_loss(seed, window),
        classification_model=make_classification_model_baseline(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def mse_in_pattern_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"mse_in_pattern_small_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_MSE_loss(seed, window),
        classification_model=make_classification_model_baseline(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def zero_gauss_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"zero_gauss_small_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_no_gauss(seed, window),
        classification_model=make_classification_model_no_gauss(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def gauss_0_15_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"0_15_gauss_small_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_0_15_gauss(seed, window),
        classification_model=make_classification_model_0_15_gauss(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def gauss_0_6_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"0_6_gauss_small_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_0_6_gauss(seed, window),
        classification_model=make_classification_model_0_6_gauss(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def lr_2x_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"2x_lr_small_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_2x_lr(seed, window),
        classification_model=make_classification_model_2x_lr(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def lr_0_5x_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"0_5x_lr_small_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_0_5x_lr(seed, window),
        classification_model=make_classification_model_0_5x_lr(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def drop_0_2_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"0_2_drop_small_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_0_1_droput(seed, window),
        classification_model=make_classification_model_2x_lr(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def drop_0_75_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"0_75_drop_small_spec_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_model_0_5_droput(seed, window),
        classification_model=make_classification_model_0_5_droput(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def more_cnn_layers_small_spec(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"more_cnn_layers_model_{seed}",
        dataset=dataset,
        pattern_model=make_pattern_more_cnn_layers_model(seed, window),
        classification_model=make_classification_more_cnn_layers_model(seed, window)
    )
    return ConfigEntry(config, dataset_root)


@mark_config
def best(seed):
    dataset, dataset_root, window = get_126x64_dataset(seed)
    config = RCNNModelConfig(
        seed=seed,
        name=f"best_BowelRCNN_{seed}",
        dataset=dataset,
        pattern_model=make_best_pattern(seed, window),
        classification_model=make_best_classification(seed, window)
    )
    return ConfigEntry(config, dataset_root)


PRIMARY_SEED = 42
ADDITIONAL_SEEDS = [43, 44, 45, 46]


def get_all_configs() -> dict[str, ConfigEntry]:
    ALL_CONFIGS_LIST = [
        *[conf(PRIMARY_SEED) for conf in _ALL_GENERATORS],
        *[baseline_small_spec(s) for s in ADDITIONAL_SEEDS]
    ]
    return {
        conf.id: conf for conf in ALL_CONFIGS_LIST
    }
