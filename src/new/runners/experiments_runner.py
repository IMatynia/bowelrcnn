import argparse
import csv
from collections import defaultdict
import json
import torchinfo
from src.new.config.examples_code.all_configs import ADDITIONAL_SEEDS, PRIMARY_SEED, get_all_configs
from itertools import product
import numpy as np
from matplotlib import pyplot as plt

from src.new.config.examples_code.config_entry import PATHS
from src.new.models.cnn_model_builder import CNNModelBuilder
from src.new.config.examples_code.config_entry import ConfigEntry
from src.new.runners.train_model import MODELS, TrainModelCommand
from src.new.runners.test_model import ModelTestCommand
from src.new.runners.model_inference import ModelInferenceCommand
from src.new.runners.process_data import DATA, ProcessDataCommand
from src.new.runners.compare_preds import ComparePredsCommand
from src.new.runners.meta_algorithm import CSVPredictionsCombiner
from src.new.config.logging_config import LOG_FORMAT, LOG_LEVEL
import logging as lg
logging = lg.getLogger("experiment_runner")
lg.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


[getattr(PATHS, pth).mkdir(parents=True, exist_ok=True) for pth in filter(lambda s: not s.startswith("_"), dir(PATHS))]


class MODEL_NAMES:
    PREVIOUS_MODEL_ID = "CRNN"
    META_ALG_ID_HEADER = "meta_algorithm"
    META_ALG_ID_SUM = "meta_algorithm_sum"
    META_ALG_ID_INTERSECT = "meta_algorithm_intersect"
    BEST_MODEL_ID = "best_BowelRCNN_42"
    BIG_SPEC_BASELINE = "baseline_big_spectrogram_{seed}"
    SMALL_SPEC_BASELINE = "baseline_small_spectrogram_{seed}"


CONFIGS = get_all_configs()


class PredParamExperimentEntry:
    ALL_TRESHOLDS = [0.5, 0.75, 0.9]
    ALL_VOTE = [0.05, 0.1, 0.2, 0.4]
    ALL_OVERLAPS = [1, 5, 10, 25]

    treshold: float
    vote: float
    overlap: int

    def __init__(self, treshold, vote, overlap, config_entry: ConfigEntry):
        self.treshold = treshold
        self.vote = vote
        self.overlap = overlap

        self.id = f"pred_param_experiment;{config_entry.config.seed};{self.treshold};{self.vote};{self.overlap}".replace(".", "_")
        self.entry_ref = config_entry

    @ property
    def dataset(self):
        return self.entry_ref.dataset

    @ property
    def config(self):
        return self.entry_ref.config

    def get_config_path(self):
        return self.entry_ref.get_config_path()

    def get_model_output_location(self, model):
        return self.entry_ref.get_model_output_location(model)

    def get_predictions_path_default(self):
        return PATHS.PRED_PARAM_EXPERIMENT_PREDS_PATH / f"{self.id}.predictions.csv"

    def get_basic_stats_path(self):
        return PATHS.PRED_PARAM_EXPERIMENT_RESULTS / self.id

    @ classmethod
    def gen_experiments(cls, config_entry: ConfigEntry) -> dict[str, "PredParamExperimentEntry"]:
        experiments = {}
        for treshold, vote, overlap in product(cls.ALL_TRESHOLDS, cls.ALL_VOTE, cls.ALL_OVERLAPS):
            entry = cls(treshold, vote, overlap, config_entry)
            experiments[entry.id] = entry
        return experiments


BEST_MODEL_CONFIG = CONFIGS[MODEL_NAMES.BEST_MODEL_ID]
BASELINE_CONFIG = CONFIGS[MODEL_NAMES.SMALL_SPEC_BASELINE.format(seed=42)]
# PRED_PARAM_EXPERIMENT_ENTRIES = PredParamExperimentEntry.gen_experiments(BEST_MODEL_CONFIG)

previous_preds = (PATHS.PREVIOUS_MODEL / "predictions.csv").read_text()
PREV_CONFIG = ConfigEntry(
    id=MODEL_NAMES.PREVIOUS_MODEL_ID,
    config=BEST_MODEL_CONFIG.config,
    dataset=BEST_MODEL_CONFIG.dataset
)
PREV_CONFIG.get_predictions_path_default().write_text(previous_preds)
META_INTERSECT_CONFIG = ConfigEntry(
    id=MODEL_NAMES.META_ALG_ID_INTERSECT,
    config=BEST_MODEL_CONFIG.config,
    dataset=BEST_MODEL_CONFIG.dataset
)
META_SUM_CONFIG = ConfigEntry(
    id=MODEL_NAMES.META_ALG_ID_SUM,
    config=BEST_MODEL_CONFIG.config,
    dataset=BEST_MODEL_CONFIG.dataset
)


def data_gen():
    for seed in [PRIMARY_SEED, *ADDITIONAL_SEEDS]:
        small_config = CONFIGS[MODEL_NAMES.SMALL_SPEC_BASELINE.format(seed=seed)]
        args = ProcessDataCommand.ProcessDataArgs(
            config=small_config.get_config_path(),
            data_root=small_config.dataset,
        )
        processor = ProcessDataCommand(args)
        processor.run()

        # Bigger res dataset gen - primary seed only
    big_config = CONFIGS[MODEL_NAMES.BIG_SPEC_BASELINE.format(seed=PRIMARY_SEED)]
    args = ProcessDataCommand.ProcessDataArgs(
        config=big_config.get_config_path(),
        data_root=big_config.dataset,
    )
    processor = ProcessDataCommand(args)
    processor.run()


def main(args):
    SKIP_MODELS: list[str] = args.skip_models
    SKIP_EXPERIMENTS: list[str] = args.skip_experiments

    if not "data_gen" in SKIP_EXPERIMENTS:
        # Smaller res dataset gen
        data_gen()

    # Skip models
    for s in SKIP_MODELS:
        if s in CONFIGS:
            del CONFIGS[s]

    # Print summaries
    for config in CONFIGS.values():
        print_model_summaries(config)

    if not "training" in SKIP_EXPERIMENTS:
        training_step()

    PRED_PARAM_EXPERIMENT_ENTRIES = generate_all_pred_param_experimetns()

    if not "pred_param_experiment_preds" in SKIP_EXPERIMENTS:
        pred_param_experiment_predictions(PRED_PARAM_EXPERIMENT_ENTRIES)

    if not "preds_base" in SKIP_EXPERIMENTS:
        generate_base_predictions()

    if not "meta_algorithm" in SKIP_EXPERIMENTS:
        generate_meta_algorithm_results()

    if not "testing_basic" in SKIP_EXPERIMENTS:
        # All experiments tested
        for entry in CONFIGS.values():
            test_models_predictions_basic(entry)

    if not "testing_pred_param_experiment" in SKIP_EXPERIMENTS:
        for entry in PRED_PARAM_EXPERIMENT_ENTRIES.values():
            test_models_predictions_basic(entry)

    if not "testing_basic_additional" in SKIP_EXPERIMENTS:
        # Test additional csv predictions
        extras = [
            PREV_CONFIG,
            META_INTERSECT_CONFIG,
            META_SUM_CONFIG,
        ]
        for entry in extras:
            test_models_predictions_basic(entry)

    if not "basic_stats_summary" in SKIP_EXPERIMENTS:
        summarize_basic_stat_jsons_to_csv()

    if not "testing_visualizations" in SKIP_EXPERIMENTS:
        test_models_predictions_visuals(BEST_MODEL_CONFIG)
        test_models_predictions_visuals(PREV_CONFIG)

    if not "pred_param_experiment_heatmap" in SKIP_EXPERIMENTS:
        logging.info("Creating inference parameter heatmap")
        inference_experiment_heatmap(PRED_PARAM_EXPERIMENT_ENTRIES)

    if not "model_comparison" in SKIP_EXPERIMENTS:
        # Compare best and previous and plot their predictions to see how they compare
        model_comparison_step()


def model_comparison_step():
    args = ComparePredsCommand.ComparePredsArgs(
        config=BEST_MODEL_CONFIG.get_config_path(),
        output=PATHS.BEST_VS_PREV_COMPARISON_RESULTS / "compare_best_and_previous",
        predictions=BEST_MODEL_CONFIG.get_predictions_path_default(),
        predictions_compare=PREV_CONFIG.get_predictions_path_default(),
        ground_truth=BEST_MODEL_CONFIG.dataset / DATA.PROCESSED.value / "test.csv",
    )
    comparer = ComparePredsCommand(args)
    comparer.run()


def generate_meta_algorithm_results():
    logging.info("Creating meta algorithm predictions")
    args = CSVPredictionsCombiner.CSVPredictionsCombinerArgs(
        combined_predictions=PATHS.PREDICTIONS_PATH / MODEL_NAMES.META_ALG_ID_HEADER,
        predictions_A=BEST_MODEL_CONFIG.get_predictions_path_default(),
        predictions_B=PREV_CONFIG.get_predictions_path_default(),
    )
    combiner = CSVPredictionsCombiner(args)
    combiner.run()


def generate_base_predictions():
    best_treshold = 0.9
    best_vote_frac = 0.1
    best_region_overlap = 25
    for config_entry in CONFIGS.values():
        logging.info(f"Inference for {config_entry.id}")
        try:

            args = ModelInferenceCommand.ModelInferenceArgs(
                config=config_entry.get_config_path(),
                spectrogram_dump=config_entry.dataset / DATA.PROCESSED.value / "test.bin",
                output=config_entry.get_predictions_path_default(),
                wav_file=None,
                pattern_model_weights=config_entry.get_model_output_location(MODELS.pattern_model),
                classification_model_weights=config_entry.get_model_output_location(MODELS.classification_model),
                detection_treshold=best_treshold,
                min_vote_fraction=best_vote_frac,
                region_overlap=best_region_overlap
            )
            inferer = ModelInferenceCommand(args)
            inferer.run()
        except Exception as e:
            logging.critical(e)


def generate_all_pred_param_experimetns():
    PRED_PARAM_EXPERIMENT_ENTRIES = {}
    for seed in [PRIMARY_SEED, *ADDITIONAL_SEEDS]:
        config = CONFIGS[MODEL_NAMES.SMALL_SPEC_BASELINE.format(seed=seed)]
        PRED_PARAM_EXPERIMENT_ENTRIES |= PredParamExperimentEntry.gen_experiments(config)
    return PRED_PARAM_EXPERIMENT_ENTRIES


def pred_param_experiment_predictions(PRED_PARAM_EXPERIMENT_ENTRIES):
    for experiment_entry in PRED_PARAM_EXPERIMENT_ENTRIES.values():
        logging.info(f"Inference heatmap for {experiment_entry.id}")
        try:
            args = ModelInferenceCommand.ModelInferenceArgs(
                config=experiment_entry.get_config_path(),
                spectrogram_dump=experiment_entry.dataset / DATA.PROCESSED.value / "test.bin",
                output=(PATHS.PRED_PARAM_EXPERIMENT_PREDS_PATH / experiment_entry.id).with_suffix(".predictions.csv"),
                wav_file=None,
                pattern_model_weights=experiment_entry.get_model_output_location(MODELS.pattern_model),
                classification_model_weights=experiment_entry.get_model_output_location(MODELS.classification_model),
                detection_treshold=experiment_entry.treshold,
                min_vote_fraction=experiment_entry.vote,
                region_overlap=experiment_entry.overlap
            )
            inferer = ModelInferenceCommand(args)
            inferer.run()
        except Exception as e:
            logging.critical(e)


def training_step():
    for config in CONFIGS.values():
        logging.info(f"Training for {config.id}")
        try:
            class_model_loc = config.get_model_output_location(MODELS.classification_model)
            if class_model_loc.exists():
                logging.warning(f"COMPATIBLE CLASS MODEL ALREADY TRAINED FOR {config.id}")
            else:
                args = TrainModelCommand.TrainModelArgs(
                    config=config.get_config_path(),
                    data_root=config.dataset,
                    model_to_train=MODELS.classification_model,
                    model_output=config.get_model_output_location(MODELS.classification_model),
                    wandb=True
                )
                trainer = TrainModelCommand(args)
                trainer.run()

            patt_model_loc = config.get_model_output_location(MODELS.pattern_model)
            if patt_model_loc.exists():
                logging.warning(f"COMPATIBLE PATT MODEL ALREADY TRAINED FOR {config.id}")
            else:
                args = TrainModelCommand.TrainModelArgs(
                    config=config.get_config_path(),
                    data_root=config.dataset,
                    model_to_train=MODELS.pattern_model,
                    model_output=patt_model_loc,
                    wandb=True
                )
                trainer = TrainModelCommand(args)
                trainer.run()
        except Exception as e:
            logging.critical(e)


def print_model_summaries():
    for s in CONFIGS.values():
        print_model_summaries(s)


def print_model_summaries(entry: ConfigEntry):
    pattern_model = CNNModelBuilder(entry.config.pattern_model)
    class_model = CNNModelBuilder(entry.config.classification_model)
    summary_patt = torchinfo.summary(pattern_model, (1, 1, *entry.config.dataset.audio_properties.window_shape), verbose=0)
    summary_class = torchinfo.summary(class_model, (1, 1, *entry.config.dataset.audio_properties.window_shape), verbose=0)
    (PATHS.BASIC_STATS_RESULTS / f"{entry.id}_patt.txt").write_text(str(summary_patt))
    (PATHS.BASIC_STATS_RESULTS / f"{entry.id}_class.txt").write_text(str(summary_class))


def summarize_basic_stat_jsons_to_csv():
    logging.info("Generating summary CSV files")
    header, rows = summarize_stats(
        models=[
            CONFIGS["baseline_big_spectrogram_42"],
            CONFIGS["bigger_net_big_spectrogram_42"],
            CONFIGS["smaller_net_big_spectrogram_42"],
            CONFIGS["mse_in_pattern_big_spec_42"],
        ],
        metrics=[
            "avg_iou", "accuracy", "precision", "recall", "specificity", "f1_score"
        ])

    with open(PATHS.RESULTS_PATH / "compare_big_spectrogram.csv", "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    ################################
    header, rows = summarize_stats(
        models=[
            CONFIGS["zero_gauss_small_spec_42"],
            CONFIGS["0_15_gauss_small_spec_42"],
            CONFIGS["baseline_small_spectrogram_42"],
            CONFIGS["0_6_gauss_small_spec_42"],
        ],
        metrics=[
            "avg_iou", "accuracy", "precision", "recall", "specificity", "f1_score"
        ])

    with open(PATHS.RESULTS_PATH / "compare_small_gauss.csv", "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    ################################

    header, rows = summarize_stats(
        models=[
            CONFIGS["0_5x_lr_small_spec_42"],
            CONFIGS["baseline_small_spectrogram_42"],
            CONFIGS["2x_lr_small_spec_42"],
        ],
        metrics=[
            "avg_iou", "accuracy", "precision", "recall", "specificity", "f1_score"
        ])

    with open(PATHS.RESULTS_PATH / "comare_small_lr.csv", "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    ################################

    header, rows = summarize_stats(
        models=[
            CONFIGS["baseline_small_spectrogram_42"],
            PREV_CONFIG,
            META_INTERSECT_CONFIG,
            META_SUM_CONFIG
        ],
        metrics=[
            "avg_iou", "accuracy", "precision", "recall", "specificity", "f1_score"
        ])

    with open(PATHS.RESULTS_PATH / "compare_standalone_meta_old.csv", "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    ################################

    header, rows = summarize_stats(
        models=[
            CONFIGS["0_2_drop_small_spec_42"],
            CONFIGS["baseline_small_spectrogram_42"],
            CONFIGS["0_75_drop_small_spec_42"],
        ],
        metrics=[
            "avg_iou", "accuracy", "precision", "recall", "specificity", "f1_score"
        ])

    with open(PATHS.RESULTS_PATH / "compare_small_dropout.csv", "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    ################################

    header, rows = summarize_stats(
        models=[
            CONFIGS["baseline_small_spectrogram_42"],
            CONFIGS["bigger_net_small_spectrogram_42"],
            CONFIGS["smaller_net_small_spectrogram_42"],
            CONFIGS["more_cnn_layers_model_42"],
            CONFIGS["mse_in_pattern_small_spec_42"],
        ],
        metrics=[
            "avg_iou", "accuracy", "precision", "recall", "specificity", "f1_score"
        ])

    with open(PATHS.RESULTS_PATH / "compare_small_architecture.csv", "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    ################################

    header, rows = summarize_stats(
        models=[
            BEST_MODEL_CONFIG,
            PREV_CONFIG,
            META_INTERSECT_CONFIG,
            META_SUM_CONFIG
        ],
        metrics=[
            "avg_iou", "accuracy", "precision", "recall", "specificity", "f1_score"
        ])

    with open(PATHS.RESULTS_PATH / "compare_final_previous.csv", "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    ################################


def test_models_predictions_basic(entry: ConfigEntry):
    logging.info(f"Testing model {entry.id}")
    try:
        args = ModelTestCommand.ModelTestArgs(
            config=entry.get_config_path(),
            wav_file=entry.dataset / DATA.PROCESSED.value / "test.wav",
            spectrogram_dump=entry.dataset / DATA.PROCESSED.value / "test.bin",
            ground_truth=entry.dataset / DATA.PROCESSED.value / "test.csv",
            output=entry.get_basic_stats_path(),
            predictions=entry.get_predictions_path_default(),
            mode="basic_only"
        )
        tester = ModelTestCommand(args)
        tester.run()
    except Exception as e:
        logging.critical(e)


def test_models_predictions_visuals(entry: ConfigEntry):
    logging.info(f"Testing model {entry.id}")
    args = ModelTestCommand.ModelTestArgs(
        config=entry.get_config_path(),
        wav_file=entry.dataset / DATA.PROCESSED.value / "test.wav",
        spectrogram_dump=entry.dataset / DATA.PROCESSED.value / "test.bin",
        ground_truth=entry.dataset / DATA.PROCESSED.value / "test.csv",
        output=PATHS.BEST_MODEL_POI_RESULTS / f"{entry.id}",
        predictions=entry.get_predictions_path_default(),
        mode="visuals_only"
    )
    tester = ModelTestCommand(args)
    tester.run()


def inference_experiment_heatmap(experiment_entries: dict[str, PredParamExperimentEntry]):
    stats_all = defaultdict(lambda: defaultdict(list))

    for experiment_entry in experiment_entries.values():
        basic_stats_loc = PATHS.PRED_PARAM_EXPERIMENT_RESULTS / f"{experiment_entry.id}.basic_stats.json"
        with open(basic_stats_loc, "r") as fd:
            stats = json.load(fd)
            for stat in stats:
                stats_all[(experiment_entry.treshold, experiment_entry.vote, experiment_entry.overlap)][stat].append(stats[stat])

    heatmap_data = np.zeros((len(PredParamExperimentEntry.ALL_VOTE), len(PredParamExperimentEntry.ALL_OVERLAPS)))
    heatmap_std = np.zeros((len(PredParamExperimentEntry.ALL_VOTE), len(PredParamExperimentEntry.ALL_OVERLAPS)))
    metrics1 = ["avg_iou", "f1_score", "accuracy"]  # Metrics to plot
    metrics2 = ["precision", "recall", "specificity"]  # Metrics to plot

    plot_heatmap(stats_all, heatmap_data, heatmap_std, metrics1)
    plot_heatmap(stats_all, heatmap_data, heatmap_std, metrics2)


def plot_heatmap(stats_all, heatmap_data, heatmap_std, metrics):
    fig, axes = plt.subplots(len(metrics), len(PredParamExperimentEntry.ALL_TRESHOLDS), figsize=(4*2, len(metrics)*2 + 2), constrained_layout=True)
    vals = defaultdict(list)
    for row_idx, metric_key in enumerate(metrics):
        for col_idx, fixed_treshold in enumerate(PredParamExperimentEntry.ALL_TRESHOLDS):
            for vote_idx, vote_val in enumerate(PredParamExperimentEntry.ALL_VOTE):
                for region_idx, region_val in enumerate(PredParamExperimentEntry.ALL_OVERLAPS):
                    key = (fixed_treshold, vote_val, region_val)
                    stat = np.average(stats_all[key][metric_key])
                    std = np.std(stats_all[key][metric_key])
                    heatmap_data[vote_idx, region_idx] = stat
                    heatmap_std[vote_idx, region_idx] = std
                    vals[metric_key].append(stat)

            # Add value annotations
            for i in range(len(PredParamExperimentEntry.ALL_VOTE)):
                for j in range(len(PredParamExperimentEntry.ALL_OVERLAPS)):
                    value = heatmap_data[i, j]
                    std = heatmap_std[i, j]
                    axes[row_idx, col_idx].text(
                        j, i, f"{value:.3f}\n±{std:.3f}",
                        ha="center", va="center", color="white",
                        fontsize=8, fontweight="bold"
                    )

            min_value = min(vals[metric_key])
            max_value = max(vals[metric_key])
            val_range = max_value - min_value
            im = axes[row_idx, col_idx].imshow(heatmap_data, cmap="viridis", origin="upper", vmin=min_value, vmax=max_value + 0.25 * val_range)
            axes[row_idx, col_idx].set_xticks(range(len(PredParamExperimentEntry.ALL_OVERLAPS)))
            axes[row_idx, col_idx].set_xticklabels(PredParamExperimentEntry.ALL_OVERLAPS)
            axes[row_idx, col_idx].set_yticks(range(len(PredParamExperimentEntry.ALL_VOTE)))
            axes[row_idx, col_idx].set_yticklabels(PredParamExperimentEntry.ALL_VOTE)

            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Treshold={fixed_treshold}")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(metric_key.capitalize())

    fig.suptitle("Heatmaps of selected metrics for different tresholds\n Overlap on X axis, Vote on Y axis", fontsize=16)
    fig.savefig(PATHS.RESULTS_PATH / f"inference_experiment_heatmap_{"_".join(metrics)}.png", dpi=300)


def summarize_stats(models: list[ConfigEntry | PredParamExperimentEntry], metrics: list[str]):
    header = ["Model id", *metrics]
    rows = []
    for entry in models:
        stats_path = entry.get_basic_stats_path().with_suffix(".basic_stats.json")
        stats = {}
        with open(stats_path, "r") as fd:
            stats = json.load(fd)
        values = [
            entry.id,
            *[f"{stats[metric]:0.3f}" for metric in metrics]
        ]
        rows.append(dict(zip(header, values)))
    return header, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process skip experiments and skip models.")

    parser.add_argument(
        "--skip-experiments",
        nargs="*",
        default=[],
        help="List of experiments to skip (space-separated strings)."
    )
    parser.add_argument(
        "--skip-models",
        nargs="*",
        default=[],
        help="List of models to skip (space-separated strings)."
    )

    args = parser.parse_args()
    main(args)
