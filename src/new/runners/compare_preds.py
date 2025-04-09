import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace
from pathlib import Path
from src.new.audio.spectrogram_handler import SpectrogramHandler
from src.new.config.logging_config import LOG_FORMAT, LOG_LEVEL
from src.new.config.model_config import RCNNModelConfig
from src.new.config.set_seed import set_seeds
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from src.new.dataloaders.bowel_sound_csv_file_handler import BowelSoundCSVFileHandler
import logging
from collections import defaultdict
from src.new.models.testing.statistics import bowel_sound_iou, bowel_sound_mse
import json
import logging as lg
logging = lg.getLogger("model-compare")

matplotlib.use('tkAgg')
matplotlib.rcdefaults()


class ComparePredsCommand:
    spectrogram_handler: SpectrogramHandler
    preds_new: list[BowelSoundRaw]
    ground_truth: list[BowelSoundRaw]

    class ComparePredsArgs(Namespace):
        config: Path
        output: Path
        predictions: Path
        predictions_compare: Path
        ground_truth: Path

    def __init__(self, args: ComparePredsArgs) -> None:
        self.args = args
        self.config = RCNNModelConfig.model_validate_json(self.args.config.read_text(encoding="utf-8"))
        self.assignments = {}

    @staticmethod
    def parse_args(parser: ArgumentParser):
        parser.add_argument("--config", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)

        parser.add_argument("--predictions", required=True, type=Path)
        parser.add_argument("--predictions-compare", type=Path)
        parser.add_argument("--ground-truth", required=True, type=Path)
        return parser.parse_args(namespace=ComparePredsCommand.ComparePredsArgs())

    def load(self):
        logging.info("Loading predictions and ground truth")
        self.preds_new = list(BowelSoundCSVFileHandler(self.args.predictions).load())
        self.preds_old = list(BowelSoundCSVFileHandler(self.args.predictions_compare).load())
        self.ground_truth = list(BowelSoundCSVFileHandler(self.args.ground_truth).load())

    def analyse_per_bowel_sound(self):
        combined = defaultdict(list)
        new_model_id = "new"
        old_model_id = "old"

        positive_iou_preds = defaultdict(set)
        predicted_gt = set()
        best = defaultdict(list)

        logging.info("Calculating combined and bests")
        for ground_truth in self.ground_truth:
            for raw_prediction in self.preds_new:
                iou = bowel_sound_iou(ground_truth, raw_prediction)
                if iou > 0:
                    combined[ground_truth].append((iou, raw_prediction, new_model_id))
                    positive_iou_preds[new_model_id].add(raw_prediction)
                    predicted_gt.add(ground_truth)
            for raw_prediction in self.preds_old:
                iou = bowel_sound_iou(ground_truth, raw_prediction)
                if iou > 0:
                    combined[ground_truth].append((iou, raw_prediction, old_model_id))
                    positive_iou_preds[old_model_id].add(raw_prediction)
                    predicted_gt.add(ground_truth)
            combined[ground_truth] = sorted(combined[ground_truth], reverse=True)
            if len(combined[ground_truth]) == 0:
                continue
            _, _, which = combined[ground_truth][0]
            best[which].append(combined[ground_truth][0])

        zero_iou_preds = {
            new_model_id: set(self.preds_new).difference(positive_iou_preds[new_model_id]),
            old_model_id: set(self.preds_old).difference(positive_iou_preds[old_model_id])
        }
        undetected_gt = set(self.ground_truth).difference(predicted_gt)

        n_hits = defaultdict(set)
        all_non_zero_ious = defaultdict(list)
        all_non_zero_length_proportions = defaultdict(list)
        for gt in combined:
            for iou, raw_pred, which in combined[gt]:
                all_non_zero_ious[which].append(iou)
                all_non_zero_length_proportions[which].append(raw_pred.length/gt.length)
                n_hits[gt].add(which)
        n_both = []
        n_only_one = defaultdict(list)
        for gt in n_hits:
            if len(n_hits[gt]) == 1:
                n_only_one[list(n_hits[gt])[0]].append(gt)
            else:
                n_both.append(gt)

        self.edges_MSE(combined, n_both)

        statistics_string = f"""
total number of bowel sounds predicted by old: {len(self.preds_old)}
total number of bowel sounds predicted by new: {len(self.preds_new)}
total bowel sounds in test dataset: {len(self.ground_truth)}
preds with iou > 0: {len(predicted_gt)}
no predictions were made for {len(undetected_gt)}
bowel sounds where old model performs better: {len(best[old_model_id])}
bowel sounds where new model performs better: {len(best[new_model_id])}
both models detected {len(n_both)} sounds with accuracy of IOU > 0
previous model was the only one to detect {len(n_only_one[old_model_id])} sounds
new model was the only one to detect {len(n_only_one[new_model_id])} sounds
previous model predicted {len(zero_iou_preds[old_model_id])} regions with iou = 0
new model predicted {len(zero_iou_preds[new_model_id])} regions with iou = 0
previous model's non zero IOUs - {self.stats_on_list(all_non_zero_ious[old_model_id])}
new model's non zero IOUs - {self.stats_on_list(all_non_zero_ious[new_model_id])}
previous model's length fraction comparison - {self.stats_on_list(all_non_zero_length_proportions[old_model_id])}
new model's length fraction comparison - {self.stats_on_list(all_non_zero_length_proportions[new_model_id])}
        """
        examples_dict = {
            "old_was_better": [el[1].model_dump() for el in filter(lambda el: el[0] > 0.85, sorted(best[old_model_id]))],
            "new_was_better": [el[1].model_dump() for el in filter(lambda el: el[0] > 0.85, sorted(best[new_model_id]))],
            "old_was_the_only_one": [el.model_dump() for el in sorted(n_only_one[old_model_id])],
            "new_was_the_only_one": [el.model_dump() for el in sorted(n_only_one[new_model_id])],
            "noone_detected": [el.model_dump() for el in sorted((set(self.ground_truth) - n_hits.keys()).union(undetected_gt))],
            "old_zero_iou": [el.model_dump() for el in zero_iou_preds[old_model_id]],
            "new_zero_iou": [el.model_dump() for el in zero_iou_preds[new_model_id]],
        }
        self.args.output.with_suffix(".statistics_description.txt").write_text(statistics_string)

        self.plot_iou_histograms(new_model_id, old_model_id, all_non_zero_ious)

        with open(self.args.output.with_suffix(".comparison_examples.json"), "w") as fd:
            json.dump(examples_dict, fd, indent=4)

    def edges_MSE(self, combined, n_both):
        krance_mse = defaultdict(list)
        n_krance_best = defaultdict(list)
        for gt in n_both:
            for iou, raw_pred, which in combined[gt]:
                if iou < 0.85:
                    continue
                mse = bowel_sound_mse(gt, raw_pred)
                krance_mse[gt].append((mse, raw_pred, which))
            krance_mse[gt] = sorted(krance_mse[gt])
            if len(krance_mse[gt]) > 0:
                mse, _, which = krance_mse[gt][0]
                n_krance_best[which].append(gt)

    def plot_iou_histograms(self, new_model_id, old_model_id, all_non_zero_ious):
        fig, ax = plt.subplots()
        ax.hist(all_non_zero_ious[old_model_id], 20)
        ax.set_xlabel("IOU value")
        ax.set_ylabel("Amount")
        ax.set_ylim(0, 80)
        fig.suptitle("IOU histogram for CRNN")
        fig.savefig(self.args.output.with_suffix(".crnn_iou_histogram.png"))
        fig, ax = plt.subplots()
        ax.set_xlabel("IOU value")
        ax.set_ylabel("Amount")
        ax.set_ylim(0, 80)
        ax.hist(all_non_zero_ious[new_model_id], 20)
        fig.suptitle("IOU histogram for BowelRCNN")
        fig.savefig(self.args.output.with_suffix(".bowelrcnn_iou_histogram.png"))

    def stats_on_list(self, l: list):
        return f"avg:{np.average(l):0.4f}, std:{np.std(l):0.4f}, median:{np.median(l):0.4f}, min:{np.min(l):0.4f}, max:{np.max(l):0.4f}"

    def run(self):
        self.load()
        set_seeds(self.config.seed)
        self.analyse_per_bowel_sound()


if __name__ == "__main__":
    lg.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    parser = ArgumentParser()
    args = ComparePredsCommand.parse_args(parser)
    command = ComparePredsCommand(args)
    command.run()
