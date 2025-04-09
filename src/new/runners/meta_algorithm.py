from argparse import ArgumentParser, Namespace
from pathlib import Path
from src.new.config.logging_config import LOG_FORMAT, LOG_LEVEL
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from src.new.dataloaders.bowel_sound_csv_file_handler import BowelSoundCSVFileHandler
import logging as lg
logging = lg.getLogger("meta-algo")


class CSVPredictionsCombiner:
    class CSVPredictionsCombinerArgs(Namespace):
        combined_predictions: Path
        predictions_A: Path
        predictions_B: Path

    def __init__(self, args: CSVPredictionsCombinerArgs) -> None:
        self.args = args

    @staticmethod
    def parse_args(parser: ArgumentParser):
        parser.add_argument("--predictions-A", required=True, type=Path)
        parser.add_argument("--predictions-B", required=True, type=Path)
        parser.add_argument("--combined-predictions", required=True, type=Path)
        return parser.parse_args(namespace=CSVPredictionsCombiner.CSVPredictionsCombinerArgs())

    def run(self):
        preds_a = BowelSoundCSVFileHandler(self.args.predictions_A)
        preds_b = BowelSoundCSVFileHandler(self.args.predictions_B)

        all_preds_bounds = []
        for pred in preds_a.load():
            all_preds_bounds.append((pred.start, "A", 1))
            all_preds_bounds.append((pred.end, "A", -1))

        for pred in preds_b.load():
            all_preds_bounds.append((pred.start, "B", 1))
            all_preds_bounds.append((pred.end, "B", -1))

        all_preds_bounds = sorted(all_preds_bounds)

        total_increment = 0
        intersect_preds = []
        current_intersect_pred = None
        sum_preds = []
        current_sum_pred = None

        for timestamp, _, increment in all_preds_bounds:
            total_increment += increment
            if total_increment >= 2 and current_intersect_pred is None:
                current_intersect_pred = BowelSoundRaw(start=timestamp)
            if total_increment < 2 and current_intersect_pred is not None:
                current_intersect_pred.end = timestamp
                intersect_preds.append(current_intersect_pred)
                current_intersect_pred = None

            if total_increment >= 1 and current_sum_pred is None:
                current_sum_pred = BowelSoundRaw(start=timestamp)
            if total_increment < 1 and current_sum_pred is not None:
                current_sum_pred.end = timestamp
                sum_preds.append(current_sum_pred)
                current_sum_pred = None

        sum_csv = BowelSoundCSVFileHandler(self.args.combined_predictions.with_name("meta_algorithm_sum").with_suffix(".predictions.csv"))
        intersect_csv = BowelSoundCSVFileHandler(self.args.combined_predictions.with_name("meta_algorithm_intersect").with_suffix(".predictions.csv"))

        sum_csv.save(sum_preds)
        intersect_csv.save(intersect_preds)


if __name__ == "__main__":
    lg.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    parser = ArgumentParser()
    args = CSVPredictionsCombiner.parse_args(parser)
    command = CSVPredictionsCombiner(args)
    command.run()
