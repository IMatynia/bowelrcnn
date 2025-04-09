import matplotlib
from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle
from src.new.audio.audio_utilities import get_wav_samples
from src.new.audio.spectrogram_handler import SpectrogramHandler
from src.new.config.logging_config import LOG_FORMAT, LOG_LEVEL
from src.new.config.model_config import RCNNModelConfig
from src.new.config.set_seed import set_seeds
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from src.new.dataloaders.bowel_sound_csv_file_handler import BowelSoundCSVFileHandler
import logging
from src.new.models.testing.statistics import BasicMetricsCalculator
from src.new.visualizations.bowel_sound_visualizer import BowelSoundGraph
from matplotlib import pyplot as plt
import json
import logging as lg
logging = lg.getLogger("model-test")

matplotlib.use('tkAgg')
matplotlib.rcdefaults()


class ModelTestCommand:
    spectrogram_handler: SpectrogramHandler
    preds: list[BowelSoundRaw]
    ground_truth: list[BowelSoundRaw]

    class ModelTestArgs(Namespace):
        config: Path
        wav_file: Path
        spectrogram_dump: Path
        output: Path
        predictions: Path
        ground_truth: Path
        mode: str

    def __init__(self, args: ModelTestArgs) -> None:
        self.args = args
        self.config = RCNNModelConfig.model_validate_json(self.args.config.read_text(encoding="utf-8"))
        self.assignments = {}

    @staticmethod
    def parse_args(parser: ArgumentParser):
        parser.add_argument("--config", required=True, type=Path, help="Path to config .json")
        parser.add_argument("--wav-file", type=Path, help="Path to .wav file with the test sample")
        parser.add_argument("--spectrogram-dump", type=Path, help="Path to .bin spectrogram dump of the test sample")
        parser.add_argument("--output", required=True, type=Path, help="Base path for all outputs")
        parser.add_argument("--mode", required=True, type=str, help="Mode of operation (basic_only, visuals_only)")

        parser.add_argument("--predictions", required=True, type=Path, help="Path to predictions .csv (also indicates the raw predicrtions .bin file location)")
        parser.add_argument("--ground-truth", required=True, type=Path, help="Path to ground truth .csv")
        return parser.parse_args(namespace=ModelTestCommand.ModelTestArgs())

    def load(self):
        assert self.args.spectrogram_dump or self.args.wav_file, "Must add wav file or spectrogram dump for testing"
        logging.info("Loading predictions and ground truth")
        self.preds = list(BowelSoundCSVFileHandler(self.args.predictions).load())
        self.ground_truth = list(BowelSoundCSVFileHandler(self.args.ground_truth).load())
        logging.info("Loading raw predictions pickle")
        raw_predictions_path = self.args.predictions.with_suffix(".bin")
        if raw_predictions_path.exists():
            with open(raw_predictions_path, mode="rb") as fd:
                self.raw_preds = pickle.load(fd)
        else:
            self.raw_preds = None
        logging.info("Loading spectrogram data")
        self.spectrogram_handler = SpectrogramHandler(self.config)
        if self.args.spectrogram_dump:
            self.spectrogram_handler.load(self.args.spectrogram_dump)
        else:
            self.spectrogram_handler.generate_spectrogram_from_wav_signle(self.args.wav_file)

        logging.info("Loading raw WAV file samples")
        self.wav_file_samples = get_wav_samples(self.args.wav_file, self.config.dataset.audio_properties.sample_rate)

    def plot_POI(self, range: tuple[float, float], title=None, plot_spec=True, plot_wave=True, plot_gt=True, plot_preds=True, plot_conf=True):
        vis_start_s, vis_end_s = range
        if not title:
            title = f"Spectrogram at {range}"
        logging.info("Preparing graphs")
        fig, ax = plt.subplots()
        bsgraph = BowelSoundGraph(title, ax, self.config, self.spectrogram_handler.true_length, vis_start_s, vis_end_s)
        logging.info("Plotting bounding boxes")
        legend = []
        if plot_spec:
            bsgraph.plot_spectrogram(self.spectrogram_handler)
        if plot_wave:
            legend += bsgraph.plot_waveform(self.wav_file_samples)
        if plot_gt:
            legend += bsgraph.plot_ground_truth(self.ground_truth)
        if plot_preds:
            legend += bsgraph.plot_preds(self.preds)
        if self.raw_preds and plot_conf:
            legend += bsgraph.plot_confidence(self.raw_preds["confidence"]["x"], self.raw_preds["confidence"]["y"])

        bsgraph.twin_ax.legend(handles=legend, loc='lower right')
        fig.savefig(self.args.output.with_suffix(f".{title}.png"), dpi=300, pad_inches=0.3)

    def visualizatons(self):
        null = None
        NaN = None
        old_better = [
            {
                "start": 214.23,
                "end": 214.25,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 155.04,
                "end": 155.08,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 53.41,
                "end": 53.44,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 154.37,
                "end": 154.4,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 207.56,
                "end": 207.58,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 163.93,
                "end": 163.96,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 269.34,
                "end": 269.39,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 224.29,
                "end": 224.31,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 88.73,
                "end": 88.76,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 42.71,
                "end": 42.75,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
        ]
        new_better = [
            {
                "start": 25.064062911343004,
                "end": 25.09021358300769,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 37.3952079878913,
                "end": 37.43489592680856,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 12.996991838325584,
                "end": 13.029303721492251,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 98.59304383375814,
                "end": 98.61206878782028,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 97.25110866065536,
                "end": 97.30129283595652,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 232.84065856821243,
                "end": 232.8702273991728,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 127.85838399342602,
                "end": 127.88239398913252,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 104.1652824017027,
                "end": 104.19576644116924,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 158.49800607957064,
                "end": 158.5296652117419,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 283.2369620580995,
                "end": 283.27896208862467,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
        ]
        old_only = [
            {
                "start": 4.748252000000008,
                "end": 4.76098300000001,
                "min_frequency": 325.423737,
                "max_frequency": 701.694946,
                "category": NaN
            },
            {
                "start": 10.187937000000034,
                "end": 10.20535099999995,
                "min_frequency": 313.39032,
                "max_frequency": 1128.2052,
                "category": "s"
            },
            {
                "start": 10.52752799999996,
                "end": 10.55800499999998,
                "min_frequency": 108.262108,
                "max_frequency": 427.350433,
                "category": "s"
            },
            {
                "start": 10.656689000000028,
                "end": 10.678457999999978,
                "min_frequency": 176.638168,
                "max_frequency": 1065.5271,
                "category": "s"
            },
            {
                "start": 21.63065800000001,
                "end": 21.659683,
                "min_frequency": 297.142853,
                "max_frequency": 520.0,
                "category": "s"
            },
            {
                "start": 24.622462000000013,
                "end": 24.63466,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 30.735319000000004,
                "end": 30.756010000000003,
                "min_frequency": 97.922844,
                "max_frequency": 320.474792,
                "category": NaN
            },
            {
                "start": 31.580493999999987,
                "end": 31.601185999999984,
                "min_frequency": 845.697388,
                "max_frequency": 2029.673462,
                "category": NaN
            },
            {
                "start": 32.403529000000006,
                "end": 32.413807000000006,
                "min_frequency": 500.0,
                "max_frequency": 1641.509399,
                "category": NaN
            },
            {
                "start": 33.167496,
                "end": 33.184625,
                "min_frequency": 103.773582,
                "max_frequency": 339.62265,
                "category": NaN
            },
        ]
        new_only = [
            {
                "start": 20.147482999999998,
                "end": 20.23165499999999,
                "min_frequency": 131.428574,
                "max_frequency": 1971.428467,
                "category": "b"
            },
            {
                "start": 20.26648499999999,
                "end": 20.289705,
                "min_frequency": 188.571426,
                "max_frequency": 645.714294,
                "category": "s"
            },
            {
                "start": 26.494709,
                "end": 26.51264900000001,
                "min_frequency": 265.714294,
                "max_frequency": 308.571442,
                "category": "s"
            },
            {
                "start": 36.75245799999999,
                "end": 36.781161,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 42.0,
                "end": 42.005986,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 43.735873,
                "end": 43.773605,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 43.836009,
                "end": 43.867937,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 44.060235000000006,
                "end": 44.07773900000001,
                "min_frequency": 563.106812,
                "max_frequency": 1067.961182,
                "category": NaN
            },
            {
                "start": 56.08381,
                "end": 56.14476199999999,
                "min_frequency": 720.0,
                "max_frequency": 720.0,
                "category": "b"
            },
            {
                "start": 56.21442200000001,
                "end": 56.28698399999999,
                "min_frequency": 125.714287,
                "max_frequency": 1982.857056,
                "category": "b"
            },
        ]
        none = [
            {
                "start": 2.769601000000001,
                "end": 2.782367999999998,
                "min_frequency": 832.116821,
                "max_frequency": 1959.854004,
                "category": NaN
            },
            {
                "start": 5.327472,
                "end": 5.337019999999995,
                "min_frequency": 244.067795,
                "max_frequency": 1077.966064,
                "category": NaN
            },
            {
                "start": 5.352933000000007,
                "end": 5.365663000000012,
                "min_frequency": 254.237289,
                "max_frequency": 711.86438,
                "category": NaN
            },
            {
                "start": 21.99927400000001,
                "end": 22.0,
                "min_frequency": 177.142853,
                "max_frequency": 400.0,
                "category": "s"
            },
            {
                "start": 24.895858000000004,
                "end": 24.91595000000001,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 25.187194000000005,
                "end": 25.203698000000003,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 26.318185,
                "end": 26.353347000000014,
                "min_frequency": 322.475586,
                "max_frequency": 332.247559,
                "category": "s"
            },
            {
                "start": 31.839936000000023,
                "end": 31.86062800000002,
                "min_frequency": 1141.935425,
                "max_frequency": 2690.32251,
                "category": NaN
            },
            {
                "start": 42.759184,
                "end": 42.785306,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 48.0,
                "end": 48.001464,
                "min_frequency": -1.0,
                "max_frequency": 403.125031,
                "category": NaN
            },
        ]
        old_zero_iou = [
            {
                "start": 158.35,
                "end": 158.37,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 70.88,
                "end": 70.89,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 188.95,
                "end": 188.97,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 241.02,
                "end": 241.04,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 171.27,
                "end": 171.29,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 247.88,
                "end": 247.89,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 233.52,
                "end": 233.53,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 221.47,
                "end": 221.49,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 257.14,
                "end": 257.16,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
            {
                "start": 80.51,
                "end": 80.52,
                "min_frequency": null,
                "max_frequency": null,
                "category": null
            },
        ]
        new_zero_iou = [
            {
                "start": 126.82030128264238,
                "end": 126.85161756802174,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 153.9846818376155,
                "end": 154.01678375332128,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 284.1573500328357,
                "end": 284.1967305853253,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 251.9812586802812,
                "end": 252.008655901608,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 18.56483753506863,
                "end": 18.585050380572916,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 163.99018019175244,
                "end": 164.01837395905028,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 307.62017757485785,
                "end": 307.6444283507055,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 308.85005355609786,
                "end": 308.8758854944318,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 174.79330662127052,
                "end": 174.80279440863265,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
            {
                "start": 309.75065654764575,
                "end": 309.7557875752213,
                "min_frequency": NaN,
                "max_frequency": NaN,
                "category": NaN
            },
        ]

        POIs = [
            *[((v["start"]-0.2, v["end"]+0.2), f"CRNN better than BowelRCNN at {v["start"]:0.2f}") for v in old_better],
            *[((v["start"]-0.2, v["end"]+0.2), f"BowelRCNN better than CRNN at {v["start"]:0.2f}") for v in new_better],
            *[((v["start"]-0.2, v["end"]+0.2), f"CRNN was the only one at {v["start"]:0.2f}") for v in old_only],
            *[((v["start"]-0.2, v["end"]+0.2), f"BowelRCNN was the only one at {v["start"]:0.2f}") for v in new_only],
            *[((v["start"]-0.2, v["end"]+0.2), f"None detected at {v["start"]:0.2f}") for v in none],
            *[((v["start"]-0.2, v["end"]+0.2), f"BowelRCNN zero iou interval at {v["start"]:0.2f}") for v in new_zero_iou],
            *[((v["start"]-0.2, v["end"]+0.2), f"CRNN zero iou interval at {v["start"]:0.2f}") for v in old_zero_iou],
        ]
        for poi, title in POIs:
            self.plot_POI(poi, title)

    def plot_raw_network_out(self):
        fig, ax = plt.subplots(2, 1)
        offsets = []
        scales = []
        for _, pred_raw in self.raw_preds["detections"]:
            offsets.append(pred_raw.bounding_box_offset)
            scales.append(pred_raw.bounding_box_scale)

        ax[0].hist(offsets)
        ax[0].set_title("offset")

        ax[1].hist(scales)
        ax[1].set_title("scales")

        fig.savefig(self.args.output.with_suffix(".raw_preds_stats.png"))

    def run(self):
        self.load()
        set_seeds(self.config.seed)

        if self.args.mode == "basic_only":
            logging.info("Preparing basic metrics and saving to json file")
            self.basic_metrics = BasicMetricsCalculator(self.config, self.ground_truth, self.preds, self.spectrogram_handler.true_length).prep_stats()
            with open(self.args.output.with_suffix(".basic_stats.json"), mode="w") as fd:
                json.dump(self.basic_metrics, fd, indent=4)

        if self.args.mode == "visuals_only":
            self.visualizatons()
            if self.raw_preds:
                self.plot_raw_network_out()


if __name__ == "__main__":
    lg.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    parser = ArgumentParser()
    args = ModelTestCommand.parse_args(parser)
    command = ModelTestCommand(args)
    command.run()
