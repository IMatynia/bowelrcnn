import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

from src.new.audio.spectrogram_handler import SpectrogramHandler
from src.new.config.model_config import RCNNModelConfig
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from src.new.models.testing.inference import CombinedNetworkOutput


class BowelSoundGraph:
    def __init__(self, title: str, axis: plt.Axes, config: RCNNModelConfig, true_length: float, start: float, end: float):
        self.ax = axis
        self.twin_ax = self.ax.twinx()
        self.true_length = true_length
        self.start_lim = start
        self.end_lim = end
        self.config = config
        self.ax.set_title(title, loc="left")

    def seconds_to_index(self, seconds: float):
        return int(seconds / self.true_length * self.spectrogram_hanlder.get_all().shape[1])

    def plot_bowel_sounds(self, bowel_sounds: list[BowelSoundRaw], color, hatch):
        self.add_and_annotate_rectangles(
            map(lambda bs: (bs.start, bs.end), filter(lambda bs: bs.start > self.start_lim and bs.end < self.end_lim, bowel_sounds)),
            color=color, hatch=hatch
        )

    def plot_ground_truth(self, bowel_sounds: list[BowelSoundRaw]):
        self.plot_bowel_sounds(bowel_sounds=bowel_sounds, color="r", hatch="\\")
        return [patches.Patch(color="r", label="Ground truth")]

    def plot_preds(self, bowel_sounds: list[BowelSoundRaw]):
        self.plot_bowel_sounds(bowel_sounds=bowel_sounds, color="g", hatch="/")
        return [patches.Patch(color="g", label="Predictions")]

    def plot_spectrogram(self, spectrogram_hanlder: SpectrogramHandler):
        spectrogram_full = spectrogram_hanlder.get_all()

        idx_per_second = 1/self.config.dataset.audio_properties.spectrogram_hop_length
        start_idx = int(self.start_lim * idx_per_second)
        end_idx = int(self.end_lim * idx_per_second)

        time_bins = spectrogram_full.shape[1]
        x_values = np.linspace(self.start_lim, self.end_lim, num=end_idx-start_idx)

        num_y_bins = spectrogram_full.shape[0]
        y_values = np.linspace(0, 2000, num=num_y_bins)

        X, Y = np.meshgrid(x_values, y_values)
        c = self.ax.pcolormesh(X, Y, spectrogram_full[:, start_idx:end_idx], shading='auto', cmap='viridis')

        self.ax.set_xlim(self.start_lim, self.end_lim)

        # Set labels and title
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")

    def plot_confidence(self, conf_x, conf_y):
        X = conf_x
        Y = conf_y
        self.twin_ax.set_ylabel("Aggregated vote")
        self.twin_ax.step(X, Y, label="confidence", color="y")
        return [patches.Patch(color="y", label="Aggregated vote")]

    def plot_raw_preds(self, raw_pred_bb: list[tuple[int, CombinedNetworkOutput]]):

        hop_len = self.config.dataset.audio_properties.spectrogram_hop_length
        window_len = self.config.dataset.audio_properties.window_length

        windows = []
        preds_raw = []
        every_nth = 100
        i = 60
        for start_offset, bounding_box in raw_pred_bb:
            i += 1
            true_start_offset = start_offset * hop_len
            if not self.start_lim <= true_start_offset <= self.end_lim or not i % every_nth == 0:
                continue
            print("bb", bounding_box)
            start, end = bounding_box.relative_start_end
            print("relative", start, end)

            start *= window_len
            end *= window_len

            print("true", start, end)
            windows.append((true_start_offset, true_start_offset+window_len))
            preds_raw.append((true_start_offset+start, true_start_offset+end))

        self.add_and_annotate_rectangles(windows, "w", fill=True, alpha=0.1)
        self.add_and_annotate_rectangles(preds_raw, "y", fill=True, alpha=0.2)

    def plot_waveform(self, waveform_samples, color='w'):
        samples, sr = waveform_samples
        start_sample = int(sr * self.start_lim)
        end_sample = int(sr * self.end_lim)
        samples = samples[start_sample:end_sample]
        time_axis = np.linspace(self.start_lim, self.end_lim, num=len(samples))
        waveform_height = 2000  # Height of the spectrogram for scaling
        normalized_samples = (samples) / 2 / (np.max(samples) - np.min(samples)) + 0.5  # Normalize to [0, 1]
        scaled_samples = normalized_samples * waveform_height  # Scale to spectrogram height

        self.ax.plot(time_axis, scaled_samples, color=color, alpha=0.6, linewidth=1.5)
        return [patches.Patch(color=color, label="Waveform")]

    def add_and_annotate_rectangles(
        self, markers, color="r", fill=False, alpha=1.0, hatch=None, border_thickness=1
    ):
        y_bottom, y_top = self.ax.get_ylim()
        for marker in markers:
            rect = patches.Rectangle(
                xy=(marker[0], y_bottom-10),
                width=marker[1] - marker[0],
                height=y_top+10,
                linewidth=border_thickness,
                edgecolor=color,
                facecolor=color,
                hatch=hatch,
                alpha=alpha,
                fill=fill
            )
            self.ax.add_patch(rect)
