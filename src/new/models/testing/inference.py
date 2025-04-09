import numpy as np
import torch
from src.new.datasets.cnn_pattern_bounding_box_dataset import CNNPatternBoundingBoxNetworkOutput
from src.new.datasets.cnn_region_classification_dataset import CNNRegionClassificationNetworkOutput
from src.new.config.model_config import AudioProperties, RCNNModelConfig
from src.new.dataloaders.bowel_sound import BowelSoundRaw
from tqdm import tqdm


def calculate_window_count(total_spectrogram_samples: int, window_width: int, overlap: int):
    offset_per_window = int(window_width // overlap)
    return int((total_spectrogram_samples-window_width+offset_per_window) // offset_per_window), offset_per_window


class CombinedNetworkOutput(CNNPatternBoundingBoxNetworkOutput, CNNRegionClassificationNetworkOutput):
    ...


START_FLAG = 1
STOP_FLAG = -1

MINIMUM_BS_PREDICTION_LENGTH = 0.005


class RCNNInferenceHandler:
    def __init__(self, config: RCNNModelConfig, region_classifier: torch.nn.Module, pattern_model: torch.nn.Module, device, inference_overlap: int, vote_fraction: float, classification_threshold: float):
        self.config: RCNNModelConfig = config
        self.audio_properties_ref: AudioProperties = config.dataset.audio_properties
        self.region_classifier = region_classifier
        self.pattern_model = pattern_model
        self.device = device
        self.inference_overlap = inference_overlap
        self.vote_fraction = vote_fraction
        self.classification_threshold = classification_threshold

    def apply_padding(self, spectrogram_data, pad_mode: str):
        pad_amount = self.audio_properties_ref.window_width//2
        spectrogram_data = spectrogram_data.T
        if pad_mode == "extend":
            return np.concatenate((spectrogram_data[0]*np.ones((pad_amount, self.audio_properties_ref.window_height)), spectrogram_data, spectrogram_data[-1]*np.ones((pad_amount, self.audio_properties_ref.window_height)))).T
        else:
            raise Exception("Invalid padding mode!")

    def infer_all_sounds_in_spectrogram(self, spectrogram_data: np.array, padding=None) -> tuple[list[BowelSoundRaw], list[CombinedNetworkOutput]]:
        """
        Iterates over all windows and saves confident bowel sound bounding boxes
        """
        window_count, offset_per_window = calculate_window_count(spectrogram_data.shape[1], self.audio_properties_ref.window_width, self.inference_overlap)
        bounding_boxes: list[CombinedNetworkOutput] = []
        for window_idx in tqdm(range(window_count)):
            start_offset = window_idx * offset_per_window
            end_offset = window_idx * offset_per_window + self.audio_properties_ref.window_width
            bounding_box = self.infer_from_window(
                spectrogram_data[:, start_offset:end_offset].T
            )
            if bounding_box is None:
                continue
            bounding_boxes.append((start_offset, bounding_box))
        return *self.compute_final_classification_from_bounding_boxes(bounding_boxes), bounding_boxes

    def compute_final_classification_from_bounding_boxes(self, bounding_boxes: list[tuple[int, CombinedNetworkOutput]]) -> tuple[list[BowelSoundRaw], list, list]:
        bb_limits = sorted(self.prep_bounding_box_limits(bounding_boxes))
        votes = 0.
        current_bs_params = None
        vote_threshold = self.vote_fraction * self.inference_overlap
        all_detected_bs = []
        confidence_x = []
        confidence_y = []
        for current_offset, vote_amount in bb_limits:
            votes += vote_amount

            confidence_x.append(current_offset)
            confidence_y.append(votes)

            if current_bs_params is None and votes >= vote_threshold:
                current_bs_params = BowelSoundRaw(start=current_offset, end=None)
            if current_bs_params is not None and votes < vote_threshold:
                current_bs_params.end = current_offset
                all_detected_bs.append(current_bs_params)
                current_bs_params = None

        all_detected_bs = list(filter(lambda bs: bs.length > MINIMUM_BS_PREDICTION_LENGTH, all_detected_bs))
        return all_detected_bs, confidence_x, confidence_y

    def prep_bounding_box_limits(self, bounding_boxes: list[tuple[int, CombinedNetworkOutput]]) -> list[tuple[float, float]]:
        bb_limits = []
        for start_offset, bounding_box in bounding_boxes:
            true_start_offset = start_offset * self.audio_properties_ref.spectrogram_hop_length
            start, end = bounding_box.relative_start_end
            start *= self.audio_properties_ref.window_length
            end *= self.audio_properties_ref.window_length

            bb_limits.append(
                (true_start_offset+start, START_FLAG*bounding_box.has_bowel_sound)
            )
            bb_limits.append(
                (true_start_offset+end, STOP_FLAG*bounding_box.has_bowel_sound)
            )
        return bb_limits

    def infer_from_window(self, spectrogram_window: np.array) -> CombinedNetworkOutput | None:
        # Unsqueeze -> [batch][channels][height][width]
        spectrogram_window_tensor = torch.tensor(spectrogram_window, requires_grad=False).float().unsqueeze(0).unsqueeze(0).to(self.device)

        self.region_classifier.eval()
        self.pattern_model.eval()

        region_confidence_pred = self.region_classifier(spectrogram_window_tensor)
        region_confidence_pred_sigmoid = torch.nn.Sigmoid().to(region_confidence_pred.device).forward(region_confidence_pred)
        activation = region_confidence_pred_sigmoid.detach().cpu().numpy()[0]
        region_confidence = CNNRegionClassificationNetworkOutput.from_numpy(activation)

        if region_confidence.has_bowel_sound < self.classification_threshold:
            return None

        bounding_box_pred = self.pattern_model(spectrogram_window_tensor)
        bounding_box = CNNPatternBoundingBoxNetworkOutput.from_numpy(bounding_box_pred.detach().cpu().numpy()[0])
        return CombinedNetworkOutput(
            has_bowel_sound=region_confidence.has_bowel_sound,
            bounding_box_offset=bounding_box.bounding_box_offset,
            bounding_box_scale=bounding_box.bounding_box_scale,
        )
