from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import logging as lg
from pathlib import Path

from src.new.config.model_config import RCNNModelConfig
from src.new.dataloaders.bowel_sound import BowelSoundRaw
logging = lg.getLogger("statistics")


def bowel_sound_iou(bs1: BowelSoundRaw, bs2: BowelSoundRaw):
    start_min = min(bs1.start, bs2.start)
    start_max = max(bs1.start, bs2.start)

    end_min = min(bs1.end, bs2.end)
    end_max = max(bs1.end, bs2.end)

    return (end_min - start_max) / (end_max - start_min) if end_max - start_min > 0 and end_min - start_max > 0 else 0


def bowel_sound_mse(bs1: BowelSoundRaw, bs2: BowelSoundRaw):
    start_min = min(bs1.start, bs2.start)
    start_max = max(bs1.start, bs2.start)

    end_min = min(bs1.end, bs2.end)
    end_max = max(bs1.end, bs2.end)

    return (start_max-start_min)**2 + (end_max-end_min)**2


class BasicMetricsCalculator:
    """
    Gathers statistics on the performance of the suggested model + visualizations
    """

    def __init__(self, config: RCNNModelConfig, ground_truth_bs: BowelSoundRaw, predictions_bs: BowelSoundRaw, total_wav_length: float):
        self.config = config
        self.ground_truth_bs = ground_truth_bs
        self.predictions_bs = predictions_bs
        self.total_wav_length = total_wav_length

    def prep_stats(self):
        """
        - IOU average
        - IOU histogram
        - TP/FP/TN/FN and the lot
        """

        pred_start = 1
        pred_end = 3
        true_start = -3
        true_end = -1

        bs_limits = self.prep_bs_limits(self.predictions_bs, pred_start, pred_end, self.ground_truth_bs, true_start, true_end)
        confusion_matrix = self.calc_confusion_matrix(bs_limits, pred_start, pred_end, true_start, true_end,
                                                      self.total_wav_length)
        summed_matrix = np.array(confusion_matrix)
        total_metrics = self.calculate_metrics(*summed_matrix)
        FN, FP, TN, TP = summed_matrix

        report = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            **total_metrics
        }
        return report

    @staticmethod
    def calculate_metrics(FN: float, FP: float, TN: float, TP: float) -> dict:
        """
        Calculates metrics from FN, FP, TN and TP of the confusion matrix
        @param FN: false negative
        @param FP: false positive
        @param TN: true negative
        @param TP: true positive
        @return: dict containing (avg_iou, accuracy, precision, recall, specificity, f1_score)
        """
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        average_iou = TP / (FP + FN + TP) if FP + FN + TP > 0 else 0

        stats = {
            "avg_iou": average_iou,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1_score
        }

        return stats

    @staticmethod
    def calc_confusion_matrix(bs_limits: list[tuple], pred_start: int, pred_end: int, true_start: int, true_end: int,
                              total_length: float):
        """
        @param bs_limits: list of limits
        @param pred_start: flag
        @param pred_end: flag
        @param true_start: flag
        @param true_end: flag
        @param total_length: file length
        @return: FN, FP, TN, TP tuple of floats
        """
        TP, TN, FP, FN = 0., 0., 0., 0.
        last_time = 0
        pred_bowel_sound = False
        truth_bowel_sound = False
        for current_time, limit_flag, _ in bs_limits:
            delta = current_time - last_time
            if not pred_bowel_sound > 0 and not truth_bowel_sound > 0:
                TN += delta
            elif pred_bowel_sound > 0 and not truth_bowel_sound > 0:
                FP += delta
            elif not pred_bowel_sound > 0 and truth_bowel_sound > 0:
                FN += delta
            else:
                TP += delta

            if limit_flag == true_start:
                truth_bowel_sound += 1
            elif limit_flag == true_end:
                truth_bowel_sound -= 1
            elif limit_flag == pred_start:
                pred_bowel_sound += 1
            elif limit_flag == pred_end:
                pred_bowel_sound -= 1
            last_time = current_time
        # file end
        delta = total_length - last_time
        if not pred_bowel_sound > 0 and not truth_bowel_sound > 0:
            TN += delta
        elif pred_bowel_sound > 0 and not truth_bowel_sound > 0:
            FP += delta
        elif not pred_bowel_sound > 0 and truth_bowel_sound > 0:
            FN += delta
        else:
            TP += delta

        return FN, FP, TN, TP

    @staticmethod
    def prep_bs_limits(pred: list[BowelSoundRaw], pred_start, pred_end, truth: list[BowelSoundRaw],
                       truth_start, truth_end) -> list[
            tuple[float, int, BowelSoundRaw]]:
        bs_limits = []
        for pred_bs in pred:
            start, end = pred_bs.to_limits(pred_start, pred_end)
            bs_limits.append(start)
            bs_limits.append(end)
        for truth_bs in truth:
            start, end = truth_bs.to_limits(truth_start, truth_end)
            bs_limits.append(start)
            bs_limits.append(end)
        return sorted(bs_limits)
