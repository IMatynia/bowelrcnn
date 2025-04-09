from enum import Enum
import numpy as np


class AugmentsEnum(str, Enum):
    gauss = "gauss"


class Augments:
    @staticmethod
    def from_enum(augment: AugmentsEnum):
        assignment = {
            AugmentsEnum.gauss: Augments.gauss
        }
        return assignment[augment]

    @staticmethod
    def gauss(training_config: "TrainingSetup", data: np.array):
        std = np.random.random() * training_config.aug_std
        return np.clip(data + np.random.normal(scale=std, size=data.shape), a_min=0.0, a_max=1.0)
