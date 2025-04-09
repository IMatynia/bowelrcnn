from hashlib import sha256
from pydantic import BaseModel
from typing import Literal
from enum import Enum
from src.new.models.training.augments import AugmentsEnum


class LossFunctions(str, Enum):
    DistanceBoxIOU = "DistanceBoxIOU"
    CompleteBoxIOU = "CompleteBoxIOU"
    CompleteBoxIOUWeighted = "CompleteBoxIOUWeighted"
    BCE = "BCE"
    CE = "CE"
    MSE = "MSE"


class Activations(str, Enum):
    LeakyRELU = "LeakyRELU"
    Sigmoid = "Sigmoid"
    RELU = "RELU"
    GELU = "GELU"
    SoftMax = "SoftMax"


class Pooling(str, Enum):
    Max = "Max"


class TrainingSetup(BaseModel):
    """
    Values used in training
    """
    seed: int
    n_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    loss_fn: LossFunctions
    validation_freq: int
    samples_per_sound: int
    chance_of_sampling_empty: float
    aug_std: float | None = None
    augments: list[AugmentsEnum]


class ConvLayer(BaseModel):
    filters: int
    kernel: tuple[int, int]
    padding: tuple[int, int]
    pool_stride: tuple[int, int]
    pool_kernel: tuple[int, int]
    activation: Activations = Activations.LeakyRELU
    pooling: Pooling = Pooling.Max


class LinearLayer(BaseModel):
    size_out: int
    activation: Activations = Activations.LeakyRELU
    drop_out: float


class CNNConfig(BaseModel):
    convolutional_layers: list[ConvLayer]
    linear_layers: list[LinearLayer]
    input_size: tuple[int, int]
    output_size: int
    output_activation: Activations | None
    training: TrainingSetup

    def get_hash(self):
        return sha256(repr(self).encode()).digest().hex()
