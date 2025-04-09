from src.new.config.cnn_config import *
from src.new.config.model_config import *


def make_classification_model_baseline(seed, window):
    return CNNConfig(
        training=TrainingSetup(
            seed=seed,
            n_epochs=250,
            batch_size=256,
            lr=1.0e-4,
            weight_decay=1.0e-6,
            loss_fn=LossFunctions.BCE,
            validation_freq=5,
            samples_per_sound=5,
            chance_of_sampling_empty=0.75,
            aug_std=0.3,
            augments=[AugmentsEnum.gauss]
        ),
        input_size=window,
        output_size=1,
        convolutional_layers=[
            ConvLayer(filters=8, kernel=(5, 5), padding=(2, 2), pool_kernel=(2, 2), pool_stride=(2, 2)),
            ConvLayer(filters=16, kernel=(5, 5), padding=(2, 2), pool_kernel=(2, 2), pool_stride=(2, 2)),
            ConvLayer(filters=16, kernel=(3, 3), padding=(2, 2), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ],
        linear_layers=[
            LinearLayer(size_out=256, drop_out=0.3),
            LinearLayer(size_out=256, drop_out=0.3),
        ],
        output_activation=None
    )


def make_classification_model_smaller(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.convolutional_layers = [
        ConvLayer(filters=8, kernel=(3, 3), padding=(1, 1), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ConvLayer(filters=8, kernel=(3, 3), padding=(1, 1), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ConvLayer(filters=8, kernel=(3, 3), padding=(1, 1), pool_kernel=(2, 2), pool_stride=(2, 2)),
    ]
    best.linear_layers = [
        LinearLayer(size_out=128, drop_out=0.3),
        LinearLayer(size_out=128, drop_out=0.3),
    ]
    return best


def make_classification_bigger(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.convolutional_layers = [
        ConvLayer(filters=16, kernel=(5, 5), padding=(2, 2), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ConvLayer(filters=16, kernel=(5, 5), padding=(2, 2), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ConvLayer(filters=16, kernel=(5, 5), padding=(2, 2), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ConvLayer(filters=32, kernel=(3, 3), padding=(1, 1), pool_kernel=(2, 2), pool_stride=(2, 2)),
    ]
    best.linear_layers = [
        LinearLayer(size_out=512, drop_out=0.3),
        LinearLayer(size_out=512, drop_out=0.3),
    ]
    return best


def make_classification_model_no_gauss(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.training.aug_std = 0.
    best.training.augments = []
    return best


def make_classification_model_0_15_gauss(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.training.aug_std = 0.15
    return best


def make_classification_model_0_6_gauss(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.training.aug_std = 0.6
    return best


def make_classification_model_2x_lr(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.training.lr *= 2
    return best


def make_classification_model_0_5x_lr(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.training.lr *= 0.5
    return best


def make_classification_model_0_1_droput(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.linear_layers[0].drop_out = 0.1
    best.linear_layers[1].drop_out = 0.1
    return best


def make_classification_model_0_5_droput(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.linear_layers[0].drop_out = 0.5
    best.linear_layers[1].drop_out = 0.5
    return best


def make_classification_more_cnn_layers_model(seed, window):
    best = make_classification_model_baseline(seed, window)
    best.convolutional_layers = [
        ConvLayer(filters=16, kernel=(3, 3), padding=(1, 1), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ConvLayer(filters=16, kernel=(3, 3), padding=(1, 1), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ConvLayer(filters=16, kernel=(3, 3), padding=(1, 1), pool_kernel=(2, 2), pool_stride=(2, 2)),
        ConvLayer(filters=16, kernel=(3, 3), padding=(1, 1), pool_kernel=(2, 2), pool_stride=(2, 2)),
    ]
    return best


def make_best_classification(seed, window):
    baseline = make_classification_model_baseline(seed, window)
    baseline.training.lr *= 2
    baseline.training.aug_std = 0.15
    baseline.training.n_epochs = 400
    baseline.linear_layers[0].drop_out = 0.75
    baseline.linear_layers[1].drop_out = 0.75

    return baseline
