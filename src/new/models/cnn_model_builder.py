from torch import nn
from src.new.config.cnn_config import CNNConfig, Activations, Pooling
import torch
from src.new.utilities.profiler import log_exec_time
import logging as lg
logging = lg.getLogger("CNNBuilder")

ACTIVATION_ENUM_TO_ACTIVATION = {
    Activations.GELU: nn.GELU,
    Activations.LeakyRELU: nn.LeakyReLU,
    Activations.RELU: nn.ReLU,
    Activations.Sigmoid: nn.Sigmoid,
    Activations.SoftMax: nn.Softmax
}

POOLING_ENUM_TRANSLATION = {
    Pooling.Max: nn.MaxPool2d
}


class CNNModelBuilder(nn.Module):
    def __init__(self, config: CNNConfig):
        super().__init__()
        all_modules = []
        prev_filters = 1
        for layer in config.convolutional_layers:
            all_modules.extend([
                nn.Conv2d(prev_filters, layer.filters, kernel_size=layer.kernel, padding=layer.padding),
                ACTIVATION_ENUM_TO_ACTIVATION[layer.activation](),
                POOLING_ENUM_TRANSLATION[layer.pooling](kernel_size=layer.pool_kernel, stride=layer.pool_stride)
            ])
            prev_filters = layer.filters
        self.conv_layers = nn.Sequential(*all_modules)

        self.flatten = nn.Flatten()

        all_modules = []

        example = torch.rand((1, 1, config.input_size[1], config.input_size[0]))
        logging.debug(f"initial: {example.shape}")
        for conv_layer in self.conv_layers:
            example = conv_layer(example)
            logging.debug(f"post conv: {example.shape}")
        example = self.flatten(example)
        logging.debug(f"flatten layer size: {example.shape}")
        prev_size = example.shape[1]

        for layer in config.linear_layers:
            all_modules.extend([
                nn.Linear(prev_size, layer.size_out),
                ACTIVATION_ENUM_TO_ACTIVATION[layer.activation](),
                nn.Dropout(layer.drop_out)
            ])
            prev_size = layer.size_out
        self.linear_layers = nn.Sequential(*all_modules)

        self.output_linear = nn.Linear(prev_size, config.output_size)
        if config.output_activation:
            self.output_activation = ACTIVATION_ENUM_TO_ACTIVATION[config.output_activation]()
        else:
            self.output_activation = None

    # @log_exec_time
    def forward(self, x):
        logging.debug(f"Initial: {x.shape}")
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            logging.debug(f"post conv: {x.shape}")
        x = self.flatten(x)
        logging.debug(f"flatten in forward: {x.shape}")
        for lin_layer in self.linear_layers:
            x = lin_layer(x)
            logging.debug(f"post lin: {x.shape}")
        x = self.output_linear(x)
        if self.output_activation:
            x = self.output_activation(x)
        logging.debug(f"OUTPUT: {x.shape}")
        return x
