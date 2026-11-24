"""Kiến trúc PyTorch tương ứng với các notebook Keras."""
from torch import nn


def initialize_keras_style(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for layer in range(module.num_layers):
            nn.init.xavier_uniform_(getattr(module, f"weight_ih_l{layer}"))
            nn.init.orthogonal_(getattr(module, f"weight_hh_l{layer}"))
            bias = getattr(module, f"bias_ih_l{layer}")
            nn.init.zeros_(bias)
            nn.init.ones_(bias[module.hidden_size:2 * module.hidden_size])
            other_bias = getattr(module, f"bias_hh_l{layer}")
            nn.init.zeros_(other_bias)
            other_bias.requires_grad_(False)


def conv_stack(blocks, dropout):
    layers = []
    channels = 1
    for block in blocks:
        for filters, padding in block:
            layers.extend((nn.Conv2d(channels, filters, 3, padding=padding), nn.ReLU()))
            channels = filters
        layers.extend((nn.MaxPool2d(2), nn.Dropout(dropout)))
    return nn.Sequential(*layers)


DIGIT_BLOCKS = (((64, 1),), ((128, 1),), ((256, 1),))
GENDER_BLOCKS = tuple(((width, 1),) * count for width, count in
                      ((16, 2), (32, 3), (64, 3), (128, 3), (256, 3)))
EMOTION_BLOCKS = (((64, 1), (64, 0)), ((128, 1), (128, 0), (218, 1), (218, 0)))


class Classifier(nn.Module):
    def __init__(self, encoder, flat_size, hidden, classes, dropout):
        super().__init__()
        self.encoder = encoder
        layers = []
        for width in hidden:
            layers.extend((nn.Linear(flat_size, width), nn.ReLU(), nn.Dropout(dropout)))
            flat_size = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(flat_size, classes)

    def forward(self, images, return_features=False):
        maps = self.encoder(images)
        # Keras Flatten duyệt theo H, W, C.
        vectors = self.hidden[:-1](maps.permute(0, 2, 3, 1).flatten(1))
        logits = self.output(self.hidden[-1](vectors))
        if return_features:
            return logits, vectors, maps.mean(dim=(2, 3))
        return logits


def mnist_mlp():
    return Classifier(nn.Identity(), 784, (512, 512), 10, 0.2)


def mnist_cnn():
    return Classifier(conv_stack(DIGIT_BLOCKS, 0.2), 256 * 3 * 3, (512, 512), 10, 0.2)


def gender_cnn():
    return Classifier(conv_stack(GENDER_BLOCKS, 0.2), 256 * 2 * 2, (1024, 1024), 2, 0.2)


def emotion_cnn():
    return Classifier(conv_stack(EMOTION_BLOCKS, 0.25), 218 * 9 * 9, (1024,), 7, 0.5)


class DigitSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = conv_stack(DIGIT_BLOCKS, 0.2)
        self.lstm = nn.LSTM(256 * 3 * 3, 128, num_layers=2, batch_first=True)
        self.output = nn.Sequential(nn.Dropout(0.22), nn.Linear(128, 1))

    def forward(self, sequences):
        batch, steps = sequences.shape[:2]
        features = self.encoder(sequences.flatten(0, 1))
        features = features.permute(0, 2, 3, 1).reshape(batch, steps, -1)
        output, _ = self.lstm(features)
        return self.output(output[:, -1]).squeeze(1)


def segnet():
    layers = []
    channels = 1

    def add_convolutions(filters):
        nonlocal channels
        for width in filters:
            layers.extend((nn.Conv2d(channels, width, 3, padding=1),
                           nn.BatchNorm2d(width, eps=1e-3, momentum=0.01), nn.ReLU()))
            channels = width

    for filters in ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)):
        add_convolutions(filters)
        layers.append(nn.MaxPool2d(2))
    for filters in ((512, 512, 512), (512, 512, 256), (256, 256, 128), (128, 64), (64,)):
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        add_convolutions(filters)
    layers.extend((nn.Conv2d(64, 7, 1), nn.BatchNorm2d(7, eps=1e-3, momentum=0.01)))
    return nn.Sequential(*layers)


class VehicleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Conv2d(3, 10, 3, padding=1), nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(8), nn.Dropout(0.25),
            nn.Conv2d(10, 128, 8), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv2d(128, 1, 1), nn.Tanh(),
        )

    def forward(self, images):
        return self.detector(images / 127.5 - 1)
