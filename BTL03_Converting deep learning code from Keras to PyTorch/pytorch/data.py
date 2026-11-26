"""Đọc dữ liệu gốc, giữ train và validation của từng notebook."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
EXTRAS = ROOT / "examples" / "5. Others"
MASK_VALUES = np.array([255, 43, 85, 128, 170, 212, 0])
MASK_IGNORE = -1
MASK_COLORS = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
                        [0.5, 0, 0.5], [0.5, 0.5, 0], [1, 1, 1]])
GENDER_NAMES = ("female", "male")
EMOTION_NAMES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def mnist_pair(root):
    cache = root / "mnist.npz"
    if cache.is_file():
        with np.load(cache) as data:
            return tuple(MnistSplit(data[f"x_{split}"], data[f"y_{split}"])
                         for split in ("train", "test"))
    return tuple(datasets.MNIST(root, train=train, transform=transforms.ToTensor(), download=True)
                 for train in (True, False))


class MnistSplit(Dataset):
    def __init__(self, images, labels):
        self.data = torch.from_numpy(images)
        self.targets = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index].unsqueeze(0).float() / 255, self.targets[index]


class DigitSequences(Dataset):
    def __init__(self, mnist, seed, steps=8):
        self.mnist = mnist
        self.steps = steps
        rng = np.random.default_rng(seed)
        self.lengths = rng.integers(1, steps + 1, size=len(mnist))
        self.indices = rng.integers(len(mnist), size=(len(mnist), steps))

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        count = self.lengths[index]
        selected = self.indices[index, :count]
        frames = torch.zeros(self.steps, 1, 28, 28)
        frames[:count, 0] = self.mnist.data[selected].float() / 255
        total = self.mnist.targets[selected].sum().float()
        return frames, total


def gender_pair(root):
    result = []
    for prefix in ("64_64_11938_4098_train", "64_64_5968_4098_val"):
        x = np.load(root / f"{prefix}_x_onehot.npy", mmap_mode="r")
        y = np.load(root / f"{prefix}_y_onehot.npy")
        x = np.rot90(x.reshape(-1, 64, 64), k=3, axes=(1, 2))
        images = torch.from_numpy(np.ascontiguousarray(x[:, None], dtype=np.float32))
        result.append(TensorDataset(images, torch.from_numpy(y.argmax(axis=1))))
    return tuple(result)


class EmotionDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        rows = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, ndmin=2)
        if rows.shape[1] != 48 * 48 + 7:
            raise ValueError(f"{csv_path} phải gồm 2304 cột pixel và 7 cột one-hot.")
        self.images = torch.from_numpy(rows[:, :2304].copy().reshape(-1, 1, 48, 48))
        self.labels = torch.from_numpy(rows[:, 2304:].argmax(axis=1))
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            # Padding biên gần nhất tương ứng fill_mode='nearest' của Keras.
            padded = torch.nn.functional.pad(image, (32, 32, 32, 32), mode="replicate")
            angle = float(torch.empty(1).uniform_(-45, 45))
            offset = tuple(round(float(torch.empty(1).uniform_(-9.6, 9.6))) for _ in range(2))
            padded = transforms.functional.affine(
                padded, angle, offset, 1.0, [0.0, 0.0],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            image = transforms.functional.center_crop(padded, (48, 48))
            if torch.rand(()) < 0.5:
                image = transforms.functional.hflip(image)
        return image, self.labels[index]


def emotion_pair(root, augment=False):
    return (
        EmotionDataset(root / "fer2013_training_onehot.csv", augment),
        EmotionDataset(root / "fer2013_publictest_onehot.csv"),
    )


class ImageArchive(Dataset):
    def __init__(self, archive):
        self.archive = archive
        self.reader = None
        with ZipFile(archive) as reader:
            self.names = sorted(name for name in reader.namelist() if name.lower().endswith(".png"))

    def image(self, name):
        if self.reader is None:
            self.reader = ZipFile(self.archive)
        with Image.open(BytesIO(self.reader.read(name))) as image:
            return image.copy()

    def __len__(self):
        return len(self.names)


class HandDepth(ImageArchive):
    def __init__(self, split, archive=EXTRAS / "depth_hand.zip"):
        super().__init__(archive)
        self.names = [name for name in self.names if name.startswith(f"data/segnet/{split}/depth/")]

    def __getitem__(self, index):
        name = self.names[index]
        label_name = name.replace("/depth/", "/labels/").replace("_depth_", "_labels_")
        shape = (96, 96)
        image = np.asarray(self.image(name).resize(shape, Image.Resampling.NEAREST), dtype=np.float32)
        mask = np.asarray(self.image(label_name).resize(shape, Image.Resampling.NEAREST))
        # Một số pixel trong ZIP nằm ngoài bảng mã bảy lớp của notebook.
        classes = np.full(shape, MASK_IGNORE, dtype=np.int64)
        for label, value in enumerate(MASK_VALUES):
            classes[mask == value] = label
        return torch.from_numpy(image[None]), torch.from_numpy(classes)


class Vehicles(ImageArchive):
    def __init__(self, archive=EXTRAS / "car_data.zip"):
        super().__init__(archive)
        self.labels = [1.0 if "/vehicles/" in name else -1.0 for name in self.names]

    def __getitem__(self, index):
        image = self.image(self.names[index]).convert("RGB")
        if image.size != (64, 64):
            image = image.resize((64, 64), Image.Resampling.BILINEAR)
        return transforms.PILToTensor()(image).float(), torch.tensor(self.labels[index])


def vehicle_pair(seed, archive=EXTRAS / "car_data.zip"):
    data = Vehicles(archive)
    train, val = train_test_split(
        range(len(data)), test_size=0.3, random_state=seed, stratify=data.labels
    )
    return torch.utils.data.Subset(data, train), torch.utils.data.Subset(data, val)
