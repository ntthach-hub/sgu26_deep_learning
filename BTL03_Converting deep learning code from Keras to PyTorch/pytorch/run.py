"""Điểm chạy cho các ví dụ PyTorch của BTL03."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

import data
import gaussian
import models
import previews
from training import fit, make_loaders, plot_history


TASKS = {
    "mnist_mlp": (models.mnist_mlp, "classification", 30, 128, 0.01),
    "mnist_cnn": (models.mnist_cnn, "classification", 30, 128, 0.01),
    "gender_cnn": (models.gender_cnn, "classification", 30, 128, 0.0001),
    "gender_callbacks": (models.gender_cnn, "classification", 100000, 128, 0.001),
    "emotion_cnn": (models.emotion_cnn, "classification", 15, 128, 0.0001),
    "emotion_augmentation": (models.emotion_cnn, "classification", 100, 128, 0.0001),
    "digit_sum": (models.DigitSum, "regression", 30, 128, 0.01),
    "segnet": (models.segnet, "segmentation", 15, 8, 0.0001),
    "vehicle": (models.VehicleCNN, "vehicle", 5, 128, 0.001),
}


def positive(value):
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return integer


def arguments():
    parser = argparse.ArgumentParser(description="Run a PyTorch example from BTL03.")
    parser.add_argument("example", choices=("environment", "gaussian", *TASKS))
    parser.add_argument("--data", type=Path, help="Directory containing the example's data files.")
    parser.add_argument("--out", type=Path, default=data.ROOT / "results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=positive, help="Epochs, or a cap per Gaussian sweep trial.")
    parser.add_argument("--batch-size", type=positive)
    parser.add_argument("--limit", type=positive, help="Limit samples per split for a quick run.")
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--seed", type=int, default=2018)
    return parser.parse_args()


def load_datasets(example, root, seed):
    if example in {"mnist_mlp", "mnist_cnn", "digit_sum"}:
        train, val = data.mnist_pair(root)
        if example == "digit_sum":
            train, val = data.DigitSequences(train, seed), data.DigitSequences(val, seed + 1)
        return train, val
    if example.startswith("gender_"):
        return data.gender_pair(root)
    if example.startswith("emotion_"):
        return data.emotion_pair(root, augment=example == "emotion_augmentation")
    if example == "segnet":
        archive = root / "depth_hand.zip"
        return data.HandDepth("train", archive), data.HandDepth("test", archive)
    return data.vehicle_pair(seed, root / "car_data.zip")


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        raise SystemExit("PyTorch chưa nhận được CUDA. Hãy chọn --device cpu.")
    device = torch.device(("cuda" if cuda_available else "cpu") if args.device == "auto" else args.device)
    print(f"PyTorch {torch.__version__} | CUDA: {cuda_available} | Device: {device}", flush=True)
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    example = args.example
    if example == "environment":
        return

    out = args.out / example
    if example == "gaussian":
        out.mkdir(parents=True, exist_ok=True)
        gaussian.run(out, device, args.seed, args.epochs)
        return

    builder, task, epochs, batch_size, lr = TASKS[example]
    epochs = args.epochs or epochs
    batch_size = args.batch_size or batch_size
    default_root = data.EXTRAS if example in {"segnet", "vehicle"} else data.ROOT / "datasets"
    root = args.data if args.data is not None else default_root
    try:
        train, val = load_datasets(example, root, args.seed)
        if task == "vehicle":
            # Kiểm tra ảnh tạo heatmap trước khi bắt đầu huấn luyện.
            with (root / "test.jpg").open("rb"):
                pass
    except FileNotFoundError as error:
        missing_path = error.filename or str(error)
        raise SystemExit(f"Thiếu dữ liệu: {missing_path}. Xem pytorch/HƯỚNG_DẪN.md.") from error
    loaders = make_loaders(train, val, batch_size, args.limit, args.seed)
    train_count, val_count = (len(loader.dataset) for loader in loaders)
    model = builder().to(device)
    model.apply(models.initialize_keras_style)
    print(f"Train: {train_count} | Evaluation: {val_count} | Batch: {batch_size} | Epochs: {epochs}")
    print(f"Parameters: {sum(param.numel() for param in model.parameters()):,}")
    if example in {"mnist_mlp", "mnist_cnn", "digit_sum"}:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    if task == "segmentation":
        criterion = nn.CrossEntropyLoss(ignore_index=data.MASK_IGNORE)
    else:
        criterion = {"classification": nn.CrossEntropyLoss, "regression": nn.L1Loss,
                     "vehicle": nn.MSELoss}[task]()
    callbacks = example == "gender_callbacks"
    metadata = dict(example=example, device=str(device), seed=args.seed, batch_size=batch_size,
                    epochs=epochs, train_samples=train_count, evaluation_samples=val_count,
                    loss=criterion.__class__.__name__, learning_rate=lr, torch=torch.__version__)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    history = fit(model, loaders, criterion, optimizer, epochs, device, task, out,
                  patience=5 if callbacks or example == "emotion_augmentation" else None,
                  schedule=callbacks, tensorboard=callbacks)
    plot_history(history, out, criterion.__class__.__name__)
    if task == "vehicle":
        previews.vehicle_heatmap(model, out / "prediction.png", device, image_path=root / "test.jpg")
    else:
        names = data.GENDER_NAMES if example.startswith("gender_") else None
        if example.startswith("emotion_"):
            names = data.EMOTION_NAMES
        previews.sample_prediction(model, loaders[1].dataset, device, out / "prediction.png", task, names)
    if example in {"mnist_mlp", "mnist_cnn"}:
        previews.digit_features(model, loaders[0].dataset, device, out, example == "mnist_cnn")


if __name__ == "__main__":
    main()
