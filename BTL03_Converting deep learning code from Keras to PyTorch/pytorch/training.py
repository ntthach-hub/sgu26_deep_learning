"""Vòng lặp thay cho compile, fit, evaluate và các callback Keras."""
import csv
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


def make_loaders(train, val, batch_size, limit, seed):
    if limit is not None:
        generator = torch.Generator().manual_seed(seed)
        train, val = (Subset(ds, torch.randperm(len(ds), generator=generator)[:limit].tolist())
                      for ds in (train, val))
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=batch_size)


def epoch_pass(model, loader, criterion, device, task, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            size = targets.numel()
            if task == "segmentation":
                size = targets.ne(criterion.ignore_index).sum().item()
                if size == 0:
                    continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(images)
            if task == "vehicle":
                output = output.flatten(1).squeeze(1)
            loss = criterion(output, targets)
            if training:
                loss.backward()
                optimizer.step()

            count += size
            loss_sum += loss.item() * size
            if task in {"classification", "segmentation"}:
                correct += (output.argmax(1) == targets).sum().item()
            elif task == "vehicle":
                correct += ((output >= 0) == (targets > 0)).sum().item()

    if count == 0:
        raise ValueError("Tập dữ liệu không có nhãn hợp lệ để tính loss.")
    metrics = {"loss": loss_sum / count}
    if task != "regression":
        metrics["accuracy"] = correct / count
    return metrics


def fit(model, loaders, criterion, optimizer, epochs, device, task, out,
        patience=None, schedule=False, tensorboard=False):
    train_loader, val_loader = loaders
    # PyTorch giảm LR khi số epoch chờ vượt patience, tương ứng 5 epoch ở Keras.
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=4, threshold_mode="abs", min_lr=0.0001
    ) if schedule else None)
    log = nullcontext()
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        log = SummaryWriter(str(out / "tensorboard"))

    history = []
    checkpoint = out / "best.pt"
    best_loss = float("inf")
    waiting = 0
    with log as writer:
        for epoch in range(1, epochs + 1):
            train = epoch_pass(model, train_loader, criterion, device, task, optimizer)
            val = epoch_pass(model, val_loader, criterion, device, task)
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update({f"train_{key}": value for key, value in train.items()})
            row.update({f"val_{key}": value for key, value in val.items()})
            history.append(row)
            print("  ".join(f"{key}={value:.6g}" for key, value in row.items()), flush=True)

            if writer is not None:
                for key, value in row.items():
                    if key != "epoch":
                        writer.add_scalar(key, value, epoch)
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                waiting = 0
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(model.state_dict(), temporary)
                temporary.replace(checkpoint)
            else:
                waiting += 1
            if scheduler is not None:
                scheduler.step(val["loss"])
            if patience is not None and waiting >= patience:
                break

    with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=history[0])
        csv_writer.writeheader()
        csv_writer.writerows(history)

    # Notebook có ModelCheckpoint nạp lại epoch có val_loss tốt nhất.
    if schedule:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    else:
        torch.save(model.state_dict(), out / "last.pt")
    final = epoch_pass(model, val_loader, criterion, device, task)
    print(f"Evaluation: {final}", flush=True)
    return history


def plot_history(history, out, loss_label):
    metrics = ("loss", "accuracy") if "train_accuracy" in history[0] else ("loss",)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.5), squeeze=False)
    epochs = [row["epoch"] for row in history]
    for ax, metric in zip(axes[0], metrics):
        for split in ("train", "val"):
            ax.plot(epochs, [row[f"{split}_{metric}"] for row in history], marker=".", label=split)
        ax.set(xlabel="Epoch", ylabel=loss_label if metric == "loss" else "Accuracy")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=140)
    plt.close(fig)
