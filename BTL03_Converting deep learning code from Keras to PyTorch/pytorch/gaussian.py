"""Bản chuyển Func_apxm_MLP.ipynb."""
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import initialize_keras_style


def gaussian_grid(size, device):
    grid = torch.linspace(-3.0, 3.0, size, device=device)
    u, v = torch.meshgrid(grid, grid, indexing="ij")
    density = torch.exp(-(u.square() + v.square()) / 2) / (2 * math.pi)
    features = torch.stack((u, v), dim=-1).reshape(-1, 2)
    return features, density.reshape(-1, 1)


def run(out, device, seed, epoch_limit=None):
    x_train, y_train = gaussian_grid(32, device)
    x_test, y_test = gaussian_grid(10, device)
    generator = torch.Generator().manual_seed(seed)
    loss_fn = nn.MSELoss()
    experiments = {
        "samples": [(a, 16, 100) for a in range(100, 1001, 100)],
        "hidden": [(800, b, 100) for b in (8, 16, 32, 64, 128, 256, 512)],
        "epochs": [(800, 128, c) for c in range(10, 200, 10)],
    }
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (factor, configurations) in zip(axes, experiments.items()):
        values, train_errors, test_errors = [], [], []
        for samples, hidden, epochs in configurations:
            selected = torch.randperm(1000, generator=generator)[:samples] + 1
            batches = DataLoader(TensorDataset(x_train[selected], y_train[selected]), batch_size=128,
                                 shuffle=True, generator=generator)
            model = nn.Sequential(nn.Linear(2, hidden), nn.Sigmoid(),
                                  nn.Linear(hidden, 1), nn.Sigmoid()).to(device)
            model.apply(initialize_keras_style)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            actual_epochs = epochs if epoch_limit is None else min(epoch_limit, epochs)
            for _ in range(actual_epochs):
                for features, targets in batches:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(features), targets)
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                train_errors.append(loss_fn(model(x_train), y_train).item())
                test_errors.append(loss_fn(model(x_test), y_test).item())
            factors = {"samples": samples, "hidden": hidden, "epochs": epochs}
            values.append(factors[factor])
            rows.append({"experiment": factor, **factors, "trained_epochs": actual_epochs,
                         "train_mse": train_errors[-1], "test_mse": test_errors[-1]})
        ax.plot(values, train_errors, label="train")
        ax.plot(values, test_errors, label="test")
        ax.set(xlabel=factor, ylabel="MSE")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gaussian_sweeps.png", dpi=150)
    plt.close(fig)

    surfaces = plt.figure(figsize=(8, 4))
    for index, (size, title) in enumerate(((32, "Train"), (10, "Test")), 1):
        x, y = gaussian_grid(size, torch.device("cpu"))
        ax = surfaces.add_subplot(1, 2, index, projection="3d")
        ax.plot_surface(x[:, 0].reshape(size, size).numpy(), x[:, 1].reshape(size, size).numpy(),
                        y.reshape(size, size).numpy(), cmap="viridis")
        ax.set(title=title, xlabel="x1", ylabel="x2")
    surfaces.tight_layout()
    surfaces.savefig(out / "gaussian_data.png", dpi=140)
    plt.close(surfaces)

    with (out / "sweeps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} configurations. Lowest test MSE: {min(r['test_mse'] for r in rows):.6f}")
