"""Minh họa dự đoán và các đặc trưng của từng ví dụ."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import pil_to_tensor

from data import EXTRAS, MASK_COLORS, MASK_IGNORE


@torch.inference_mode()
def sample_prediction(model, dataset, device, path, task, labels=None):
    images, target = dataset[0]
    output = model(images.unsqueeze(0).to(device))[0].cpu()
    if task == "classification":
        predicted = output.argmax().item()
        confidence = output.softmax(0)[predicted].item()
        name = str if labels is None else labels.__getitem__
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(images[0], cmap="gray")
        ax.set_title(f"True: {name(target.item())}, predicted: {name(predicted)} ({confidence:.1%})")
        ax.axis("off")
    elif task == "segmentation":
        predicted = output.argmax(0).numpy()
        ground_truth = MASK_COLORS[target]
        ground_truth[target.numpy() == MASK_IGNORE] = (0.5, 0.5, 0.5)
        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        for ax, image, title in zip(
            axes, (images[0], ground_truth, MASK_COLORS[predicted]),
            ("Depth", "Ground truth", "Prediction")
        ):
            ax.imshow(image, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
    else:
        fig, axes = plt.subplots(1, len(images), figsize=(10, 2), squeeze=False)
        for ax, image in zip(axes[0], images):
            ax.imshow(image[0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        fig.suptitle(f"True sum: {target.item():.0f}, estimated sum: {output.item():.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.inference_mode()
def digit_features(model, dataset, device, out, cnn):
    images, _ = dataset[0]
    output, hidden, maps = model(images.unsqueeze(0).to(device), return_features=True)
    np.savez(out / "sample_features.npz", fc2=hidden.cpu().numpy(),
             probabilities=output.softmax(1).cpu().numpy(), gap=maps.cpu().numpy())
    if cnn and len(dataset) >= 2:
        data = Subset(dataset, range(min(2000, len(dataset))))
        pixels, hidden, pooled, labels = [], [], [], []
        for images, targets in DataLoader(data, batch_size=128):
            _, fc2, gap = model(images.to(device), return_features=True)
            pixels.append(images.flatten(1))
            hidden.append(fc2.cpu())
            pooled.append(gap.cpu())
            labels.append(targets)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, vectors, title in zip(axes, (pixels, hidden, pooled), ("Raw pixels", "FC2", "Pool3 + GAP")):
            xy = PCA(n_components=2).fit_transform(torch.cat(vectors).numpy())
            ax.scatter(*xy.T, c=torch.cat(labels), cmap="tab10", vmin=-0.5, vmax=9.5, s=7)
            ax.set(title=title, xlabel="PC1", ylabel="PC2")
        fig.tight_layout()
        fig.savefig(out / "mnist_pca.png", dpi=140)
        plt.close(fig)


@torch.inference_mode()
def vehicle_heatmap(model, path, device, threshold=0.99, image_path=EXTRAS / "test.jpg"):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = pil_to_tensor(image).unsqueeze(0).float().to(device)
    heatmap = model(tensor)[0, 0].cpu().numpy()
    selected = heatmap > threshold
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for y, x in np.argwhere(selected):
        left, top = int(x) * 8, int(y) * 8
        draw.rectangle((left, top, left + 63, top + 63), outline="lime", width=1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, plot, title in zip(axes, (heatmap, selected, annotated),
                               ("Heatmap", f"Score > {threshold:g}", "Detections")):
        ax.imshow(plot)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
