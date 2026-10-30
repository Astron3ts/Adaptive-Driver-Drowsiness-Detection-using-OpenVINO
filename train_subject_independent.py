"""Train OcuNet with subject-disjoint train/validation/test partitions."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_model import IMG_H, IMG_W, OcuNet, build_cache


CLASSES = {"awake": 0, "sleepy": 1}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


class SmallCNN(torch.nn.Module):
    """Small LeNet-style CNN baseline trained on the same eye crops."""

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(16, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


class TwoLayerCNN(torch.nn.Module):
    """Moderate-capacity two-convolution baseline for a fairer comparison."""

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def collect_images(root):
    grouped = {}
    for original_split in ("train", "val", "test"):
        images, labels = build_cache(str(root), original_split)
        paths = []
        for class_name in CLASSES:
            paths.extend(sorted((root / original_split / class_name).glob("*.png")))
        if len(paths) != len(labels):
            raise RuntimeError(f"Cache/file count mismatch for {original_split}")
        for index, path in enumerate(paths):
            subject = path.name.split("_")[0]
            grouped.setdefault(subject, []).append((images[index], int(labels[index])))
    if not grouped:
        raise FileNotFoundError(f"No images found below {root}")
    return grouped


def subject_counts(grouped):
    result = {}
    for subject, samples in grouped.items():
        result[subject] = np.bincount([label for _, label in samples], minlength=2)
    return result


def choose_subject_split(grouped, seed):
    counts = subject_counts(grouped)
    subjects = sorted(counts)
    n_train = round(len(subjects) * SPLIT_RATIOS["train"])
    n_val = round(len(subjects) * SPLIT_RATIOS["val"])
    n_test = len(subjects) - n_train - n_val
    total_by_class = np.sum(list(counts.values()), axis=0)

    # Search seeded subject permutations and retain the split whose class
    # counts most closely match the intended 70/15/15 sample proportions.
    best = None
    rng = random.Random(seed)
    for _ in range(5000):
        order = subjects[:]
        rng.shuffle(order)
        assignments = {
            "train": order[:n_train],
            "val": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:],
        }
        score = 0.0
        for split, ratio in SPLIT_RATIOS.items():
            split_count = np.sum([counts[s] for s in assignments[split]], axis=0)
            score += float(np.sum(((split_count / total_by_class) - ratio) ** 2))
        if best is None or score < best[0]:
            best = (score, assignments)
    return best[1]


def build_arrays(grouped, subjects):
    images, labels = [], []
    for subject in subjects:
        for image, label in grouped[subject]:
            images.append(image)
            labels.append(label)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def metrics(labels, predictions):
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    per_class = {}
    f1_values = []
    for index, name in enumerate(("awake", "sleepy")):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int(matrix[index, :].sum())}
        f1_values.append(f1)
    return {"accuracy": float(np.mean(labels == predictions)),
            "macro_f1": float(np.mean(f1_values)),
            "confusion_matrix_rows_true_cols_pred_awake_sleepy": matrix.tolist(),
            "per_class": per_class}


def evaluate_torch(model, loader, device):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            all_labels.extend(y.numpy().tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return metrics(np.asarray(all_labels), np.asarray(all_predictions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/archive/data"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--architecture", choices=("OcuNet", "SmallCNN", "TwoLayerCNN"), default="OcuNet")
    parser.add_argument("--output-prefix")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    grouped = collect_images(args.data_dir)
    split_subjects = choose_subject_split(grouped, args.seed)
    overlap = (set(split_subjects["train"]) & set(split_subjects["val"]) |
               set(split_subjects["train"]) & set(split_subjects["test"]) |
               set(split_subjects["val"]) & set(split_subjects["test"]))
    if overlap:
        raise RuntimeError(f"Subject leakage in generated split: {sorted(overlap)}")

    datasets = {name: build_arrays(grouped, subjects) for name, subjects in split_subjects.items()}
    np.save(args.data_dir / "cache_subject_independent_test_x.npy", datasets["test"][0])
    np.save(args.data_dir / "cache_subject_independent_test_y.npy", datasets["test"][1])
    loaders = {
        name: DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                         batch_size=args.batch_size, shuffle=(name == "train"),
                         num_workers=0, pin_memory=(device.type == "cuda"))
        for name, (x, y) in datasets.items()
    }

    model_classes = {"OcuNet": OcuNet, "SmallCNN": SmallCNN, "TwoLayerCNN": TwoLayerCNN}
    model = model_classes[args.architecture]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    prefix = args.output_prefix or f"{args.architecture.lower()}_subject_independent"
    report_path = args.report or Path(f"metrics_{args.architecture.lower()}_subject_independent.json")
    checkpoint = Path(f"{prefix}_best.pt")
    best_val = -1.0
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(y)
        scheduler.step()
        val_metrics = evaluate_torch(model, loaders["val"], device)
        epoch_result = {"epoch": epoch, "train_loss": loss_sum / seen,
                        "train_accuracy": correct / seen,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"]}
        history.append(epoch_result)
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            torch.save(model.state_dict(), checkpoint)
        print(json.dumps(epoch_result), flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_metrics = evaluate_torch(model, loaders["test"], device)
    elapsed = time.perf_counter() - started

    report = {
        "model": args.architecture,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dataset": "MRL eye images from the project dataset folder",
        "seed": args.seed,
        "device": str(device),
        "subject_split": {name: sorted(ids) for name, ids in split_subjects.items()},
        "split_counts": {name: {"subjects": len(split_subjects[name]),
                                "images": int(len(datasets[name][1])),
                                "awake": int(np.sum(datasets[name][1] == 0)),
                                "sleepy": int(np.sum(datasets[name][1] == 1))}
                         for name in split_subjects},
        "best_validation_accuracy": best_val,
        "test_metrics": test_metrics,
        "epochs": args.epochs,
        "training_seconds": elapsed,
        "checkpoint": str(checkpoint),
        "caveat": "Subject-independent eye-state classification; not an end-to-end driver drowsiness evaluation.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        import openvino as ov
        model.eval().to("cpu")
        ov_model = ov.convert_model(model, example_input=torch.zeros(1, 1, IMG_H, IMG_W))
        ov.save_model(ov_model, f"{prefix}.xml")
    except ImportError:
        print("OpenVINO not installed; skipped IR export.", flush=True)
    print(json.dumps({"report": str(report_path), "test_metrics": test_metrics,
                      "training_seconds": elapsed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
