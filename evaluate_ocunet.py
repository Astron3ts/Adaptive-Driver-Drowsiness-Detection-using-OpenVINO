"""Evaluate the exported OcuNet OpenVINO model on an eye-image split."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import openvino as ov


LABELS = {"awake": 0, "sleepy": 1}
IMAGE_SIZE = (34, 26)


def load_split(root: Path, split: str):
    images, labels = [], []
    for class_name, label in LABELS.items():
        paths = sorted((root / split / class_name).glob("*.png"))
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                images.append(np.asarray(image, dtype=np.float32) / 255.0)
                labels.append(label)
    if not images:
        raise FileNotFoundError(f"No PNG images found under {root / split}")
    x = np.stack(images)[:, None, :, :]
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def summarize(y_true, y_pred):
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    per_class = {}
    f1s = []
    for label, name in enumerate(("awake", "sleepy")):
        tp = int(matrix[label, label])
        fp = int(matrix[:, label].sum() - tp)
        fn = int(matrix[label, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int(matrix[label, :].sum())}
        f1s.append(f1)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix_rows_true_cols_pred_awake_sleepy": matrix.tolist(),
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/archive/data"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--model", type=Path, default=Path("ocunet.xml"))
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--images-npy", type=Path)
    parser.add_argument("--labels-npy", type=Path)
    args = parser.parse_args()

    core = ov.Core()
    if args.device not in core.available_devices:
        raise RuntimeError(f"Device {args.device!r} unavailable; found {core.available_devices}")
    model = core.read_model(args.model)
    compiled = core.compile_model(model, args.device, {"PERFORMANCE_HINT": "LATENCY"})
    output = compiled.output(0)

    if args.images_npy or args.labels_npy:
        if not args.images_npy or not args.labels_npy:
            parser.error("--images-npy and --labels-npy must be provided together")
        x = np.load(args.images_npy)
        y = np.load(args.labels_npy)
        if len(x) != len(y):
            raise ValueError("Image and label array lengths differ")
    else:
        x, y = load_split(args.data_dir, args.split)
    predictions = []
    batch_ms = []
    start = time.perf_counter()
    for begin in range(0, len(x), args.batch_size):
        batch = x[begin:begin + args.batch_size]
        tick = time.perf_counter()
        logits = compiled([batch])[output]
        batch_ms.append((time.perf_counter() - tick) * 1000.0)
        predictions.extend(np.argmax(logits, axis=1).tolist())
    evaluation_seconds = time.perf_counter() - start
    y_pred = np.asarray(predictions, dtype=np.int64)

    # Separate batch-one timing represents the live eye-crop inference shape.
    sample = x[:1]
    for _ in range(50):
        compiled([sample])
    single_ms = []
    for _ in range(500):
        tick = time.perf_counter()
        compiled([sample])
        single_ms.append((time.perf_counter() - tick) * 1000.0)

    report = {
        "model": str(args.model),
        "split": args.split if not args.images_npy else "subject-independent-held-out",
        "device": args.device,
        "device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
        "openvino_version": ov.__version__,
        "samples": int(len(y)),
        "batch_size": args.batch_size,
        "metrics": summarize(y, y_pred),
        "batched_evaluation_seconds_including_inference_only_loop": evaluation_seconds,
        "batch_inference_ms_median": float(np.median(batch_ms)),
        "batch_inference_ms_p95": float(np.percentile(batch_ms, 95)),
        "batch_one_inference_ms_median": float(np.median(single_ms)),
        "batch_one_inference_ms_p95": float(np.percentile(single_ms, 95)),
        "batch_one_inference_fps_median_latency": float(1000.0 / np.median(single_ms)),
        "caveat": ("Subject-independent held-out images; this is still eye-state classification, not full driver-drowsiness detection."
                    if args.images_npy else
                    "The repository split shares all 37 subject IDs across train, validation, and test; these are image-level results, not unseen-driver results."),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
