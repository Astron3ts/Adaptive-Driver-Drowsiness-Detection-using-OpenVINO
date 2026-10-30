import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import openvino as ov
except ImportError:
    ov = None

IMG_H, IMG_W = 26, 34
LABEL_MAP = {"awake": 0, "sleepy": 1}


def setup_logger(log_path):
    logger = logging.getLogger("ocunet")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def build_cache(data_dir, split, logger=None):
    log = logger.info if logger else print
    root = Path(data_dir) / split
    cache_x = Path(data_dir) / f"cache_{split}_x.npy"
    cache_y = Path(data_dir) / f"cache_{split}_y.npy"

    if cache_x.exists() and cache_y.exists():
        log(f"  [{split}] Loading from cache ...")
        images = np.load(cache_x)
        labels = np.load(cache_y)
        return images, labels

    log(f"  [{split}] Building cache from PNGs (one-time) ...")
    images, labels = [], []
    for cls_name, label in LABEL_MAP.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing: {cls_dir}")
        paths = sorted(cls_dir.glob("*.png"))
        for p in paths:
            img = Image.open(p).convert("L").resize((IMG_W, IMG_H), Image.BILINEAR)
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(label)

    images = np.stack(images)[:, None, :, :]   # (N, 1, H, W)
    labels = np.array(labels, dtype=np.int64)
    np.save(cache_x, images)
    np.save(cache_y, labels)
    log(f"  [{split}] Cached {len(images):,} samples -> {cache_x.name}, {cache_y.name}")
    return images, labels


def make_loader(images, labels, batch_size, shuffle, workers):
    x = torch.from_numpy(images)
    y = torch.from_numpy(labels)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True)


class OcuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += int((preds == y).sum())
            total += y.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0


def train_model(model, train_loader, val_loader, device, epochs=25, save_path="ocunet_best.pt", logger=None):
    log = logger.info if logger else print

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_epoch = 0
    epoch_log = []
    start_time = time.time()

    log("-" * 65)
    log(f"{'Epoch':>6}  {'Loss':>8}  {'Train Acc':>10}  {'Val Acc':>9}  {'LR':>10}")
    log("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += int((logits.argmax(1) == y).sum())
            total += y.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader, device)
        avg_loss = running_loss / total

        epoch_log.append({
            "epoch": epoch,
            "loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == epochs:
            log(f"{epoch:>6}  {avg_loss:>8.4f}  {train_acc:>9.2f}%  {val_acc:>8.2f}%  {current_lr:>10.6f}")

    elapsed = time.time() - start_time
    log("-" * 65)
    log(f"Training complete in {elapsed:.1f}s  |  Best Val Acc: {best_acc:.2f}%  (epoch {best_epoch})")
    log("-" * 65)

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, best_acc, best_epoch, elapsed, epoch_log


def export_to_openvino(model, output_prefix="ocunet", logger=None):
    log = logger.info if logger else print

    if ov is None:
        log("[Warning] OpenVINO is not installed; skipping IR export.")
        return None, None

    model.eval().to("cpu")
    dummy_input = torch.zeros(1, 1, IMG_H, IMG_W, dtype=torch.float32)
    ov_model = ov.convert_model(model, example_input=dummy_input)

    xml_path = f"{output_prefix}.xml"
    bin_path = f"{output_prefix}.bin"
    ov.save_model(ov_model, xml_path)

    xml_size = Path(xml_path).stat().st_size / 1024
    bin_size = Path(bin_path).stat().st_size / 1024
    log(f"[OpenVINO] Exported IR -> {xml_path} ({xml_size:.1f} KB)  |  {bin_path} ({bin_size:.1f} KB)")
    return xml_path, bin_path


def main():
    parser = argparse.ArgumentParser(description="Train OcuNet Eye Classifier and Export to OpenVINO IR")
    parser.add_argument("--data-dir", type=str, default="dataset/archive/data", help="Root folder with train/val subfolders (awake/sleepy)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--output", type=str, default="ocunet", help="Prefix for exported OpenVINO model")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker threads (0 = main thread)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"ocunet_training_{timestamp}.txt")
    logger = setup_logger(log_path)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    logger.info("=" * 65)
    logger.info("  OcuNet Training Log")
    logger.info(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)
    logger.info(f"  Device      : {device_name} ({device})")
    logger.info(f"  Data Dir    : {args.data_dir}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  Batch Size  : {args.batch_size}")
    logger.info(f"  Output      : {args.output}")
    logger.info(f"  Log File    : {log_path}")
    logger.info("=" * 65)

    logger.info("")
    logger.info("  Dataset Preparation")
    logger.info("-" * 65)
    try:
        train_x, train_y = build_cache(args.data_dir, "train", logger)
        val_x, val_y = build_cache(args.data_dir, "val", logger)
    except Exception as e:
        logger.info(f"[Error] Dataset loading failed: {e}")
        return

    logger.info(f"  Train samples : {len(train_x):,}")
    logger.info(f"  Val samples   : {len(val_x):,}")
    logger.info(f"  Image size    : {IMG_H}x{IMG_W} (grayscale)")
    logger.info("")

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_x, val_y, args.batch_size, shuffle=False, workers=args.workers)

    model = OcuNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters : {total_params:,}")
    logger.info("")

    model, best_acc, best_epoch, elapsed, epoch_log = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        save_path=f"{args.output}_best.pt",
        logger=logger,
    )

    logger.info("")
    logger.info("  Per-Epoch Summary")
    logger.info("-" * 65)
    for e in epoch_log:
        logger.info(
            f"  Epoch {e['epoch']:02d} | Loss {e['loss']:.4f} | "
            f"Train {e['train_acc']:.2f}% | Val {e['val_acc']:.2f}% | LR {e['lr']:.6f}"
        )

    logger.info("")
    logger.info("  Export")
    logger.info("-" * 65)
    export_to_openvino(model, output_prefix=args.output, logger=logger)

    logger.info("")
    logger.info("=" * 65)
    logger.info(f"  Best Epoch      : {best_epoch}")
    logger.info(f"  Best Val Acc    : {best_acc:.2f}%")
    logger.info(f"  Training Time   : {elapsed:.1f}s")
    logger.info(f"  Finished        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)
    logger.info(f"\nLog saved to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
