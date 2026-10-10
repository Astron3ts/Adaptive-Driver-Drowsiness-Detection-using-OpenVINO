import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import openvino as ov

class OcuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 26x34 -> 13x17
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 13x17 -> 6x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 6x8 -> 3x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, 2)  # [logit_closed, logit_open]
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def load_data(data_dir="dataset"):
    print("[1/5] Loading and preparing eye dataset...")
    x_train = np.load(os.path.join(data_dir, "x_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy")).squeeze().astype(np.int64)
    x_val   = np.load(os.path.join(data_dir, "x_val.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy")).squeeze().astype(np.int64)

    # Convert shape: (N, H, W, C) -> (N, C, H, W) and normalize [0, 1]
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32) / 255.0
    x_val   = np.transpose(x_val,   (0, 3, 1, 2)).astype(np.float32) / 255.0

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(x_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    print(f"      Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")
    return train_loader, val_loader

def train_model(train_loader, val_loader, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/5] Training OcuNet on target hardware: {device.type.upper()}")
    if device.type == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    model = OcuNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = "ocunet_best.pt"

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"      Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:5.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:5.1f}%")

    print(f"[3/5] Training Complete in {time.time() - t_start:.1f}s! Peak Val Accuracy: {best_val_acc:.2f}%")
    print(f"      Saved PyTorch weights: {best_weights_path}")

    # Load best weights
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()
    return model

def convert_to_openvino(torch_model, output_name="ocunet"):
    print("[4/5] Converting PyTorch model -> Native OpenVINO IR (.xml + .bin)...")
    torch_model.eval().to("cpu")
    dummy_input = torch.randn(1, 1, 26, 34, dtype=torch.float32)

    ov_model = ov.convert_model(torch_model, example_input=dummy_input)
    xml_path = f"{output_name}.xml"
    bin_path = f"{output_name}.bin"
    ov.save_model(ov_model, xml_path)
    print(f"      Exported OpenVINO Model: {xml_path} and {bin_path}")
    return xml_path, bin_path

def benchmark_openvino(xml_path):
    print("[5/5] Benchmarking OpenVINO compiled inference...")
    core = ov.Core()
    devices = core.available_devices
    print(f"      Detected OpenVINO Devices: {devices}")

    dummy_tensor = np.random.randn(1, 1, 26, 34).astype(np.float32)

    print("-" * 65)
    print(f"{'Target Device':<20} | {'Mean Latency':<18} | {'Throughput (FPS)':<15}")
    print("-" * 65)

    for dev in devices:
        try:
            compiled = core.compile_model(xml_path, dev)
            out_layer = compiled.output(0)

            # Warmup
            for _ in range(50):
                _ = compiled([dummy_tensor])[out_layer]

            # Measure
            iters = 500
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = compiled([dummy_tensor])[out_layer]
            total_sec = time.perf_counter() - t0

            mean_ms = (total_sec / iters) * 1000.0
            fps = iters / total_sec
            print(f"{dev:<20} | {mean_ms:6.3f} ms           | {fps:8.1f} FPS")
        except Exception as e:
            print(f"{dev:<20} | Skipped ({e})")
    print("-" * 65)

if __name__ == "__main__":
    train_loader, val_loader = load_data()
    model = train_model(train_loader, val_loader, epochs=25)
    xml_path, _ = convert_to_openvino(model)
    benchmark_openvino(xml_path)
    print("[Done] Custom eye model trained on CUDA and ready for OpenVINO deployment!")
