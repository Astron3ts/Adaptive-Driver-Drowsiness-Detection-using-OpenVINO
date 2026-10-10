# OpenVINO Driver Drowsiness Detection System

Real-time driver fatigue monitoring system featuring **Dual-Stream Eye Verification (EAR + Custom OcuNet CNN)**, **Mouth Aspect Ratio (MAR)**, and **3D Head Pose Estimation** with an onboarding statistical calibration model.

---

## Features

- **Dual-Stream Eye Verification**:
  - **Geometric EAR**: 6-point Euclidean eye aspect ratio with dynamic thresholding.
  - **OcuNet Neural Classifier**: Custom-trained lightweight CNN (~72 KB) running on OpenVINO at 0.12ms latency.
- **Yawn Detection**: Real-time Mouth Aspect Ratio (MAR) tracking prolonged mouth opening.
- **Head Nodding Detection**: Perspective-n-Point (`cv2.solvePnP`) 3D head pose estimation (Pitch < -18°).
- **Phase 1 Baseline Calibration (60s)**: Personalizes driver EAR thresholds with 10th-percentile blink outlier filtering.
- **Hardware Acceleration**: Dynamic target re-compilation (`CPU`, `GPU`, `AUTO`) via Intel OpenVINO.

---

## Installation

```bash
git clone <repository-url>
cd project
pip install -r requirements.txt
```

---

## Usage

### 1. Run Real-Time Detection
```bash
python drowsiness_detector.py
```

#### Controls:
- **`q`** / **`Esc`**: Quit application cleanly.
- **`1`**: Switch OpenVINO inference target to **CPU**.
- **`2`**: Switch OpenVINO inference target to **GPU**.
- **`3`**: Switch OpenVINO inference target to **AUTO**.
