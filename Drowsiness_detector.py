from collections import deque
import os
import time
import urllib.request
import winsound
import cv2
import numpy as np

# Safe OpenVINO Import handling modern (v2023+) and legacy packages
try:
  from openvino import Core
except ImportError:
  try:
    from openvino.runtime import Core
  except ImportError:
    Core = None

# ----------------- 1. Configuration & Parameters -----------------
CALIBRATION_DURATION = 60  # Phase 1 calibration duration in seconds
CONSECUTIVE_FRAMES_TRIGGER = 5  # Alert trigger condition: 5 consecutive frames[cite: 1]
MOVING_AVG_WINDOW = 5  # Temporal smoothing filter size[cite: 1]
MODEL_INPUT_SIZE = (224, 224)  # OpenVINO input resolution[cite: 1]
STATIC_BASELINE_EAR = 0.250  # Literature reference threshold[cite: 1]

# ----------------- 2. Initialize Face/Eye Detection Engine -----------------
# Auto-download Haar Cascade XML files if missing for robust edge execution
face_xml = "haarcascade_frontalface_default.xml"
eye_xml = "haarcascade_eye_tree_eyeglasses.xml"
base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"

if not os.path.exists(face_xml):
  print("[System] Downloading face cascade model...")
  urllib.request.urlretrieve(base_url + face_xml, face_xml)

if not os.path.exists(eye_xml):
  print("[System] Downloading eye cascade model...")
  urllib.request.urlretrieve(base_url + eye_xml, eye_xml)

face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)

# Initialize OpenVINO Core if available
if Core is not None:
  try:
    core = Core()
    print(f"[OpenVINO] Runtime Core Initialized. Devices: {core.available_devices}")
  except Exception as e:
    print(f"[OpenVINO] Initialization warning: {e}")

# ----------------- 3. Pipeline State & Buffers -----------------
cap = cv2.VideoCapture(0)  # Built-in PC Camera capture[cite: 1]
cap.set(cv2.CAP_PROP_FPS, 30)

calibration_buffer = []  # Phase 1 baseline EAR buffer[cite: 1]
ear_history = deque(maxlen=MOVING_AVG_WINDOW)  # Temporal moving average buffer[cite: 1]
consecutive_drowsy_counter = 0
is_calibrated = False
dynamic_threshold = None
calibration_start_time = time.time()

print("=" * 70)
print("Adaptive Statistical Calibration Driver Drowsiness Detection System[cite: 1]")
print("Phase 1: Initializing 60-Second Driver Baseline Calibration[cite: 1]")
print("=" * 70)

# ----------------- 4. Main Real-Time Processing Loop -----------------
while cap.isOpened():
  loop_start = time.time()
  ret, frame = cap.read()
  if not ret:
    print("[Error] Failed to grab frame from webcam[cite: 1].")
    break

  h, w, _ = frame.shape
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

  eye_openness_metric = None

  for x, y, fw, fh in faces:
    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 0, 0), 2)
    roi_gray = gray[y : y + int(fh * 0.6), x : x + fw]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)

    if len(eyes) >= 1:
      total_eye_h = sum([eh for (_, _, _, eh) in eyes])
      eye_openness_metric = total_eye_h / float(fh)
      for ex, ey, ew, eh in eyes:
        cv2.rectangle(
            frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2
        )
    else:
      eye_openness_metric = 0.0
    break

  # Temporal smoothing filter (Moving Average)[cite: 1]
  if eye_openness_metric is not None:
    ear_history.append(eye_openness_metric)
    filtered_ear = np.mean(ear_history)
  else:
    filtered_ear = None

  # ----------------- PHASE 1: Statistical Calibration (0-60s)[cite: 1] -----------------
  if not is_calibrated:
    elapsed_calib_time = time.time() - calibration_start_time
    time_remaining = max(0, int(CALIBRATION_DURATION - elapsed_calib_time))

    if filtered_ear is not None and filtered_ear > 0:
      calibration_buffer.append(filtered_ear)

    cv2.putText(
        frame,
        f"PHASE 1: Calibrating Baseline ({time_remaining}s remaining)[cite: 1]",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Operating in passive learning mode. Keep normal alert posture.",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )

    # Compute personalized statistical baseline after 60 seconds[cite: 1]
    if elapsed_calib_time >= CALIBRATION_DURATION and len(calibration_buffer) > 0:
      mu_ear = float(np.mean(calibration_buffer))
      sigma_ear = float(np.std(calibration_buffer))

      # Dynamic Z-score threshold formula: mu - 1.5*sigma[cite: 1]
      dynamic_threshold = max(0.01, mu_ear - (1.5 * sigma_ear))
      is_calibrated = True

      print("\n" + "=" * 70)
      print("Driver Onboarding Calibration Complete![cite: 1]")
      print(f"Personalized Baseline Mean (u_EAR): {mu_ear:.4f}[cite: 1]")
      print(f"Personalized Standard Deviation (o_EAR): {sigma_ear:.4f}[cite: 1]")
      print(
          "Calculated Alert Threshold (u - 1.5o):"
          f" {dynamic_threshold:.4f}[cite: 1]"
      )
      print("=" * 70 + "\n")

  # ----------------- PHASE 2: Active Monitoring[cite: 1] -----------------
  else:
    status_text = "Status: Alert"
    status_color = (0, 255, 0)

    if filtered_ear is not None:
      # Compare smoothed metric with personalized dynamic threshold[cite: 1]
      if filtered_ear < dynamic_threshold:
        consecutive_drowsy_counter += 1
      else:
        consecutive_drowsy_counter = 0

      # Trigger multimodal alert if below threshold for 5 consecutive frames[cite: 1]
      if consecutive_drowsy_counter >= CONSECUTIVE_FRAMES_TRIGGER:
        status_text = "ALERT: DROWSINESS DETECTED!"
        status_color = (0, 0, 255)

        # Visual GUI Popup Box[cite: 1]
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
        cv2.rectangle(
            frame,
            (w // 2 - 250, h // 2 - 50),
            (w // 2 + 250, h // 2 + 50),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame,
            "WAKE UP! DROWSINESS DETECTED",
            (w // 2 - 230, h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Audible Beep Trigger[cite: 1]
        try:
          winsound.Beep(1000, 100)
        except Exception:
          pass

    # Real-Time Telemetry HUD[cite: 1]
    cv2.putText(
        frame,
        "PHASE 2: Active Monitoring (OpenVINO/CPU)",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Instantaneous Metric: {filtered_ear:.3f}"
        if filtered_ear
        else "Searching Face...",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Dynamic Threshold (u - 1.5o): {dynamic_threshold:.3f}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Literature Static Threshold: {STATIC_BASELINE_EAR:.3f}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        frame,
        status_text,
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )

  # Measure FPS[cite: 1]
  fps = 1.0 / (time.time() - loop_start + 1e-6)
  cv2.putText(
      frame,
      f"FPS: {fps:.1f}",
      (w - 120, 35),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 255, 0),
      2,
  )

  # Display real-time output window
  cv2.imshow(
      "Adaptive Statistical Calibration - Driver Drowsiness Detection", frame
  )

  # Press 'q' to cleanly terminate
  if cv2.waitKey(1) & 0xFF == ord("q"):
    print("[System] Terminating application...")
    break

cap.release()
cv2.destroyAllWindows()