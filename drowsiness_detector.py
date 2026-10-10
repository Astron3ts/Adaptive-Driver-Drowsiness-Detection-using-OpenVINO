# pyright: reportMissingImports=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalOperand=false
# pyright: reportAttributeAccessIssue=false

import os
import time
import math
import threading
import urllib.request
import winsound
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode as VisionRunningMode

try:
    from openvino import Core
except ImportError:
    try:
        from openvino.runtime import Core
    except ImportError:
        Core = None

def draw_text(img, text, pos, scale=0.55, color=(0, 255, 255), thickness=1):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

_last_beep_time = 0.0

def beep_async(freq=1000, duration_ms=300):
    global _last_beep_time
    now = time.time()
    if now - _last_beep_time >= 0.4:
        _last_beep_time = now
        try:
            threading.Thread(target=winsound.Beep, args=(freq, duration_ms), daemon=True).start()
        except Exception:
            pass

def calculate_ear(pts):
    v1 = math.hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1])
    v2 = math.hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1])
    h  = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

def calculate_mar(top, bot, left, right):
    v1 = math.hypot(top[0][0] - bot[0][0], top[0][1] - bot[0][1])
    v2 = math.hypot(top[1][0] - bot[1][0], top[1][1] - bot[1][1])
    v3 = math.hypot(top[2][0] - bot[2][0], top[2][1] - bot[2][1])
    h  = math.hypot(left[0] - right[0], left[1] - right[1])
    return (v1 + v2 + v3) / (2.0 * h) if h > 0 else 0.0

def extract_eye_crop(pts, gray_img, target_w=34, target_h=26, pad=5):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = max(0, int(min(xs) - pad)), min(gray_img.shape[1], int(max(xs) + pad))
    y1, y2 = max(0, int(min(ys) - pad)), min(gray_img.shape[0], int(max(ys) + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = gray_img[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_w, target_h))
    return (resized.astype(np.float32) / 255.0)[None, None, ...]

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Right eye outer
    (225.0, 170.0, -135.0),    # Left eye outer
    (-150.0, -150.0, -125.0),  # Right mouth corner
    (150.0, -150.0, -125.0)    # Left mouth corner
], dtype=np.float64)

POSE_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)
LEFT_EYE_INDICES      = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_INDICES     = (33, 160, 158, 133, 153, 144)
MOUTH_TOP_INDICES     = (81, 13, 311)
MOUTH_BOT_INDICES     = (178, 14, 402)
MOUTH_CORNER_LEFT     = 61
MOUTH_CORNER_RIGHT    = 291

def estimate_head_pose(landmarks_2d, camera_matrix, dist_coeffs):
    try:
        success, rvec, _ = cv2.solvePnP(
            MODEL_POINTS_3D, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(angles[0]), float(angles[1]), float(angles[2])
    except Exception:
        return 0.0, 0.0, 0.0

CALIBRATION_DURATION       = 60
CONSECUTIVE_FRAMES_TRIGGER = 35
MOVING_AVG_WINDOW          = 5
PERCLOS_WINDOW_FRAMES      = 1800
PERCLOS_ALARM_THRESHOLD    = 15.0
PERCLOS_MIN_BUFFER         = 300
YAWN_MAR_THRESHOLD         = 0.58
YAWN_CONSEC_FRAMES         = 28
HEAD_NOD_PITCH_THRESHOLD   = -18.0
HEAD_NOD_CONSEC_FRAMES     = 22

FACE_LANDMARKER_MODEL = "face_landmarker.task"
FACE_LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
if not os.path.exists(FACE_LANDMARKER_MODEL):
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)

face_landmarker = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
)

if Core is None:
    raise RuntimeError("OpenVINO is not installed. Run: pip install openvino")

core = Core()
available_devices = core.available_devices
gpu_device = next((d for d in available_devices if "GPU" in d), None)
current_device = "AUTO" if "AUTO" in available_devices else "CPU"
last_switch_time_ms = 0.0

base_url  = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/face-detection-retail-0004/FP16/"
model_xml = "face-detection-retail-0004.xml"
model_bin = "face-detection-retail-0004.bin"

if not os.path.exists(model_xml): urllib.request.urlretrieve(base_url + model_xml, model_xml)
if not os.path.exists(model_bin): urllib.request.urlretrieve(base_url + model_bin, model_bin)

face_model     = core.read_model(model=model_xml, weights=model_bin)
compiled_face  = core.compile_model(model=face_model, device_name=current_device)
face_out_layer = compiled_face.output(0)
_, _, net_h, net_w = compiled_face.input(0).shape

ocunet_xml = "ocunet.xml"
ocunet_bin = "ocunet.bin"
compiled_ocunet = None
ocunet_out_layer = None

if os.path.exists(ocunet_xml) and os.path.exists(ocunet_bin):
    ocunet_model = core.read_model(model=ocunet_xml, weights=ocunet_bin)
    compiled_ocunet = core.compile_model(model=ocunet_model, device_name=current_device)
    ocunet_out_layer = compiled_ocunet.output(0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera at index 0.")

cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, first_frame = cap.read()
if not ret or first_frame is None:
    raise RuntimeError("Failed to read initial frame from camera.")

h, w = first_frame.shape[:2]
camera_matrix = np.array([
    [float(w), 0.0, w / 2.0],
    [0.0, float(w), h / 2.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1), dtype=np.float64)
pose_2d_points = np.empty((6, 2), dtype=np.float64)

calibration_buffer_ear = []
calibration_buffer_mar = []
ear_history            = deque(maxlen=MOVING_AVG_WINDOW)
mar_history            = deque(maxlen=MOVING_AVG_WINDOW)
perclos_buffer         = deque(maxlen=PERCLOS_WINDOW_FRAMES)

consecutive_drowsy_counter = 0
consecutive_yawn_counter   = 0
consecutive_nod_counter    = 0
yawn_total_count           = 0
is_calibrated              = False
dynamic_threshold_ear      = 0.0
dynamic_threshold_mar      = YAWN_MAR_THRESHOLD
calibration_start_time     = time.time()

max_fps = 0.0
frames_since_switch = 0
consecutive_failures = 0

WINDOW_NAME = "Adaptive Multi-Cue Driver Fatigue Detection (OpenVINO)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

print("[System] Detection Pipeline Started. Hotkeys: [q/Esc]: Quit | [1]: CPU | [2]: GPU | [3]: AUTO")

try:
    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print("[System] Video feed lost.")
                break
            time.sleep(0.01)
            continue
        consecutive_failures = 0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t_ov = time.perf_counter()
        resized = cv2.resize(frame, (net_w, net_h))
        input_tensor = resized.transpose(2, 0, 1)[None, ...]
        detections = compiled_face([input_tensor])[face_out_layer][0][0]
        openvino_ms = (time.perf_counter() - t_ov) * 1000.0

        for det in detections:
            if det[2] > 0.6:
                xmin, ymin = max(0, int(det[3] * w)), max(0, int(det[4] * h))
                xmax, ymax = min(w, int(det[5] * w)), min(h, int(det[6] * h))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 120, 0), 1)
                break

        mp_results = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))

        filtered_ear = None
        filtered_mar = None
        pitch_deg, yaw_deg, roll_deg = None, None, None
        ocunet_p_closed = 0.0

        if mp_results.face_landmarks:
            lm = mp_results.face_landmarks[0]

            left_eye_pts  = [(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE_INDICES]
            right_eye_pts = [(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE_INDICES]
            ear_history.append((calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) * 0.5)
            filtered_ear = float(np.mean(ear_history))

            if compiled_ocunet is not None:
                eye_crop = extract_eye_crop(left_eye_pts, gray_frame)
                if eye_crop is not None:
                    logits = compiled_ocunet([eye_crop])[ocunet_out_layer][0]
                    e0, e1 = math.exp(logits[0]), math.exp(logits[1])
                    ocunet_p_closed = e0 / (e0 + e1)

            top_m = [(lm[i].x * w, lm[i].y * h) for i in MOUTH_TOP_INDICES]
            bot_m = [(lm[i].x * w, lm[i].y * h) for i in MOUTH_BOT_INDICES]
            c_l   = (lm[MOUTH_CORNER_LEFT].x * w, lm[MOUTH_CORNER_LEFT].y * h)
            c_r   = (lm[MOUTH_CORNER_RIGHT].x * w, lm[MOUTH_CORNER_RIGHT].y * h)
            mar_history.append(calculate_mar(top_m, bot_m, c_l, c_r))
            filtered_mar = float(np.mean(mar_history))

            for idx, p_idx in enumerate(POSE_LANDMARK_INDICES):
                pose_2d_points[idx, 0] = lm[p_idx].x * w
                pose_2d_points[idx, 1] = lm[p_idx].y * h
            pitch_deg, yaw_deg, roll_deg = estimate_head_pose(pose_2d_points, camera_matrix, dist_coeffs)

            for pt in left_eye_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
            for pt in right_eye_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
            for pt in top_m:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 200, 255), -1)
            for pt in bot_m:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 200, 255), -1)

        if not is_calibrated:
            status_text = "CALIBRATING"
            status_color = (0, 255, 255)
            perclos_score = 0.0

            elapsed = time.time() - calibration_start_time
            time_remaining = max(0, int(CALIBRATION_DURATION - elapsed))

            if filtered_ear is not None:
                calibration_buffer_ear.append(filtered_ear)
            if filtered_mar is not None:
                calibration_buffer_mar.append(filtered_mar)

            draw_text(frame, f"PHASE 1: Baseline Calibration ({time_remaining}s remaining)", (15, 125), 0.6, (0, 255, 255), 2)
            draw_text(frame, "Establishing personalized EAR & MAR baseline. Stay alert.", (15, 150), 0.5, (255, 255, 255), 1)

            if elapsed >= CALIBRATION_DURATION and len(calibration_buffer_ear) > 0:
                q10 = float(np.percentile(calibration_buffer_ear, 10))
                clean_ear = [v for v in calibration_buffer_ear if v >= q10]
                mu_ear = float(np.mean(clean_ear))
                sigma_ear = float(np.std(clean_ear))
                dynamic_threshold_ear = float(np.clip(max(mu_ear - 2.2 * sigma_ear, 0.72 * mu_ear), 0.18, 0.24))

                if len(calibration_buffer_mar) > 0:
                    mu_mar = float(np.mean(calibration_buffer_mar))
                    sigma_mar = float(np.std(calibration_buffer_mar))
                    dynamic_threshold_mar = max(0.55, mu_mar + 3.0 * sigma_mar)

                is_calibrated = True
        else:
            status_text = "Status: Alert"
            status_color = (0, 255, 0)
            perclos_score = 0.0

            if filtered_ear is not None:
                is_closed = 1 if (filtered_ear < dynamic_threshold_ear or ocunet_p_closed > 0.75) else 0
                perclos_buffer.append(is_closed)
                consecutive_drowsy_counter = consecutive_drowsy_counter + 1 if is_closed else 0

                if len(perclos_buffer) >= PERCLOS_MIN_BUFFER:
                    perclos_score = (sum(perclos_buffer) / len(perclos_buffer)) * 100.0

            if filtered_mar is not None and filtered_mar >= dynamic_threshold_mar:
                consecutive_yawn_counter += 1
            else:
                if consecutive_yawn_counter >= YAWN_CONSEC_FRAMES:
                    yawn_total_count += 1
                consecutive_yawn_counter = 0

            if pitch_deg is not None and pitch_deg <= HEAD_NOD_PITCH_THRESHOLD:
                consecutive_nod_counter += 1
            else:
                consecutive_nod_counter = 0

            if consecutive_drowsy_counter >= CONSECUTIVE_FRAMES_TRIGGER:
                status_text  = "ALERT: MICROSLEEP DETECTED!"
                status_color = (0, 0, 255)
                beep_async(1000, 300)
            elif consecutive_nod_counter >= HEAD_NOD_CONSEC_FRAMES:
                status_text  = "ALERT: HEAD NODDING / DROOP!"
                status_color = (0, 69, 255)
                beep_async(850, 300)
            elif perclos_score >= PERCLOS_ALARM_THRESHOLD:
                status_text  = "ALERT: HIGH PERCLOS FATIGUE"
                status_color = (0, 165, 255)
                beep_async(950, 300)
            elif consecutive_yawn_counter >= YAWN_CONSEC_FRAMES:
                status_text  = "WARNING: YAWNING DETECTED"
                status_color = (0, 215, 255)

            if "ALERT" in status_text:
                cv2.rectangle(frame, (0, 0), (w, h), status_color, 8)
                cv2.rectangle(frame, (w // 2 - 250, h // 2 - 45), (w // 2 + 250, h // 2 + 45), status_color, -1)
                draw_text(frame, "DRIVER FATIGUE!", (w // 2 - 140, h // 2 + 10), 0.9, (255, 255, 255), 2)

            draw_text(frame, "PHASE 2: Multi-Cue Monitoring", (15, 120), 0.6, (255, 255, 255), 2)
            ear_txt = f"{filtered_ear:.3f}" if filtered_ear is not None else "N/A"
            mar_txt = f"{filtered_mar:.3f}" if filtered_mar is not None else "N/A"
            pitch_txt = f"{pitch_deg:.1f} deg" if pitch_deg is not None else "N/A"
            ocu_txt = f"{ocunet_p_closed * 100.0:.0f}%" if compiled_ocunet is not None else "N/A"
            draw_text(frame, f"EAR: {ear_txt} (Thr: {dynamic_threshold_ear:.3f}) | OcuNet Closed: {ocu_txt}", (15, 145), 0.52, (0, 255, 255), 1)
            draw_text(frame, f"Head Pitch: {pitch_txt} | Yawns: {yawn_total_count} | MAR: {mar_txt}", (15, 170), 0.52, (255, 200, 100), 1)
            draw_text(frame, f"PERCLOS: {perclos_score:.1f}%", (15, 195), 0.52, (255, 200, 0), 1)
            draw_text(frame, status_text, (15, 225), 0.68, status_color, 2)

        cv2.rectangle(frame, (10, 10), (470, 95), (25, 25, 25), -1)
        cv2.rectangle(frame, (10, 10), (470, 95), (180, 180, 180), 1)
        draw_text(frame, f"OpenVINO Target: {current_device} | Infer: {openvino_ms:.1f} ms", (20, 32), 0.55, (0, 255, 0), 2)
        gpu_label = "[2]: GPU" if gpu_device else "[2]: N/A"
        draw_text(frame, f"Keys -> [1]: CPU  {gpu_label}  [3]: AUTO", (20, 56), 0.48, (255, 255, 255), 1)

        fps = 1.0 / (time.time() - loop_start + 1e-6)
        frames_since_switch += 1
        if frames_since_switch > 5 and fps > max_fps:
            max_fps = fps

        draw_text(frame, f"FPS: {fps:.1f}",      (w - 150, 32), 0.65, (0, 255, 0),    2)
        draw_text(frame, f"PEAK: {max_fps:.1f}", (w - 150, 60), 0.55, (0, 200, 255),  1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key in (ord("1"), ord("2"), ord("3")):
            target = None
            if key == ord("1") and current_device != "CPU":
                target = "CPU"
            elif key == ord("2") and gpu_device and current_device != gpu_device:
                target = gpu_device
            elif key == ord("3") and current_device != "AUTO":
                target = "AUTO"

            if target:
                t_switch = time.time()
                current_device  = target
                compiled_face   = core.compile_model(model=face_model, device_name=current_device)
                face_out_layer  = compiled_face.output(0)
                if compiled_ocunet is not None:
                    compiled_ocunet  = core.compile_model(model=ocunet_model, device_name=current_device)
                    ocunet_out_layer = compiled_ocunet.output(0)
                last_switch_time_ms = (time.time() - t_switch) * 1000.0
                max_fps = 0.0
                frames_since_switch = 0
                print(f"[System] Switched to {current_device} in {last_switch_time_ms:.1f}ms")

except KeyboardInterrupt:
    print("\n[System] Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        face_landmarker.close()
    except Exception:
        pass
    print("[System] Exited cleanly.")