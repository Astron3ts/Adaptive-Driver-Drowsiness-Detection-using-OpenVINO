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


FACE_MODEL_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/face-detection-retail-0004/FP16/"
LANDMARK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# 3D facial model for Perspective-n-Point pose estimation
HEAD_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Right eye outer corner
    (225.0, 170.0, -135.0),    # Left eye outer corner
    (-150.0, -150.0, -125.0),  # Right mouth corner
    (150.0, -150.0, -125.0)    # Left mouth corner
], dtype=np.float64)

POSE_INDICES = (1, 152, 33, 263, 61, 291)
LEFT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
MOUTH_TOP_INDICES = (81, 13, 311)
MOUTH_BOT_INDICES = (178, 14, 402)
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291

_last_beep = 0.0


def play_alert(freq=1000, duration_ms=300):
    global _last_beep
    now = time.time()
    if now - _last_beep >= 0.4:
        _last_beep = now
        try:
            threading.Thread(target=winsound.Beep, args=(freq, duration_ms), daemon=True).start()
        except Exception:
            pass


def draw_hud_text(img, text, pos, scale=0.55, color=(0, 255, 255), thickness=1):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def calculate_ear(points):
    v1 = math.hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
    v2 = math.hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])
    h = math.hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def calculate_mar(top_pts, bot_pts, left_pt, right_pt):
    v1 = math.hypot(top_pts[0][0] - bot_pts[0][0], top_pts[0][1] - bot_pts[0][1])
    v2 = math.hypot(top_pts[1][0] - bot_pts[1][0], top_pts[1][1] - bot_pts[1][1])
    v3 = math.hypot(top_pts[2][0] - bot_pts[2][0], top_pts[2][1] - bot_pts[2][1])
    h = math.hypot(left_pt[0] - right_pt[0], left_pt[1] - right_pt[1])
    return (v1 + v2 + v3) / (2.0 * h) if h > 0 else 0.0


def extract_eye_crop(points, gray_frame, target_w=34, target_h=26, padding=5):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = max(0, int(min(xs) - padding)), min(gray_frame.shape[1], int(max(xs) + padding))
    y1, y2 = max(0, int(min(ys) - padding)), min(gray_frame.shape[0], int(max(ys) + padding))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = gray_frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_w, target_h))
    return (resized.astype(np.float32) / 255.0)[None, None, ...]


def estimate_head_pose(landmarks_2d, camera_matrix, dist_coeffs):
    try:
        success, rvec, _ = cv2.solvePnP(
            HEAD_MODEL_3D, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(angles[0]), float(angles[1]), float(angles[2])
    except Exception:
        return 0.0, 0.0, 0.0


def main():
    if Core is None:
        raise RuntimeError("OpenVINO is not installed. Please run: pip install openvino")

    core = Core()
    devices = core.available_devices
    gpu_device = next((d for d in devices if "GPU" in d), None)
    current_device = "AUTO" if "AUTO" in devices else "CPU"

    # Ensure model assets are available
    if not os.path.exists("face-detection-retail-0004.xml"):
        urllib.request.urlretrieve(FACE_MODEL_URL + "face-detection-retail-0004.xml", "face-detection-retail-0004.xml")
    if not os.path.exists("face-detection-retail-0004.bin"):
        urllib.request.urlretrieve(FACE_MODEL_URL + "face-detection-retail-0004.bin", "face-detection-retail-0004.bin")
    if not os.path.exists("face_landmarker.task"):
        urllib.request.urlretrieve(LANDMARK_MODEL_URL, "face_landmarker.task")

    face_model = core.read_model("face-detection-retail-0004.xml", "face-detection-retail-0004.bin")
    compiled_face = core.compile_model(face_model, current_device)
    face_out_layer = compiled_face.output(0)
    _, _, net_h, net_w = compiled_face.input(0).shape

    ocunet_xml = "ocunet.xml"
    ocunet_bin = "ocunet.bin"
    compiled_ocunet, ocunet_out_layer, ocunet_model = None, None, None

    if os.path.exists(ocunet_xml) and os.path.exists(ocunet_bin):
        ocunet_model = core.read_model(ocunet_xml, ocunet_bin)
        compiled_ocunet = core.compile_model(ocunet_model, current_device)
        ocunet_out_layer = compiled_ocunet.output(0)

    device_cache = {
        current_device: {
            "face": compiled_face,
            "face_out": face_out_layer,
            "ocunet": compiled_ocunet,
            "ocunet_out": ocunet_out_layer
        }
    }

    # MediaPipe Face Mesh initialization
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        raise RuntimeError("Failed to read video frame from camera.")

    h, w = first_frame.shape[:2]
    camera_matrix = np.array([[float(w), 0.0, w / 2.0], [0.0, float(w), h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    pose_2d_points = np.empty((6, 2), dtype=np.float64)

    # Operational thresholds and state buffers
    calibration_duration = 60
    drowsy_trigger_frames = 35
    yawn_threshold_mar = 0.58
    yawn_consec_frames = 28
    head_nod_pitch_threshold = -18.0
    head_nod_consec_frames = 22
    perclos_alarm_threshold = 15.0

    calib_ear_buffer = []
    calib_mar_buffer = []
    ear_history = deque(maxlen=5)
    mar_history = deque(maxlen=5)
    perclos_buffer = deque(maxlen=1800)

    drowsy_counter = 0
    yawn_counter = 0
    nod_counter = 0
    total_yawns = 0
    is_calibrated = False
    dyn_ear_threshold = 0.20
    dyn_mar_threshold = yawn_threshold_mar
    calib_start_time = time.time()

    peak_fps = 0.0
    frames_since_switch = 0
    last_switch_ms = 0.0
    window_name = "Adaptive Multi-Cue Driver Fatigue Detection (OpenVINO)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"[System] Pipeline running on {current_device}. Keys: [q/Esc]: Quit | [1]: CPU | [2]: GPU | [3]: AUTO")

    try:
        while cap.isOpened():
            t_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # OpenVINO Face Detection
            t_infer = time.perf_counter()
            resized = cv2.resize(frame, (net_w, net_h))
            input_tensor = resized.transpose(2, 0, 1)[None, ...]
            detections = compiled_face([input_tensor])[face_out_layer][0][0]
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            for det in detections:
                if det[2] > 0.6:
                    xmin, ymin = max(0, int(det[3] * w)), max(0, int(det[4] * h))
                    xmax, ymax = min(w, int(det[5] * w)), min(h, int(det[6] * h))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 120, 0), 1)
                    break

            # MediaPipe Facial Mesh
            mp_res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
            filtered_ear = None
            filtered_mar = None
            pitch_deg, yaw_deg, roll_deg = None, None, None
            ocunet_prob_closed = 0.0

            if mp_res.face_landmarks:
                lm = mp_res.face_landmarks[0]
                left_eye = [(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE_INDICES]
                right_eye = [(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE_INDICES]
                ear_val = (calculate_ear(left_eye) + calculate_ear(right_eye)) * 0.5
                ear_history.append(ear_val)
                filtered_ear = float(np.mean(ear_history))

                if compiled_ocunet is not None:
                    left_crop = extract_eye_crop(left_eye, gray_frame)
                    right_crop = extract_eye_crop(right_eye, gray_frame)
                    probs = []
                    for crop in (left_crop, right_crop):
                        if crop is not None:
                            logits = compiled_ocunet([crop])[ocunet_out_layer][0]
                            max_l = max(logits[0], logits[1])
                            e0 = math.exp(logits[0] - max_l)
                            e1 = math.exp(logits[1] - max_l)
                            probs.append(e1 / (e0 + e1))
                    if probs:
                        ocunet_prob_closed = float(np.mean(probs))

                top_m = [(lm[i].x * w, lm[i].y * h) for i in MOUTH_TOP_INDICES]
                bot_m = [(lm[i].x * w, lm[i].y * h) for i in MOUTH_BOT_INDICES]
                c_left = (lm[MOUTH_CORNER_LEFT].x * w, lm[MOUTH_CORNER_LEFT].y * h)
                c_right = (lm[MOUTH_CORNER_RIGHT].x * w, lm[MOUTH_CORNER_RIGHT].y * h)
                mar_history.append(calculate_mar(top_m, bot_m, c_left, c_right))
                filtered_mar = float(np.mean(mar_history))

                for i, p_idx in enumerate(POSE_INDICES):
                    pose_2d_points[i, 0] = lm[p_idx].x * w
                    pose_2d_points[i, 1] = lm[p_idx].y * h
                pitch_deg, yaw_deg, roll_deg = estimate_head_pose(pose_2d_points, camera_matrix, dist_coeffs)

                # Draw landmark anchors
                for pt in left_eye + right_eye:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
                for pt in top_m + bot_m:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 200, 255), -1)

            # Phase 1: Onboarding Baseline Calibration
            if not is_calibrated:
                status_text = "CALIBRATING"
                status_color = (0, 255, 255)
                perclos_val = 0.0

                elapsed = time.time() - calib_start_time
                remaining = max(0, int(calibration_duration - elapsed))

                if filtered_ear is not None:
                    calib_ear_buffer.append(filtered_ear)
                if filtered_mar is not None:
                    calib_mar_buffer.append(filtered_mar)

                draw_hud_text(frame, f"PHASE 1: Baseline Calibration ({remaining}s remaining)", (15, 125), 0.6, (0, 255, 255), 2)
                draw_hud_text(frame, "Establishing personalized EAR & MAR baseline. Stay alert.", (15, 150), 0.5, (255, 255, 255), 1)

                if elapsed >= calibration_duration and calib_ear_buffer:
                    q10 = float(np.percentile(calib_ear_buffer, 10))
                    clean_ears = [v for v in calib_ear_buffer if v >= q10]
                    mu = float(np.mean(clean_ears))
                    sigma = float(np.std(clean_ears))
                    dyn_ear_threshold = float(np.clip(max(mu - 2.2 * sigma, 0.72 * mu), 0.18, 0.24))

                    if calib_mar_buffer:
                        mu_m = float(np.mean(calib_mar_buffer))
                        sigma_m = float(np.std(calib_mar_buffer))
                        dyn_mar_threshold = max(0.55, mu_m + 3.0 * sigma_m)

                    is_calibrated = True

            # Phase 2: Active Multi-Cue Monitoring
            else:
                status_text = "Status: Alert"
                status_color = (0, 255, 0)
                perclos_val = 0.0

                if filtered_ear is not None:
                    closed = 1 if (filtered_ear < dyn_ear_threshold or ocunet_prob_closed > 0.75) else 0
                    perclos_buffer.append(closed)
                    drowsy_counter = drowsy_counter + 1 if closed else 0

                    if len(perclos_buffer) >= 300:
                        perclos_val = (sum(perclos_buffer) / len(perclos_buffer)) * 100.0

                if filtered_mar is not None and filtered_mar >= dyn_mar_threshold:
                    yawn_counter += 1
                else:
                    if yawn_counter >= yawn_consec_frames:
                        total_yawns += 1
                    yawn_counter = 0

                if pitch_deg is not None and pitch_deg <= head_nod_pitch_threshold:
                    nod_counter += 1
                else:
                    nod_counter = 0

                # Alarm state machine
                if drowsy_counter >= drowsy_trigger_frames:
                    status_text = "ALERT: MICROSLEEP DETECTED!"
                    status_color = (0, 0, 255)
                    play_alert(1000, 300)
                elif nod_counter >= head_nod_consec_frames:
                    status_text = "ALERT: HEAD NODDING / DROOP!"
                    status_color = (0, 69, 255)
                    play_alert(850, 300)
                elif perclos_val >= perclos_alarm_threshold:
                    status_text = "ALERT: HIGH PERCLOS FATIGUE"
                    status_color = (0, 165, 255)
                    play_alert(950, 300)
                elif yawn_counter >= yawn_consec_frames:
                    status_text = "WARNING: YAWNING DETECTED"
                    status_color = (0, 215, 255)

                if "ALERT" in status_text:
                    cv2.rectangle(frame, (0, 0), (w, h), status_color, 8)
                    cv2.rectangle(frame, (w // 2 - 250, h // 2 - 45), (w // 2 + 250, h // 2 + 45), status_color, -1)
                    draw_hud_text(frame, "DRIVER FATIGUE!", (w // 2 - 140, h // 2 + 10), 0.9, (255, 255, 255), 2)

                draw_hud_text(frame, "PHASE 2: Multi-Cue Monitoring", (15, 120), 0.6, (255, 255, 255), 2)
                ear_s = f"{filtered_ear:.3f}" if filtered_ear is not None else "N/A"
                mar_s = f"{filtered_mar:.3f}" if filtered_mar is not None else "N/A"
                pitch_s = f"{pitch_deg:.1f} deg" if pitch_deg is not None else "N/A"
                ocu_s = f"{ocunet_prob_closed * 100.0:.0f}%" if compiled_ocunet is not None else "N/A"

                draw_hud_text(frame, f"EAR: {ear_s} (Thr: {dyn_ear_threshold:.3f}) | OcuNet Closed: {ocu_s}", (15, 145), 0.52, (0, 255, 255), 1)
                draw_hud_text(frame, f"Head Pitch: {pitch_s} | Yawns: {total_yawns} | MAR: {mar_s}", (15, 170), 0.52, (255, 200, 100), 1)
                draw_hud_text(frame, f"PERCLOS: {perclos_val:.1f}%", (15, 195), 0.52, (255, 200, 0), 1)
                draw_hud_text(frame, status_text, (15, 225), 0.68, status_color, 2)

            # System telemetry panel
            cv2.rectangle(frame, (10, 10), (490, 95), (25, 25, 25), -1)
            cv2.rectangle(frame, (10, 10), (490, 95), (180, 180, 180), 1)
            draw_hud_text(frame, f"OpenVINO Target: {current_device} | Infer: {infer_ms:.1f} ms", (20, 32), 0.55, (0, 255, 0), 2)
            gpu_str = "[2]: GPU" if gpu_device else "[2]: N/A"
            draw_hud_text(frame, f"Hotkeys: [1]: CPU  {gpu_str}  [3]: AUTO", (20, 56), 0.48, (255, 255, 255), 1)
            draw_hud_text(frame, "(Click video window to focus keys)", (20, 78), 0.40, (180, 180, 180), 1)

            fps = 1.0 / (time.time() - t_start + 1e-6)
            frames_since_switch += 1
            if frames_since_switch > 5 and fps > peak_fps:
                peak_fps = fps

            draw_hud_text(frame, f"FPS: {fps:.1f}", (w - 150, 32), 0.65, (0, 255, 0), 2)
            draw_hud_text(frame, f"PEAK: {peak_fps:.1f}", (w - 150, 60), 0.55, (0, 200, 255), 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            # Dynamic OpenVINO Hardware Switching
            if key in (ord("1"), ord("2"), ord("3")):
                target = None
                if key == ord("1"):
                    target = "CPU"
                elif key == ord("2"):
                    target = gpu_device if gpu_device else "GPU"
                elif key == ord("3"):
                    target = "AUTO"

                if target == current_device:
                    print(f"[System] Already running on {current_device}.")
                else:
                    print(f"[System] Switching to {target}...")
                    t_sw = time.time()
                    try:
                        if target not in device_cache:
                            print(f"[System] Compiling models for {target} (first time may take a few seconds)...")
                            c_face = core.compile_model(face_model, target)
                            c_ocu = core.compile_model(ocunet_model, target) if ocunet_model is not None else None
                            device_cache[target] = {
                                "face": c_face,
                                "face_out": c_face.output(0),
                                "ocunet": c_ocu,
                                "ocunet_out": c_ocu.output(0) if c_ocu is not None else None
                            }

                        current_device = target
                        compiled_face = device_cache[target]["face"]
                        face_out_layer = device_cache[target]["face_out"]
                        compiled_ocunet = device_cache[target]["ocunet"]
                        ocunet_out_layer = device_cache[target]["ocunet_out"]
                        last_switch_ms = (time.time() - t_sw) * 1000.0
                        peak_fps = 0.0
                        frames_since_switch = 0
                        print(f"[System] Switched to {current_device} in {last_switch_ms:.1f}ms")
                    except Exception as e:
                        print(f"[System] Failed to switch to {target}: {e}")

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        try:
            landmarker.close()
        except Exception:
            pass
        print("[System] Exited cleanly.")


if __name__ == "__main__":
    main()
