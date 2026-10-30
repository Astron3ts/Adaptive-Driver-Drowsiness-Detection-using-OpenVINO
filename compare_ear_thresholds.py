"""Compare fixed and calibrated EAR using labeled frame-level eye data.

Input CSV columns: subject_id,video_id,timestamp_seconds,ear,eye_closed
eye_closed must be 1 for closed and 0 for open. Each video must include the
initial calibration period and continue beyond it.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def scores(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"samples": len(labels), "accuracy": (tp + tn) / len(labels) if len(labels) else 0.0,
            "precision_closed": precision, "recall_closed": recall,
            "f1_closed": f1, "specificity_open": specificity,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def dynamic_threshold(calibration_ears):
    values = np.asarray(calibration_ears, dtype=np.float64)
    q10 = float(np.percentile(values, 10))
    cleaned = values[values >= q10]
    mean = float(np.mean(cleaned))
    sigma = float(np.std(cleaned))
    return float(np.clip(max(mean - 2.2 * sigma, 0.72 * mean), 0.18, 0.24))


def event_runs(values):
    runs = []
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def read_groups(csv_path):
    grouped = defaultdict(list)
    required = {"subject_id", "video_id", "timestamp_seconds", "ear", "eye_closed"}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            key = (row["subject_id"], row["video_id"])
            grouped[key].append({"time": float(row["timestamp_seconds"]),
                                 "ear": float(row["ear"]),
                                 "closed": int(row["eye_closed"])})
    for rows in grouped.values():
        rows.sort(key=lambda item: item["time"])
        if any(row["closed"] not in (0, 1) for row in rows):
            raise ValueError("eye_closed values must be 0 or 1")
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument("--static-threshold", type=float, default=0.20)
    parser.add_argument("--calibration-seconds", type=float, default=60.0)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--alarm-frames", type=int, default=35)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.smoothing_window < 1 or args.alarm_frames < 1:
        parser.error("smoothing-window and alarm-frames must be positive")

    videos = read_groups(args.labels_csv)
    all_labels = {"static": [], "dynamic": []}
    all_predictions = {"static": [], "dynamic": []}
    alarm_summary = {"static": {"false_alarms": 0, "missed_closures": 0, "delays_seconds": []},
                     "dynamic": {"false_alarms": 0, "missed_closures": 0, "delays_seconds": []}}
    scored_minutes = 0.0
    thresholds = {}

    for (subject, video), rows in sorted(videos.items()):
        calibration = [r["ear"] for r in rows if r["time"] < args.calibration_seconds]
        evaluated = [r for r in rows if r["time"] >= args.calibration_seconds]
        if len(calibration) < 2 or not evaluated:
            raise ValueError(f"{subject}/{video}: need calibration frames and post-calibration frames")
        thresholds[f"{subject}/{video}"] = dynamic_threshold(calibration)

        # Mirror the detector's rolling average. Include pre-calibration EAR
        # values in the rolling history, then score only after calibration.
        smoothed = np.convolve(np.asarray([r["ear"] for r in rows], dtype=np.float64),
                               np.ones(args.smoothing_window) / args.smoothing_window, mode="full")
        smoothed = smoothed[:len(rows)]
        scored_indices = [i for i, row in enumerate(rows) if row["time"] >= args.calibration_seconds]
        labels = [rows[i]["closed"] for i in scored_indices]
        static_pred = [int(smoothed[i] < args.static_threshold) for i in scored_indices]
        dynamic_pred = [int(smoothed[i] < thresholds[f"{subject}/{video}"]) for i in scored_indices]
        times = [rows[i]["time"] for i in scored_indices]
        scored_minutes += max(0.0, (times[-1] - times[0]) / 60.0) if len(times) > 1 else 0.0

        for name, predictions in (("static", static_pred), ("dynamic", dynamic_pred)):
            all_labels[name].extend(labels)
            all_predictions[name].extend(predictions)
            predicted_alarms = [(a, b) for a, b in event_runs(predictions)
                                if b - a + 1 >= args.alarm_frames]
            true_events = event_runs(labels)
            matched_true = set()
            for a, b in predicted_alarms:
                match = next((j for j, (ta, tb) in enumerate(true_events)
                              if j not in matched_true and not (b < ta or a > tb)), None)
                if match is None:
                    alarm_summary[name]["false_alarms"] += 1
                else:
                    matched_true.add(match)
                    true_start = times[true_events[match][0]]
                    alarm_time = times[min(a + args.alarm_frames - 1, b)]
                    alarm_summary[name]["delays_seconds"].append(max(0.0, alarm_time - true_start))
            alarm_summary[name]["missed_closures"] += len(true_events) - len(matched_true)

    results = {}
    for name in ("static", "dynamic"):
        result = scores(all_labels[name], all_predictions[name])
        alarms = alarm_summary[name]
        result.update({"false_alarm_events_per_minute": alarms["false_alarms"] / scored_minutes if scored_minutes else 0.0,
                       "closure_event_sensitivity": 1 - alarms["missed_closures"] / max(1, alarms["missed_closures"] + len(alarms["delays_seconds"])),
                       "mean_detected_closure_delay_seconds": float(np.mean(alarms["delays_seconds"])) if alarms["delays_seconds"] else None})
        results[name] = result

    report = {"input_csv": str(args.labels_csv),
              "method": "Same per-video rows, smoothing window, calibration duration, and alarm persistence; only EAR threshold method differs.",
              "static_threshold": args.static_threshold,
              "calibration_seconds": args.calibration_seconds,
              "smoothing_window": args.smoothing_window,
              "alarm_frames": args.alarm_frames,
              "dynamic_thresholds_by_video": thresholds,
              "scored_video_minutes": scored_minutes,
              "results": results,
              "warning": "Eye-closure event metrics are not full driver-drowsiness metrics. Keep every subject's videos in one evaluation split and tune settings only on validation subjects."}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
