# Initial Model and Device Experiments

## Scope

These experiments evaluate eye-state classification on the MRL eye-image dataset in `dataset/archive/data`. They do not evaluate complete driver-drowsiness detection. The source dataset identifies open/closed eye images from 37 people; its project fork labels the directories `awake` and `sleepy`.

## Subject-independent split

The original train, validation, and test folders each contained images from the same 37 subject IDs. To avoid subject leakage, all images were regrouped by subject and split with seed 42:

| Split | Subjects | Images | Awake | Sleepy |
|---|---:|---:|---:|---:|
| Train | 26 | 59,376 | 29,871 | 29,505 |
| Validation | 6 | 13,252 | 6,995 | 6,257 |
| Test | 5 | 12,270 | 6,086 | 6,184 |

No subject appears in more than one split. The subject IDs and full confusion matrices are recorded in the JSON reports.

## Classification results

All models used the same subject split, 25 epochs, batch size 128, AdamW, learning rate 0.001, cosine learning-rate schedule, and seed 42. The reported accuracies below are OpenVINO inference on the same 12,270 held-out eye crops. OcuNet and the two-layer CNN are the more useful comparison; the 1,282-parameter CNN is included as a deliberately small capacity baseline.

| Model | Parameters | CPU accuracy | Intel GPU accuracy | CPU macro-F1 | Intel GPU macro-F1 |
|---|---:|---:|---:|---:|---:|
| OcuNet | 25,666 | 96.95% | 96.95% | 96.95% | 96.95% |
| Two-layer CNN | 5,458 | 88.38% | 88.43% | 88.30% | 88.35% |
| Small CNN | 1,282 | 77.94% | 78.00% | 77.68% | 77.75% |

On this one held-out split, OcuNet exceeds the two-layer baseline by 8.57 percentage points and the small baseline by 19.01 points. These are internal baselines trained under one shared fixed schedule, not reproductions of the papers in the survey. The two-layer model also has fewer parameters and lower CPU latency, so the comparison shows an accuracy/runtime trade-off rather than universal superiority.

The original random-image test split scored 99.10% with the existing OcuNet. Because every subject was also present during training, treat that number as an image-level result only. The subject-independent retrain scored 96.94% in PyTorch and 96.95% after OpenVINO export. This gap is consistent with the need to separate people when estimating generalization.

## Batch-one model latency

Measurements used OpenVINO 2026.4.1 on a 13th Gen Intel Core i5-13420H and Intel UHD Graphics iGPU. Each result is model-only batch-one inference over preprocessed tensors, after 50 warm-up calls and 500 timed calls. It excludes camera capture, face detection, MediaPipe landmarks, eye cropping, and the rest of the application.

| Model | CPU median / p95 | Intel GPU median / p95 |
|---|---:|---:|
| OcuNet | 0.086 / 0.149 ms | 0.818 / 1.189 ms |
| Two-layer CNN | 0.073 / 0.093 ms | 0.663 / 1.203 ms |
| Small CNN | 0.064 / 0.090 ms | 0.571 / 0.941 ms |

For this tiny batch-one workload, Intel GPU inference was slower than CPU inference. The GPU produced essentially the same classification accuracy, but it did not improve OcuNet latency. This is not a full-system FPS measurement and does not establish that the whole application runs faster or slower on either target.

## EAR comparison status

The current dataset consists of isolated eye crops. It has no face frames, video sequences, timestamps, facial landmarks, or frame-level closure annotations. It cannot support the static-versus-dynamic EAR experiment. The live detector implements the dynamic calibration threshold, but there is no labeled video in the project to calculate precision, recall, false alarms, or detection delay for either threshold method.

For that comparison, use the same face videos and frame labels for both methods, keep all video from each person in one split, calibrate dynamic EAR only from the first 60 seconds of each evaluation video, and score only the remaining segment. Hold smoothing, temporal counters, and all other cues constant. Report eye-closure precision/recall/F1 and false alarms per minute; if drowsiness event labels are available, also report event sensitivity and warning delay.

`compare_ear_thresholds.py` is ready for a CSV with columns `subject_id,video_id,timestamp_seconds,ear,eye_closed`. It applies the project's calibration formula and rolling window against a fixed 0.20 EAR threshold. Example: `python compare_ear_thresholds.py labeled_eye_frames.csv --json-out ear_comparison.json`.

The official DROZY page describes videos, timestamps, frame-level facial landmarks, and subject-level sleepiness scores. Its KSS scores are not frame-level eye-closure labels, so additional eye-state annotation or a clearly defined sequence-level protocol is needed. Check its data license before using the download. Another option is to obtain a frame-labeled driver video set through its authors.

## External-paper comparison limit

The table's CNN reference [9] is Hashemi et al. Their paper reports 98.15% for its FD-NN on the ZJU eye-image dataset and 97.54% test accuracy on its separate extended dataset. Those results use different data and protocols, so the percentages cannot be treated as a direct head-to-head comparison with OcuNet on MRL. A direct claim against that published model would require both models to be evaluated on the same subject-disjoint eye images.

## Reproducibility files

- `train_subject_independent.py` creates the subject-disjoint split, trains a selected CNN, evaluates the held-out people, and exports OpenVINO IR.
- `evaluate_ocunet.py` measures classification metrics and batch-one CPU or GPU latency.
- `compare_ear_thresholds.py` compares fixed and calibrated EAR once frame-level labels are available.
- `metrics_ocunet_subject_independent.json`, `metrics_twolayercnn_subject_independent.json`, and `metrics_smallcnn_subject_independent.json` record the PyTorch training and test results.
- `metrics_*_cpu.json` and `metrics_*_intel_gpu.json` record the OpenVINO device measurements.
