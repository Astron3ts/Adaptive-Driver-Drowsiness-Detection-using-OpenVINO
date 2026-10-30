# Adaptive Driver Drowsiness Detection

This is a curated project handoff for GitHub/course sharing. The paper in `docs/update_reports` is an interim update report, not a final IEEE submission. The raw eye-image dataset is not included; see `docs/dataset/MRL_dataset_notes.md` and check its terms before downloading or redistributing it.

## Contents

- `drowsiness_detector.py`: real-time Windows camera application using facial landmarks, EAR/MAR, head pose, and OcuNet.
- `ocunet.xml` and `ocunet.bin`: subject-independent OcuNet export used by the application.
- `face-detection-retail-0004.*` and `face_landmarker.task`: detector and landmark models used at runtime.
- `train_model.py` and `train_subject_independent.py`: OcuNet training and subject-disjoint model comparisons.
- `evaluate_ocunet.py`: OpenVINO eye-state evaluation on CPU or GPU.
- `compare_ear_thresholds.py`: comparison harness for static and calibrated EAR. It needs labeled video-frame data; those results are not yet available.
- `results/`: subject-independent metrics and experiment notes. The reports distinguish eye-state classification from end-to-end driver-drowsiness detection.
- `models/benchmarks` and `checkpoints/`: trained baseline models and subject-independent weights.
- `docs/requirements`: the completion checklist. The original course requirements PDF was not available at its supplied Downloads path when this package was assembled; see the folder note.
- `docs/update_reports`: the supplied IEEE update report, included as an interim document. Its remaining inconsistencies are listed beside it.
- `figures`: the current Gantt chart. Old diagrams and unverified benchmark plots were excluded pending correction.

## Run the application (Windows)

1. Install Python 3.10 or 3.11 and create a virtual environment.
2. Install dependencies:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Start the detector:

   ```powershell
   python drowsiness_detector.py
   ```

   Press `1` for CPU, `2` for the available OpenVINO GPU, `3` for AUTO, or `q`/`Esc` to quit. GPU availability depends on the installed Intel graphics driver and OpenVINO runtime.

## Reproduce training and evaluation

Place the MRL eye images under `dataset/archive/data/{train,val,test}/{awake,sleepy}`. The dataset itself is intentionally not bundled. Then run:

```powershell
python train_subject_independent.py --data-dir dataset/archive/data --architecture OcuNet --epochs 25
python evaluate_ocunet.py --data-dir dataset/archive/data --model ocunet.xml --device CPU --json-out results/ocunet_cpu_reproduced.json
```

For Intel GPU evaluation, use `--device GPU.0` if that is the Intel iGPU name reported by `openvino.Core().available_devices`. The supplied measurements are from one machine and one held-out subject split; reruns can vary by software version and device.

## Evidence limits and remaining work

- OcuNet scored about 96.95% accuracy on held-out eye crops and exceeded the two in-project CNN baselines under the same split. This supports a comparison among these trained eye-state models; it is not a head-to-head reproduction of published models and does not establish end-to-end drowsiness accuracy.
- CPU and Intel GPU reports are model-only inference measurements. They do not measure full application FPS; on this tiny batch-one model the CPU was faster.
- The static-versus-dynamic EAR harness is included, but a labeled face-video dataset is still needed to produce a valid comparison.
- FPGA work is a plan, not a completed hardware implementation.
- The paper, system diagrams, and result figures still need revision. Do not submit the draft PDF as final.

