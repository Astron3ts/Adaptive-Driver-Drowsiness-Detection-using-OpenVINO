# Project Requirements Checklist

Source: MACSE517 Computer Architecture and Organization project-document requirements supplied as `project Document.pdf`. This checklist reviews the current project and paper draft; it does not edit the IEEE paper.

## Requirement status

| Requirement | Current status | Work still needed |
|---|---|---|
| Title | Present in the paper draft. | Keep consistent with the final implementation. |
| Abstract with problem statement and results | Present, but stale. It says the eye dataset and evaluation do not exist. | Replace that statement with the subject-independent OcuNet eye-state results; do not label them full drowsiness accuracy. |
| Introduction with citations | Present. | Check citation numbering and source details during final paper editing. |
| Technical, economic, social, environmental, political, and demographic motivation | All six are discussed. | Copyedit wording and keep claims proportional to a prototype. |
| Individual member contributions | Contributors are named, but individual task ownership is not clearly assigned. | Add a short member-to-task contribution statement. |
| Project plan / Gantt chart | `images/project_gantt_chart.png` exists. | Confirm the timeline matches work actually completed and cite it as a numbered figure. |
| Literature review with at least 20 journal papers | The draft has a 20-paper survey table and 30 references. | Verify journal quality/quartiles, reference metadata, table-to-reference mapping, and reported values. |
| Research gaps | Present. | Update the gap wording to reflect the new eye-state experiment while retaining the missing video evaluation. |
| Proposed work and block diagram | Simple and AI-generated diagram drafts exist. | Both diagrams need correction before use: see the diagram issues below. |
| Dataset description and software installation | README installation steps exist; MRL eye data and training caches are in the project. | Replace the paper’s missing-data/provenance claims with the MRL source, subject split, labels, preprocessing, and exact software versions. |
| Updated simple and AI-generated diagrams, separately available | Two old PNGs are retained under `images/drafts_unverified/`. | Rebuild both from the current code, export at 300 dpi or as vector files, and keep the two final files distinct in the repository. |
| Module-wise implementation with mathematical functions | The paper describes modules, but equations and threshold values need a code-accurate, consolidated explanation. | Add the equations listed below and define every variable. |
| CPU, GPU, and FPGA implementation | CPU screenshots exist; OcuNet has been measured on Intel CPU and Intel UHD GPU. FPGA is a plan, not an implementation. | Separate implemented CPU/GPU results from planned FPGA work. Full-system GPU and FPGA measurements are not available. |
| Performance parameters | Face-inference snapshots and model-only CPU/GPU latency are available. | Add methodology and repeated full-system measurements; distinguish model latency, face-detection latency, and end-to-end FPS. |
| Graphs, tables, and images at 300 dpi with captions/citations | The Gantt chart reports 300 dpi. Old diagrams and benchmark charts are in a quarantined draft folder; `System_Arch_COA.png` reports 25.4 dpi. | Create current result charts, re-export corrected diagrams at 300 dpi or vector quality, and number/cite every figure and table. |
| Results explanation, minimum two pages | The draft has multiple results subsections and state-by-state discussion. | Recheck page count and expand with the new split, model comparison, and device results without overstating them. |
| Conclusion | Present, but says no accuracy evaluation exists and GPU was not measured. | Update after the final experiments; keep EAR comparison and full-system GPU limitations explicit. |
| 25–30 references, mainly Q1/Q2 journals plus selected conferences | 30 references are listed. | Verify current quartiles/indexing and clean citation/title mismatches. |
| Instructor-accessible GitHub with IEEE document, code, and images organized separately | An `origin` GitHub remote is configured. | Verify instructor access and organize the final paper, source, and images into clearly named locations before submission. |

## Diagrams that must be corrected

The current AI-generated `images/System_Arch_COA.png` does not match the implementation. It shows an EAR threshold of `mean - 1.5 × standard deviation`, a five-frame alarm, unsupported FPS claims, and FPGA/bitstream execution. The code uses a clipped personalized EAR threshold, 35 consecutive closure frames, and has no FPGA bitstream or CAN trigger. The simple diagram also needs the current OcuNet fusion, clipped EAR calibration, MAR calibration, PERCLOS, and correct alert logic represented precisely.

The old benchmark images also need to stay out of the results section until their source data is verified. The FPS chart reports a 26.5 mean, while the paper's five screenshot observations average 35.0 FPS. The latency chart contains no GPU measurements, and the multi-cue plot shows head-pose wrapping rather than a validated drowsiness sequence. These five images were moved to `images/drafts_unverified/`; do not cite them as measured results.

Treat both diagrams as drafts until redrawn from the current source. Do not include their unsupported FPS figures in the paper. The Gantt chart remains in the active `images/` folder. `.gitignore` now allows active images under `images/` into Git while keeping the unverified draft folder ignored.

## Equations to include in the module description

Use the landmark ordering from the implementation and define all points and thresholds in the text.

### Eye Aspect Ratio (EAR)

For one six-point eye, with points ordered around the eye as in the implementation:

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 ||p1 - p4||)
EAR_bilateral = (EAR_left + EAR_right) / 2
```

The code smooths the bilateral EAR with a five-frame moving average.

### Personalized EAR calibration

For calibration EAR samples `E`, discard values below the 10th percentile, then calculate the mean `μ_E` and standard deviation `σ_E` of the remaining samples:

```text
q10 = percentile(E, 10)
E_clean = {e in E : e >= q10}
μ_E = mean(E_clean)
σ_E = std(E_clean)
τ_E = clip(max(μ_E - 2.2 σ_E, 0.72 μ_E), 0.18, 0.24)
```

Calibration lasts 60 seconds. Explain that `clip(x, a, b)` limits `x` to `[a,b]`.

### Mouth Aspect Ratio (MAR)

The implementation uses three vertical mouth distances and the horizontal corner-to-corner distance:

```text
MAR = (||m1_top - m1_bottom|| + ||m2_top - m2_bottom||
       + ||m3_top - m3_bottom||) / (2 ||mouth_left - mouth_right||)
τ_M = max(0.55, μ_M + 3 σ_M)
```

Yawning is counted after MAR remains at or above `τ_M` for 28 consecutive frames.

### OcuNet closed-eye probability and eye-cue fusion

For logits `z_awake` and `z_sleepy`, describe the softmax probability used by the detector:

```text
p_closed = exp(z_sleepy) / (exp(z_awake) + exp(z_sleepy))
closed_t = (EAR_t < τ_E) OR (p_closed > 0.75)
```

The actual code uses a numerically stable equivalent softmax. A microsleep alert occurs after `closed_t` remains true for 35 consecutive frames.

### PERCLOS

For a rolling buffer of `N` frames with `c_i = 1` when the eye is considered closed and `0` otherwise:

```text
PERCLOS (%) = 100 × (Σ c_i / N)
```

The buffer holds up to 1,800 frames; PERCLOS is displayed after at least 300 frames. Explain that frame-count windows correspond to different time durations when FPS changes.

### Head pose

If the paper explains the solvePnP module mathematically, state the reprojection objective:

```text
(R*, t*) = argmin_(R,t) Σ_i ||u_i - π(K (R X_i + t))||²
```

Here `X_i` are the six 3D model points, `u_i` their 2D facial landmarks, `K` the camera matrix, and `π` the projection operation. The code triggers the nod branch at pitch `<= -18°` for 22 consecutive frames. The paper already notes Euler-angle wrapping; preserve that limitation.

### Training objective (if retaining training details)

For two-class cross-entropy over `N` labeled eye crops:

```text
L = -(1/N) Σ_i log(p_(i, y_i))
```

State that labels are awake/open and sleepy/closed as represented by the project folders; this is eye-state classification, not a direct ground truth label for driver drowsiness.

## Cleanup performed

- `train.py` was byte-for-byte identical to `train_model.py`; the duplicate was removed. `train_model.py` remains as the README-referenced entry point.
- The incomplete October 21 training log contained setup lines but no completed epoch or final metrics; it was removed. The completed October 7 log is retained.
- The root `__pycache__` contained generated Python bytecode and was removed.
- `kernel.errors.txt` is retained as an Intel GPU diagnostic until the full application GPU run is reviewed.
- Dataset, source code, active OpenVINO models, experiment scripts, metrics, Gantt chart, and FPGA planning document are retained because they support implementation or course requirements.
- Unverified benchmark charts and outdated diagrams were moved to `images/drafts_unverified/`; they are excluded from the normal Git image set pending replacement.
