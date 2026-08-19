```
OpenVINO Adaptive Driver Drowsiness Detection System
```

```
Real-time driver fatigue monitoring pipeline featuring a 60-second dynamic
onboarding calibration phase ($\mu - 1.5\sigma$) to eliminate demographic false
alarms.
```

```
Features
```

```
-Phase 1 Baseline Calibration: Computes driver-specific mean (μ) and standard
deviation (σ) during the first 60 seconds.
```

```
-Phase 2 Active Monitoring:Temporal moving average filter with a 5-consecutive-
frame threshold.
```

```
-Hardware Acceleration:Optimized for CPU inference via Intel OpenVINO.
```

```
Installation:
```

```
pip install -r requirements.txt
```

