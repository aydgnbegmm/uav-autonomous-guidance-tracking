# 🛸 UAV Autonomous Guidance & Target Tracking Suite

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-YOLOv8%20%7C%20OpenCV%20%7C%20MAVLink-orange.svg)](https://github.com/)
[![Autopilot](https://img.shields.io/badge/Autopilot-ArduPilot%20%2F%20PX4-red.svg)](https://ardupilot.org/)

An end-to-end, real-time autonomous target tracking, guidance, and mission engagement suite engineered for Fixed-Wing and Multi-Rotor Unmanned Aerial Vehicles (UAVs).

Developed and flight-tested in demanding competitive and operational scenarios (including TEKNOFEST Combat UAV & METU VTOL competitions), this architecture fuses Deep Learning object detection (YOLOv8), multi-target correlation (Hungarian Algorithm + CSRT Tracker Pool), Discrete Kalman Filtering, and Proportional Navigation guidance commands delivered directly via MAVLink.

---

## 📌 Architecture Overview

```
                        ┌────────────────────────┐
                        │   Onboard RGB Camera   │
                        └───────────┬────────────┘
                                    │ Frames
                                    ▼
                        ┌────────────────────────┐
                        │  YOLOv8 Object Detector │
                        └───────────┬────────────┘
                                    │ Bounding Boxes
                                    ▼
                        ┌────────────────────────┐
                        │ Hungarian Data Assoc.  │
                        │    (IoU Matcher)       │
                        └───────────┬────────────┘
                                    │ Matches
                                    ▼
                        ┌────────────────────────┐
                        │  SortCSRT Multi-Track  │
                        └───────────┬────────────┘
                                    │ Raw Positions
                                    ▼
┌──────────────────┐    ┌────────────────────────┐
│  Autopilot Telemetry │───▶│  Discrete Kalman Filter │
│ (GNSS / IMU / Baro)│    │   (State Estimation)   │
└──────────────────┘    └───────────┬────────────┘
                                    │ Filtered [x, y, vx, vy]
                                    ▼
                        ┌────────────────────────┐
                        │ Guidance & PID Engine  │
                        │ (FSM: Search / Track)  │
                        └───────────┬────────────┘
                                    │ Velocity / Angle Setpoints
                                    ▼
                        ┌────────────────────────┐
                        │ Drone SDK / MAVLink    │
                        │ (Flight Controller)    │
                        └────────────────────────┘
```

---

## ✨ Key Capabilities

*   **Multi-Object Visual Tracker (`sort_csrt.py`):**
    *   Replaces brittle single-tracker setups with a SORT-inspired CSRT tracker pool.
    *   Hungarian Algorithm with IoU matching accurately associates YOLO detections with active tracker states.
    *   Grace period and streak management prevent false track loss during short-term occlusions and high-G maneuvers.
*   **Discrete Kalman Filter (`kalman_filter.py`):**
    *   Fuses visual bounding box centroids to smooth out detection jitter.
    *   Predicts target velocity vector during dropped camera frames or inference latency.
*   **Hybrid GNSS-Visual Guidance FSM (`guidance_controller.py` & `angajman_mix.py`):**
    *   Seamless finite state machine transitions between wide-area GNSS waypoints and terminal visual engagement.
    *   State seeding ensures PID integral and derivative terms are preserved during mode handovers, preventing control spikes.
*   **GNSS Reliability Monitor (`gnss_reliability.py`):**
    *   Real-time monitoring of HDOP, satellite count, fix type, and velocity variance to identify sensor degradation or spoofing.
*   **MAVLink Abstraction Layer (`drone_sdk.py`):**
    *   Clean, thread-safe interface for vehicle telemetry, attitude control, guided waypoint injection, and emergency overrides.

---

## 📂 Repository Structure

```
├── angajman.py                 # Core engagement logic & CLI entry point
├── angajman_mix.py             # Hybrid GNSS + Computer Vision mission coordinator
├── drone_sdk.py                # MAVLink / Autopilot hardware abstraction layer
├── guidance_controller.py      # Proportional Navigation & PID trajectory controller
├── gnss_reliability.py         # GNSS integrity, HDOP & satellite constellation monitor
├── kalman_filter.py            # 2D/3D state estimation & velocity prediction
├── sort_csrt.py                # SORT-inspired multi-CSRT tracker with Hungarian association
├── vision_tracker.py           # OSD filtering, dark-blob mitigation & detector pipeline
├── kamikaze_vision_v3.py       # Terminal visual pursuit and interception controller
├── kamikaze_gnss_controller.py # Waypoint alignment and vector approach controller
├── requirements.txt            # Dependency definitions
└── README.md                   # System documentation
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/<your-username>/uav-autonomous-guidance-tracking.git
cd uav-autonomous-guidance-tracking
pip install -r requirements.txt
```

### 2. Running Simulation / Benchmarking
Run the mission coordinator with your trained YOLO weights and camera index:
```bash
python angajman_mix.py --weights weights/best.pt --camera-index 0 --show --conf 0.45
```

### 3. Key CLI Flags
*   `--no-visual-track`: Run strictly on GNSS guidance.
*   `--no-csrt`: Disable tracker pool (pure frame-by-frame YOLO inference).
*   `--detector-interval 8`: Run YOLO detector every 8 frames, relying on CSRT for intermediate high-FPS tracking.
*   `--imgsz 640`: Input inference resolution (use `320` for resource-constrained edge SBCs like Raspberry Pi 5 / Radxa Rock 5B).

---

## 🛠️ Supported Hardware & Platforms

*   **Companion Computers:** Radxa Rock 5B (RK3588 NPU), Raspberry Pi 5, NVIDIA Jetson Orin / Nano.
*   **Flight Controllers:** Pixhawk Cube Orange / Black, Holybro Pixhawk 6C (ArduPilot Firmware / PX4).
*   **Cameras:** Sony IMX477 / Global Shutter USB3 Cameras / RTSP Video Streams.

---

## 👩‍💻 Author

**Begüm Aydoğan Çiçin**  
*Mechatronics Engineer | Embedded Systems & Autonomous UAV Specialist*  
* [LinkedIn](https://www.linkedin.com/in/begum-aydogan/)  
* Contact: `aydgnbegm@gmail.com`
