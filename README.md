# YOLOv8 Helmet Detection & Compliance System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An End-to-End Computer Vision System for Motorcycle Helmet Detection and Safety Compliance**

</div>

---

## Overview

YOLOv8 Helmet Detection & Compliance System is an end-to-end computer vision project designed to detect motorcycle riders and classify their helmet usage using deep learning and object detection techniques.

The project utilizes a custom-trained **YOLOv8 Nano (YOLOv8n)** model to identify two categories:

- **With Helmet:** Motorcycle riders wearing helmets.
- **Without Helmet:** Motorcycle riders not wearing helmets.

The pipeline covers the complete machine learning workflow, including raw PASCAL VOC XML annotation processing, YOLO-format dataset conversion, model training, performance evaluation, and video-based inference.

The system generates annotated video outputs with bounding boxes and classification labels to support automated helmet compliance monitoring.

---

## Tech Stack & Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Ultralytics YOLOv8 | Deep learning and object detection |
| PyTorch | Deep learning framework |
| OpenCV | Image and video processing |
| Scikit-learn | Dataset splitting and preprocessing |
| Pandas | Data manipulation |
| XML ElementTree | PASCAL VOC XML annotation parsing |

---

## Repository Structure

```text
YOLOv8-Helmet-Detection/
│
├── dataset/
│   └── data.yaml
│
├── runs/
│   └── detect/
│       └── helmet_custom_model-5/
│           └── weights/
│               └── best.pt
│
├── convert_to_yolo.py
├── train.py
├── detect.py
├── requirements.txt
└── README.md
```

### Key Files

| File / Directory | Description |
|---|---|
| `dataset/data.yaml` | Dataset configuration containing class names and dataset paths |
| `best.pt` | Best-performing trained YOLOv8 model weights |
| `convert_to_yolo.py` | Converts PASCAL VOC XML annotations into YOLO format |
| `train.py` | Model training pipeline |
| `detect.py` | Performs video inference and renders detection bounding boxes |
| `requirements.txt` | Project dependencies |
| `README.md` | Project documentation |

---

## Model Architecture & Training

The project uses the **YOLOv8 Nano (YOLOv8n)** architecture, initialized with pretrained weights and fine-tuned on a custom helmet detection dataset.

### Training Configuration

| Parameter | Value |
|---|---|
| Model Architecture | YOLOv8n |
| Model Type | Object Detection |
| Pretrained Weights | YOLOv8n |
| Training Epochs | 30 |
| Number of Classes | 2 |
| Framework | Ultralytics YOLOv8 |
| Deep Learning Backend | PyTorch |

### Detection Classes

| Class | Description |
|---|---|
| With Helmet | Detects riders wearing helmets |
| Without Helmet | Detects riders without helmets |

---

## Model Performance & Evaluation

The trained model achieved the following reported evaluation results:

| Evaluation Metric | Performance |
|---|---:|
| Overall mAP50 | 84.7% |
| With Helmet mAP50 | 90.0% |
| Without Helmet mAP50 | 79.4% |

### Performance Highlights

- Achieved an overall mAP50 of approximately **84.7%**.
- Demonstrated stronger detection performance for riders wearing helmets.
- Successfully learned to distinguish between helmet-compliant and non-compliant riders.
- Supports video-based object detection and visual result generation.

---

## Installation & Setup

Follow these steps to set up the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/MobeenFatimaa/YOLOv8-Helmet-Detection.git
```

Navigate to the project directory:

```bash
cd YOLOv8-Helmet-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Dataset Preparation

Ensure that your dataset is correctly organized and that the dataset configuration file is available:

```text
dataset/
└── data.yaml
```

If your annotations are in PASCAL VOC XML format, execute the conversion script:

```bash
python convert_to_yolo.py
```

This converts the XML annotations into YOLO-compatible text annotation files.

### 2. Train the Model

To train the YOLOv8 model using your configured dataset:

```bash
python train.py
```

The training process generates model checkpoints and evaluation results in the Ultralytics `runs/` directory.

### 3. Run Video Inference

Place your test video file (for example, `test_video.mp4`) in the project root directory.

Execute the detection script:

```bash
python detect.py
```

The script processes the input video, performs helmet detection, and generates an annotated video containing bounding boxes and classification labels.

---

## Applications

This project can be used as a foundation for:

- Automated motorcycle helmet compliance monitoring.
- Intelligent traffic surveillance systems.
- Road safety monitoring applications.
- Smart city traffic management solutions.
- Computer vision-based safety analytics.
- Real-time object detection research and development.

---

## Future Improvements

Potential enhancements include:

- Real-time detection through CCTV camera streams.
- Improved detection accuracy for distant and partially occluded riders.
- Integration with vehicle detection and tracking systems.
- Automated helmet violation reporting.
- Deployment through web-based monitoring dashboards.
- Optimization for edge devices and embedded systems.

---

## License

This project is distributed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

---

<div align="center">

**Developed by Mobeen Fatima**

AI & Machine Learning | Deep Learning | Computer Vision

[GitHub](https://github.com/MobeenFatimaa) · [LinkedIn](https://www.linkedin.com/in/mobeen-fatima-599a35347/)

</div>
