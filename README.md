# Traffic Object Detection & Tracking

Real-time computer vision system for detecting, tracking, and counting road-traffic objects across multiple video scenes.

---

## Project Structure

```
Final_project/
├── main.py                   # CLI entry point (all models)
├── kaggle_finetune           # Training on Kaggle
├── streamlit_app.py          # Streamlit real-time visualization
├── fine_tune.py              # YOLOv11 fine-tuning script
├── download_model.py         # Download base YOLO weights
├── download_samples.py       # Download sample test data
├── requirements.txt
├── data/
│   ├── traffic.yaml          # Dataset config for fine-tuning
│   ├── images/train/         # Training images
│   ├── images/val/           # Validation images
│   └── labels/train/         # YOLO .txt annotations
├── models/
│   ├── yolo11n.pt                              # Base weights
│   ├── stop_data.xml                           # Cascade weights
│   ├── MobileNetSSD_deploy.prototxt
│   ├── MobileNetSSD_deploy.caffemodel
│   └── yolo11n_traffic_finetuned/weights/
│       ├── best.pt                             # Best fine-tuned model
│       └── last.pt
├── utils/
│   ├── __init__.py           # Empty — marks utils/ as Python package
│   ├── cascade.py
│   ├── yolo_image.py
│   ├── yolo_video.py
│   ├── ssd_video.py
│   ├── sort_tracker.py
│   └── logger.py
├── logs/
└── uploads/

```

---

## Follow steps bellow to run the code

### STEP 1 — Create and intall a virtual environment

```bash
conda create -n aims_cv python=3.10
conda activate aims_cv
pip install -r requirements.txt
```

### STEP 2 — Download the YOLOv11 basic model

```bash
python download_model.py
```

You will found the result in : `models/yolo11n.pt`

### STEP 3 — Download test videos

```bash
python download_samples.py
```

Or manually via https://www.pexels.com/search/videos/traffic/
Put that/those videos in `data/videos/`

### STEP 4 — Test with the basic model

```bash
python main.py --model yolov11 \
    --model_path models/yolo11n.pt \
    --filepath data/videos/traffic1.mp4 \
    --scene intersection_1 \
    --show

```bash
python main.py --model yolov11 \
    --model_path models/yolo11n.pt \
    --filepath data/videos/traffic2.mp4 \
    --scene intersection_1 \
    --show

python main.py --model cascade \
    --filepath data/images/stop_sign.jpg \
    --model_path models/stop_data.xml

python main.py --model watershed \
    --filepath data/images/stop_sign.jpg
```

### STEP 5 — Fine-tuning

Annotate your images with Roboflow (YOLO format).
Place in `data/images/train/`, `data/images/val/`, `data/labels/train/`, `data/labels/val/`.

```bash
python fine_tune.py \
    --base_model models/yolo11n.pt \
    --data_yaml data/traffic.yaml \
    --epochs 30 \
    --batch 8 \
    --run_name yolo11n_traffic_finetuned \
    --validate
```

Fine-Tune backup model has : `models/yolo11n_traffic_finetuned/weights/best.pt`

### STEP 6 — Use the fine-tune model

```bash
python main.py --model yolov11 \
    --model_path models/yolo11n_traffic_finetuned/weights/best.pt \
    --filepath data/videos/traffic1.mp4 \
    --scene intersection_finetuned \
    --show \
    --output outputs/result_finetuned.avi
```

### STEP 7 — Streamlit interface (real time)

```bash
streamlit run streamlit_app.py
```

Open : http://localhost:8501


### STEP 9 — SSD MobileNet

Download the SSD weights from https://github.com/chuanqi305/MobileNet-SSD
Place in `models/`.

```bash
python main.py --model ssd \
    --filepath data/videos/traffic.mp4 \
    --scene roundabout_1 \
    --show
```

---

## Modele fine-tune

| | Path |
|---|---|
| Basic model | `models/yolo11n.pt` |
| Modele fine-tune (best) | `models/yolo11n_traffic_finetuned/weights/best.pt` |
| Modele fine-tune (last) | `models/yolo11n_traffic_finetuned/weights/last.pt` |

---

## Log Schema

| Colonne | Description |
|---|---|
| timestamp | Detection time |
| scene | Scene name |
| frame | Frame Index |
| track_id | persistent unique ID per object |
| class_name | Class detected |
| x1, y1, x2, y2 | Bounding box |
| confidence | Trust Score |
| crossed | 1 if the object crosses the counting line |

---


# CV_Final_Project
