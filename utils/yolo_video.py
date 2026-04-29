import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO

from utils.sort_tracker import SORTTracker
from utils.logger import TrafficLogger

# Classes for the base YOLOv11 model (COCO ids)
COCO_TRAFFIC_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Classes for the fine-tuned model (bdd100k — 7 sequential classes)
FINETUNED_CLASSES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motor",
    4: "bike",
    5: "person",
    6: "rider",
}

FINETUNED_MODEL = "models/yolo11n_traffic_finetuned/weights/best.pt"
BASE_MODEL = "models/yolo11n.pt"


def resolve_model_path(path=None):
    candidate = path or FINETUNED_MODEL
    if os.path.exists(candidate):
        return candidate
    print(f"[WARNING] Model not found at '{candidate}'. Falling back to base model '{BASE_MODEL}'.")
    if os.path.exists(BASE_MODEL):
        return BASE_MODEL
    raise FileNotFoundError(
        f"No model found. Expected one of:\n  {candidate}\n  {BASE_MODEL}\n"
        f"Run: python download_model.py"
    )

DEFAULT_COLORS = {
    "person":     (255, 128,   0),
    "bicycle":    (  0, 200, 255),
    "car":        (  0, 255, 100),
    "motorcycle": (255,  50, 200),
    "bus":        ( 50, 150, 255),
    "truck":      (200,  50,  50),
}


class YOLOVideoDetector:
    def __init__(self, model_path=None, scene_name="scene", target_classes=None, conf_threshold=0.4):
        path = resolve_model_path(model_path)
        print(f"[INFO] Loading model: {path}")
        self.model = YOLO(path)
        nc = len(self.model.names)
        self._class_dict = FINETUNED_CLASSES if nc <= 10 else COCO_TRAFFIC_CLASSES
        print(f"[INFO] Model has {nc} classes -> using {'FINETUNED_CLASSES' if nc <= 10 else 'COCO_TRAFFIC_CLASSES'}")
        if target_classes is None:
            self.target_classes = list(self._class_dict.values())
        self.scene_name = scene_name
        self._class_dict = COCO_TRAFFIC_CLASSES  # updated after model load
        self.target_classes = target_classes or list(COCO_TRAFFIC_CLASSES.values())
        self.conf_threshold = conf_threshold
        self.tracker = SORTTracker(max_age=30, min_hits=3)
        self.logger = TrafficLogger(scene_name)
        self.crossed_ids = set()
        self.count_per_class = {}

    def _get_class_id_filter(self):
        return [k for k, v in self._class_dict.items() if v in self.target_classes]

    def _draw_overlay(self, frame, tracks, class_map, line_y):
        cv.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
        for trk in tracks:
            x1, y1, x2, y2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
            cls_name = class_map.get(tid, "unknown")
            color = DEFAULT_COLORS.get(cls_name, (180, 180, 180))
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f"{cls_name} #{tid}", (x1, y1 - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        y_offset = 30
        cv.putText(frame, "Counts:", (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for cls_name, count in self.count_per_class.items():
            y_offset += 25
            cv.putText(frame, f"  {cls_name}: {count}", (10, y_offset),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        total = sum(self.count_per_class.values())
        if total == 0:
            cv.putText(frame, "No objects detected", (10, y_offset + 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
        return frame

    def process_frame(self, frame, frame_idx, line_y, class_filter, class_map):
        results = self.model(frame, classes=class_filter, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        det_classes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self._class_dict.get(cls_id, "unknown")
            if cls_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])
            det_classes.append(cls_name)

        tracks = self.tracker.update([[d[0], d[1], d[2], d[3]] for d in detections])

        for i, det in enumerate(detections):
            for trk in tracks:
                if self._iou(det[:4], trk[:4]) > 0.3:
                    tid = int(trk[4])
                    cls_name = det_classes[i]
                    class_map[tid] = cls_name
                    cy = (trk[1] + trk[3]) / 2
                    crossed = cy > line_y
                    is_new = crossed and tid not in self.crossed_ids
                    if is_new:
                        self.crossed_ids.add(tid)
                        self.count_per_class[cls_name] = self.count_per_class.get(cls_name, 0) + 1
                    self.logger.log(frame_idx, tid, cls_name, trk[:4], det[4], crossed=is_new)
                    break

        return tracks, class_map

    def process(self, video_path, output_path=None, show=False):
        cap = cv.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"

        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        line_y = int(h * 0.55)

        writer = None
        if output_path:
            writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        class_filter = self._get_class_id_filter()
        frame_idx = 0
        class_map = {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            tracks, class_map = self.process_frame(frame, frame_idx, line_y, class_filter, class_map)
            frame = self._draw_overlay(frame, tracks, class_map, line_y)

            if writer:
                writer.write(frame)
            if show:
                cv.imshow("Traffic Detection", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        cv.destroyAllWindows()
        return self.count_per_class, self.logger.get_log_path()

    def stream_frames(self, video_path):
        cap = cv.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"

        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        line_y = int(h * 0.55)
        class_filter = self._get_class_id_filter()
        frame_idx = 0
        class_map = {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            tracks, class_map = self.process_frame(frame, frame_idx, line_y, class_filter, class_map)
            frame = self._draw_overlay(frame, tracks, class_map, line_y)
            _, buffer = cv.imencode(".jpg", frame)
            yield buffer.tobytes(), dict(self.count_per_class)
            frame_idx += 1

        cap.release()

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0.0