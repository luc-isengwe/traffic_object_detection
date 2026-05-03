import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils.sort_tracker import SORTTracker
from utils.logger import TrafficLogger
from utils.bonus import (SpeedEstimator, DirectionPredictor, HeatmapGenerator,
                         plot_traffic_flow, plot_class_distribution)

COCO_TRAFFIC_CLASSES = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}

FINETUNED_CLASSES = {
    0: "person", 1: "bike", 2: "car",
    3: "motor", 4: "bus", 5: "truck", 6: "rider",
}

FINETUNED_MODEL = "models/yolo11n_traffic_finetuned/weights/best.pt"
BASE_MODEL      = "models/yolo11n.pt"

DEFAULT_COLORS = {
    "person":   (255, 128,   0), "bicycle": (  0, 200, 255),
    "car":      (  0, 255, 100), "motorcycle": (255,  50, 200),
    "bus":      ( 50, 150, 255), "truck":  (200,  50,  50),
    "motor":    (255,  50, 200), "bike":   (  0, 200, 255),
    "rider":    (128, 255,   0),
}


def resolve_model_path(path=None):
    candidate = path or FINETUNED_MODEL
    if os.path.exists(candidate):
        return candidate
    print(f"[WARNING] '{candidate}' not found. Falling back to '{BASE_MODEL}'.")
    if os.path.exists(BASE_MODEL):
        return BASE_MODEL
    raise FileNotFoundError(
        f"No model found.\n  Expected: {candidate}\n  Fallback: {BASE_MODEL}\n"
        "Run: python download_model.py"
    )


class YOLOVideoDetector:
    def __init__(self, model_path=None, scene_name="scene",
                 target_classes=None, conf_threshold=0.4,
                 group_id="group_02", video_name="unknown"):
        path = resolve_model_path(model_path)
        print(f"[INFO] Loading: {path}")
        self.model = YOLO(path)
        nc = len(self.model.names)
        self._class_dict = FINETUNED_CLASSES if nc <= 10 else COCO_TRAFFIC_CLASSES
        print(f"[INFO] nc={nc} -> {'FINETUNED' if nc<=10 else 'COCO'} classes")

        self.scene_name      = scene_name
        self.target_classes  = target_classes or list(self._class_dict.values())
        self.conf_threshold  = conf_threshold
        self.tracker         = SORTTracker(max_age=30, min_hits=3)
        self.crossed_ids     = set()
        self.count_per_class = {}
        self.speed_est       = None
        self.dir_pred        = DirectionPredictor(history_len=8)
        self.heatmap         = None
        self.count_history   = []
        self.frame_w         = 0
        self.frame_h         = 0
        self.fps             = 25
        self.logger          = None
        self.group_id        = group_id
        self.video_name      = video_name

    def _init_logger(self):
        self.logger = TrafficLogger(
            scene_name=self.scene_name,
            group_id=self.group_id,
            video_name=self.video_name,
            fps=self.fps,
        )

    def _get_class_id_filter(self):
        return [k for k, v in self._class_dict.items() if v in self.target_classes]

    def _draw_overlay(self, frame, tracks, class_map, line_y, speed_map=None):
        if self.heatmap:
            frame = self.heatmap.get_overlay(frame, alpha=0.35)
        cv.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

        for trk in tracks:
            x1, y1, x2, y2, tid = (int(trk[0]), int(trk[1]),
                                    int(trk[2]), int(trk[3]), int(trk[4]))
            cls_name = class_map.get(tid, "unknown")
            color    = DEFAULT_COLORS.get(cls_name, (180, 180, 180))
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            spd   = speed_map.get(tid, 0.0) if speed_map else 0.0
            arrow = self.dir_pred.get_arrow(tid)
            label = f"{cls_name}#{tid} {arrow}" + (f" {spd:.0f}px/s" if spd > 0 else "")
            cv.putText(frame, label, (x1, y1-6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        y_off = 30
        cv.putText(frame, "Counts:", (10, y_off),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for cls_name, count in self.count_per_class.items():
            y_off += 25
            cv.putText(frame, f"  {cls_name}: {count}", (10, y_off),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        if sum(self.count_per_class.values()) == 0:
            cv.putText(frame, "No objects detected", (10, y_off+30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
        return frame

    def process_frame(self, frame, frame_idx, line_y, class_filter, class_map):
        results = self.model(frame, classes=class_filter,
                             conf=self.conf_threshold, verbose=False)[0]
        detections, det_classes = [], []

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self._class_dict.get(cls_id, "unknown")
            if cls_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append([x1, y1, x2, y2, float(box.conf[0])])
            det_classes.append(cls_name)

        tracks = self.tracker.update([[d[0],d[1],d[2],d[3]] for d in detections])
        if self.heatmap:
            self.heatmap.update(tracks)

        speed_map = {}
        for i, det in enumerate(detections):
            for trk in tracks:
                if self._iou(det[:4], trk[:4]) > 0.3:
                    tid      = int(trk[4])
                    cls_name = det_classes[i]
                    class_map[tid] = cls_name

                    cx = int((trk[0]+trk[2])/2)
                    cy = int((trk[1]+trk[3])/2)

                    speed_px_s = 0.0
                    if self.speed_est:
                        speed_px_s = self.speed_est.update(tid, cx, cy, frame_idx)
                        speed_map[tid] = speed_px_s

                    direction = self.dir_pred.update(tid, cx, cy)

                    crossed = cy > line_y
                    is_new  = crossed and tid not in self.crossed_ids
                    if is_new:
                        self.crossed_ids.add(tid)
                        self.count_per_class[cls_name] = \
                            self.count_per_class.get(cls_name, 0) + 1

                    if self.logger:
                        self.logger.log(
                            frame_idx=frame_idx,
                            track_id=tid,
                            class_name=cls_name,
                            bbox=trk[:4],
                            confidence=det[4],
                            frame_w=self.frame_w,
                            frame_h=self.frame_h,
                            crossed=is_new,
                            direction=direction,
                            speed_px_s=speed_px_s,
                        )
                    break

        self.count_history.append({"frame": frame_idx, **self.count_per_class})
        return tracks, class_map, speed_map

    def _save_plots(self):
        os.makedirs("outputs", exist_ok=True)
        if self.heatmap:
            self.heatmap.save_plot(f"outputs/{self.scene_name}_heatmap.png", self.scene_name)
        plot_traffic_flow(self.count_history, self.scene_name,
                          f"outputs/{self.scene_name}_traffic_flow.png")
        plot_class_distribution(self.count_per_class, self.scene_name,
                                f"outputs/{self.scene_name}_class_dist.png")

    def process(self, video_path, output_path=None, show=False,
                enable_speed=True, enable_heatmap=True):
        cap = cv.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open: {video_path}"

        self.frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps     = int(cap.get(cv.CAP_PROP_FPS)) or 25
        self.video_name = os.path.basename(video_path)
        self._init_logger()
        line_y = int(self.frame_h * 0.55)

        if enable_speed:
            self.speed_est = SpeedEstimator(self.fps)
        if enable_heatmap:
            self.heatmap = HeatmapGenerator(self.frame_h, self.frame_w)

        writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"),
                                self.fps, (self.frame_w, self.frame_h)) \
                 if output_path else None

        class_filter, frame_idx, class_map = self._get_class_id_filter(), 0, {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            tracks, class_map, speed_map = self.process_frame(
                frame, frame_idx, line_y, class_filter, class_map)
            frame = self._draw_overlay(frame, tracks, class_map, line_y, speed_map)
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
        self._save_plots()
        return self.count_per_class, self.logger.get_log_path()

    def stream_frames(self, video_path, enable_speed=True, enable_heatmap=True):
        cap = cv.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open: {video_path}"

        self.frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.fps     = int(cap.get(cv.CAP_PROP_FPS)) or 25
        self.video_name = os.path.basename(video_path)
        self._init_logger()
        line_y = int(self.frame_h * 0.55)

        if enable_speed:
            self.speed_est = SpeedEstimator(self.fps)
        if enable_heatmap:
            self.heatmap = HeatmapGenerator(self.frame_h, self.frame_w)

        class_filter, frame_idx, class_map = self._get_class_id_filter(), 0, {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            tracks, class_map, speed_map = self.process_frame(
                frame, frame_idx, line_y, class_filter, class_map)
            frame = self._draw_overlay(frame, tracks, class_map, line_y, speed_map)
            _, buffer = cv.imencode(".jpg", frame)
            yield buffer.tobytes(), dict(self.count_per_class)
            frame_idx += 1

        cap.release()
        self._save_plots()

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0.0