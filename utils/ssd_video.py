import cv2 as cv
import numpy as np

from utils.sort_tracker import SORTTracker
from utils.logger import TrafficLogger

SSD_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

TRAFFIC_SSD_CLASSES = {"bicycle", "bus", "car", "motorbike", "person"}

SSD_COLORS = {
    "person": (255, 128, 0),
    "bicycle": (0, 200, 255),
    "car": (0, 255, 100),
    "motorbike": (255, 50, 200),
    "bus": (50, 150, 255),
}


class SSDVideoDetector:
    def __init__(self, prototxt_path, weights_path, scene_name, target_classes=None, conf_threshold=0.4):
        self.net = cv.dnn.readNetFromCaffe(prototxt_path, weights_path)
        self.scene_name = scene_name
        self.target_classes = target_classes or list(TRAFFIC_SSD_CLASSES)
        self.conf_threshold = conf_threshold
        self.tracker = SORTTracker(max_age=30, min_hits=3)
        self.logger = TrafficLogger(scene_name)
        self.crossed_ids = set()
        self.count_per_class = {}

    def _detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections_raw = self.net.forward()
        detections = []
        det_classes = []
        for i in range(detections_raw.shape[2]):
            conf = float(detections_raw[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            cls_id = int(detections_raw[0, 0, i, 1])
            cls_name = SSD_CLASSES[cls_id] if cls_id < len(SSD_CLASSES) else "unknown"
            if cls_name not in self.target_classes:
                continue
            box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            detections.append([x1, y1, x2, y2, conf])
            det_classes.append(cls_name)
        return detections, det_classes

    def _draw_overlay(self, frame, tracks, class_map, line_y):
        cv.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
        for trk in tracks:
            x1, y1, x2, y2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
            cls_name = class_map.get(tid, "unknown")
            color = SSD_COLORS.get(cls_name, (180, 180, 180))
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f"{cls_name} #{tid}", (x1, y1 - 6), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y_offset = 30
        cv.putText(frame, "Counts:", (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for cls_name, count in self.count_per_class.items():
            y_offset += 25
            cv.putText(frame, f"  {cls_name}: {count}", (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        return frame

    def process(self, video_path, output_path=None, show=False):
        cap = cv.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        line_y = int(h * 0.55)
        writer = None
        if output_path:
            writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        frame_idx = 0
        class_map = {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            detections, det_classes = self._detect(frame)
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

            frame = self._draw_overlay(frame, tracks, class_map, line_y)
            if writer:
                writer.write(frame)
            if show:
                cv.imshow("SSD Detection", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        cv.destroyAllWindows()
        return self.count_per_class, self.logger.get_log_path()

    def process_frame(self, frame, frame_idx, line_y, class_filter, class_map):
        detections, det_classes = self._detect(frame)
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

    def _get_class_id_filter(self):
        return []

    def stream_frames(self, video_path):
        cap = cv.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"

        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        line_y = int(h * 0.55)
        frame_idx = 0
        class_map = {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            tracks, class_map = self.process_frame(frame, frame_idx, line_y, [], class_map)
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