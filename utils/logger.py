import csv
import os
import time
from datetime import datetime


class TrafficLogger:
    SCHEMA = ["timestamp", "scene", "frame", "track_id", "class_name", "x1", "y1", "x2", "y2", "confidence", "crossed"]

    def __init__(self, scene_name, output_dir="logs"):
        os.makedirs(output_dir, exist_ok=True)
        self.scene_name = scene_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"{scene_name}_{timestamp}.csv")
        self.stats = {}
        self.crossed_ids = set()
        self._init_file()

    def _init_file(self):
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SCHEMA)
            writer.writeheader()

    def log(self, frame_idx, track_id, class_name, bbox, confidence, crossed=False):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = {
            "timestamp": ts,
            "scene": self.scene_name,
            "frame": frame_idx,
            "track_id": track_id,
            "class_name": class_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": round(float(confidence), 4),
            "crossed": int(crossed),
        }
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SCHEMA)
            writer.writerow(row)

        if crossed and track_id not in self.crossed_ids:
            self.crossed_ids.add(track_id)
            self.stats[class_name] = self.stats.get(class_name, 0) + 1

    def get_stats(self):
        return dict(self.stats)

    def get_log_path(self):
        return self.filepath