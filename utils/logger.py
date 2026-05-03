import csv
import os


class TrafficLogger:
    SCHEMA = [
        "frame", "timestamp_sec", "scene_name", "group_id", "video_name",
        "track_id", "class_name", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "cx", "cy", "frame_width", "frame_height",
        "crossed_line", "direction", "speed_px_s",
    ]

    def __init__(self, scene_name, group_id="group_02", video_name="unknown",
                 fps=25, output_dir="logs"):
        os.makedirs(output_dir, exist_ok=True)
        self.scene_name  = scene_name
        self.group_id    = group_id
        self.video_name  = video_name
        self.fps         = fps
        self.crossed_ids = set()
        self.stats       = {}

        safe_scene = scene_name.replace(" ", "_")
        safe_video = os.path.splitext(os.path.basename(video_name))[0]
        self.filepath = os.path.join(output_dir, f"{safe_scene}_{safe_video}.csv")

        with open(self.filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.SCHEMA).writeheader()

    def log(self, frame_idx, track_id, class_name, bbox,
            confidence, frame_w, frame_h,
            crossed=False, direction="", speed_px_s=0.0):

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        ts = round(frame_idx / max(self.fps, 1), 3)

        row = {
            "frame":         frame_idx,
            "timestamp_sec": ts,
            "scene_name":    self.scene_name,
            "group_id":      self.group_id,
            "video_name":    self.video_name,
            "track_id":      track_id,
            "class_name":    class_name,
            "confidence":    round(float(confidence), 3),
            "bbox_x1":       x1,
            "bbox_y1":       y1,
            "bbox_x2":       x2,
            "bbox_y2":       y2,
            "cx":            cx,
            "cy":            cy,
            "frame_width":   frame_w,
            "frame_height":  frame_h,
            "crossed_line":  str(crossed).lower(),
            "direction":     direction,
            "speed_px_s":    round(float(speed_px_s), 1),
        }

        with open(self.filepath, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.SCHEMA).writerow(row)

        if crossed and track_id not in self.crossed_ids:
            self.crossed_ids.add(track_id)
            self.stats[class_name] = self.stats.get(class_name, 0) + 1

    def get_stats(self):
        return dict(self.stats)

    def get_log_path(self):
        return self.filepath