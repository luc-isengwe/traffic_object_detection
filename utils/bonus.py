import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


class SpeedEstimator:
    def __init__(self, fps):
        self.fps = fps
        self.prev_positions = {}
        self.speeds = {}

    def update(self, track_id, cx, cy, frame_idx):
        if track_id in self.prev_positions:
            prev_cx, prev_cy, prev_frame = self.prev_positions[track_id]
            dt = (frame_idx - prev_frame) / max(self.fps, 1)
            if dt > 0:
                dist_px = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                self.speeds[track_id] = round(dist_px / dt, 1)
        self.prev_positions[track_id] = (cx, cy, frame_idx)
        return self.speeds.get(track_id, 0.0)


class DirectionPredictor:
    def __init__(self, history_len=8):
        self.history = defaultdict(list)
        self.history_len = history_len

    def update(self, track_id, cx, cy):
        self.history[track_id].append((cx, cy))
        if len(self.history[track_id]) > self.history_len:
            self.history[track_id].pop(0)
        return self._compute(track_id)

    def _compute(self, track_id):
        pts = self.history[track_id]
        if len(pts) < 2:
            return ""
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        if abs(dx) < 2 and abs(dy) < 2:
            return ""
        if abs(dy) >= abs(dx):
            return "down" if dy > 0 else "up"
        return "right" if dx > 0 else "left"

    def get_arrow(self, track_id):
        arrows = {"up": "↑", "down": "↓", "left": "←", "right": "→", "": "•"}
        return arrows.get(self._compute(track_id), "•")


class HeatmapGenerator:
    def __init__(self, frame_h, frame_w):
        self.h = frame_h
        self.w = frame_w
        self.accumulator = np.zeros((frame_h, frame_w), dtype=np.float32)

    def update(self, tracks):
        for trk in tracks:
            cx = int(np.clip((trk[0]+trk[2])/2, 0, self.w-1))
            cy = int(np.clip((trk[1]+trk[3])/2, 0, self.h-1))
            self.accumulator[cy, cx] += 1.0

    def get_overlay(self, frame, alpha=0.4):
        if self.accumulator.max() == 0:
            return frame
        norm    = cv.normalize(self.accumulator, None, 0, 255, cv.NORM_MINMAX)
        blurred = cv.GaussianBlur(norm.astype(np.uint8), (51, 51), 0)
        heatmap = cv.applyColorMap(blurred, cv.COLORMAP_JET)
        return cv.addWeighted(frame, 1-alpha, heatmap, alpha, 0)

    def save_plot(self, output_path="outputs/heatmap.png", scene_name="scene"):
        fig, ax = plt.subplots(figsize=(10, 6))
        blurred = cv.GaussianBlur(self.accumulator, (51, 51), 0)
        im = ax.imshow(blurred, cmap="hot", interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="Detection density")
        ax.set_title(f"Object Position Heatmap — {scene_name}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
#######################################################################

def plot_traffic_flow(count_history, scene_name, output_path="outputs/traffic_flow.png"):
    if not count_history:
        return
    import pandas as pd
    df = pd.DataFrame(count_history).fillna(0)
    if "frame" not in df.columns:
        return
    df = df.set_index("frame")
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=2)
    ax.set_title(f"Traffic Flow Over Time — {scene_name}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cumulative count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_class_distribution(counts, scene_name, output_path="outputs/class_dist.png"):
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(counts.keys())
    values  = list(counts.values())
    colors  = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Object Class Distribution — {scene_name}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
