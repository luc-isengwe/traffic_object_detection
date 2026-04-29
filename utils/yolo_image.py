import cv2 as cv
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt

COCO_TRAFFIC_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
}

COLORS = {
    "person": (255, 128, 0), "bicycle": (0, 200, 255), "car": (0, 255, 100),
    "motorcycle": (255, 50, 200), "bus": (50, 150, 255), "truck": (200, 50, 50),
}


class YOLOImageDetector:
    def __init__(self, model_path, conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image_path, target_classes=None):
        img = cv.imread(image_path)
        assert img is not None, f"Cannot read image: {image_path}"
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result_img = img_rgb.copy()

        class_filter = [k for k, v in COCO_TRAFFIC_CLASSES.items()
                        if target_classes is None or v in target_classes]

        results = self.model(img, classes=class_filter, conf=self.conf_threshold, verbose=False)[0]

        found = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = COCO_TRAFFIC_CLASSES.get(cls_id, "unknown")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            color = COLORS.get(cls_name, (180, 180, 180))
            cv.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
            cv.putText(result_img, f"{cls_name} {conf:.2f}", (x1, y1 - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            found.append({"class": cls_name, "conf": conf, "bbox": [x1, y1, x2, y2]})

        self._plot(img_rgb, result_img, found)
        return found

    def _plot(self, original, detected, found):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(original)
        axes[1].imshow(detected)
        axes[0].set_title("Original")
        axes[1].set_title(f"Detected ({len(found)} objects)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("yolo_image_result.png", dpi=150)
        plt.show()
        for d in found:
            print(f"  {d['class']} — conf: {d['conf']:.3f} — bbox: {d['bbox']}")