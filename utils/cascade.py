import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


class Classifier:
    def __init__(self, filepath, model_path):
        self.filepath = filepath
        self.model_path = model_path

    def forward(self):
        img = cv.imread(self.filepath)
        assert img is not None, f"Cannot read image: {self.filepath}"
        self.img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.img_detected = self.img_rgb.copy()
        model = cv.CascadeClassifier(self.model_path)
        found = model.detectMultiScale(img_gray, minSize=(20, 20))
        return found

    def plot(self):
        found = self.forward()
        for (x, y, w, h) in found:
            cv.rectangle(self.img_detected, (x, y), (x + w, y + h), (0, 255, 0), 4)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(self.img_rgb)
        axes[1].imshow(self.img_detected)
        axes[0].set_title("Original")
        axes[1].set_title(f"Detected ({len(found)} objects)")
        axes[0].axis("off")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig("cascade_result.png", dpi=150)
        plt.show()
        print(f"Found {len(found)} object(s).")
        return found


class WatershedSegmenter:
    def __init__(self, filepath):
        self.filepath = filepath

    def process(self):
        img = cv.imread(self.filepath)
        assert img is not None, f"Cannot read image: {self.filepath}"
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        _, otsu_thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(otsu_thresh, cv.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv.watershed(img, markers)

        result = img.copy()
        result[markers == -1] = [0, 0, 255]

        self._plot(img, otsu_thresh, result)
        return markers

    def _plot(self, original, otsu, watershed):
        original_rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB)
        watershed_rgb = cv.cvtColor(watershed, cv.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original")
        axes[1].imshow(otsu, cmap="gray")
        axes[1].set_title("Otsu Threshold")
        axes[2].imshow(watershed_rgb)
        axes[2].set_title("Watershed Segmentation")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("watershed_result.png", dpi=150)
        plt.show()