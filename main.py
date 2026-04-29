import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Traffic object detection and tracking")
    parser.add_argument("--model", type=str, choices=["cascade", "watershed", "yolov11", "ssd"],
                        default="yolov11", help="Detection model to use")
    parser.add_argument("--filepath", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--model_path", type=str, default="models/yolo11n.pt", help="Path to model weights")
    parser.add_argument("--scene", type=str, default="scene_1", help="Scene name for logging")
    parser.add_argument("--classes", type=str, default="car,person,bus,truck,motorcycle,bicycle",
                        help="Comma-separated list of target classes")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output video path (optional)")
    parser.add_argument("--show", action="store_true", help="Display video while processing")
    parser.add_argument("--prototxt", type=str, default="models/MobileNetSSD_deploy.prototxt",
                        help="SSD prototxt path")
    parser.add_argument("--weights", type=str, default="models/MobileNetSSD_deploy.caffemodel",
                        help="SSD caffemodel path")
    return parser.parse_args()


def main():
    args = parse_args()
    target_classes = [c.strip() for c in args.classes.split(",")]

    if args.model == "cascade":
        from utils.cascade import Classifier
        clf = Classifier(args.filepath, args.model_path)
        clf.plot()

    elif args.model == "watershed":
        from utils.cascade import WatershedSegmenter
        seg = WatershedSegmenter(args.filepath)
        seg.process()

    elif args.model == "yolov11":
        filepath = args.filepath
        if filepath.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            from utils.yolo_image import YOLOImageDetector
            det = YOLOImageDetector(args.model_path, args.conf)
            found = det.detect(filepath, target_classes)
            print(f"\nDetected {len(found)} object(s).")
        else:
            from utils.yolo_video import YOLOVideoDetector
            det = YOLOVideoDetector(args.model_path, args.scene, target_classes, args.conf)
            counts, log_path = det.process(filepath, output_path=args.output, show=args.show)
            print("\nCounting results:")
            for cls, n in counts.items():
                print(f"  {cls}: {n}")
            print(f"Log saved to: {log_path}")

    elif args.model == "ssd":
        from utils.ssd_video import SSDVideoDetector
        det = SSDVideoDetector(args.prototxt, args.weights, args.scene, target_classes, args.conf)
        counts, log_path = det.process(args.filepath, output_path=args.output, show=args.show)
        print("\nCounting results:")
        for cls, n in counts.items():
            print(f"  {cls}: {n}")
        print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()