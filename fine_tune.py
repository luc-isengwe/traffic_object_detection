import os
import yaml
import argparse
import torch
from ultralytics import YOLO

TRAFFIC_CLASSES = ["person", "bike", "car", "motor", "bus", "truck", "rider"]

ORIGINAL_CLASSES_BDD100K = [
    '1','10','2','3','4','5','6','7','8','9',
    'bike', 'bus', 'car', 'motor', 'person', 'rider',
    'tl_green', 'tl_none', 'tl_red', 'tl_yellow', 'train', 'truck'
]

ORIG_TO_NEW = {}
for new_id, cls_name in enumerate(TRAFFIC_CLASSES):
    if cls_name in ORIGINAL_CLASSES_BDD100K:
        ORIG_TO_NEW[ORIGINAL_CLASSES_BDD100K.index(cls_name)] = new_id


def detect_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"[INFO] GPU detected: {names}")
        return "0,1" if n >= 2 else "0"
    print("[INFO] No GPU found. Training on CPU (will be slow — use Kaggle for faster training).")
    return "cpu"


def auto_batch(device):
    if device == "cpu":
        return 4
    if "," in str(device):
        return 32
    try:
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 32 if mem >= 14 else 16 if mem >= 8 else 8
    except Exception:
        return 8


def filter_label(src_path):
    if not os.path.exists(src_path):
        return []
    lines = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            orig_id = int(parts[0])
            if orig_id in ORIG_TO_NEW:
                lines.append(f"{ORIG_TO_NEW[orig_id]} {' '.join(parts[1:])}")
    return lines


def prepare_filtered_dataset(src_root, dst_root="data_filtered"):
    import glob, shutil
    print(f"[INFO] Filtering dataset from '{src_root}' -> '{dst_root}'")
    for split in ["train", "val"]:
        src_img = os.path.join(src_root, split, "images")
        src_lbl = os.path.join(src_root, split, "labels")
        dst_img = os.path.join(dst_root, split, "images")
        dst_lbl = os.path.join(dst_root, split, "labels")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        if not os.path.exists(src_img):
            print(f"  [{split}] source not found: {src_img}")
            continue
        imgs = sorted(glob.glob(f"{src_img}/*.jpg") + glob.glob(f"{src_img}/*.png"))
        kept = skipped = 0
        for img_path in imgs:
            stem    = os.path.splitext(os.path.basename(img_path))[0]
            lbl_src = os.path.join(src_lbl, stem + ".txt")
            filtered = filter_label(lbl_src)
            if not filtered:
                skipped += 1
                continue
            shutil.copy2(img_path, os.path.join(dst_img, os.path.basename(img_path)))
            with open(os.path.join(dst_lbl, stem + ".txt"), "w") as f:
                f.write("\n".join(filtered))
            kept += 1
        print(f"  [{split}] kept={kept}, skipped={skipped}")
    return dst_root


def build_yaml(data_root="data", output_path="data/traffic.yaml"):
    config = {
        "path":  os.path.abspath(data_root),
        "train": "train/images",
        "val":   "val/images",
        "nc":    len(TRAFFIC_CLASSES),
        "names": TRAFFIC_CLASSES,
    }
    os.makedirs(data_root, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[INFO] YAML written to {output_path}")
    return output_path


def fine_tune(
    base_model="yolo11n.pt",
    data_yaml="data/traffic.yaml",
    epochs=30,
    imgsz=640,
    batch=None,
    project="models",
    run_name="yolo11n_traffic_finetuned",
    device=None,
):
    device = device or detect_device()
    batch  = batch  or auto_batch(device)

    print(f"[INFO] device={device} | batch={batch} | epochs={epochs}")

    model   = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        exist_ok=True,
        patience=10,
        save=True,
        verbose=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
    )

    best = os.path.join(project, run_name, "weights", "best.pt")
    print(f"\nFine-tuning complete. Best model: {best}")
    return best


def validate(weights_path, data_yaml="data/traffic.yaml"):
    device  = detect_device()
    model   = YOLO(weights_path)
    metrics = model.val(data=data_yaml, device=device)
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")
    print("\nPer-class AP50:")
    for i, name in enumerate(TRAFFIC_CLASSES):
        try:
            print(f"  {name:10s}: {metrics.box.ap50[i]:.4f}")
        except Exception:
            pass
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on traffic classes")
    parser.add_argument("--base_model",  default="yolo11n.pt")
    parser.add_argument("--data_yaml",   default="data/traffic.yaml")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--imgsz",       type=int, default=640)
    parser.add_argument("--batch",       type=int, default=None,
                        help="Batch size (auto if not set: CPU=4, 1GPU=8-16, 2GPU=32)")
    parser.add_argument("--device",      type=str, default=None,
                        help="Device: cpu | 0 | 0,1 (auto-detected if not set)")
    parser.add_argument("--run_name",    default="yolo11n_traffic_finetuned")
    parser.add_argument("--validate",    action="store_true")
    parser.add_argument("--build_yaml",  action="store_true",
                        help="Build data/traffic.yaml (use with --data_root)")
    parser.add_argument("--data_root",   default="data",
                        help="Root of prepared dataset (with train/ and val/ subfolders)")
    parser.add_argument("--filter_dataset", type=str, default=None,
                        help="Path to raw BDD100K dataset to filter into data_filtered/")
    args = parser.parse_args()

    if args.filter_dataset:
        filtered_root = prepare_filtered_dataset(args.filter_dataset, "data_filtered")
        build_yaml(filtered_root, "data/traffic.yaml")

    elif args.build_yaml:
        build_yaml(args.data_root)

    best = fine_tune(
        base_model=args.base_model,
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.run_name,
        device=args.device,
    )

    if args.validate:
        validate(best, args.data_yaml)