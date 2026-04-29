import os
import urllib.request
import subprocess
import sys


MODELS = [
    {
        "name": "yolo11n.pt",
        "dest": "models/yolo11n.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "description": "YOLOv11 nano — base model (pre-trained on COCO)",
    },
    {
        "name": "MobileNetSSD_deploy.prototxt",
        "dest": "models/MobileNetSSD_deploy.prototxt",
        "url": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/923b3128f25262b5010cef67e4fb9e4b6728ae7b/voc/MobileNetSSD_deploy.prototxt",
        "description": "SSD MobileNet — network config (pinned commit)",
    },
    {
        "name": "haarcascade_car.xml",
        "dest": "models/stop_data.xml",
        "url": "https://raw.githubusercontent.com/souravdeyone/OpenCV-Reference/master/Haarcascades/haarcascade_car.xml",
        "description": "Haar cascade — car detection (used as cascade model)",
    },
]

SSD_CAFFEMODEL = {
    "name": "MobileNetSSD_deploy.caffemodel",
    "dest": "models/MobileNetSSD_deploy.caffemodel",
    "kaggle_note": "https://www.kaggle.com/datasets/hiteshkumars/mobilenetssd-deploycaffemodel",
    "github_lfs": "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel",
    "description": "SSD MobileNet — weights (~23MB)",
}


def _progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(int(count * block_size * 100 / total_size), 100)
        print(f"\r  Progress: {pct}%", end="", flush=True)


def download_file(url, dest, description):
    print(f"\nDownloading: {description}")
    print(f"  URL : {url}")
    print(f"  Dest: {dest}")
    if os.path.exists(dest):
        print(f"  Already exists, skipping.")
        return True
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        size = os.path.getsize(dest)
        print(f"\n  Done. ({size/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_caffemodel():
    dest = SSD_CAFFEMODEL["dest"]
    description = SSD_CAFFEMODEL["description"]
    print(f"\nDownloading: {description}")

    if os.path.exists(dest):
        print(f"  Already exists, skipping.")
        return True

    url = SSD_CAFFEMODEL["github_lfs"]
    print(f"  Trying GitHub LFS: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                chunk = 8192
                while True:
                    buf = response.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = min(int(downloaded * 100 / total), 100)
                        print(f"\r  Progress: {pct}%", end="", flush=True)
        size = os.path.getsize(dest)
        if size < 1_000_000:
            os.remove(dest)
            raise Exception(f"Downloaded file too small ({size} bytes) — probably a redirect page, not the model.")
        print(f"\n  Done. ({size/1024/1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)

    print("\n  Trying gdown (Google Drive)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
        gdown.download(id="0B3gersZ2cHIxRm5PMWRoTkdHdHc", output=dest, quiet=False)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            return True
        raise Exception("gdown failed or file too small")
    except Exception as e:
        print(f"  gdown also failed: {e}")

    print(f"""
  MANUAL DOWNLOAD REQUIRED for caffemodel:
  Option 1 — Kaggle (free account):
    {SSD_CAFFEMODEL['kaggle_note']}
    Download MobileNetSSD_deploy.caffemodel and place at:
    {dest}

  Option 2 — Direct page (may work in browser):
    https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view
""")
    return False


def main():
    os.makedirs("models", exist_ok=True)

    results = {}
    for m in MODELS:
        ok = download_file(m["url"], m["dest"], m["description"])
        results[m["name"]] = ok

    ok = download_caffemodel()
    results[SSD_CAFFEMODEL["name"]] = ok

    print("\n\nSummary:")
    for name, ok in results.items():
        status = "OK" if ok else "MISSING — see instructions above"
        print(f"  [{status}] models/{name}")

    print("\nNote: Fine-tuned model will be created at:")
    print("  models/yolo11n_traffic_finetuned/weights/best.pt")
    print("  Run: python fine_tune.py --epochs 30 --batch 8")
    print("\nYOLO is the primary model. SSD is optional.")


if __name__ == "__main__":
    main()