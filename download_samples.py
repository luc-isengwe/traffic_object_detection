import os
import urllib.request


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.pexels.com/",
}

SAMPLE_VIDEOS = [
    {
        "name": "traffic_scene1_bangkok.mp4",
        "dest": "data/videos/traffic_scene1_bangkok.mp4",
        "url": "https://videos.pexels.com/video-files/27694000/12207144_1920_1080_30fps.mp4",
        "description": "Bangkok busy street traffic (scene 1 — intersection)",
    },
    {
        "name": "traffic_scene2_jakarta.mp4",
        "dest": "data/videos/traffic_scene2_jakarta.mp4",
        "url": "https://videos.pexels.com/video-files/14552311/14552311-hd_1920_1080_50fps.mp4",
        "description": "Jakarta traffic jam on street (scene 2 — urban)",
    },
    {
        "name": "traffic_scene3_highway.mp4",
        "dest": "data/videos/traffic_scene3_highway.mp4",
        "url": "https://videos.pexels.com/video-files/29607724/12742244_1920_1080_50fps.mp4",
        "description": "Busy urban highway with flowing traffic (scene 3 — highway)",
    },
    {
        "name": "traffic_scene4_intersection.mp4",
        "dest": "data/videos/traffic_scene4_intersection.mp4",
        "url": "https://videos.pexels.com/video-files/30609021/13105511_2560_1440_30fps.mp4",
        "description": "Urban traffic flow at intersection (scene 4 — intersection)",
    },
]

SAMPLE_IMAGES = [
    {
        "name": "stop_sign.jpg",
        "dest": "data/images/stop_sign.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f9/STOP_sign.jpg",
        "description": "Stop sign for cascade/watershed demo",
    },
]


def _progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(int(count * block_size * 100 / total_size), 100)
        print(f"\r  Progress: {pct}%", end="", flush=True)


def download_file(url, dest, description):
    print(f"\nDownloading: {description}")
    print(f"  URL : {url}")
    print(f"  Dest: {dest}")

    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print(f"  Already exists ({os.path.getsize(dest)/1024/1024:.1f} MB), skipping.")
        return True

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                while True:
                    buf = response.read(65536)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = min(int(downloaded * 100 / total), 100)
                        mb = downloaded / 1024 / 1024
                        print(f"\r  Progress: {pct}% ({mb:.1f} MB)", end="", flush=True)
        size = os.path.getsize(dest)
        print(f"\n  Done. ({size/1024/1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    for folder in [
        "data/videos",
        "data/images",
        "data/images/train",
        "data/images/val",
        "data/labels/train",
        "data/labels/val",
    ]:
        os.makedirs(folder, exist_ok=True)

    results = {}

    for item in SAMPLE_IMAGES:
        ok = download_file(item["url"], item["dest"], item["description"])
        results[item["name"]] = ok

    for item in SAMPLE_VIDEOS:
        ok = download_file(item["url"], item["dest"], item["description"])
        results[item["name"]] = ok

    print("\n\nSummary:")
    for name, ok in results.items():
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {name}")

    print("\nTo test immediately (yolo11n.pt already downloaded):")
    print("  python main.py --model yolov11 --model_path models/yolo11n.pt \\")
    print("      --filepath data/videos/traffic_scene1_bangkok.mp4 --scene scene_1 --show")
    print("")
    print("  python main.py --model yolov11 --model_path models/yolo11n.pt \\")
    print("      --filepath data/videos/traffic_scene2_jakarta.mp4 --scene scene_2 --show")
    print("")
    print("  python main.py --model watershed --filepath data/images/stop_sign.jpg")


if __name__ == "__main__":
    main()