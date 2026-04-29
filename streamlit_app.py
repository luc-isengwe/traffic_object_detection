import streamlit as st
import cv2 as cv
import numpy as np
import tempfile
import os
import pandas as pd
import time
import glob
from pathlib import Path

from utils.yolo_video import YOLOVideoDetector, FINETUNED_MODEL, COCO_TRAFFIC_CLASSES
from utils.ssd_video import SSDVideoDetector

st.set_page_config(
    page_title="Traffic Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1 { font-family: 'Space Mono', monospace !important; color: #00e5c8 !important; font-size: 2.5rem !important; }
h2, h3 { font-family: 'Space Mono', monospace !important; color: #00e5c8 !important; }

.nav-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #00e5c8;
    letter-spacing: 2px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e2535;
}
.badge-live {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    background: rgba(255,50,50,0.15); border: 1px solid #ff3232;
    color: #ff3232; font-family: 'Space Mono', monospace;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;
}
.badge-upload {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    background: rgba(0,229,200,0.1); border: 1px solid #00e5c8;
    color: #00e5c8; font-family: 'Space Mono', monospace;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;
}
.count-card {
    background: #131826; border: 1px solid #1e2535;
    border-radius: 10px; padding: 14px; text-align: center; margin-bottom: 8px;
}
.count-val { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #00e5c8; font-weight: 700; }
.count-lbl { color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.no-obj {
    background: rgba(255,107,53,0.1); border: 1px solid #ff6b35;
    border-radius: 8px; padding: 10px 14px; color: #ff6b35;
    font-family: 'Space Mono', monospace; font-size: 0.75rem; margin-top: 8px;
}
.stat-card {
    background: #131826; border: 1px solid #1e2535;
    border-radius: 10px; padding: 20px; text-align: center;
}
.stat-val { font-family: 'Space Mono', monospace; font-size: 2.2rem; color: #00e5c8; font-weight: 700; }
.stat-lbl { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }
.scene-row { padding: 10px 0; border-bottom: 1px solid #1e2535; }
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; letter-spacing: 1px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace; font-size: 0.8rem; letter-spacing: 1px;
}
section[data-testid="stSidebar"] { background: #f0f0f0; border-right: 1px solid #1e2535; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="nav-title">TRAFFIC_MONITOR</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Detection", "Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if page == "Detection":
        st.markdown("**Model**")
        model_choice = st.selectbox(
            "Model",
            [
                "YOLOv11 — fine-tuned",
                "YOLOv11 — base",
                "SSD MobileNet",
            ],
            label_visibility="collapsed",
        )

        if model_choice == "YOLOv11 — fine-tuned":
            #model_path = st.text_input("Weights path", value=FINETUNED_MODEL)
            model_path = st.text_input("Weights path", value="models/yolo11n_traffic_finetuned/weights/best.pt")
            use_ssd = False
        elif model_choice == "YOLOv11 — base":
            model_path = st.text_input("Weights path", value="models/yolo11n.pt")
            use_ssd = False
        else:
            prototxt = st.text_input("Prototxt", value="models/MobileNetSSD_deploy.prototxt")
            ssd_weights = st.text_input("Caffemodel", value="models/MobileNetSSD_deploy.caffemodel")
            model_path = None
            use_ssd = True

        st.markdown("---")
        st.markdown("**Scene name**")
        scene_name = st.text_input("Scene", value="scene_1", label_visibility="collapsed")
        conf = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05)

        st.markdown("**Object classes**")
        all_classes = list(COCO_TRAFFIC_CLASSES.values()) + ["Rider"]
        selected_classes = []
        cols = st.columns(2)
        for i, cls in enumerate(all_classes):
            with cols[i % 2]:
                if st.checkbox(cls.capitalize(), value=True, key=f"cls_{cls}"):
                    selected_classes.append(cls)

        st.markdown("---")
        st.markdown("**Live camera**")
        camera_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0)
        phone_url = st.text_input(
            "Phone URL",
            placeholder="http://192.168.x.x:8080/video",
            help="Android: IP Webcam app. iPhone: DroidCam.",
        )
        skip_live = st.slider("Process every N frames (live)", 1, 5, 2)


def build_detector(scene, classes, conf_threshold):
    if use_ssd:
        return SSDVideoDetector(
            prototxt_path=prototxt,
            weights_path=ssd_weights,
            scene_name=scene,
            target_classes=classes,
            conf_threshold=conf_threshold,
        )
    actual_path = model_path
    if actual_path and not os.path.exists(actual_path):
        st.warning(
            f"Model not found at `{actual_path}`. Falling back to `models/yolo11n.pt`."
        )
        actual_path = "models/yolo11n.pt"
    return YOLOVideoDetector(
        model_path=actual_path,
        scene_name=scene,
        target_classes=classes,
        conf_threshold=conf_threshold,
    )


def render_counts(counts, placeholder):
    html = ""
    for cls in selected_classes:
        n = counts.get(cls, 0)
        html += f"""<div class="count-card">
            <div class="count-val">{n}</div>
            <div class="count-lbl">{cls}</div>
        </div>"""
    total = sum(counts.values())
    html += f"""<div class="count-card" style="border-color:#00e5c8">
        <div class="count-val" style="color:#fff">{total}</div>
        <div class="count-lbl">TOTAL</div>
    </div>"""
    placeholder.markdown(html, unsafe_allow_html=True)


def show_log_and_chart(detector, count_history):
    log_path = detector.logger.get_log_path()
    if os.path.exists(log_path):
        st.markdown("#### Detection Log")
        df = pd.read_csv(log_path)
        st.dataframe(df, use_container_width=True)
        with open(log_path, "rb") as f:
            st.download_button(
                "Download CSV log",
                f,
                file_name=os.path.basename(log_path),
                mime="text/csv",
            )
    if count_history:
        st.markdown("#### Traffic Flow Over Time")
        df_hist = pd.DataFrame(count_history).set_index("frame").fillna(0)
        st.line_chart(df_hist)


def parse_log(filepath):
    counts = {}
    temporal = {}
    try:
        with open(filepath, newline="") as f:
            reader = pd.read_csv(f)
            crossed = reader[reader["crossed"] == 1]
            for _, row in crossed.iterrows():
                cls = row["class_name"]
                counts[cls] = counts.get(cls, 0) + 1
            for _, row in reader.iterrows():
                bucket = int(row["frame"]) // 30
                temporal[bucket] = temporal.get(bucket, 0) + 1
    except Exception:
        pass
    return counts, temporal


if page == "Detection":
    st.title("Detection & Tracking")
    st.caption("Upload a video or use your phone camera to detect and count traffic objects in real time.")

    tab_upload, tab_live = st.tabs(["Upload Video", "Live Camera"])

    with tab_upload:
        st.markdown('<div class="badge-upload">VIDEO UPLOAD MODE</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose a traffic video",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )

        if uploaded:
            st.video(uploaded)
            skip_upload = st.slider("Process every N frames", 1, 5, 1, key="skip_upload")

            if st.button("Start Detection", type="primary", use_container_width=True, key="btn_upload"):
                if not selected_classes:
                    st.error("Select at least one class in the sidebar.")
                    st.stop()

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                    tmp.write(uploaded.read())
                    video_path = tmp.name

                col_vid, col_counts = st.columns([3, 1])
                with col_vid:
                    frame_ph = st.empty()
                    status_ph = st.empty()
                with col_counts:
                    st.markdown("**Live Counts**")
                    counts_ph = st.empty()
                    no_obj_ph = st.empty()

                detector = build_detector(scene_name, selected_classes, conf)
                count_history = []
                frame_idx = 0
                status_ph.info("Detection running...")

                for frame_bytes, counts in detector.stream_frames(video_path):
                    if frame_idx % skip_upload != 0:
                        frame_idx += 1
                        continue
                    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame_rgb = cv.cvtColor(cv.imdecode(arr, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
                    frame_ph.image(frame_rgb, channels="RGB", use_container_width=True)
                    render_counts(counts, counts_ph)
                    if sum(counts.values()) == 0:
                        no_obj_ph.markdown('<div class="no-obj">NO OBJECTS DETECTED IN THIS FRAME</div>', unsafe_allow_html=True)
                    else:
                        no_obj_ph.empty()
                    count_history.append({"frame": frame_idx, **counts})
                    frame_idx += 1

                status_ph.success("Detection complete. Go to **Dashboard** to see aggregated results across all scenes.")
                os.unlink(video_path)
                show_log_and_chart(detector, count_history)
        else:
            st.info("Upload a video file above to start.")

    with tab_live:
        st.markdown('<div class="badge-live">● LIVE CAMERA MODE</div>', unsafe_allow_html=True)

        st.info("""
**To use your phone camera:**
1. Install **IP Webcam** (Android — Play Store) or **DroidCam** (iPhone)
2. Open the app → tap **Start server**
3. Enter the URL shown (e.g. `http://192.168.1.5:8080/video`) in the sidebar
4. Your phone and PC must be on the **same WiFi network**

**Webcam only:** Leave URL empty, set camera index to 0.
        """)

        if "live_running" not in st.session_state:
            st.session_state.live_running = False

        col_start, col_stop = st.columns(2)
        if col_start.button("Start Live Detection", type="primary", use_container_width=True):
            st.session_state.live_running = True
        if col_stop.button("Stop", use_container_width=True):
            st.session_state.live_running = False

        if st.session_state.live_running:
            if not selected_classes:
                st.error("Select at least one class in the sidebar.")
                st.session_state.live_running = False
                st.stop()

            source = phone_url.strip() if phone_url.strip() else int(camera_index)
            cap = cv.VideoCapture(source)

            if not cap.isOpened():
                st.error(f"Cannot open camera: `{source}`. Check that IP Webcam is running and the URL is correct.")
                st.session_state.live_running = False
                st.stop()

            col_vid, col_counts = st.columns([3, 1])
            with col_vid:
                frame_ph = st.empty()
                status_ph = st.empty()
                fps_ph = st.empty()
            with col_counts:
                st.markdown("**Live Counts**")
                counts_ph = st.empty()
                no_obj_ph = st.empty()

            detector = build_detector(scene_name, selected_classes, conf)
            h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            line_y = int(h * 0.55)
            class_filter = detector._get_class_id_filter()
            class_map = {}
            frame_idx = 0
            prev_time = time.time()
            count_history = []

            status_ph.success("Camera connected. Press **Stop** to end session.")

            while st.session_state.live_running:
                ret, frame = cap.read()
                if not ret:
                    status_ph.error("Lost connection to camera.")
                    break
                if frame_idx % skip_live == 0:
                    tracks, class_map = detector.process_frame(frame, frame_idx, line_y, class_filter, class_map)
                    frame = detector._draw_overlay(frame, tracks, class_map, line_y)
                    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    frame_ph.image(frame_rgb, channels="RGB", use_container_width=True)
                    render_counts(detector.count_per_class, counts_ph)
                    if sum(detector.count_per_class.values()) == 0:
                        no_obj_ph.markdown('<div class="no-obj">NO OBJECTS DETECTED</div>', unsafe_allow_html=True)
                    else:
                        no_obj_ph.empty()
                    curr_time = time.time()
                    fps_ph.caption(f"FPS: {1.0 / max(curr_time - prev_time, 0.001):.1f}")
                    prev_time = curr_time
                    count_history.append({"frame": frame_idx, **detector.count_per_class})
                frame_idx += 1

            cap.release()
            st.session_state.live_running = False
            show_log_and_chart(detector, count_history)


elif page == "Dashboard":
    st.title("Global Traffic Dashboard")
    st.caption("Aggregated detection results across all processed scenes.")

    log_files = sorted(glob.glob("logs/*.csv"))

    if not log_files:
        st.warning("No detection logs found. Run a detection first, then come back here.")
        st.stop()

    all_scenes = []
    for filepath in log_files:
        stem = Path(filepath).stem
        parts = stem.rsplit("_", 2)
        scene = parts[0] if len(parts) >= 3 else stem
        counts, temporal = parse_log(filepath)
        total = sum(counts.values())
        all_scenes.append({
            "scene": scene,
            "file": Path(filepath).name,
            "counts": counts,
            "temporal": temporal,
            "total": total,
        })

    total_objects = sum(s["total"] for s in all_scenes)
    all_counts = {}
    for s in all_scenes:
        for cls, n in s["counts"].items():
            all_counts[cls] = all_counts.get(cls, 0) + n
    top_class = max(all_counts, key=all_counts.get) if all_counts else "—"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-val">{total_objects}</div>
            <div class="stat-lbl">Total Objects Counted</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-val">{len(all_scenes)}</div>
            <div class="stat-lbl">Scenes Processed</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-val">{top_class}</div>
            <div class="stat-lbl">Most Detected Class</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Object Class Distribution")
        if all_counts:
            df_classes = pd.DataFrame(
                list(all_counts.items()), columns=["Class", "Count"]
            ).set_index("Class")
            st.bar_chart(df_classes)

    with col_chart2:
        st.markdown("#### Total Counts per Scene")
        if all_scenes:
            df_scenes = pd.DataFrame(
                [{"Scene": s["scene"], "Total": s["total"]} for s in all_scenes]
            ).set_index("Scene")
            st.bar_chart(df_scenes)

    st.markdown("---")
    st.markdown("#### Scene Breakdown")

    CLASS_ORDER = ["car", "person", "bus", "truck", "moto", "bike", "rider"]
    header_cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
    headers = ["Scene", "Car", "Person", "Bus", "Truck", "Moto", "Bike", "Rider" "Total"]
    for col, h in zip(header_cols, headers):
        col.markdown(f"**{h}**")

    st.markdown("<hr style='border-color:#1e2535;margin:4px 0'>", unsafe_allow_html=True)

    for s in all_scenes:
        row_cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
        row_cols[0].markdown(f"`{s['scene']}`")
        for i, cls in enumerate(CLASS_ORDER):
            row_cols[i + 1].markdown(str(s["counts"].get(cls, 0)))
        row_cols[7].markdown(f"**{s['total']}**")

    st.markdown("---")
    st.markdown("#### Traffic Intensity Over Time (all scenes)")

    all_temporal = {}
    for s in all_scenes:
        for bucket, count in s["temporal"].items():
            all_temporal[bucket] = all_temporal.get(bucket, 0) + count

    if all_temporal:
        df_temporal = pd.DataFrame(
            sorted(all_temporal.items()), columns=["Time bucket (30 frames)", "Detections"]
        ).set_index("Time bucket (30 frames)")
        st.line_chart(df_temporal)

    st.markdown("---")
    st.markdown("#### Download Logs")
    for s in all_scenes:
        filepath = os.path.join("logs", s["file"])
        with open(filepath, "rb") as f:
            st.download_button(
                label=f"Download — {s['scene']}",
                data=f,
                file_name=s["file"],
                mime="text/csv",
                key=f"dl_{s['file']}",
            )