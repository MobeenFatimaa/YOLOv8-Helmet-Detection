import os
import tempfile
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HelmetGuard AI | Compliance Dashboard",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CYBER-DARK GLASSMORPHISM CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a0b12, #0a0a0f 60%);
        color: #f0f2f6;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 15, 22, 0.85);
        border-right: 1px solid rgba(255, 42, 85, 0.2);
    }
    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 42, 85, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a8b8 !important;
        font-size: 0.9rem !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
    }
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .accent-text {
        color: #ff2a55;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ff2a55 0%, #b81435 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(255, 42, 85, 0.6);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

MODEL_PATH = "runs/detect/helmet_custom_model-5/weights/best.pt"

# --- HEADER ---
st.markdown("<h1>🪖 HelmetGuard <span class='accent-text'>AI</span> — Real-Time Safety Analytics</h1>", unsafe_allow_html=True)
st.markdown("Automated motorcycle helmet compliance monitoring, telemetry tracking, and violation logging powered by YOLOv8.")
st.divider()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
    
    st.markdown("### 📹 Video Input Source")
    input_mode = st.radio("Select Source:", ["Use Default (test_video1.mp4)", "Upload Custom Video"])
    
    video_path = None
    if input_mode == "Upload Custom Video":
        uploaded_file = st.file_uploader("Upload MP4, AVI, or MOV", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
    else:
        if os.path.exists("test_video1.mp4"):
            video_path = "test_video1.mp4"
        else:
            st.warning("'test_video1.mp4' not found in project root.")

    start_btn = st.button("🚀 Run Detection & Analytics", use_container_width=True)

# --- MAIN DASHBOARD LAYOUT ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model weights not found at `{MODEL_PATH}`. Please check your folder path.")
    st.stop()

model = load_model(MODEL_PATH)

# Top KPI Placeholders
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
metric_total = kpi1.empty()
metric_with = kpi2.empty()
metric_without = kpi3.empty()
metric_compliance = kpi4.empty()

# Initial default metrics
metric_total.metric("Total Detections", "0")
metric_with.metric("With Helmet ✅", "0")
metric_without.metric("Without Helmet ⚠️", "0")
metric_compliance.metric("Compliance Rate", "0.0%")

st.markdown("<br>", unsafe_allow_html=True)

# Video & Live Feed Columns
col_video, col_live = st.columns([1.3, 1])
with col_video:
    st.markdown("#### 🎥 Live Inference Feed")
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

with col_live:
    st.markdown("#### 📊 Live Detection Distribution")
    live_chart_placeholder = st.empty()

# --- PROCESSING LOOP ---
if start_btn and video_path:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_video_path = "dashboard_output.mp4"
    # Use 'avc1' or 'mp4v' for browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    telemetry_data = []
    violation_snapshots = []
    
    total_with_helmet = 0
    total_without_helmet = 0
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Count classes in current frame
        frame_with = 0
        frame_without = 0
        confidences = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id].lower()
            confidences.append(conf)

            if "without" in cls_name:
                frame_without += 1
                total_without_helmet += 1
                # Capture up to 6 violation crops for the gallery
                if len(violation_snapshots) < 6 and frame_idx % 15 == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[max(0, y1-20):min(height, y2+20), max(0, x1-20):min(width, x2+20)]
                    if crop.size > 0:
                        violation_snapshots.append((frame_idx, conf, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            else:
                frame_with += 1
                total_with_helmet += 1

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        telemetry_data.append({
            "Frame": frame_idx,
            "Time (s)": round(frame_idx / fps, 2),
            "With Helmet": frame_with,
            "Without Helmet": frame_without,
            "Avg Confidence": round(avg_conf, 2)
        })

        # Update UI every 3 frames for smooth performance
        if frame_idx % 3 == 0 or frame_idx == total_frames:
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            total_det = total_with_helmet + total_without_helmet
            comp_rate = (total_with_helmet / total_det * 100) if total_det > 0 else 100.0

            metric_total.metric("Total Detections", f"{total_det:,}")
            metric_with.metric("With Helmet ✅", f"{total_with_helmet:,}")
            metric_without.metric("Without Helmet ⚠️", f"{total_without_helmet:,}")
            metric_compliance.metric("Compliance Rate", f"{comp_rate:.1f}%")

            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            # Update Live Donut Chart
            fig_donut = go.Figure(data=[go.Pie(
                labels=["With Helmet", "Without Helmet"],
                values=[total_with_helmet, total_without_helmet],
                hole=0.6,
                marker_colors=["#00e5a0", "#ff2a55"]
            )])
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f2f6"),
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                showlegend=True
            )
            live_chart_placeholder.plotly_chart(fig_donut, use_container_width=True, key=f"donut_{frame_idx}")

    cap.release()
    out.release()
    st.success("✅ Video processing complete! Explore the detailed analytics below.")

    # --- POST-PROCESSING ANALYTICS & DOWNLOADS ---
    st.divider()
    df = pd.DataFrame(telemetry_data)

    tab1, tab2, tab3 = st.tabs(["📈 Time-Series Telemetry", "🚨 Violation Snapshots", "📥 Export & Video"])

    with tab1:
        st.markdown("#### Detections Over Video Timeline")
        fig_line = px.area(
            df,
            x="Time (s)",
            y=["With Helmet", "Without Helmet"],
            color_discrete_map={"With Helmet": "#00e5a0", "Without Helmet": "#ff2a55"},
            template="plotly_dark"
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis_title="Video Timestamp (Seconds)",
            yaxis_title="Detected Count per Frame"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.dataframe(df, use_container_width=True, height=250)

    with tab2:
        st.markdown("#### Flagged Non-Compliance Captures (`Without Helmet`)")
        if violation_snapshots:
            cols = st.columns(3)
            for idx, (f_num, conf_score, img_crop) in enumerate(violation_snapshots):
                with cols[idx % 3]:
                    st.image(img_crop, caption=f"Frame #{f_num} | Confidence: {conf_score:.2f}", use_container_width=True)
        else:
            st.info("🎉 Zero helmet violations detected in this video!")

    with tab3:
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.markdown("#### Download Processed Video")
            if os.path.exists(output_video_path):
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Annotated MP4",
                        data=file,
                        file_name="helmet_detection_output.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
        with col_dl2:
            st.markdown("#### Download Telemetry Data")
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Frame Analytics (CSV)",
                data=csv_data,
                file_name="helmet_telemetry.csv",
                mime="text/csv",
                use_container_width=True
            )
