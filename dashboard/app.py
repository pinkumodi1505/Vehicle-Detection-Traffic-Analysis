import hashlib

import sys
import subprocess
import subprocess

import tempfile

import time

from pathlib import Path



import cv2

import numpy as np

import pandas as pd

import plotly.express as px

import streamlit as st



HERE = Path(__file__).resolve()

PROJECT_ROOT = HERE.parent.parent if HERE.parent.name.lower() == "dashboard" else HERE.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))



from PINKU.ai_yolo.vehicle_detector import VehicleDetector

from src.video.video_processor import VideoProcessor



st.set_page_config(

    page_title="TrafficVision",

    page_icon="🚦",

    layout="wide",

    initial_sidebar_state="expanded",

)



for key, default in {

    "theme": "Dark",

    "page": "Overview",

    "video_result": None,

    "image_result": None,

    "webcam_result": None,

}.items():

    if key not in st.session_state:

        st.session_state[key] = default



dark = st.session_state.theme == "Dark"

if dark:

    C = {

        "bg": "#080B12", "sidebar": "#0D121D", "panel": "#111824",

        "panel2": "#172235", "border": "#252F40", "text": "#F4F6FB",

        "muted": "#A0AEC2", "accent": "#4F8CFF", "blue": "#76A5FF",

        "green": "#42D6AD", "orange": "#F6B94C", "red": "#FF7188",

        "purple": "#B69AFF",

    }

else:

    C = {

        "bg": "#F4F6FA", "sidebar": "#FFFFFF", "panel": "#FFFFFF",

        "panel2": "#EEF2F8", "border": "#DCE3ED", "text": "#182438",

        "muted": "#596A82", "accent": "#3478F6", "blue": "#2868D7",

        "green": "#16845A", "orange": "#995400", "red": "#B82940",

        "purple": "#6A49B8",

    }



st.markdown(f"""

<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{{--bg:{C['bg']};--panel:{C['panel']};--panel2:{C['panel2']};--border:{C['border']};--text:{C['text']};--muted:{C['muted']};--accent:{C['accent']};}}

html, body, [class*="css"], .stApp{{font-family:'DM Sans',sans-serif!important;font-size:16px}}

.stApp{{background:var(--bg);color:var(--text)}}

[data-testid="stHeader"]{{background:transparent}}

[data-testid="stSidebar"]{{background:{C['sidebar']};border-right:1px solid var(--border)}}

[data-testid="stSidebar"]>div{{padding-top:1.1rem}}

.block-container{{max-width:1450px;padding-top:1.5rem;padding-bottom:3rem}}

h1,h2,h3,h4{{font-family:'Space Grotesk',sans-serif!important;color:var(--text)!important;letter-spacing:-.35px}}

h1{{font-size:2.35rem!important}} h2{{font-size:1.7rem!important}} h3{{font-size:1.28rem!important}}

p,li,label,div[data-testid="stMarkdownContainer"]{{line-height:1.6}}

[data-testid="stCaptionContainer"]{{color:{C['muted']};font-size:.9rem}}

[data-testid="stSidebar"] label, [data-testid="stSidebar"] p{{font-size:1rem!important}}

.stButton>button,.stDownloadButton>button{{border-radius:10px;min-height:45px;font-weight:700;font-size:1rem;border:1px solid var(--border);transition:transform .18s ease,box-shadow .18s ease,background .18s ease}}

.stButton>button:hover,.stDownloadButton>button:hover{{transform:translateY(-2px);box-shadow:0 7px 20px #0002;border-color:var(--accent)}}

.stButton>button[kind="primary"]{{background:var(--accent);color:#071B19;border:0}}

div[data-testid="stMetric"]{{background:var(--panel);border:1px solid var(--border);padding:15px;border-radius:13px}}

div[data-testid="stFileUploader"]{{background:var(--panel);border:1px dashed {C['blue']};border-radius:14px;padding:9px}}

div[data-testid="stFileUploader"] section{{background:transparent}}

hr{{border-color:var(--border)}}

.kicker{{font-size:.78rem;color:var(--accent);font-weight:700;letter-spacing:1.25px;text-transform:uppercase}}

.hero{{background:linear-gradient(115deg,{C['panel']} 0%,{C['panel2']} 78%,{C['panel']} 100%);border:1px solid var(--border);border-radius:18px;padding:28px 30px;margin-bottom:24px;animation:fadeUp .55s ease both}}

.hero h1{{font-size:2.55rem!important;margin:10px 0 8px;line-height:1.16}}

.hero p{{color:{C['muted']};margin:0;font-size:1.08rem}}

.panel{{background:var(--panel);border:1px solid var(--border);border-radius:15px;padding:20px;transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;animation:fadeUp .45s ease both}}

.panel:hover,.team-card:hover{{transform:translateY(-3px);border-color:{C['accent']};box-shadow:0 10px 28px #0002}}

.panel-title{{font-family:'Space Grotesk',sans-serif;font-weight:700;color:var(--text);font-size:1.15rem}}

.mini-label{{color:{C['muted']};font-size:.78rem;font-weight:700;letter-spacing:1px;text-transform:uppercase}}

.kpi{{background:var(--panel);border:1px solid var(--border);border-radius:13px;padding:16px;min-height:104px}}

.kpi-label{{color:{C['muted']};font-size:.78rem;letter-spacing:.7px;text-transform:uppercase;font-weight:700}}

.kpi-value{{font-family:'Space Grotesk',sans-serif;font-size:1.95rem;font-weight:700;color:var(--text);margin-top:8px;line-height:1.1}}

.team-card{{height:100%;margin-bottom:16px;background:var(--panel);border:1px solid var(--border);border-radius:15px;padding:20px;transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;animation:fadeUp .5s ease both}}

.team-initials{{width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:13px;background:var(--panel2);color:var(--accent);font-weight:800;font-family:'Space Grotesk',sans-serif;font-size:1.05rem;margin-bottom:14px}}

@keyframes fadeUp{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}

@media (prefers-reduced-motion: reduce){{*,*::before,*::after{{animation:none!important;transition:none!important;scroll-behavior:auto!important}}}}



[data-testid="stSidebar"] .stButton>button {{

    min-height:48px;

    justify-content:flex-start;

    text-align:left;

    padding:0 15px;

    font-size:1.02rem;

    font-weight:700;

    border-radius:10px;

}}

[data-testid="stSidebar"] .stButton>button[kind="primary"] {{

    background:linear-gradient(100deg,{C['accent']},#376FE3);

    color:white;

    box-shadow:0 5px 15px #3478f626;

}}

[data-testid="stSidebar"] .stRadio label {{

    font-size:1rem!important;

    font-weight:600;

}}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{

    gap:10px;

}}



footer{{visibility:hidden}} #MainMenu{{visibility:hidden}}



</style>

""", unsafe_allow_html=True)





def kpi(label, value, color=None, hint=None):

    color = color or C["accent"]

    hint_html = f'<div style="font-size:.86rem;color:{C["muted"]};margin-top:6px">{hint}</div>' if hint else ""

    st.markdown(

        f'<div class="kpi"><div class="kpi-label">{label}</div>'

        f'<div class="kpi-value" style="color:{color}">{value}</div>{hint_html}</div>',

        unsafe_allow_html=True,

    )





def heading(tag, title, note=""):

    st.markdown(

        f'<div style="display:flex;align-items:baseline;gap:12px;margin:17px 0 10px">'

        f'<span class="kicker">{tag}</span><span class="panel-title">{title}</span></div>',

        unsafe_allow_html=True,

    )

    if note:

        st.caption(note)





def chart_style(fig):

    fig.update_layout(

        paper_bgcolor=C["panel"], plot_bgcolor=C["panel"], font_color=C["text"],

        margin=dict(l=8, r=8, t=28, b=8), legend=dict(orientation="h", y=1.08),

    )

    fig.update_xaxes(showgrid=False, linecolor=C["border"])

    fig.update_yaxes(gridcolor=C["border"], zeroline=False)

    return fig





@st.cache_resource
def get_detector():
    return VehicleDetector(model_path=str(PROJECT_ROOT / "yolo11n.pt"))


def convert_to_browser_mp4(source_path: Path, destination_path: Path) -> Path:
    """Convert OpenCV's MP4 output to browser-friendly H.264 MP4."""
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y", "-i", str(source_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        str(destination_path),
    ]
    subprocess.run(
        command, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
    )
    if not destination_path.exists() or destination_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not create a browser-compatible video.")
    return destination_path





NAV = {

    "Overview": ("⌂", "A simple starting point for TrafficVision"),

    "Video Analysis": ("▣", "Upload, inspect, and count vehicles in traffic footage"),

    "Image Inspection": ("▧", "Detect vehicles in a still image"),

    "Live Camera": ("◉", "Run a short capture from a connected camera"),

    "About & Team": ("ⓘ", "Project information and contributor roles"),

}



with st.sidebar:

    st.markdown(

        f'<div style="display:flex;align-items:center;gap:12px;margin:8px 0 5px">'

        f'<div style="width:44px;height:44px;border-radius:14px;background:linear-gradient(140deg,{C["accent"]},{C["purple"]});'

        f'display:flex;align-items:center;justify-content:center;color:white;font-size:1.35rem;font-weight:800;box-shadow:0 6px 18px #3478f633">🚦</div>'

        f'<div><div style="font-family:Space Grotesk,sans-serif;font-size:1.45rem;font-weight:700;color:{C["text"]};line-height:1.15">'

        f'Traffic<span style="color:{C["accent"]}">Vision</span></div>'

        f'<div style="font-size:.88rem;color:{C["muted"]};margin-top:4px">Traffic intelligence</div></div></div>',

        unsafe_allow_html=True,

    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{C["border"]};margin-bottom:22px"></div>', unsafe_allow_html=True)

    st.markdown(

        f'<div style="font-size:.78rem;color:{C["muted"]};font-weight:800;letter-spacing:1.4px;margin-bottom:10px">WORKSPACE</div>',

        unsafe_allow_html=True,

    )



    nav_items = [

        ("Overview", "⌂", "Dashboard home"),

        ("Video Analysis", "▣", "Vehicle tracking & counts"),

        ("Image Inspection", "▧", "Single-image detection"),

        ("Live Camera", "◉", "Webcam detection"),

        ("About & Team", "ⓘ", "Project & contributors"),

    ]

    for destination, icon, subtitle in nav_items:

        selected = st.session_state.page == destination

        if st.button(

            f"{icon}   {destination}",

            key=f"sidebar_nav_{destination}",

            use_container_width=True,

            type="primary" if selected else "secondary",

        ):

            st.session_state.page = destination

            st.rerun()

        st.markdown(

            f'<div style="font-size:.84rem;color:{C["muted"]};margin:-7px 0 9px 15px">{subtitle}</div>',

            unsafe_allow_html=True,

        )



    st.markdown(f'<div style="height:1px;background:{C["border"]};margin:19px 0"></div>', unsafe_allow_html=True)

    st.markdown(

        f'<div style="font-size:.78rem;color:{C["muted"]};font-weight:800;letter-spacing:1.4px;margin-bottom:10px">APPEARANCE</div>',

        unsafe_allow_html=True,

    )

    st.radio("Color theme", ["Dark", "Light"], horizontal=True, key="theme", label_visibility="collapsed")



    st.markdown(f'<div style="height:1px;background:{C["border"]};margin:19px 0"></div>', unsafe_allow_html=True)

    st.markdown(

        f'<div style="border:1px solid {C["border"]};border-radius:13px;padding:14px;background:{C["panel"]};'

        f'display:flex;gap:10px;align-items:flex-start">'

        f'<div style="font-size:1.2rem;color:{C["green"]}">●</div>'

        f'<div><div style="font-size:1rem;font-weight:700;color:{C["text"]}">Ready to analyze</div>'

        f'<div style="font-size:.86rem;color:{C["muted"]};margin-top:4px;line-height:1.5">Choose a workspace above to get started.</div></div></div>',

        unsafe_allow_html=True,

    )





page = st.session_state.page

st.markdown(

    f'<div style="display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:12px">'

    f'<div class="kicker">TRAFFICVISION WORKSPACE</div>'

    f'<div style="border:1px solid {C["border"]};border-radius:99px;padding:8px 13px;color:{C["muted"]};font-size:.86rem;background:{C["panel"]}">'

    f'<span style="color:{C["green"]}">●</span>&nbsp; Detection workspace</div></div>'

    f'<div class="hero"><div class="kicker">{page.upper()}</div>'

    f'<h1>{"TrafficVision Dashboard" if page == "Overview" else page}</h1>'

    f'<p>{NAV[page][1]}.</p></div>',

    unsafe_allow_html=True,

)



if page == "Overview":

    # Reference-style overview: case/session header, prominent action panel,

    # concise KPI cards and two operational panels.

    recent = st.session_state.video_result

    video_data = (recent or {}).get("result", {}) if recent else {}

    class_counts = video_data.get("class_counts", {}) if isinstance(video_data, dict) else {}

    total_crossed = int(video_data.get("total_count", 0) or 0) if isinstance(video_data, dict) else 0

    image_data = st.session_state.image_result

    image_total = int(image_data.get("total", 0) or 0) if image_data else 0

    total_items = total_crossed + image_total

    suspicious_items = 0

    missing_items = 0

    contradictions = 0



    top_left, top_right = st.columns([1.5, 1], vertical_alignment="center")

    with top_left:

        st.markdown('<div class="kicker">TRAFFIC OPERATIONS WORKSPACE</div><h1 style="margin:6px 0 0">Traffic Analysis Dashboard</h1>', unsafe_allow_html=True)

    with top_right:

        st.markdown(

            f'<div style="display:flex;justify-content:flex-end;align-items:center;gap:10px;flex-wrap:wrap">'

            f'<div style="border:1px solid {C["border"]};border-radius:30px;padding:10px 14px;background:{C["panel"]};font-size:.86rem;color:{C["muted"]}">'

            f'<span style="color:{C["green"]}">●</span>&nbsp; Session active</div></div>',

            unsafe_allow_html=True,

        )

    st.markdown(f'<div style="height:1px;background:{C["border"]};margin:22px 0 26px"></div>', unsafe_allow_html=True)



    cta_left, cta_right = st.columns([4, 1], vertical_alignment="center")

    with cta_left:

        st.markdown(

            f'<div class="hero" style="margin-bottom:0">'

            f'<div class="kicker">TRAFFIC INTELLIGENCE</div>'

            f'<h2 style="font-size:1.8rem!important;margin:8px 0">Turn traffic footage into clear, useful insights.</h2>'

            f'<p>Upload a traffic video or image to detect vehicles, track movement, count crossings and review the results in one workspace.</p>'

            f'</div>',

            unsafe_allow_html=True,

        )

    with cta_right:

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("＋  Analyze Traffic", type="primary", use_container_width=True):

            st.session_state.page = "Video Analysis"

            st.rerun()



    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    metric_cols = st.columns(4, gap="medium")

    with metric_cols[0]:

        kpi("Vehicles counted", f"{total_crossed}", C["blue"], "Line crossings · latest video")

    with metric_cols[1]:

        kpi("Image detections", f"{image_total}", C["red"], "Objects · latest image")

    with metric_cols[2]:

        kpi("Input modes", "3", C["orange"], "Video · Image · Webcam")

    with metric_cols[3]:

        kpi("Analyses completed", f"{int(recent is not None) + int(image_data is not None) + int(st.session_state.webcam_result is not None)}", C["purple"], "This app session")



    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    timeline_col, findings_col = st.columns([1.2, .9], gap="medium")

    with timeline_col:

        st.markdown(

            f'<div class="panel" style="padding:0;overflow:hidden">'

            f'<div style="padding:18px 20px;border-bottom:1px solid {C["border"]};display:flex;justify-content:space-between;gap:10px;align-items:center">'

            f'<div><div class="panel-title">Recent Activity</div><div style="color:{C["muted"]};font-size:.88rem">Latest actions in this session</div></div>'

            f'<span class="kicker">ACTIVITY</span></div>'

            f'<div style="padding:18px 20px">',

            unsafe_allow_html=True,

        )

        if recent is None and image_data is None and st.session_state.webcam_result is None:

            st.markdown(

                f'<div style="display:flex;gap:14px;padding:8px 0 18px">'

                f'<div style="color:{C["blue"]};font-size:1.1rem">●</div>'

                f'<div><b>No analysis run yet</b><div style="color:{C["muted"]};font-size:.9rem">Your completed video, image and webcam analyses will appear here.</div></div></div>',

                unsafe_allow_html=True,

            )

        else:

            activities = []

            if recent:

                activities.append(("Video analysis completed", f"{total_crossed} vehicle crossing(s) counted", "VIDEO"))

            if image_data:

                activities.append(("Image inspection completed", f"{image_total} supported vehicle detection(s)", "IMAGE"))

            if st.session_state.webcam_result:

                activities.append(("Webcam capture completed", "A short camera capture was processed", "CAMERA"))

            for title, desc, label in activities:

                st.markdown(

                    f'<div style="display:flex;gap:14px;padding:8px 0 16px">'

                    f'<div style="color:{C["green"]};font-size:1.1rem">●</div>'

                    f'<div style="flex:1"><b>{title}</b><div style="color:{C["muted"]};font-size:.9rem">{desc}</div></div>'

                    f'<div class="mini-label">{label}</div></div>',

                    unsafe_allow_html=True,

                )

        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        a, b = st.columns(2, gap="small")

        with a:

            if st.button("Open Video Analysis →", use_container_width=True):

                st.session_state.page = "Video Analysis"; st.rerun()

        with b:

            if st.button("Open Image Inspection →", use_container_width=True):

                st.session_state.page = "Image Inspection"; st.rerun()



    with findings_col:

        st.markdown(

            f'<div class="panel" style="padding:0;overflow:hidden">'

            f'<div style="padding:18px 20px;border-bottom:1px solid {C["border"]}">'

            f'<div class="panel-title">Quick Actions</div><div style="color:{C["muted"]};font-size:.88rem">Choose the next task</div></div>'

            f'<div style="padding:12px 18px">',

            unsafe_allow_html=True,

        )

        actions = [

            ("▣", "Analyze a video", "Track vehicles and count line crossings", "Video Analysis"),

            ("▧", "Inspect an image", "Detect and classify vehicles in one frame", "Image Inspection"),

            ("◉", "Use live camera", "Run a short local webcam capture", "Live Camera"),

            ("ⓘ", "About the project", "Read project details and team roles", "About & Team"),

        ]

        for icon, title, desc, destination in actions:

            if st.button(f"{icon}  {title}", key=f"quick_{destination}", use_container_width=True):

                st.session_state.page = destination

                st.rerun()

            st.markdown(f'<div style="color:{C["muted"]};font-size:.86rem;margin:-7px 0 10px 4px">{desc}</div>', unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)





elif page == "Video Analysis":

    heading("STEP 1", "Upload traffic video", "Supported formats: MP4, AVI, MOV, MKV, MPEG.")

    uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"], key="video_upload")

    if uploaded is not None:

        temp_root = Path(tempfile.gettempdir()) / "vehicle_detection_app"

        temp_root.mkdir(parents=True, exist_ok=True)

        blob = uploaded.getvalue()

        file_hash = hashlib.sha256(blob).hexdigest()[:12]

        in_path = temp_root / f"input_{file_hash}{Path(uploaded.name).suffix or '.mp4'}"

        out_path = temp_root / f"annotated_{file_hash}.mp4"

        browser_path = temp_root / f"annotated_{file_hash}_h264.mp4"
        if not in_path.exists() or in_path.stat().st_size != len(blob):

            in_path.write_bytes(blob)

        cap = cv2.VideoCapture(str(in_path))

        if not cap.isOpened():

            st.error("We couldn't open this video. Try another file.")

            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ok, first = cap.read()

        cap.release()

        if not ok:

            st.error("No readable frames were found in this file.")

            st.stop()

        duration = frames / fps if fps > 0 else 0

        info = st.columns(4)

        for col, label, value in zip(info, ["Resolution", "Frame rate", "Frames", "Duration"], [f"{w} × {h}", f"{fps:.1f} FPS", f"{frames:,}", f"{int(duration//60)}:{int(duration%60):02d}"]):

            with col:

                kpi(label, value, C["blue"])

        left, right = st.columns([.75, 1.25], gap="large")

        with left:

            line_y = st.slider("Counting line position", 0, max(h - 1, 1), min(int(h * .55), max(h - 1, 1)), key=f"line_{file_hash}")

            st.caption(f"Counting line: {line_y} px from the top.")

            run = st.button("Process video", type="primary", use_container_width=True)

        with right:

            preview = first.copy()

            cv2.line(preview, (0, line_y), (w, line_y), (0, 220, 255), 3)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="First frame preview with selected counting line.", use_column_width=True)

        if run:

            try:

                with st.status("Processing video…", expanded=True) as status:

                    status.write("Loading detector.")

                    detector = get_detector()

                    status.write("Detecting, tracking and counting vehicles.")

                    result = VideoProcessor(detector=detector, line_y=int(line_y), frame_rate=float(fps if fps > 0 else 30)).process_video(str(in_path), str(out_path))
                    status.write("Converting annotated video for browser playback.")
                    convert_to_browser_mp4(out_path, browser_path)
                    status.update(label="Analysis complete", state="complete", expanded=False)

                if browser_path.exists():

                    st.session_state.video_result = {"hash": file_hash, "line_y": int(line_y), "result": result, "output": str(browser_path)}

                    st.success("Video analysis completed.")

                else:

                    st.error("No annotated output video was created.")

            except Exception as exc:

                st.error(f"Video analysis failed: {exc}")

                with st.expander("Technical details"):

                    st.exception(exc)

        rec = st.session_state.video_result

        if rec and rec.get("hash") == file_hash and rec.get("line_y") == int(line_y):

            result = rec["result"] or {}

            counts = result.get("class_counts") or {}

            heading("RESULTS", "Vehicle counts")

            values = [

                ("Total crossed", int(result.get("total_count", 0) or 0), C["accent"]),

                ("Cars", int(counts.get("car", 0) or 0), C["blue"]),

                ("Motorcycles", int(counts.get("motorcycle", 0) or 0), C["orange"]),

                ("Buses", int(counts.get("bus", 0) or 0), C["purple"]),

                ("Trucks", int(counts.get("truck", 0) or 0), C["red"]),

            ]

            cols = st.columns(5)

            for col, (label, value, color) in zip(cols, values):

                with col:

                    kpi(label, value, color)

            vcol, ccol = st.columns([1.15, 1], gap="large")

            with vcol:

                st.markdown('<div class="panel-title">Annotated video</div>', unsafe_allow_html=True)

                # Pass encoded bytes directly, as in the successful playback diagnostic.
                browser_video_bytes = Path(rec["output"]).read_bytes()
                st.video(browser_video_bytes, format="video/mp4")
                st.download_button(
                    "Download annotated video",
                    browser_video_bytes,
                    file_name="trafficvision_annotated.mp4",
                    mime="video/mp4",
                )

            with ccol:

                st.markdown('<div class="panel-title">Vehicle distribution</div>', unsafe_allow_html=True)

                df = pd.DataFrame({"Vehicle": ["Car", "Motorcycle", "Bus", "Truck"], "Count": [values[1][1], values[2][1], values[3][1], values[4][1]]})

                nonzero = df[df["Count"] > 0]

                if not nonzero.empty:

                    fig = px.pie(nonzero, names="Vehicle", values="Count", hole=.6)

                    fig.update_traces(textinfo="label+value")

                    st.plotly_chart(chart_style(fig), use_container_width=True)

                else:

                    st.info("No vehicles crossed the selected line in this clip.")

                st.download_button("Download counts (CSV)", df.to_csv(index=False).encode(), "vehicle_counts.csv", "text/csv")

    else:

        st.info("Upload a traffic video to preview it, choose a counting line and start analysis.")



elif page == "Image Inspection":

    heading("STEP 1", "Upload image", "Supported formats: JPG, JPEG, PNG, BMP, WEBP.")

    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="image_upload")

    if image_file is not None:

        image_bytes = image_file.getvalue()

        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:

            st.error("We couldn't read this image. Try another file.")

        else:

            left, right = st.columns([1.2, .8], gap="large")

            with right:

                confidence = st.slider("Minimum detection confidence", .10, .95, .50, .05, key="image_conf")

                run = st.button("Detect vehicles", type="primary", use_container_width=True)

                st.caption("Image mode detects objects in one image; it does not track movement.")

            if run:

                try:

                    with st.spinner("Detecting vehicles…"):

                        detector = get_detector()

                        prediction = detector.model.predict(source=frame, conf=confidence, verbose=False)[0]

                        annotated = prediction.plot()

                        class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

                        class_counts = {name: 0 for name in class_names.values()}

                        if prediction.boxes is not None:

                            for cls in prediction.boxes.cls.tolist():

                                name = class_names.get(int(cls))

                                if name:

                                    class_counts[name] += 1

                        st.session_state.image_result = {"hash": hashlib.sha256(image_bytes).hexdigest(), "annotated": annotated, "counts": class_counts, "total": sum(class_counts.values())}

                except Exception as exc:

                    st.error(f"Image detection failed: {exc}")

                    with st.expander("Technical details"):

                        st.exception(exc)

            current_hash = hashlib.sha256(image_bytes).hexdigest()

            rec = st.session_state.image_result

            with left:

                if rec and rec["hash"] == current_hash:

                    st.markdown('<div class="panel-title">Annotated image</div>', unsafe_allow_html=True)

                    st.image(cv2.cvtColor(rec["annotated"], cv2.COLOR_BGR2RGB), use_column_width=True)

                    ok, enc = cv2.imencode(".jpg", rec["annotated"])

                    if ok:

                        st.download_button("Download annotated image", enc.tobytes(), "trafficvision_image.jpg", "image/jpeg")

                else:

                    st.markdown('<div class="panel-title">Image preview</div>', unsafe_allow_html=True)

                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            if rec and rec["hash"] == current_hash:

                heading("RESULTS", "Vehicle detections")

                vals = [("Total", rec["total"], C["accent"]), ("Cars", rec["counts"]["car"], C["blue"]), ("Motorcycles", rec["counts"]["motorcycle"], C["orange"]), ("Buses", rec["counts"]["bus"], C["purple"]), ("Trucks", rec["counts"]["truck"], C["red"])]

                cols = st.columns(5)

                for col, (label, value, color) in zip(cols, vals):

                    with col:

                        kpi(label, value, color)



elif page == "Live Camera":

    heading("CAMERA", "Short webcam capture", "Uses the camera attached to the computer running Streamlit.")

    st.info("Camera processing happens on the host computer. This page captures a limited number of frames and does not stream continuously.")

    left, right = st.columns([1.15, .85], gap="large")

    with right:

        confidence = st.slider("Minimum detection confidence", .10, .95, .50, .05, key="live_conf")

        max_frames = st.slider("Frames to capture", 30, 180, 60, 30)

        run = st.button("Start webcam capture", type="primary", use_container_width=True)

    with left:

        image_slot = st.empty()

        progress_slot = st.empty()

        if run:

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():

                st.error("Could not open the webcam. Check that it is connected and not in use.")

            else:

                try:

                    detector = get_detector()

                    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

                    total = 0

                    snapshots = []

                    started = time.time()

                    for i in range(max_frames):

                        ok, frame = cap.read()

                        if not ok:

                            break

                        pred = detector.model.predict(source=frame, conf=confidence, verbose=False)[0]

                        annotated = pred.plot()

                        if pred.boxes is not None:

                            for cls in pred.boxes.cls.tolist():

                                name = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}.get(int(cls))

                                if name:

                                    counts[name] += 1

                                    total += 1

                        image_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

                        progress_slot.caption(f"Captured {i + 1} / {max_frames} frames")

                        if i % 5 == 0:

                            snapshots.append(annotated.copy())

                    st.session_state.webcam_result = {"counts": counts, "total": total, "snapshots": snapshots, "elapsed": time.time() - started}

                    st.success("Webcam capture completed.")

                except Exception as exc:

                    st.error(f"Webcam processing failed: {exc}")

                    with st.expander("Technical details"):

                        st.exception(exc)

                finally:

                    cap.release()

        rec = st.session_state.webcam_result

        if rec:

            st.caption(f"Capture completed in {rec['elapsed']:.1f} seconds. Counts are per-frame detections, not unique vehicles.")

            if rec["snapshots"]:

                ok, enc = cv2.imencode(".jpg", rec["snapshots"][-1])

                if ok:

                    st.download_button("Download last annotated frame", enc.tobytes(), "trafficvision_webcam.jpg", "image/jpeg")

            vals = [("Frame detections", rec["total"], C["accent"]), ("Cars", rec["counts"]["car"], C["blue"]), ("Motorcycles", rec["counts"]["motorcycle"], C["orange"]), ("Buses", rec["counts"]["bus"], C["purple"]), ("Trucks", rec["counts"]["truck"], C["red"])]

            cols = st.columns(5)

            for col, (label, value, color) in zip(cols, vals):

                with col:

                    kpi(label, value, color)



elif page == "About & Team":

    heading("PROJECT", "About TrafficVision", "A student-built computer vision system for vehicle detection and traffic-flow analysis.")

    st.markdown(

        f'<div class="panel"><h3>Project objective</h3><p style="color:{C["muted"]};font-size:1.02rem">'

        'TrafficVision analyzes traffic videos, images and local webcam captures to detect supported vehicle classes. '

        'In video mode, tracking and a configurable virtual line help count vehicle crossings. The application presents '

        'annotated media and summary statistics for review.</p></div>',

        unsafe_allow_html=True,

    )

    heading("CAPABILITIES", "Key features")

    features = [

        ("Vehicle detection", "Identify cars, motorcycles, buses and trucks using YOLO."),

        ("Tracking & counting", "Follow vehicles through video and count virtual-line crossings."),

        ("Image preprocessing", "Optional frame validation, resizing, brightness/contrast adjustment and noise reduction."),

        ("Results & exports", "View annotated outputs, class totals and downloadable reports."),

    ]

    cols = st.columns(2, gap="medium")

    for idx, (title, desc) in enumerate(features):

        with cols[idx % 2]:

            st.markdown(f'<div class="panel" style="min-height:125px;margin-bottom:12px"><h3>{title}</h3><p style="font-size:1rem;color:{C["muted"]}">{desc}</p></div>', unsafe_allow_html=True)



    heading("TECHNOLOGY", "Built with")

    st.markdown("YOLO / Ultralytics · OpenCV · ByteTrack · Python · Streamlit")



    heading("TEAM", "Meet the project team", "Team roles reflect the latest agreed work distribution.")

    members = [

        ("RJ", "RAJ JHA", "Team Lead · Core Integration & Dashboard UI/Visualization", "Integrates detection, tracking, counting and video processing into the main application workflow, and also worked on the dashboard UI and visualization."),

        ("PM", "PINKU MODI", "Developed the YOLO-based vehicle detection module using Python and OpenCV to detect and classify cars, motorcycles, buses, and trucks from images and video. Implemented bounding-box visualization, confidence scoring, and structured detection outputs for integration with the project's tracking and traffic-counting pipeline."),

        ("PS", "PRITAM SINGH", "Image Processing & Preprocessing", "Implements frame validation, resizing, brightness/contrast adjustment and noise reduction."),

        ("DP", "DEBANSHU PATRA", "Dashboard UI & Visualization Contributor", "Contributed to the dashboard interface, visual presentation and user-facing results."),

        ("ST", "SANSKRITI THAKUR", "Documentation & Results Demo", "Organizes project documentation, results and demonstration materials."),

        ("SP", "SANATANU PAL", "Traffic Data Collection & Testing", "Collects traffic videos/images and supports testing across different traffic conditions."),

    ]

    cols = st.columns(3, gap="medium")

    for idx, (initials, name, role, description) in enumerate(members):

        with cols[idx % 3]:

            st.markdown(

                f'<div class="team-card"><div class="team-initials">{initials}</div>'

                f'<div class="mini-label">PROJECT CONTRIBUTOR</div><h3 style="margin:8px 0 5px">{name}</h3>'

                f'<div style="font-weight:700;color:{C["accent"]};font-size:.98rem;margin-bottom:10px">{role}</div>'

                f'<div style="font-size:.97rem;color:{C["muted"]};line-height:1.6">{description}</div></div>',

                unsafe_allow_html=True,

            )



st.markdown("---")

st.markdown(

    f'<div style="text-align:center;color:{C["muted"]};font-size:.88rem">'

    f'<b style="color:{C["text"]}">TRAFFICVISION</b><br>'

    f'Vehicle Detection & Traffic Analysis</div>',

    unsafe_allow_html=True,

)
