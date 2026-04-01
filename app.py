import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tempfile
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model.signatures['serving_default']

with st.spinner("Loading AI Model... Please wait ⏳"):
    movenet = load_model()

# ================= POSE =================
def detect_pose(frame):
    h, w, _ = frame.shape
    img = cv2.resize(frame, (192, 192))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = np.expand_dims(rgb, axis=0)
    input_img = tf.cast(input_img, dtype=tf.int32)

    outputs = movenet(input_img)
    kps = outputs['output_0'].numpy()[0][0]

    pts = []
    for kp in kps:
        y = int(kp[0] * h)
        x = int(kp[1] * w)
        pts.append((x, y) if kp[2] > 0.4 else None)
    return pts

# ================= DRAW =================
def draw_skeleton(frame, points):
    edges = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16)
    ]

    for p in points:
        if p:
            cv2.circle(frame, p, 4, (0,255,0), -1)

    for e in edges:
        p1, p2 = points[e[0]], points[e[1]]
        if p1 and p2:
            cv2.line(frame, p1, p2, (255,0,0), 2)

    return frame

def angle(a,b,c):
    a,b,c = np.array(a),np.array(b),np.array(c)
    ba,bc = a-b,c-b
    cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

# ================= PROCESS =================
def process_video(video_file):

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(3))

    series, frames = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pts = detect_pose(frame)
        series.append(pts)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # ===== METRICS =====
    hip = [p[11][0] for p in series if p and p[11]]
    disp = np.mean(np.abs(np.diff(hip))) if len(hip)>2 else 0

    run = max(10,min((disp/w)*fps*50,30))
    speed = max(90,min(run*2.8+60,150))

    arm = np.mean([abs(series[i][9][0]-series[i-1][9][0])
                   for i in range(1,len(series))
                   if series[i][9] and series[i-1][9]] or [0])

    knee = np.mean([angle(p[11],p[13],p[15])
                    for p in series if p and p[11] and p[13] and p[15]] or [0])

    rhythm = np.std(np.diff(hip)) if len(hip)>2 else 0

    ankle = [p[15][1] for p in series if p and p[15]]
    jump = (max(ankle)-min(ankle))/200 if ankle else 0

    # ACTION
    elbow = [angle(p[5],p[7],p[9]) for p in series if p and p[5] and p[7] and p[9]]
    elbow = np.mean(elbow) if elbow else 0
    action = "Fair" if elbow>150 else "Moderate" if elbow>130 else "Suspect"

    # RELEASE
    wrist = [p[9][0] if p and p[9] else 0 for p in series]
    release_idx = np.argmax(np.diff(wrist)) if len(wrist)>2 else 0

    try:
        head = series[release_idx][0]
        hand = series[release_idx][9]
        release = "Front" if hand and head and hand[0]>head[0] else "Behind"
    except:
        release = "Front"

    # TYPE
    bowl_type = "Fast" if speed>130 else "Medium" if speed>115 else "Slow"

    # SCORE
    score = min(run*2,25)+min(jump*50,20)+min(arm*2,20)
    score += 15 if release=="Front" else 5
    score += 20 if action=="Fair" else 5
    performance = int(min(score,100))

    risk_score = (2 if knee<30 else 0)+(2 if action=="Suspect" else 0)+(1 if rhythm>15 else 0)
    injury = "High" if risk_score>=3 else "Medium" if risk_score>=1 else "Low"
    confidence = int(min(70+(run*0.5),95))

    tips = []
    if action=="Suspect": tips.append("Fix elbow")
    if release=="Behind": tips.append("Release forward")
    if jump<0.3: tips.append("Improve jump")
    if knee<30: tips.append("Strong front leg")
    if run<15: tips.append("Increase run-up")
    tips = " | ".join(tips) if tips else "Excellent"

    # ===== VIDEO OUTPUT (FULL) =====
    h,w,_ = frames[0].shape
    video_path = "FINAL_VIDEO_FULL.avi"

    out = cv2.VideoWriter(video_path,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          fps,
                          (w,h))

    for i,f in enumerate(frames):

        if i < len(series):
            f = draw_skeleton(f, series[i])

        # LEFT SIDE
        y = 30
        for txt in [
            f"Run:{run:.1f}",
            f"Speed:{speed:.1f}",
            f"Type:{bowl_type}",
            f"Score:{performance}",
            f"Risk:{injury}",
            f"Confidence:{confidence}%"
        ]:
            cv2.putText(f, txt, (20, y), 0, 0.6, (255,255,255), 2)
            y += 25

        # RIGHT SIDE
        y = 30
        for txt in [
            f"Arm:{arm:.2f}",
            f"Brace:{knee:.1f}",
            f"Rhythm:{rhythm:.2f}",
            f"Jump:{jump:.2f}"
        ]:
            cv2.putText(f, txt, (w-260, y), 0, 0.6, (200,200,200), 2)
            y += 25

        # BOTTOM
        cv2.putText(f, "Action:"+action, (20, h-70), 0, 0.6, (255,255,0), 2)
        cv2.putText(f, "Release:"+release, (20, h-45), 0, 0.6, (255,255,0), 2)
        cv2.putText(f, "Tips:"+tips, (20, h-15), 0, 0.5, (0,255,0), 2)

        out.write(f)

    out.release()

    # ===== EXCEL =====
    excel_path = "result.xlsx"
    df = pd.DataFrame([{
        "RunSpeed":run,"BallSpeed":speed,"Type":bowl_type,
        "Performance":performance,"Risk":injury,"Confidence":confidence,
        "Arm":arm,"Brace":knee,"Rhythm":rhythm,"Jump":jump,
        "Action":action,"Release":release,"Tips":tips
    }])
    df.to_excel(excel_path,index=False)

    return video_path, excel_path, action, release, tips

# ================= UI =================
st.title("🏏 Cricket Bowling Analysis AI")

uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi"])

if uploaded_file:
    try:
        st.info("Processing... ⏳")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        result = process_video(tfile.name)

        if result:
            video_path, excel_path, action, release, tips = result

            st.success("✅ Done")

            st.video(video_path)

            st.write("Action:", action)
            st.write("Release:", release)
            st.write("Tips:", tips)

            with open(video_path, "rb") as f:
                st.download_button("⬇️ Download Video", f,
                                   file_name="FINAL_VIDEO_FULL.avi",
                                   mime="video/x-msvideo")

            with open(excel_path, "rb") as f:
                st.download_button("⬇️ Download Excel", f,
                                   file_name="result.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        else:
            st.error("❌ Failed to process video")

    except Exception as e:
        st.error(f"Error: {e}")
