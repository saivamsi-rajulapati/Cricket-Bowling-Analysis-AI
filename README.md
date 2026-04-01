# 🏏 Cricket Bowling Analysis AI

🚀 AI-powered system to analyze cricket bowling using pose estimation and machine learning.

---

## 🚀 Project Highlights

* 🎯 Real-time cricket bowling analysis using AI
* 🧠 Pose estimation powered by MoveNet (TensorFlow)
* ⚡ Speed, action, and performance metrics extraction
* 🎥 Annotated video output with skeleton tracking
* 📊 Automated Excel performance report generation

---

## 📸 Demo

### 🎯 Application UI

![UI](images/ui.png)

### 🎥 Video Output

![Video](images/video.png)

### 📊 Excel Output

![Excel](images/excel.png)

---

## 🚀 Features

* 🎯 Pose detection using MoveNet
* ⚡ Bowling speed estimation
* 🧠 Action classification (Fair / Moderate / Suspect)
* ⚠️ Injury risk analysis
* 📊 Performance scoring system
* 🎥 Video output with full overlay (metrics + skeleton)
* 📄 Excel report generation

---

## 🧠 How It Works

1. Upload bowling video
2. Extract body keypoints using MoveNet
3. Compute metrics:

   * Run-up speed
   * Arm speed
   * Jump height
   * Rhythm & balance
4. Classify bowling action
5. Generate outputs:

   * Annotated video
   * Excel performance report

---

## 📊 Sample Output

| Metric            | Value    |
| ----------------- | -------- |
| Run Speed         | 22.5     |
| Ball Speed        | 132 km/h |
| Bowling Type      | Fast     |
| Performance Score | 87       |
| Injury Risk       | Low      |
| Action            | Fair     |

---

## 💡 Why This Project?

Cricket bowling analysis is traditionally done manually by coaches.
This project automates the process using AI, helping players:

* Improve performance
* Detect illegal bowling actions
* Reduce injury risks
* Get instant feedback from video

---

## 🛠 Tech Stack

* Python
* TensorFlow
* OpenCV
* Streamlit
* NumPy & Pandas

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 🌐 Deployment

* Streamlit (Local App)
* Cloudflare Tunnel (for public sharing)

---

## 👨‍💻 Author

**Sai Vamsi Rajulapati**
AI & ML Student

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
