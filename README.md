# 🧠 Person Re-Identification & Scene Classification Pipeline

This repository implements a **multi-video pipeline** for:
- Person **Re-Identification (ReID)** across different videos.  
- **Global ID assignment** using embedding similarity and Hungarian matching.
- **Suspicious activity detection** based on optical flow magnitude. 
- Scene-level classification: ***Crime* vs *Normal*** on detected motion.

---

## 📁 Project Structure

```
project_root/
│
├── main.py                # Entry point (argparse, pipeline orchestration)
├── videos/                # Folder for input videos
├── results/               # Output folder (auto-created)
│   ├── person_catalogue/  # Processed videos with Global IDs
│   └── suspicious_frames/ # Saved frames with suspicious activity
├── PersonCatalogue.json   # Global ID catalogue (auto-generated)
└── SceneLabelling.json    # Scene classification results (auto-generated)
```

## ⚙️ Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your_username>/person-reid-scene-pipeline.git
cd person-reid-scene-pipeline
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download external assets
- YOLO model weights (`yolo11m.pt`) will be automatically downloaded by Ultralytics on the first run.

---

## ▶️ How to Run

Run the pipeline on all videos in `./videos`:

```bash
python main.py     --video_dir ./videos     --output_dir ./results     --json_globalID PersonIDCatalogue.json     --json_labelling SceneLabelling.json     --sim_thresh 0.65     --momentum 0.85     --max_thres 24     --min_frames 5
```

For each video, the pipeline will produce:
- A processed video with **Global IDs** overlaid  
- Saved **suspicious frames** (e.g., abrupt motion)  
- Updated **JSON reports** for Global IDs and Scene Labels  

---

## 💾 Example of Output

After running the pipeline:
```
results/
├── person_catalogue/
│   ├── globalID_1.mp4
│   └── globalID_2.mp4
└── suspicious_frames/
    ├── 1.mp4/frame_no200_GID4_max25.3.jpg
    └── 1.mp4/frame_no134_GID7_max29.8.jpg
```

`SceneLabelling.json` example:
```json
[
  {
    "clip_id": "video1.mp4",
    "label": "Crime",
    "justification": {
      "GID:4": [
        {"frame": 200, "max_val": 25.3},
        {"frame": 201, "max_val": 28.1}
      ]
    }
  }
]
```
---

## 🏁 License & Citation

If you use this project for research or academic work, please cite:

- **TorchReID:** [https://github.com/KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)  
- **Ultralytics YOLO:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
