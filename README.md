# 🧠 Person Re-Identification & Scene Classification Pipeline

This repository implements a **multi-video pipeline** for:
- Person **Re-Identification (ReID)** across different videos  
- **Global ID assignment** using embedding similarity and Hungarian matching  
- **Suspicious activity detection** based on optical flow magnitude  
- Scene-level classification: *Crime* vs *Normal*

---

## 📁 Project Structure

```
project_root/
│
├── main.py                # Entry point (argparse, pipeline orchestration)
├── reid_pipeline.py       # Core logic: GlobalIdentityTracker, ReID model, utilities, process_video()
├── videos/                # Folder for input videos
├── results/               # Output folder (auto-created)
│   ├── person_catalogue/  # Processed videos with Global IDs
│   └── suspicious_frames/ # Saved frames with suspicious activity
│
├── PersonCatalogue.json   # Global ID catalogue (auto-generated)
└── SceneLabelling.json    # Scene classification results (auto-generated)
```

---

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

Example `requirements.txt`:
```txt
torch
torchreid
ultralytics
scikit-learn
opencv-python
numpy
scipy
torchvision
```

### 4. Download external assets
- YOLO model weights (`yolo11m.pt`) will be automatically downloaded by Ultralytics on first run.

---

## ▶️ How to Run

Run the pipeline on all videos in `./videos`:

```bash
python main.py     --video_dir ./videos     --output_dir ./results     --json_globalID PersonCatalogue.json     --json_labelling SceneLabelling.json     --sim_thresh 0.65     --momentum 0.85     --max_thres 24     --min_frames 5
```

Each video will produce:
- A processed video with **Global IDs** overlaid  
- Saved **suspicious frames** (e.g., high motion)  
- Updated **JSON reports** for Global IDs and Scene Labels  

---

## 🧩 Argument Descriptions

| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--video_dir` | str | `./videos` | Input directory containing videos |
| `--output_dir` | str | `./results` | Output directory for results |
| `--json_globalID` | str | `PersonCatalogue.json` | Global identity catalogue JSON |
| `--json_labelling` | str | `SceneLabelling.json` | Scene-level labelling JSON |
| `--sim_thresh` | float | `0.65` | Similarity threshold for ReID matching |
| `--momentum` | float | `0.85` | Momentum for updating global embeddings |
| `--max_thres` | float | `24` | Max optical flow threshold for suspicious motion |
| `--min_frames` | float | `5` | Minimum consecutive frames to flag suspicious activity |

---

## 💾 Example Output

After running the pipeline:
```
results/
├── person_catalogue/
│   ├── globalID_video1.mp4
│   └── globalID_video2.mp4
└── suspicious_frames/
    ├── video1/frame_no200_GID4_max25.3.jpg
    └── video2/frame_no134_GID7_max29.8.jpg
```

`SceneLabelling.json` example:
```json
[
  {
    "clip_id": "video1.mp4",
    "label": "Crime",
    "justification": {
      "4": [
        {"frame": 200, "max_val": 25.3},
        {"frame": 201, "max_val": 28.1}
      ]
    }
  }
]
```

---

## 🧠 Implementation Highlights

- **GlobalIdentityTracker** – Maintains consistent IDs across multiple videos via cosine similarity and Hungarian matching.  
- **ReID embeddings** – Extracted using OSNet (`torchreid`) for appearance-based matching.  
- **Optical Flow Analysis** – Detects suspicious movement via Farneback dense flow.  
- **Scene Classification** – Labels each clip as *Crime* or *Normal* based on detected motion.  

---

## 🧰 Tips & Troubleshooting

- Ensure your videos are named with **unique filenames** across runs.  
- If CUDA is unavailable, the pipeline automatically falls back to CPU (slower).  
- Adjust `--max_thres` and `--min_frames` to control detection sensitivity.  
- Logs are printed to console; redirect output to file for batch runs.

---

## 🏁 License & Citation

If you use this project for research or academic work, please cite:

- **TorchReID:** [https://github.com/KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)  
- **Ultralytics YOLO:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
