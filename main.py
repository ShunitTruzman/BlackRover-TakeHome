import os
import cv2
import json
import numpy as np
import torch
import argparse
from torchreid import models
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import torchvision.transforms as TV

# Define GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================
#  Global Identity Tracker
# ========================================
class GlobalIdentityTracker:
    """
    Tracks and maintains consistent Global IDs across multiple videos.
    Uses cosine similarity between embeddings and the Hungarian matching algorithm.
    """

    def __init__(self, similarity_threshold=0.65, embedding_momentum=0.85):
        self.similarity_threshold = similarity_threshold
        self.embedding_momentum = embedding_momentum
        self.global_counter = 1
        self.global_embeddings = {}
        self.local_to_global = {}

    def assign_global_ids(self, video_name: str, local_ids: np.ndarray, embeddings: np.ndarray) -> dict:
        """
        Assigns local track IDs to Global IDs using Hungarian matching.
        Returns a dict: {local_id: gid}
        """
        num_detections = embeddings.shape[0]
        result = {}

        # Case 1: no global IDs yet
        if not self.global_embeddings:
            for i in range(num_detections):
                gid = self._create_gid(embeddings[i])
                self.local_to_global[(video_name, local_ids[i])] = gid
                result[local_ids[i]] = gid
            return result

        # Build gallery of known global embeddings
        gids = list(self.global_embeddings.keys())
        gallery = np.stack(list(self.global_embeddings.values()))

        # Compute cosine similarity between current embeddings and global gallery
        sims = cosine_similarity(embeddings, gallery)
        cost_matrix = 1 - sims  # Hungarian minimizes cost

        # Optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        used_dets, used_gids = set(), set()

        for det_idx, gid_idx in zip(row_ind, col_ind):
            sim = sims[det_idx, gid_idx]
            if sim >= self.similarity_threshold:
                gid = gids[gid_idx]
                tid = local_ids[det_idx]
                self.local_to_global[(video_name, tid)] = gid
                self._update_embedding(gid, embeddings[det_idx])
                result[tid] = gid
                used_dets.add(det_idx)
                used_gids.add(gid_idx)

        # Create new GIDs for unmatched detections
        for i in range(num_detections):
            if i not in used_dets:
                gid = self._create_gid(embeddings[i])
                tid = local_ids[i]
                self.local_to_global[(video_name, tid)] = gid
                result[tid] = gid
        return result

    def _create_gid(self, embedding: np.ndarray) -> int:
        """Creates a new Global ID with an  initial normalized embedding."""
        gid = self.global_counter
        self.global_counter += 1
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        self.global_embeddings[gid] = embedding.copy()
        return gid

    def _update_embedding(self, gid: int, new_embedding: np.ndarray):
        """Updates an existing Global ID embedding using momentum averaging."""
        old_embedding = self.global_embeddings[gid]
        updated = self.embedding_momentum * old_embedding + (1.0 - self.embedding_momentum) * new_embedding
        updated /= (np.linalg.norm(updated) + 1e-6)
        self.global_embeddings[gid] = updated

# ========================================
# ReID Model Setup
# ========================================
# Load pretrained OSNet model for person re-identification
reid_model = models.build_model(name="osnet_ibn_x1_0", num_classes=1000, pretrained=True).to(DEVICE).eval()

# Image preprocessing transforms
reid_transform = TV.Compose(
    [TV.ToPILImage(), TV.Resize((256, 128)), TV.ToTensor(), TV.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225]), ])

@torch.no_grad()
def extract_reid_embedding(frame_bgr, xyxy):
    """
    Extracts a normalized ReID embedding for a given bounding box.
    Returns a numpy array of size (d,).
    """
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(512, dtype=np.float32)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(512, dtype=np.float32)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = reid_transform(crop_rgb).unsqueeze(0).to(DEVICE)
    embedding = reid_model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    embedding /= (np.linalg.norm(embedding) + 1e-6)
    return embedding

# ========================================
# Utility Functions
# ========================================
def compute_optical_flow_magnitude(prev_gray, gray, roi=None):
    """Computes maximum optical flow magnitude inside ROI."""
    a, b = prev_gray, gray
    if roi is not None:
        x1, y1, x2, y2 = roi
        a, b = prev_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2]
        if a.size == 0 or b.size == 0:
            return 0.0
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.max(mag))

def update_identity_catalogue(catalogue_path, gid, video_name, frame_idx):
    """Updates the identity catalogue JSON with (GID → video/frame range)."""
    if os.path.exists(catalogue_path):
        with open(catalogue_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    gid_str = str(gid)
    if gid_str in data:
        clips = data[gid_str].setdefault("clip_id & frame ranges", [])
        exists = any(c[0] == video_name for c in clips)
        if exists:
            idx = next(i for i, item in enumerate(clips) if item[0] == video_name)
            clips[idx] = (video_name, (clips[idx][1][0], frame_idx))
        else:
            clips.append((video_name, (frame_idx, frame_idx)))
    else:
        data[gid_str] = {"clip_id & frame ranges": [(video_name, (frame_idx, frame_idx))]}

    with open(catalogue_path, "w") as f:
        json.dump(data, f, indent=1)

def mark_suspicious_activity(max_val, gid, frame_idx, id_catalogue_dict, roi, frame_susp, output_path, video_name):
    """Marks a suspicious GID on the frame and saves an annotated image."""
    x1, y1, x2, y2 = roi
    id_catalogue_dict.setdefault(str(gid), []).append({"frame": frame_idx, "max_val": round(max_val, 1)})
    label = f"GID {gid} | Max={max_val:.1f}"

    cv2.rectangle(frame_susp, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame_susp, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    out_path_susp = os.path.join(f"{output_path}/suspicious_frames", f"{video_name}")
    os.makedirs(out_path_susp, exist_ok=True)
    cv2.imwrite(f"{out_path_susp}/frame_no{frame_idx}_GID{gid}_max{max_val:.2f}.jpg", frame_susp)

def saves_suspicious_results(id_catalogue_dict, video_name, output_path, json_labelling, min_frames):
    """Filters short detections and saves suspicious activity metadata."""
    suspicious = False

    for gid, frames in list(id_catalogue_dict.items()):
        if len(frames) < min_frames:
            id_catalogue_dict.pop(str(gid))
            out_path_susp = os.path.join(f"{output_path}/suspicious_frames", f"{video_name}")

            for fname in os.listdir(out_path_susp):
                if f"GID{gid}_" in fname:
                    os.remove(os.path.join(out_path_susp, fname))
            if not os.listdir(out_path_susp):
                os.rmdir(out_path_susp)

        else:
            suspicious = True

    data_entry = {"clip_id": video_name,
                  "label": "Crime" if suspicious else "Normal",
                  "justification": id_catalogue_dict if suspicious else {}}

    if os.path.exists(json_labelling):
        with open(json_labelling, "r") as f:
            data = json.load(f)
    else:
        data = []

    if not isinstance(data, list):  data = [data]

    data.append(data_entry)
    with open(json_labelling, "w") as f:
        json.dump(data, f, indent=2)

# ========================================
# Video Processing Pipeline
# ========================================
def process_video(video_path, video_name, output_path, json_globalID, max_thres):
    """
    - Tracks persons (YOLO + BoT-SORT)
    - Assigns Global IDs using ReID embeddings
    - Detects suspicious activity via optical flow
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path_globalID = os.path.join(output_path, "person_catalogue")
    os.makedirs(out_path_globalID, exist_ok=True)
    writer = cv2.VideoWriter(f"{out_path_globalID}/globalID_{video_name}", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))

    gid_tracker = process_video.gid_tracker
    yolo = YOLO("yolo11m.pt")

    results_gen = yolo.track(source=video_path, tracker="botsort.yaml", stream=True, conf=0.4, iou=0.5, classes=[0],
                             verbose=False)

    prev_gray = None
    id_catalogue_dict = {}
    frame_idx = 0

    for res in results_gen:
        frame = res.orig_img.copy()
        frame_susp = res.orig_img.copy()
        gray = cv2.cvtColor(frame_susp, cv2.COLOR_BGR2GRAY)
        if prev_gray is None: prev_gray = gray

        if res.boxes is None or res.boxes.xyxy is None:
            writer.write(frame)
            continue

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id

        if ids is None:
            writer.write(frame)
            continue

        local_ids = ids.int().cpu().numpy()
        embeddings = np.stack([extract_reid_embedding(frame, b) for b in xyxy], axis=0)

        gid_map = gid_tracker.assign_global_ids(video_name, local_ids, embeddings)

        for i in range(len(local_ids)):
            lid = int(local_ids[i])
            gid = gid_map[lid]
            x1, y1, x2, y2 = map(int, xyxy[i])

            # Compute optical flow magnitude within bounding box ROI
            if prev_gray is not None:
                roi = [x1, y1, x2, y2]
                max_val = compute_optical_flow_magnitude(prev_gray, gray, roi)
                if max_val >= max_thres:
                    mark_suspicious_activity(max_val, gid, frame_idx, id_catalogue_dict, roi, frame_susp, output_path, video_name)

            # Draw bounding boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
            label = f"GID {gid} | LID {lid}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)

            update_identity_catalogue(json_globalID, gid, video_name, frame_idx)

        prev_gray = gray
        frame_idx += 1
        writer.write(frame)

    writer.release()
    return id_catalogue_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Person Re-Identification & Scene Classification Pipeline")

    parser.add_argument("--video_dir", type=str, default="./videos", help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save videos with global ID and suspicious frames")
    parser.add_argument("--json_globalID", type=str, default="PersonCatalogue.json",
                        help="Path to global identity catalogue JSON file")
    parser.add_argument("--json_labelling", type=str, default="SceneLabelling.json",
                        help="Path to scene labelling JSON output file")
    parser.add_argument("--sim_thresh", type=float, default=0.65, help="Similarity threshold for ReID matching (0–1)")
    parser.add_argument("--momentum", type=float, default=0.85, help="Momentum for updating global embeddings (0–1)")
    parser.add_argument("--max_thres", type=float, default=24, help="Thershold for maximum magnitude")
    parser.add_argument("--min_frames", type=float, default=5, help="Minimum frames to detect crime")
    args, _ = parser.parse_known_args()

    # Create output folders
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Global Tracker
    process_video.gid_tracker = GlobalIdentityTracker(similarity_threshold=args.sim_thresh,
                                                      embedding_momentum=args.momentum)

    # Get list of videos
    videos = sorted([f for f in os.listdir(args.video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))])

    # Process each video
    for video_file in videos:
        video_path = os.path.join(args.video_dir, video_file)
        id_catalogue_dict = process_video(video_path, video_file, args.output_dir, args.json_globalID, args.max_thres)
        saves_suspicious_results(id_catalogue_dict, video_file, args.output_dir, args.json_labelling, args.min_frames)

        print(f"✅ Finished processing video: {video_file}")

    print("Done! All videos processed.")