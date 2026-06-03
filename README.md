# TrashCam — AI-Powered Littering Detection & Enforcement System

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Repository Structure](#4-repository-structure)
5. [AI/ML Pipeline — Deep Dive](#5-aiml-pipeline--deep-dive)
   - 5.1 [Litter Detection (YOLOv8 + SORT)](#51-litter-detection-yolov8--sort)
   - 5.2 [Face Detection & Recognition (YOLOv8 + FaceNet/VGGFace2)](#52-face-detection--recognition-yolov8--facenetvggface2)
   - 5.3 [License Plate Detection & OCR (YOLOv8 + TrOCR)](#53-license-plate-detection--ocr-yolov8--trocr)
   - 5.4 [Car & Pedestrian Detection (YOLOv8 + SORT)](#54-car--pedestrian-detection-yolov8--sort)
   - 5.5 [The SORT Tracking Algorithm](#55-the-sort-tracking-algorithm)
   - 5.6 [Post-Processing & JSON Filtering](#56-post-processing--json-filtering)
   - 5.7 [Entity Association & Offender Identification](#57-entity-association--offender-identification)
   - 5.8 [LLM Report Generation (Ollama + Qwen2.5)](#58-llm-report-generation-ollama--qwen25)
6. [Video Processing Orchestration](#6-video-processing-orchestration)
7. [Model Training & Performance](#7-model-training--performance)
8. [ORL Face Database](#8-orl-face-database)
9. [Backend API](#9-backend-api)
10. [Database Schema](#10-database-schema)
11. [Frontend Application](#11-frontend-application)
12. [Cloud Infrastructure](#12-cloud-infrastructure)
13. [Environment Variables & Configuration](#13-environment-variables--configuration)
14. [Installation & Setup](#14-installation--setup)
15. [Data Flow Reference](#15-data-flow-reference)
16. [Design Decisions & Trade-offs](#16-design-decisions--trade-offs)
17. [Performance Metrics](#17-performance-metrics)
18. [Future Work](#18-future-work)

---

## 1. Project Overview

TrashCam is an end-to-end AI surveillance system designed to automatically detect littering events from video footage, identify the offender by face and/or vehicle license plate, and generate a formal incident report with an associated fine. The system integrates four independent computer vision models, a Kalman-filter-based multi-object tracker, a Transformer OCR model, a deep metric learning face recogniser, and a locally-hosted large language model — all orchestrated into a single video processing pipeline.

The web application provides law enforcement officers with a dashboard showing incident statistics, an interactive heatmap of offense locations, a searchable report list, and offender profile pages with identity card images pulled from AWS S3.

---

## 2. The Problem It Solves

Pakistan currently generates approximately **36 million tons of solid waste per year**, a figure projected to reach **85 million tons by 2050**. Enforcement is almost entirely manual and reactive. Existing solutions (Litterati, Clean Swell, LitterCam) either require citizen participation or are passive logging tools with no automated identification or fine-issuance capability.

TrashCam fills this gap by:
- Processing CCTV or dashcam footage **automatically** — no human review required
- Identifying **who** discarded litter via facial recognition against a known-persons database
- Identifying the **vehicle** via license plate OCR, even when the face is obscured
- Producing a **legally-formatted incident report** and issuing a fine record, all without human intervention

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS S3 Bucket                               │
│   to_process/ ──► raw video files (uploaded by cameras / officers) │
│   processed/  ◄── annotated output videos                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ poll every 10 s
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Python Backend (FastAPI + Uvicorn)                  │
│                                                                     │
│  ┌──────── Detection Worker Thread (daemon) ────────────────────┐  │
│  │                                                               │  │
│  │  download_and_process_video()                                 │  │
│  │        │                                                      │  │
│  │        ├─► process_litter()          → litter.json           │  │
│  │        ├─► process_number_plate()    → plate.json            │  │
│  │        ├─► process_face()            → faces.json            │  │
│  │        └─► process_car_pedestrian()  → car_pedestrian.json   │  │
│  │                   │                                           │  │
│  │        preprocess_json_files()  (filter & de-noise)          │  │
│  │                   │                                           │  │
│  │        generate_processed_video()   (annotated MP4)          │  │
│  │                   │                                           │  │
│  │        identify_associations()      (who → what → vehicle)   │  │
│  │                   │                                           │  │
│  │        generate_report()            (LLM → text report)      │  │
│  │                   │                                           │  │
│  │        write to PostgreSQL                                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  REST API                                                           │
│  /auth  /dashboard  /reports  /offenders  /detect                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP / JSON
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                React + Vite Frontend (port 5173)                   │
│  Dashboard · Report List · Report Detail · Offender Profile        │
│  Video Upload · Heatmap (Leaflet) · Charts (Recharts)              │
└─────────────────────────────────────────────────────────────────────┘
```

The detection worker runs as a **background daemon thread** separate from the FastAPI request loop, so API requests are never blocked by the (potentially minutes-long) video processing job.

---

## 4. Repository Structure

```
trashcam-app/
│
├── trashcam-backend/
│   ├── requirements.txt
│   └── src/
│       ├── main.py                        # FastAPI app, router wiring
│       │
│       ├── apis/
│       │   ├── auth/
│       │   │   ├── router.py              # POST /auth/login
│       │   │   ├── model.py               # Pydantic: UserLogin
│       │   │   ├── sessions.py            # JWT creation
│       │   │   └── dependencies.py        # OAuth2 bearer validation
│       │   ├── dashboard/
│       │   │   └── router.py              # Statistics & heatmap endpoints
│       │   ├── reports/
│       │   │   └── router.py              # CRUD for incident reports
│       │   ├── offenders/
│       │   │   └── router.py              # Offender profiles & ID cards
│       │   └── detect/
│       │       └── router.py              # POST /detect/ (video upload)
│       │
│       ├── detect/                        ◄── ALL AI LOGIC LIVES HERE
│       │   ├── video.py                   # Pipeline orchestrator
│       │   ├── detect_litter.py           # YOLO + SORT litter tracker
│       │   ├── detect_face.py             # YOLO + FaceNet face recogniser
│       │   ├── detect_number_plate.py     # YOLO + TrOCR plate reader
│       │   ├── detect_car_pedestrian.py   # YOLO + SORT vehicle/person tracker
│       │   ├── identify.py                # Cross-modal entity association
│       │   └── report_gen.py              # Ollama LLM report formatter
│       │
│       ├── utils/
│       │   ├── database.py                # Async PostgreSQL helpers
│       │   ├── helpers.py                 # Reverse geocoding (Nominatim)
│       │   └── s3_storage.py              # Boto3 S3 helpers
│       │
│       ├── orl_faces/                     # Face database (one dir per person)
│       │   ├── Abdullah/  (10 × .pgm)
│       │   ├── Ahmad/     (10 × .pgm)
│       │   └── …
│       └── temp/                          # Intermediate JSON files
│
└── trashcam-frontend/
    └── src/
        ├── pages/
        │   ├── dashboard/
        │   ├── detect/
        │   ├── listReports/
        │   ├── reportDetails/
        │   ├── offenderProfile/
        │   ├── landing/
        │   └── login/
        └── layout/sidebar/
```

---

## 5. AI/ML Pipeline — Deep Dive

This is the core of the system. Four independent detection modules run **sequentially** over the same video file, each writing a JSON file of per-frame bounding boxes. A post-processing pass then cleans and validates those files before an entity-association layer reasons across all four streams to answer: *"Who threw what, and from which vehicle?"*

### 5.1 Litter Detection (YOLOv8 + SORT)

**File:** [`trashcam-backend/src/detect/detect_litter.py`](trashcam-backend/src/detect/detect_litter.py)

#### Model

A custom YOLOv8 model fine-tuned on the **pStreetLitter** dataset. The training set was augmented to **222,000 images** over 58 epochs and achieved:
- **mAP50:** 0.611
- **Precision:** 0.696
- **Recall:** 0.566

The model runs on CPU via Ultralytics at `imgsz=640` with:
- Confidence threshold: `0.45`
- IoU NMS threshold: `0.45`
- Maximum detections per frame: `100`

#### Tracking

Raw per-frame YOLO detections are fed into a **SORT** (Simple Online and Realtime Tracking) tracker configured as:
```python
Sort(max_age=5, min_hits=2, iou_threshold=0.3)
```
- `max_age=5`: a track is kept alive for up to 5 consecutive frames with no matching detection, handling momentary occlusions
- `min_hits=2`: a track must be confirmed by at least 2 consecutive detections before being assigned a stable ID, suppressing one-frame spurious detections
- `iou_threshold=0.3`: the minimum bounding box overlap required to associate a detection to an existing Kalman prediction

SORT returns `[x1, y1, x2, y2, track_id]` for each frame. For each tracked box, the code back-matches it to the nearest original detection by IoU (threshold `> 0.5`) to recover the YOLO class label, storing the winning class at each frame in `class_history[obj_id]`.

#### Output Format

After all frames are processed, `save_tracking_data()` emits `temp/litter.json`:
```json
[
  {
    "track_id": 3,
    "frame": 42,
    "x1": 310, "y1": 420, "x2": 375, "y2": 465,
    "center_x": 342, "center_y": 442,
    "class_name": "plastic_bottle"
  },
  ...
]
```
The `class_name` is determined by majority-vote across all frames a track was alive: the single most-frequently detected class wins, preventing flickering labels.

---

### 5.2 Face Detection & Recognition (YOLOv8 + FaceNet/VGGFace2)

**File:** [`trashcam-backend/src/detect/detect_face.py`](trashcam-backend/src/detect/detect_face.py)

This is a **two-stage pipeline**: first detect face bounding boxes in each frame, then compute a deep embedding for each detected face crop and compare it against a pre-built database of known faces.

#### Stage 1 — Detection (YOLOv8 + ByteTrack)

The face detection model is another fine-tuned YOLOv8 instance. Unlike the other detectors, face processing uses Ultralytics' **ByteTrack** tracker (invoked via `.track()`) rather than SORT:
```python
results = face_model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    persist=True,
    verbose=False
)
```
ByteTrack is preferred here because it handles the high density of face detections better — it retains low-confidence detections in a secondary buffer and only commits them when they get re-confirmed by a high-confidence detection in a later frame, dramatically reducing ID switches on partially-occluded faces.

#### Stage 2 — Recognition (FaceNet / InceptionResnetV1)

For every detected face crop in every frame, the system:

1. Converts the OpenCV BGR crop to a PIL RGB image
2. Resizes to **160×160** pixels (FaceNet's required input size)
3. Normalises pixel values to `[-1, 1]` using `mean=[0.5, 0.5, 0.5]` and `std=[0.5, 0.5, 0.5]`
4. Passes through **InceptionResnetV1** pretrained on **VGGFace2** (a dataset of 3.3 million face images of 9,131 identities):
   ```python
   facenet = InceptionResnetV1(pretrained='vggface2').eval()
   ```
5. Extracts the 512-dimensional unit-normalised embedding vector

#### Running Average Embedding

Rather than comparing a single-frame crop, the system accumulates all embeddings seen so far for a given `track_id` and **averages them**:
```python
avg_embedding = np.mean(face_images[track_id], axis=0)
```
This running mean reduces noise caused by pose variation, motion blur, and partial occlusion. The identification decision is deferred to each new frame, so early frames with poor quality don't permanently "lock in" a wrong identity.

#### Database Matching

The averaged embedding is compared against every entry in the ORL database using **Euclidean distance**:
```python
dist = np.linalg.norm(avg_embedding - db_emb)
identity = name if dist < 1.0 else "Unknown"
```
The threshold of `1.0` in embedding space corresponds roughly to ~60% cosine similarity. Any person with no close match is labelled `"Unknown"`. The database embeddings themselves are also means — computed once at startup from all 10 reference images per person.

#### Output Format

`temp/faces.json`:
```json
[
  {
    "track_id": 7,
    "frame": 15,
    "x1": 200, "y1": 80, "x2": 250, "y2": 140,
    "identity": "Ahmad"
  },
  ...
]
```

---

### 5.3 License Plate Detection & OCR (YOLOv8 + TrOCR)

**File:** [`trashcam-backend/src/detect/detect_number_plate.py`](trashcam-backend/src/detect/detect_number_plate.py)

This module detects and reads license plate numbers. It is architecturally the most complex of the four because it chains detection → tracking → OCR into a single loop.

#### Detection

A dedicated YOLOv8 model trained to detect license plates runs at a **lower confidence threshold of 0.3** (vs. 0.45 for litter) because plates are small, may be partially obscured, and a missed detection is worse than a false positive — false positives are filtered out by the 15-frame track-length requirement at post-processing time.

SORT is configured with more generous parameters than the litter tracker to handle the fact that a parked or slow-moving car may disappear behind other objects:
```python
Sort(max_age=30, min_hits=3, iou_threshold=0.3)
```
`max_age=30` means a plate track is maintained for 30 frames (approximately 1 second at 30fps) of non-detection before being discarded. `min_hits=3` requires three consecutive detections for confirmation.

#### TrOCR — Transformer OCR

Every 10th frame (and whenever a new track is first seen), the plate crop is passed to **Microsoft TrOCR** (`microsoft/trocr-base-printed`):
```python
pixel_values = trocr_processor(pil_img, return_tensors="pt").pixel_values
generated_ids = trocr_model.generate(pixel_values)
ocr_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
```
TrOCR is a **Vision Encoder + Text Decoder** model (a ViT image encoder paired with a RoBERTa-style language decoder) fine-tuned for printed text recognition. It outperforms classical OCR (Tesseract) on license plates because it:
- Handles irregular fonts, glare, and partial blur natively
- Has a language model prior — it "knows" that license plate text follows certain patterns
- Produces end-to-end results without needing character segmentation

OCR runs only every 10 frames to avoid the inference cost on every frame. The most recently successful OCR result is stored in `ocr_history[obj_id]` and attached to every frame entry in the final JSON, so downstream consumers always have the best available plate text.

#### Output Format

`temp/plate.json`:
```json
[
  {
    "track_id": 2,
    "frame": 30,
    "x1": 450, "y1": 510, "x2": 530, "y2": 540,
    "center_x": 490, "center_y": 525,
    "ocr_text": "ABC-1234"
  },
  ...
]
```

---

### 5.4 Car & Pedestrian Detection (YOLOv8 + SORT)

**File:** [`trashcam-backend/src/detect/detect_car_pedestrian.py`](trashcam-backend/src/detect/detect_car_pedestrian.py)

A standard YOLOv8 model (pretrained on COCO-128) detects persons and vehicles. The model predicts 80 COCO classes but the code **filters to only two**:
- Class `0` → `person`
- Class `2` → `car`

```python
if cls_id in [0, 2]:
    detections.append([x1, y1, x2, y2, conf, cls_id])
```

SORT configuration matches litter detection (`max_age=5, min_hits=2, iou_threshold=0.3`). The same class-assignment logic (majority vote across track lifetime) applies.

**Why this module exists separately from face detection:** Face detection identifies *who* is in frame. Car/pedestrian detection provides the *spatial bounding boxes* of cars and persons used by the entity association logic to:
1. Determine whether a recognised face is *inside* a vehicle (which changes the nature of the offense — is the litter thrown from a car window?)
2. Resolve which car a license plate belongs to (by IoU overlap in the same frame)
3. Infer the direction a pedestrian was moving relative to where litter appeared

Output: `temp/car_pedestrian.json` (same schema as `litter.json` with `class_name` of either `"person"` or `"car"`).

---

### 5.5 The SORT Tracking Algorithm

All four detection modules use SORT (with ByteTrack as an alternative for faces). Understanding SORT is essential to understanding why the system produces stable track IDs across hundreds of frames.

SORT works in three steps per frame:

**Step 1 — Predict:** Each existing Kalman filter is propagated forward one time step. The state vector is `[u, v, s, r, u̇, v̇, ṡ]` where `(u,v)` is the box center, `s` is the scale (area), and `r` is the aspect ratio. Velocity components are initialised to zero for new tracks. The Kalman prediction gives an *expected* bounding box location even if no detection arrives in this frame.

**Step 2 — Associate:** Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) matches the N predicted boxes to the M YOLO detections using 1 - IoU as the cost matrix. Any detection with IoU below `iou_threshold` relative to its matched prediction is treated as unmatched.

**Step 3 — Update:** Matched filters are updated with the measurement (the actual YOLO box). Unmatched tracks increment their `age` counter; tracks that exceed `max_age` are deleted. Unmatched detections that satisfy `min_hits` are promoted to confirmed tracks with new IDs.

The end result: each litter item, car, plate, and person gets a **persistent integer track ID** that stays consistent across frames even through partial occlusions, allowing the post-processing layer and entity-association layer to reason about the same object over time.

---

### 5.6 Post-Processing & JSON Filtering

**File:** [`trashcam-backend/src/detect/video.py`](trashcam-backend/src/detect/video.py) — `preprocess_json_files()`

Raw detection JSON files contain noise: short-lived spurious tracks, persons wrongly classified as litter, and conflicting face identities. This function cleans them before any reasoning happens.

#### Filter 1 — Minimum Track Length (≥ 15 frames)

```python
def get_valid_tracks(data, min_frames=15):
    counter = defaultdict(int)
    for item in data:
        counter[item['track_id']] += 1
    return {tid for tid, count in counter.items() if count >= min_frames}
```

Any track observed in fewer than 15 frames is discarded. At 30fps this is 0.5 seconds — long enough to confirm a real detection but short enough not to miss fast-moving objects. This single filter removes most false positives.

#### Filter 2 — Person-Litter IoU Check

A critical source of false positives is the litter model occasionally predicting a "litter" box that overlaps heavily with a person's body (e.g., a bag held in someone's hand before it is discarded). This filter computes IoU between each litter box and every valid person box in the same frame:

```python
filtered_litter = [
    item for item in litter_data
    if item['track_id'] in valid_litter
    and not any(
        calculate_iou(
            (item['x1'], item['y1'], item['x2'], item['y2']),
            p_bbox
        ) > 0.5
        for p_bbox in person_boxes.get(item['frame'], [])
    )
]
```

If the litter box overlaps a person box by more than 50%, that frame entry is removed. This prevents the system from "detecting" a person carrying a bag as litter.

#### Filter 3 — Face Identity Conflict Resolution

Face recognition across many frames produces identity conflicts: the same physical person may be labelled differently across frames due to pose changes or partial occlusion. The system resolves this in two passes:

**Pass 1 — Minimum Identity Frequency:**
For a given `track_id`, count how many times each `identity` label appears. If the most common identity appears fewer than 15 times, the system looks for a *spatially overlapping* track (IoU > 0.5 in the same frame) whose dominant identity does exceed 15 appearances, and borrows that identity.

**Pass 2 — Spatial Area Clustering:**
The frame is divided into a coarse 100×100-pixel grid. For each grid cell, a count is maintained of how many times each identity has been seen. If a face in that cell is currently labelled identity A, but identity B has been seen more often in that same cell, the label is replaced with B.

**Final Gate:**
After both replacement passes, any `(track_id, identity)` combination that still has fewer than 15 occurrences is removed from the face data entirely.

This multi-pass approach handles the case where a person moves across the frame (different spatial cells) while maintaining consistent identification in the region where they are most visible.

---

### 5.7 Entity Association & Offender Identification

**File:** [`trashcam-backend/src/detect/identify.py`](trashcam-backend/src/detect/identify.py)

After all four detection streams have been filtered and validated, `identify_associations()` reasons across them to produce a structured answer:

```json
{
  "cars":    {"Car5": "ABC-1234"},
  "persons": {"Ahmad": true},
  "trash":   {"Trash3": "Ahmad"}
}
```

This is done in three sub-analyses:

#### 5.7.1 Entity Consolidation (`consolidate_entities`)

Different SORT instances may assign different track IDs to the same physical entity across the video (e.g., if it briefly leaves frame). Consolidation merges these:

- **Faces:** grouped by `identity` string (all track IDs sharing the same name are merged into one set)
- **Plates:** grouped by `ocr_text` (all track IDs that read the same plate number are merged)
- **Cars ↔ Plates:** For each `car` entry and each `plate` entry in the same frame, if their bounding boxes overlap by IoU > 0.3, the plate is associated with the car. Cars sharing at least one common plate number are then merged into a single group.

#### 5.7.2 Car–Plate Matching (`analyze_car_plates`)

Within each merged car group, the most frequently seen plate text is selected as the canonical plate number for that vehicle. The car group is then represented by the lowest-numbered track ID (`Car{min(group['cars'])}`):
```python
canonical_car = f"Car{min(group['cars'])}"
car_plates[canonical_car] = main_plate
```

#### 5.7.3 Person-in-Car Test (`analyze_person_status`)

For each recognised face, the function checks whether the face centre point falls within any car bounding box in the same frame:
```python
if (car_bbox[0] <= face_center[0] <= car_bbox[2] and
    car_bbox[1] <= face_center[1] <= car_bbox[3]):
    in_car[identity] = True
```
The result is a boolean map `{"Ahmad": True}` meaning Ahmad was detected inside a car. This is relevant for fine calculation (throwing litter from a vehicle is a different legal category) and helps link the face to a vehicle even if the plate is briefly unreadable.

#### 5.7.4 Trash Movement Analysis (`analyze_trash_movement`)

This is the key reasoning step that determines **who threw the litter**.

The system computes a **trajectory** for each litter track — the sequence of `center_x` positions across its observed frames. The direction of movement is:
- `"right"` if `end_x > start_x + 20`
- `"left"` if `end_x < start_x - 20`
- `"stationary"` otherwise

Then, for each known person (from the face recognition output), the person's average horizontal position over all frames is computed. The offender is the person positioned such that the litter appears to be **moving away from them**:

- If litter is moving right → the offender is the person with the largest average X coordinate *less than* the litter's start position (i.e., they are to the left of the litter, which is moving rightward away from them)
- If litter is moving left → symmetric logic

```python
if direction == "right" and avg_x < start:
    dist = start - avg_x
elif direction == "left" and avg_x > start:
    dist = avg_x - start
else:
    dist = abs(avg_x - start)
```

The person with the minimum `dist` is declared the offender for that litter item.

This spatial-trajectory approach is deterministic, requires no additional model, and works purely from the bounding box centroids already produced by earlier stages.

---

### 5.8 LLM Report Generation (Ollama + Qwen2.5)

**File:** [`trashcam-backend/src/detect/report_gen.py`](trashcam-backend/src/detect/report_gen.py)

Once offender identification is complete, the structured incident data is passed to a locally-hosted LLM to produce a human-readable, legally-formatted incident report.

#### Model

The system uses **Qwen2.5:0.5b** via Ollama — a 500-million-parameter quantised language model that runs entirely on-device (no external API calls, no data leaving the server):

```python
response = ollama.chat(
    model='qwen2.5:0.5b',
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_summary}
    ]
)
```

The 500M parameter size was chosen to balance report quality with inference speed on CPU hardware. The model produces coherent formal prose without requiring a GPU.

#### System Prompt Engineering

The system prompt instructs the model with explicit formatting rules:
- Begin with date and time
- Name the offender (or "an unidentified individual" if Unknown)
- Describe the littering action
- Include license plate if available
- Include location (or `[REDACTED]` if unavailable)
- State fine amount and status
- Single formal paragraph, no bullet points, no invented details

#### Input Structure

```
Incident Date: 2025-06-01
Incident Time: 14:32:07
Offender: Ahmad
Object Discarded: plastic_bottle
Car Plate: ABC-1234
Location: Sector G-9/3, Islamabad
Fine Amount: PKR 5000
Fine Status: Pending
```

This structured input is assembled from the identification result, the geolocation lookup, and the database configuration for fine amounts. The LLM's role is purely **natural language formatting** — it never makes reasoning decisions. All factual content is provided by the deterministic pipeline stages above.

---

## 6. Video Processing Orchestration

**File:** [`trashcam-backend/src/detect/video.py`](trashcam-backend/src/detect/video.py)

### Model Loading (`main_loop`)

On startup, `main_loop()` initialises all models once and keeps them in memory for the lifetime of the process:

```
1. setup_environment()           — create .ocr_model cache dir, set env vars
2. YOLO(MODEL_PATH)              — litter model
3. YOLO(NUMBER_PLATE_MODEL)      — plate model
4. TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
5. VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
6. YOLO(FACE_MODEL_PATH)         — face detector
7. InceptionResnetV1('vggface2') — FaceNet
8. build_orl_database(...)       — compute mean embeddings for all known faces
9. YOLO(CAR_OR_PERSON_MODEL_PATH)— car/pedestrian model
```

Models are loaded lazily (only if not already loaded) so restarts or hot-reloads don't force redundant re-loading.

### Processing Loop

```python
while True:
    if not download_and_process_video():
        time.sleep(10)
```

The loop polls S3 every 10 seconds when idle. When a video is found:

1. **Download:** Streams the video in 8KB chunks to a temp file
2. **Sequential detection:** All four detectors run one after another on the same temp file, each writing their JSON to `temp/`
3. **Post-process:** `preprocess_json_files()` cleans all four JSONs in-place
4. **Visualise:** `generate_processed_video()` re-reads the original video frame-by-frame and draws all detections, then writes an annotated MP4
5. **Upload:** The annotated MP4 is uploaded to `processed/` on S3 with up to 3 retries
6. **Delete original:** The source file is removed from `to_process/` on S3
7. **Identify:** `identify_associations()` runs cross-modal reasoning
8. **Report:** `generate_report()` formats the LLM report
9. **Store:** Results written to PostgreSQL

### Annotated Video Generation (`generate_processed_video`)

Bounding boxes are drawn with a colour-coded scheme for immediate visual interpretation:

| Detection Type | Colour    | RGB         |
|----------------|-----------|-------------|
| Litter         | Green     | (0, 255, 0) |
| License Plates | Blue      | (255, 0, 0) |
| Faces          | Red       | (0, 0, 255) |
| Car/Pedestrian | Yellow    | (255, 255, 0)|
| Litter↔Plate connection | Magenta | (255, 0, 255)|

A **magenta line** is drawn between every litter center point and every plate center point visible in the same frame, plus a filled circle at each end. This visually confirms the spatial association between litter events and nearby vehicles.

---

## 7. Model Training & Performance

| Model | Architecture | Dataset | Epochs | mAP50 | Precision | Recall |
|-------|-------------|---------|--------|-------|-----------|--------|
| Litter Detector | YOLOv8 (custom) | pStreetLitter (222k images, augmented) | 58 | 0.611 | 0.696 | 0.566 |
| License Plate Detector | YOLOv8 (custom) | License plate dataset | — | — | — | — |
| Face Detector | YOLOv8 (custom) | Face detection dataset | — | — | — | — |
| Car/Pedestrian | YOLOv8 (COCO pretrain) | COCO-128 | pretrained | — | — | — |
| Face Embedder | InceptionResnetV1 | VGGFace2 (3.3M images, 9131 IDs) | pretrained | — | — | — |
| Plate OCR | TrOCR ViT+RoBERTa | Printed text (Microsoft pretrain) | pretrained | — | — | — |

**System-level tracking metric:**
- **MOTA (Multi-Object Tracking Accuracy):** 76.4% on the litter tracker
- **ID Switch Rate:** 1.2 per minute

**Non-functional targets from requirements:**
- Detection latency: < 5 seconds per frame batch
- Report generation: < 10 seconds
- System MTBF: 200+ hours

---

## 8. ORL Face Database

The `orl_faces/` directory contains the system's known-person reference database. Each subdirectory is named after a person and contains exactly **10 grayscale `.pgm` images** of that person from different angles and expressions.

At startup, `build_orl_database()` processes every image:

```python
for person_dir in dataset_path.iterdir():
    for img_path in person_dir.glob("*"):
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)   # FaceNet requires 3-channel
        embeddings_list.append(extract_embedding(image, facenet_model))
    embeddings[person_dir.name] = np.mean(embeddings_list, axis=0)
```

The **mean embedding** across all 10 images is stored as the canonical reference for that person. This averaging strategy makes the reference robust to the variation in the 10 reference images.

To add a new person to the database:
1. Create a directory `orl_faces/<PersonName>/`
2. Add 10 representative face images (any format OpenCV can read; `.pgm` is conventional)
3. Restart the backend — the database is rebuilt on startup

---

## 9. Backend API

**Framework:** FastAPI 0.115 / Uvicorn 0.34  
**Auth:** JWT (PyJWT) via OAuth2 Bearer tokens

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Email + password → JWT access token |

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard/get_all_report_dates` | Monthly incident counts for bar charts |
| GET | `/dashboard/get_longitude_latitude` | Lat/lon list for Leaflet heatmap |

### Reports

| Method | Endpoint | Query Params | Description |
|--------|----------|-------------|-------------|
| GET | `/reports/get_list_of_reports` | `searchTerm`, `fineStatus`, `location`, `startRow`, `endRow` | Paginated, filtered report list |
| GET | `/reports/get_single_report` | `report_id` | Full report detail including offender |
| POST | `/reports/create_report` | — | Manually create a report (body: JSON) |

### Offenders

| Method | Endpoint | Query Params | Description |
|--------|----------|-------------|-------------|
| GET | `/offenders/get_offender_profile` | `offender_id` | Profile + full report history |
| GET | `/offenders/get_offender_personal_details` | `offender_id` | Name, CNIC, address |
| GET | `/offenders/get_offender_idcard` | `cnic` | Fetch ID card image from S3 |

### Detection

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect/` | Upload a video file for processing |

---

## 10. Database Schema

PostgreSQL database managed via raw psycopg2 with async wrappers.

```
┌──────────┐       ┌──────────────────┐       ┌───────────────┐
│  users   │       │     reports      │       │   offenders   │
├──────────┤       ├──────────────────┤       ├───────────────┤
│ userid   │       │ reportid (PK)    │       │ offenderid(PK)│
│ email    │       │ latitude         │       │ name          │
│ passwordH│       │ longitude        │       │ cnic          │
└──────────┘       │ locationStr      │       │ address       │
                   │ timestamp        │       └───────┬───────┘
                   │ fineStatus       │               │
                   │ fineIssued       │               │
                   │ infodetails      │       ┌───────┴──────────┐
                   └────────┬─────────┘       │ report_offenders │
                            │                 ├──────────────────┤
                   ┌────────┴─────────┐       │ reportid  (FK)   │
                   │   user_reports   │       │ offenderid (FK)  │
                   ├──────────────────┤       └──────────────────┘
                   │ userid   (FK)    │
                   │ reportid (FK)    │       ┌──────────────────┐
                   └──────────────────┘       │  report_litter   │
                                              ├──────────────────┤
                                              │ reportid  (FK)   │
                                              │ litterid         │
                                              └──────────────────┘
```

**Database helper methods** (`utils/database.py`):
- `read_from_db(query, params)` — parameterised SELECT, returns rows as list of dicts
- `write_to_db(query, params, batch=False)` — INSERT/UPDATE/DELETE; supports batch mode for bulk inserts
- `generate_id()` — UUID v4 generation for new record IDs

**Geolocation** (`utils/helpers.py`):
- `reverse_geocode_geopy(lat, lon)` — converts coordinates to a human-readable address using the **Nominatim** (OpenStreetMap) API
- `get_city_sectors(city_name)` — queries the **Overpass API** for neighbourhood/sector polygons for a given city

---

## 11. Frontend Application

**Stack:** React 19 · Vite 6 · React Router DOM · Material-UI (MUI v5) · Recharts · Leaflet + react-leaflet · Axios

### Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Landing | Product overview |
| `/login` | Login | JWT auth form |
| `/dashboard` | Dashboard | Monthly bar charts (Recharts) + heatmap (Leaflet) |
| `/detect` | Video Upload | Upload video file → POST `/detect/` |
| `/reports` | Report List | Searchable/filterable paginated table |
| `/reportDetails/:id` | Report Detail | Full incident info, offender card, fine status |
| `/offenderProfile/:id` | Offender Profile | Personal details, ID card image, history |

All API calls go through Axios configured with the base URL in `src/config/config.js`. The JWT token is stored client-side and attached as `Authorization: Bearer <token>` on each request.

---

## 12. Cloud Infrastructure

| Service | Purpose |
|---------|---------|
| **AWS S3** | Video storage (`to_process/` → `processed/`), ID card image storage |
| **PostgreSQL** | Relational database for reports, offenders, users |
| **Ollama** | Local LLM inference server (no cloud dependency) |

S3 access uses boto3 for ID card image retrieval and plain HTTP `requests.get/put/delete` for video operations (presigned or public bucket URLs configured via `AWS_URL`).

---

## 13. Environment Variables & Configuration

Create a `.env` file in `trashcam-backend/src/`:

```bash
# ── Database ──────────────────────────────────────────
DB_NAME=trashcam
DB_USER=postgres
DB_PASS=your_password
DB_HOST=localhost
DB_PORT=5432

# ── AWS ───────────────────────────────────────────────
AWS_URL=https://your-bucket.s3.amazonaws.com/
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# ── Model Paths ───────────────────────────────────────
LITTER_MODEL_PATH=/path/to/litter_model.pt
NUMBER_PLATE_MODEL_PATH=/path/to/plate_model.pt
FACE_DETECTION_MODEL_PATH=/path/to/face_model.pt
CAR_OR_PERSON_MODEL_PATH=/path/to/coco_yolov8.pt
ORL_DATASET_PATH=/path/to/trashcam-backend/src/orl_faces

# ── JWT ───────────────────────────────────────────────
SECRET_KEY=a_long_random_secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ── OCR Cache ─────────────────────────────────────────
MODEL_CACHE=.ocr_model    # TrOCR weights cached here
```

---

## 14. Installation & Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- [Ollama](https://ollama.ai/) with `qwen2.5:0.5b` pulled (`ollama pull qwen2.5:0.5b`)
- Custom YOLO `.pt` model files for litter and license plate detection (trained separately)

### Backend

```bash
cd trashcam-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env           # edit with your values

# Run the API server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

To start the detection worker manually (it is normally launched automatically via startup event):
```bash
python -m src.detect.video
```

### Frontend

```bash
cd trashcam-frontend
npm install
npm run dev       # dev server at http://localhost:5173
```

### Database

Run the following SQL to create the required tables (schema inferred from query patterns):

```sql
CREATE TABLE users (
    userid UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    "passwordHash" TEXT NOT NULL
);

CREATE TABLE offenders (
    offenderid UUID PRIMARY KEY,
    name TEXT,
    cnic TEXT,
    address TEXT
);

CREATE TABLE reports (
    reportid UUID PRIMARY KEY,
    latitude FLOAT,
    longitude FLOAT,
    "locationStr" TEXT,
    "timestamp" TIMESTAMP,
    "fineStatus" TEXT,
    "fineIssued" FLOAT,
    infodetails TEXT
);

CREATE TABLE report_offenders (
    reportid UUID REFERENCES reports(reportid),
    offenderid UUID REFERENCES offenders(offenderid)
);

CREATE TABLE report_litter (
    reportid UUID REFERENCES reports(reportid),
    litterid TEXT
);

CREATE TABLE user_reports (
    userid UUID REFERENCES users(userid),
    reportid UUID REFERENCES reports(reportid)
);
```

---

## 15. Data Flow Reference

```
Video file uploaded by officer / CCTV camera
        │
        ▼  (HTTP PUT)
AWS S3: to_process/<filename>.mp4
        │
        ▼  (poll every 10s)
get_video_list() parses S3 XML listing via BeautifulSoup
        │
        ▼  (HTTP GET, 8KB streaming)
Temp file: /tmp/<filename>.mp4
        │
        ├──────────────────────────────────────────┐
        │ process_litter()                         │ process_number_plate()
        │   YOLO predict → SORT track              │   YOLO predict → SORT track
        │   majority-vote class labels             │   TrOCR every 10 frames
        │   → temp/litter.json                     │   → temp/plate.json
        │                                          │
        │ process_face()                           │ process_car_pedestrian()
        │   YOLO+ByteTrack → FaceNet embedding     │   YOLO predict → SORT track
        │   running mean → Euclidean match ORL     │   filter classes 0,2 only
        │   → temp/faces.json                      │   → temp/car_pedestrian.json
        │──────────────────────────────────────────┘
        │
        ▼  preprocess_json_files()
        │   ├─ remove tracks < 15 frames
        │   ├─ remove litter overlapping persons (IoU > 0.5)
        │   └─ resolve face identity conflicts (spatial + frequency)
        │
        ▼  generate_processed_video()
        │   re-read original MP4, draw colour-coded bboxes
        │   draw magenta litter↔plate connection lines
        │   → /tmp/processed_<filename>.mp4
        │
        ▼  (HTTP PUT, up to 3 retries)
AWS S3: processed/processed_<filename>.mp4
        │
        ▼  delete_video_from_bucket()
AWS S3: to_process/<filename>.mp4  ← deleted
        │
        ▼  identify_associations()
        │   consolidate_entities() → face/plate/car groups
        │   analyze_car_plates()   → {CarN: "plate_text"}
        │   analyze_person_status()→ {name: in_car_bool}
        │   analyze_trash_movement()→ {TrashN: offender_name}
        │
        ▼  generate_report()  (Ollama + Qwen2.5:0.5b)
        │   structured dict → formal incident report text
        │
        ▼  write_to_db()
PostgreSQL: reports, offenders, report_offenders, report_litter
        │
        ▼  (GET /reports, GET /reportDetails/:id)
React frontend displays the incident report
```

---

## 16. Design Decisions & Trade-offs

### Sequential vs. Parallel Detection

All four detection modules run **sequentially** rather than in parallel threads. This is a deliberate trade-off: running four YOLO models concurrently on CPU would cause memory contention and unpredictable total processing time. Sequential execution gives predictable memory usage and simpler error recovery. On GPU hardware, parallel execution would be straightforward to enable by wrapping each call in `threading.Thread`.

### SORT vs. DeepSORT

The system uses SORT (appearance-free) rather than DeepSORT (appearance-based). DeepSORT embeds a re-identification network that learns to match objects across large gaps in appearance, but requires an additional neural network inference per tracked object per frame. At the detection volumes involved (O(10) objects per frame), SORT's Kalman-only approach is sufficient and substantially faster on CPU.

### Local LLM (Qwen2.5:0.5b)

A locally-hosted 500M parameter model was chosen over a cloud API (GPT-4, Claude, etc.) for:
- **Zero ongoing cost** — no per-token billing for potentially thousands of reports
- **Data privacy** — incident data (names, plates, locations) never leaves the server
- **Offline operation** — the system functions without internet connectivity (except S3)
- **Latency** — a small local model produces a short formatted report in ~1–2 seconds on CPU

The trade-off is lower linguistic quality than a frontier model, but the output is constrained enough (single formal paragraph, templated structure) that the small model performs acceptably.

### TrOCR over Tesseract

`pytesseract` is present in requirements as a fallback, but TrOCR is the primary OCR engine. Tesseract was tested and found unreliable on license plates due to non-standard fonts, low contrast, and perspective distortion. TrOCR's vision encoder handles these conditions without pre-processing pipelines.

### Running Average Embeddings

FaceNet embeddings are averaged across all frames of a track rather than matching frame-by-frame. This has a latency cost (identity is not confirmed until multiple frames are accumulated) but produces significantly more stable identifications, especially when early frames in a track have poor image quality.

---

## 17. Performance Metrics

| Metric | Value |
|--------|-------|
| Litter detection mAP50 | 0.611 |
| Litter detection precision | 0.696 |
| Litter detection recall | 0.566 |
| Multi-object tracking accuracy (MOTA) | 76.4% |
| ID switch rate | 1.2 per minute |
| Face recognition distance threshold | 1.0 (Euclidean in 512-D space) |
| Minimum confirmed track length | 15 frames |
| License plate IoU match threshold | 0.5 |
| Person-in-car test | point-in-bbox |
| Trash trajectory threshold (px) | ±20 pixels |
| OCR frequency | every 10 frames |

---

## 18. Future Work

From the project report and architectural analysis:

1. **3D Trajectory Reconstruction:** Current trash movement analysis uses only horizontal (X-axis) displacement. Stereo camera input or depth estimation could enable 3D trajectory analysis for more accurate offender attribution in crowded scenes.

2. **Improved Dataset Diversity:** The litter model was trained on the pStreetLitter dataset, which skews toward Western urban environments. A Pakistan-specific dataset (different road materials, lighting conditions, litter types) would improve recall from 0.566.

3. **IoT Sensor Fusion:** Integrating with environmental sensors (wind direction, vibration) could provide additional signal for attributing stationary litter to a specific passing vehicle.

4. **Privacy-Preserving Face Recognition:** Replace the raw face database with a federated or encrypted embedding store. Faces should only be matched, never stored as images in production.

5. **GPU Acceleration:** The current pipeline runs entirely on CPU. Moving YOLO inference and FaceNet embedding extraction to GPU would reduce per-video processing time from minutes to seconds.

6. **Extended Use Cases:** The pipeline architecture is modular enough to extend to illegal dumping detection, fly-tipping from vehicles, and abandoned-vehicle identification by swapping or adding YOLO models.

7. **Mobile App:** The project report specifies a planned mobile companion app for citizen-reported littering (litter photos, plate photos, incident history) that was not implemented in this release.

8. **Edge Deployment:** Quantise and prune the YOLO models for deployment on embedded hardware (Jetson Nano, Raspberry Pi with Coral TPU) to eliminate the S3 upload/download latency entirely.

---

## License

This project was developed as an academic Final Year Project at FAST-NUCES Islamabad. All rights reserved by the authors. Contact the team for any usage inquiries.
