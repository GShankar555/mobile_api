from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
from utils import read_video, save_video
from trackers import Tracker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = BASE_DIR / "models/best_seg.pt"
DETECTIONS_DIR = BASE_DIR / "detections"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file received")

    input_path = UPLOAD_DIR / file.filename
    output_video_path = OUTPUT_DIR / f"processed_{file.filename}"
    detections_path = DETECTIONS_DIR / f"{Path(file.filename).stem}_detections.pkl"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_frames, fps = read_video(str(input_path))

    tracker = Tracker(str(MODEL_PATH), str(detections_path))
    tracks = tracker.get_object_tracks(video_frames)
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, str(output_video_path), fps)

    return {"download_url": f"/download_video/{output_video_path.name}"}

@app.get("/download_video/{file_name}")
async def download_video(file_name: str):
    file_path = OUTPUT_DIR / file_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="video/mp4", filename=file_name)
