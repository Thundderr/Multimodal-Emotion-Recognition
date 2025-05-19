import os
from glob import glob
from tqdm import tqdm
import subprocess

RAW_DIR = "Data/raw"
AUDIO_DIR = "Data/audio"
FRAMES_DIR = "Data/frames"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

mp4_files = glob(os.path.join(RAW_DIR, "*.mp4"))

for mp4_path in tqdm(mp4_files, desc="Extracting audio and frames"):
    filename = os.path.basename(mp4_path)
    audio_path = os.path.join(AUDIO_DIR, filename.replace(".mp4", ".wav"))
    frame_path = os.path.join(FRAMES_DIR, filename.replace(".mp4", ".png"))

    # Extract audio as 16kHz mono WAV
    subprocess.run([
        "ffmpeg", "-i", mp4_path,
        "-ar", "16000", "-ac", "1",
        "-y", audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract a frame at 1 second
    subprocess.run([
        "ffmpeg", "-i", mp4_path,
        "-ss", "00:00:01.000", "-vframes", "1",
        "-y", frame_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
