import os
from glob import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# Folder structure
AUDIO_DIR = "data/audio"
MFCC_DIR = "data/mfcc"
os.makedirs(MFCC_DIR, exist_ok=True)

# Constants
SR = 16000
N_MFCC = 40
MAX_LEN = 300

label_dict = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def extract_mfcc(filepath):
    y, _ = librosa.load(filepath, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc

# Process all .wav files
metadata = []
for fpath in tqdm(glob(os.path.join(AUDIO_DIR, "*.wav")), desc="Extracting MFCCs"):
    fname = os.path.basename(fpath)
    emotion_code = fname.split("-")[2]
    label = label_dict.get(emotion_code, "unknown")

    mfcc = extract_mfcc(fpath)
    out_path = os.path.join(MFCC_DIR, fname.replace(".wav", ".npy"))
    np.save(out_path, mfcc)
    
    metadata.append({"file": out_path, "label": label})

# Save metadata
df = pd.DataFrame(metadata)
df.to_csv(os.path.join(MFCC_DIR, "metadata.csv"), index=False)
print("✅ Saved MFCC features and metadata.")
