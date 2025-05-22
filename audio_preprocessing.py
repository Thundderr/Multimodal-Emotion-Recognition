import os
from glob import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from scipy.stats import zscore

# Folder structure
AUDIO_DIR = "data/audio"
MFCC_DIR = "data/mfcc"
os.makedirs(MFCC_DIR, exist_ok=True)

# Constants — Feel free to tune!
SR = 16000          # Sample rate
N_MFCC = 40         # Number of MFCCs
MAX_LEN = 300       # Number of frames after padding/truncation
AUG_PER_SAMPLE = 2  # How many noisy copies to generate per file

# Emotion mapping from RAVDESS filename codes
label_dict = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

# Adds white noise at random SNRs to increase robustness
def noisy_signal(y, snr_low=15, snr_high=30, num_augmented=AUG_PER_SAMPLE):
    signal_len = len(y)
    if signal_len == 0:
        return []

    noise = np.random.normal(size=(num_augmented, signal_len))
    s_power = np.mean(y ** 2)
    n_power = np.mean(noise ** 2, axis=1)
    snr = np.random.uniform(snr_low, snr_high, size=num_augmented)
    k = np.sqrt((s_power / n_power) * 10 ** (-snr / 10)).reshape(-1, 1)
    return [y + k[i] * noise[i] for i in range(num_augmented)]

# Extract MFCC + delta + delta2; normalize, trim silence, filter bad audio
def extract_mfcc(y):
    y, _ = librosa.effects.trim(y)

    if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
        return None  # Skip silent or broken clips

    y = zscore(y)
    y = np.nan_to_num(y)

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_combined = np.vstack([mfcc, delta, delta2])
    mfcc_combined = np.nan_to_num(mfcc_combined)

    if mfcc_combined.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc_combined.shape[1]
        mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_combined = mfcc_combined[:, :MAX_LEN]

    return mfcc_combined

# Main loop: process original + augmentations, write metadata
metadata = []
for fpath in tqdm(glob(os.path.join(AUDIO_DIR, "*.wav")), desc="Extracting MFCCs"):
    fname = os.path.basename(fpath)
    emotion_code = fname.split("-")[2]
    label = label_dict.get(emotion_code, "unknown")

    y, _ = librosa.load(fpath, sr=SR)

    # Original
    mfcc = extract_mfcc(y)
    if mfcc is not None:
        out_path = os.path.join(MFCC_DIR, fname.replace(".wav", ".npy"))
        np.save(out_path, mfcc)
        metadata.append({"file": out_path, "label": label})

    # Augmentations
    #for i, y_aug in enumerate(noisy_signal(y)):
    #    mfcc_aug = extract_mfcc(y_aug)
    #    if mfcc_aug is not None:
    #        aug_name = fname.replace(".wav", f"_aug{i}.npy")
    #        aug_path = os.path.join(MFCC_DIR, aug_name)
    #        np.save(aug_path, mfcc_aug)
    #        metadata.append({"file": aug_path, "label": label})

# Save metadata
df = pd.DataFrame(metadata)
print(df.head())
df.to_csv(os.path.join(MFCC_DIR, "metadata.csv"), index=False)
print(f"✅ Saved {len(df)} MFCC samples and metadata (with augmentation)")
