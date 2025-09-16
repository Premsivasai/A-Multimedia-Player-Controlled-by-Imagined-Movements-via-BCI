# scripts/extract_real_eeg_features.py
import os
import numpy as np
import pandas as pd
import mne
import pywt
from scipy.signal import welch

# ----------------------------
# CONFIG
# ----------------------------
RAW_DIR = "data/raw"
OUT_CSV = "models/real_eeg_features.csv"
OUT_NPY = "models/real_eeg_features.npy"

# Sliding window parameters
WINDOW_SIZE = 2.0   # seconds
STEP_SIZE = 0.5     # seconds

# Event → Action mapping
EVENT_TO_ACTION = {
    "T0": "pause",
    "T1": "play",
    "T2": "next",
    # optionally extend later
}

# ----------------------------
# Feature extraction functions
# ----------------------------
def compute_bandpower(epoch, sfreq):
    freqs, psd = welch(epoch, sfreq, nperseg=int(sfreq*2))
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
    return [np.mean(psd[(freqs >= low) & (freqs <= high)]) for low, high in bands.values()]

def hjorth_params(epoch):
    d1 = np.diff(epoch)
    d2 = np.diff(d1)
    var0, var1, var2 = np.var(epoch), np.var(d1), np.var(d2)
    activity = var0
    mobility = np.sqrt(var1/var0) if var0 > 0 else 0
    complexity = np.sqrt(var2/var1)/mobility if var1 > 0 and mobility > 0 else 0
    return [activity, mobility, complexity]

def wavelet_features(epoch, wavelet="db4", level=3):
    coeffs = pywt.wavedec(epoch, wavelet, level=level)
    return [np.sum(np.square(c)) for c in coeffs]

def extract_features(epoch, sfreq):
    return np.concatenate([compute_bandpower(epoch, sfreq),
                           hjorth_params(epoch),
                           wavelet_features(epoch)])

# ----------------------------
# Main
# ----------------------------
all_features, all_labels = [], []

for fname in os.listdir(RAW_DIR):
    if not fname.endswith(".edf"):
        continue
    path = os.path.join(RAW_DIR, fname)
    print(f"📂 Processing {path}")
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]

        # annotations → events
        if raw.annotations is None or len(raw.annotations) == 0:
            print(f"❌ Skipping {fname} | no annotations")
            continue

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        for onset, _, code in events:
            # map to label
            label = None
            for key, val in event_id.items():
                if val == code and key in EVENT_TO_ACTION:
                    label = EVENT_TO_ACTION[key]
            if label is None:
                continue

            start = int(onset)
            win_samples = int(WINDOW_SIZE * sfreq)
            step_samples = int(STEP_SIZE * sfreq)
            end = start + win_samples

            while end < raw.n_times:
                epoch, _ = raw[:, start:end]
                epoch = epoch[0]  # first channel
                feats = extract_features(epoch, sfreq)
                all_features.append(feats)
                all_labels.append(label)
                start += step_samples
                end = start + win_samples

        print(f"✅ {fname} → {len(all_features)} total samples so far")

    except Exception as e:
        print(f"❌ Error {fname} | {e}")

# ----------------------------
# Save
# ----------------------------
if len(all_features) == 0:
    print("⚠️ No features extracted. Check EVENT_TO_ACTION and annotations.")
else:
    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")

    os.makedirs("models", exist_ok=True)
    np.save(OUT_NPY, {"X": X, "y": y})
    pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])]).assign(label=y).to_csv(OUT_CSV, index=False)
    print(f"💾 Saved to {OUT_CSV} and {OUT_NPY}")
