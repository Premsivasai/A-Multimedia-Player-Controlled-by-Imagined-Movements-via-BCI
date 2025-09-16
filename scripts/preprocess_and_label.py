import os
import numpy as np
import mne
from scipy.signal import butter, lfilter
from sklearn.preprocessing import LabelEncoder

def bandpass_filter(data, low=8, high=30, fs=160):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)

def extract_features_from_file(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.filter(8., 30., fir_design='firwin')
    data, _ = raw[:, :]  # (n_channels, n_times)
    bandpassed = bandpass_filter(data)
    return np.mean(bandpassed, axis=1)  # average across time

def get_label_from_filename(filename):
    if "R02" in filename:
        return "play"
    elif "R03" in filename:
        return "pause"
    elif "R04" in filename:
        return "next"
    elif "R05" in filename:
        return "volume_up"
    else:
        return None

def process_all(data_dir="data/raw"):
    X, y = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".edf"):
            label = get_label_from_filename(file)
            if label:
                path = os.path.join(data_dir, file)
                features = extract_features_from_file(path)
                X.append(features)
                y.append(label)
    return np.array(X), LabelEncoder().fit_transform(y)

if __name__ == "__main__":
    X, y = process_all()
    os.makedirs("models", exist_ok=True)
    np.save("models/X.npy", X)
    np.save("models/y.npy", y)
