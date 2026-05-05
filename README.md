# A Multimedia Player Controlled by Imagined Movements via Brain–Computer Interface (BCI)

A Python proof-of-concept that maps **motor-imagery EEG classifications** to **multimedia player controls**. The repository includes:
- a small **Tkinter GUI** to simulate EEG input,
- scripts to **generate synthetic EEG-like features**,
- scripts to **extract features from real EEG EDF files** (optional),
- training code for an **SVM classifier**, and
- a **VLC-based media controller** for play/pause/next/volume.

> Note: This repo currently contains **pre-generated datasets and a pre-trained SVM** under `models/`.

---

## What’s in this repo

### Entry point
- `main.py` launches the GUI.

### GUI (simulation)
- `gui/gui.py`: Tkinter window with a **Simulate EEG** button. It randomly selects one feature vector from `models/X.npy` and runs prediction + media control.

### Core scripts
- `scripts/predict_action.py`: loads `models/svm_model.pkl` and predicts an action for a feature vector, then triggers the mapped media command.
- `scripts/media_control.py`: implements `MediaController` using **python-vlc**. Defaults to a Windows music folder.
- `scripts/train_model.py`: trains an SVM from `models/augmented_features.csv` and saves a bundle to `models/svm_model.pkl`.
- `scripts/train_svm.py`: alternative SVM training script using `models/X.npy` + `models/y.npy`.
- `scripts/preprocess_and_label.py`: basic EDF preprocessing & feature extraction (bandpass + mean features) and saves `models/X.npy`, `models/y.npy`.
- `scripts/extract_real_eeg_features.py`: richer real-EEG feature extraction (bandpower, Hjorth, wavelets) from EDF + annotations.

### Synthetic feature generation
- `synthetic_data/generate_features.py`: generates synthetic EEG-like features for 5 classes and writes:
  - `synthetic_data/features.csv`
  - `models/X.npy`, `models/y.npy`

### Models / artifacts
The `models/` folder contains datasets/artifacts such as:
- `X.npy`, `y.npy` (feature matrix + labels)
- `svm_model.pkl` (trained SVM)
- `conf_matrix.png`
- `augmented_features.csv` (large)
- other generated arrays/CSVs

---

## Installation

### Option A — Synthetic demo (recommended)
```bash
git clone https://github.com/Premsivasai/A-Multimedia-Player-Controlled-by-Imagined-Movements-via-BCI.git
cd A-Multimedia-Player-Controlled-by-Imagined-Movements-via-BCI
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### Option B — Real EEG feature extraction
If you want to run the EDF-based feature extraction, install the pinned versions:
```bash
pip install -r requirements_real_eeg.txt
```

---

## Run the GUI demo (synthetic)
```bash
python main.py
```
1. Click **Simulate EEG**
2. The app prints the predicted label (play/pause/next/volume up/down)
3. The corresponding media control function is executed

---

## Train / retrain the model

### Train from augmented CSV (recommended if present)
This trains an RBF SVM from `models/augmented_features.csv` and writes `models/svm_model.pkl`.
```bash
python scripts/train_model.py
```

### Train from `X.npy`/`y.npy`
```bash
python scripts/train_svm.py
```

---

## Generate synthetic features
Regenerates the synthetic dataset and overwrites `models/X.npy` and `models/y.npy`.
```bash
python synthetic_data/generate_features.py
```

---

## Real EEG pipeline (optional)

### 1) Place EDF files
Put EDF files in:
- `data/raw/`

### 2) Extract features (annotation-based)
```bash
python scripts/extract_real_eeg_features.py
```
Outputs:
- `models/real_eeg_features.csv`
- `models/real_eeg_features.npy`

### Alternative: simple EDF preprocessing
```bash
python scripts/preprocess_and_label.py
```

---

## Configuration notes

### Media folder
`scripts/media_control.py` defaults to:
- `C:\Users\prems\Music`

If you are on a different machine/OS, change:
- `MediaController(music_folder=...)`

### Requirements note
`requirements.txt` currently lists `tk`, but Tkinter is usually bundled with Python on many systems. If you face install issues, you can remove `tk` and ensure Tk is installed via your OS/Python distribution.

---

## License
No license specified.