import numpy as np
import joblib
from scripts.media_control import MediaController

# Load trained model and features
clf = joblib.load("models/svm_model.pkl")
X = np.load("models/X.npy")
y = np.load("models/y.npy")

controller = MediaController()

# Mapping of class labels to actions
# 0: play, 1: pause, 2: next, 3: volume_up, 4: volume_down
actions = {
    0: controller.play,
    1: controller.pause,
    2: controller.next,
    3: controller.volume_up,
    4: controller.volume_down
}

labels = ["Play", "Pause", "Next", "Volume Up", "Volume Down"]

def predict_and_control(sample=None):
    """Predict an action from EEG features and control media player."""
    if sample is None:
        # pick a random EEG feature vector for testing
        idx = np.random.randint(0, len(X))
        sample = X[idx].reshape(1, -1)
    else:
        # ensure correct shape if passed from GUI
        sample = np.array(sample).reshape(1, -1)

    pred = clf.predict(sample)[0]
    print(f"[PREDICTED] {labels[pred].lower()}")
    
    action = actions.get(pred, None)
    if action:
        action()

    
    action = actions.get(pred, None)
    if action:
        action()
