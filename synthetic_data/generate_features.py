import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Define classes
CLASSES = ["play", "pause", "next", "volume_up", "volume_down"]
N_SAMPLES_PER_CLASS = 2000   #2000 per class = 5000 total
N_FEATURES = 20

os.makedirs("synthetic_data", exist_ok=True)

def generate_class_features(class_idx):
    """Generate noisy overlapping features for a given class"""
    base_mean = 2.0 * class_idx  # spread class centers
    features = np.random.normal(base_mean, 1.0, (N_SAMPLES_PER_CLASS, N_FEATURES))

    # Inject Gaussian noise (simulates EEG sensor noise)
    noise = np.random.normal(0, 0.5, features.shape)
    features += noise

    # Inject class overlap (distort different feature ranges per class)
    if class_idx == 0:  # play
        features[:, :5] += np.random.normal(0, 1.2, (N_SAMPLES_PER_CLASS, 5))
    elif class_idx == 1:  # pause
        features[:, 5:10] += np.random.normal(0, 1.2, (N_SAMPLES_PER_CLASS, 5))
    elif class_idx == 2:  # next
        features[:, 10:15] += np.random.normal(0, 1.2, (N_SAMPLES_PER_CLASS, 5))
    elif class_idx == 3:  # volume_up
        features[:, :10] += np.random.normal(0, 0.8, (N_SAMPLES_PER_CLASS, 10))
    elif class_idx == 4:  # volume_down
        features[:, 10:] += np.random.normal(0, 0.8, (N_SAMPLES_PER_CLASS, 10))

    return features

# Build dataset
X = []
y = []
for i, cls in enumerate(CLASSES):
    feats = generate_class_features(i)
    X.append(feats)
    y.extend([i] * N_SAMPLES_PER_CLASS)

X = np.vstack(X)
y = np.array(y)

# Add 3% random label noise (simulate EEG mislabeling)
n_noise = int(0.03 * len(y))
noise_idx = np.random.choice(len(y), n_noise, replace=False)
y[noise_idx] = np.random.choice(len(CLASSES), n_noise)

# Save dataset
np.save("models/X.npy", X)
np.save("models/y.npy", y)

df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(N_FEATURES)])
df["label"] = [CLASSES[i] for i in y]
df.to_csv("synthetic_data/features.csv", index=False)

print(f"✅ Generated dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(CLASSES)} classes")
print("📂 Saved to synthetic_data/features.csv and models/X.npy, models/y.npy")
