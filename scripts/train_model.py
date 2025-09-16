import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------
# Load dataset
# -----------------------
CSV_PATH = Path("models/augmented_features.csv")
MODEL_PATH = Path("models/svm_model.pkl")

print(f"📂 Loading dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

X = df.drop("label", axis=1).values
y = df["label"].values

# -----------------------
# Encode labels
# -----------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"✅ Classes: {list(le.classes_)}")

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------
# Scale features
# -----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------
# Train SVM
# -----------------------
clf = SVC(kernel="rbf", probability=True, random_state=42)
clf.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
y_pred = clf.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -----------------------
# Save model + scaler + label encoder
# -----------------------
joblib.dump({
    "model": clf,
    "scaler": scaler,
    "label_encoder": le
}, MODEL_PATH)

print(f"💾 Model saved → {MODEL_PATH}")
