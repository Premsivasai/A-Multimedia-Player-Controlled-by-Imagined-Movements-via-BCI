import tkinter as tk
import numpy as np
import random
from scripts.predict_action import predict_and_control

def run_simulation():
    # Load dataset
    X = np.load("models/X.npy")
    # Pick a random sample
    idx = random.randint(0, len(X)-1)
    feature = X[idx]
    predict_and_control(feature)

def launch_gui():
    root = tk.Tk()
    root.title("BCI Multimedia Player - Synthetic")
    root.geometry("300x200")

    tk.Label(root, text="Simulated BCI Control").pack(pady=10)
    tk.Button(root, text="Simulate EEG", command=run_simulation).pack(pady=20)
    tk.Button(root, text="Exit", command=root.destroy).pack(pady=10)

    root.mainloop()
