import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# file paths
x_path='X_plaid.npy'
y_path='y_plaid.npy'

# ---------- load model ----------
X, y = np.load(x_path), np.load(y_path)

# hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
PATH = 'model.keras'
NUM_CLASSES = len(np.unique(y))

print("Classes found:", np.unique(y))
unique_classes, class_counts = np.unique(y, return_counts=True)
for cls, count in zip(unique_classes, class_counts):
    print(f"Class: {cls}, Count: {count}")  