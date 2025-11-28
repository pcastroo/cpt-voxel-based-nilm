import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# file paths
x_path = './preprocessed_data/X_plaid.npy'
y_path = './preprocessed_data/y_plaid.npy'

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

model = tf.keras.models.load_model(PATH)

# split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
# transform labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_true = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_true, NUM_CLASSES) 