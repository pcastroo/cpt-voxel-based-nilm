import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

 # hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
PATH = 'model.keras'
NUM_CLASSES = 16

# file paths
x_path='X.npy'
y_path='y.npy'

# ---------- load model ----------
model = tf.keras.models.load_model(PATH)

X, y = np.load(x_path), np.load(y_path)

# split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
# transform labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_true = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)