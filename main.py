import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models.architectures.FocalLoss import FocalLoss

from sklearn.metrics import classification_report, accuracy_score

# file paths
x_path = './preprocessed_data/X_plaid-whited.npy'
y_path = './preprocessed_data/y_plaid-whited.npy'

X, y = np.load(x_path), np.load(y_path)

# ---------- load model ----------
# hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = './models/checkpoints/RESNET3D_PLAID-WHITED_FL.keras'
NUM_CLASSES = len(np.unique(y))

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})

# split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---------- encode labels ----------    
# transform labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_test_int = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_test_int, NUM_CLASSES)

# --------- embedding extractor ----------
embedding_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

X_train_emb = embedding_model.predict(X_train, verbose=0)
X_test_emb  = embedding_model.predict(X_test,  verbose=0)

# =========================================================
# AE 32D + SVM (com normalização e GridSearch)
# =========================================================

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# ------------------ Autoencoder 32D ------------------
def build_autoencoder_32(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))

    # Encoder
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    latent = tf.keras.layers.Dense(32, activation="linear", name="latent")(x)

    # Decoder
    x = tf.keras.layers.Dense(128, activation="relu")(latent)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(input_dim, activation="linear")(x)

    autoencoder = tf.keras.Model(inp, out, name="autoencoder_32")
    encoder = tf.keras.Model(inp, latent, name="encoder_32")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse"
    )

    return autoencoder, encoder

# ------------------ Train Autoencoder ------------------
ae, encoder = build_autoencoder_32(X_train_emb.shape[1])

ae.fit(
    X_train_emb, X_train_emb,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    shuffle=True,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=0
)

# ------------------ Extract Latent Space ------------------
Xtr_latent = encoder.predict(X_train_emb, verbose=0)
Xte_latent = encoder.predict(X_test_emb,  verbose=0)

# ------------------ Normalization (CRITICAL) ------------------
scaler = StandardScaler()
Xtr_latent = scaler.fit_transform(Xtr_latent)
Xte_latent = scaler.transform(Xte_latent)

# ------------------ apply knn (k=5) ------------------
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr_latent, y_train_int)

y_pred_int = knn.predict(Xte_latent)

# ---------- reports and evaluation ----------
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

test_acc = accuracy_score(y_test_int, y_pred_int)
print(f"\nAccuracy: {test_acc:.4f}")

report = classification_report(
    y_test_int, 
    y_pred_int, 
    labels=range(NUM_CLASSES), 
    target_names=le.classes_,
    zero_division=0
)
print("\nClassification Report:")
print(report) 

cm = confusion_matrix(y_test_int, y_pred_int)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show() 