import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing.dataset_builder import load_or_process_data, debug_load_data

# constants
BATCH_SIZE = 32
EPOCHS = 20
PATH = 'appliance_classifier_voxel_steady.keras'
NUM_CLASSES = 16  
VOXEL_RESOLUTION = 32  

# ---------- preprocess data ----------
X, y = load_or_process_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # split data into train and test 
    
# transform labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_true = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)

# ---------- build and train model ----------
# build model
model = tf.keras.models.Sequential([
    # Bloco convolucional 1
    tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

    # Bloco convolucional 2
    tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

    # Bloco convolucional 3
    tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

    # Classificador
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # exit layer with 16 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_test, y_true_onehot),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

model.save(PATH)

# ---------- reports and evaluation ----------
# make predictions
y_pred_prob = model.predict(X_test)
y_pred_int = np.argmax(y_pred_prob, axis=1)

# geral accuracy
test_loss, test_acc = model.evaluate(X_test, y_true_onehot, verbose=0)
print(f"Accuracy: {test_acc:.4f}")

# classification report
report = classification_report(
    y_true, 
    y_pred_int, 
    labels=range(NUM_CLASSES),  # forçar todas as 16 classes
    target_names=le.classes_,
    zero_division=0  # evitar warnings se alguma classe não aparecer
)
print("Classification Report:")
print(report) 

# confusion matrix
cm = confusion_matrix(y_true, y_pred_int)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()