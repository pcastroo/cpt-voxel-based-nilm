import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from data_processing.dataset_builder import load_or_process_data, debug_load_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
PATH = 'appliance_classifier_voxel_steady.keras'
NUM_CLASSES = 16

# ---------- load model ----------
model = tf.keras.models.load_model(PATH)

X, y = load_or_process_data()

unique_classes, class_counts = np.unique(y, return_counts=True)
for cls, count in zip(unique_classes, class_counts):
    print(f"{cls:30s}: {count:4d} samples")
print(f"Data shape: {X.shape}")
print(f"Total samples: {len(y)}")
print(f"Number of classes: {len(unique_classes)}")

# split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y,)
    
# transform labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_true = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)

# ---------- reports and evaluation ----------
# make predictions
y_pred_prob = model.predict(X_test)
y_pred_int = np.argmax(y_pred_prob, axis=1)

# geral accuracy
test_loss, test_acc = model.evaluate(X_test, y_true_onehot, verbose=0)
print(f"Accuracy: {test_acc:.4f}")

# classification report
report = classification_report(y_true, y_pred_int, target_names=le.classes_, zero_division=0)
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
