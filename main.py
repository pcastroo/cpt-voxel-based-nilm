import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models.architectures.FocalLoss import FocalLoss

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

# --------- normalization ----------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_emb = scaler.fit_transform(X_train_emb)
X_test_emb  = scaler.transform(X_test_emb)

# --------------- classifier function ---------------
from sklearn.metrics import accuracy_score

def run_classifier(Xtr, Xte, ytr, yte, clf):
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    return accuracy_score(yte, y_pred)

# --------------- baseline (without reduction) ---------------
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

results = []

# SVM baseline
svm = SVC(kernel="rbf", C=10, gamma="scale")
acc = run_classifier(X_train_emb, X_test_emb, y_train_int, y_test_int, svm)

results.append({
    "Reduction": "None",
    "Dim": X_train_emb.shape[1],
    "Classifier": "SVM",
    "Accuracy": acc
})

# kNN baseline
knn = KNeighborsClassifier(n_neighbors=3)
acc = run_classifier(X_train_emb, X_test_emb, y_train_int, y_test_int, knn)

results.append({
    "Reduction": "None",
    "Dim": X_train_emb.shape[1],
    "Classifier": "kNN",
    "Accuracy": acc
})

# --------------- PCA reduction ---------------
from sklearn.decomposition import PCA

pca_dims = [16, 32, 64, 128]

for d in pca_dims:
    pca = PCA(n_components=d, random_state=42)
    Xtr_pca = pca.fit_transform(X_train_emb)
    Xte_pca = pca.transform(X_test_emb)

    svm = SVC(kernel="rbf", C=10, gamma="scale")
    acc = run_classifier(Xtr_pca, Xte_pca, y_train_int, y_test_int, svm)

    results.append({
        "Reduction": "PCA",
        "Dim": d,
        "Classifier": "SVM",
        "Accuracy": acc
    })
# --------------- LDA reduction ---------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_dim = NUM_CLASSES - 1

lda = LDA(n_components=lda_dim)
Xtr_lda = lda.fit_transform(X_train_emb, y_train_int)
Xte_lda = lda.transform(X_test_emb)

svm = SVC(kernel="rbf", C=10, gamma="scale")
acc = run_classifier(Xtr_lda, Xte_lda, y_train_int, y_test_int, svm)

results.append({
    "Reduction": "LDA",
    "Dim": lda_dim,
    "Classifier": "SVM",
    "Accuracy": acc
})
# --------------- UMAP ---------------  
import umap

umap_dims = [2, 5, 10, 20]
neighbors = [10, 30]

for d in umap_dims:
    for n in neighbors:
        reducer = umap.UMAP(
            n_components=d,
            n_neighbors=n,
            random_state=42
        )

        Xtr_umap = reducer.fit_transform(X_train_emb)
        Xte_umap = reducer.transform(X_test_emb)

        svm = SVC(kernel="rbf", C=10, gamma="scale")
        acc = run_classifier(Xtr_umap, Xte_umap, y_train_int, y_test_int, svm)

        results.append({
            "Reduction": "UMAP",
            "Dim": d,
            "Classifier": "SVM",
            "Accuracy": acc,
            "Neighbors": n
        })

# --------------- Autoencoder reduction ---------------

def build_autoencoder(input_dim, bottleneck):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    bott = tf.keras.layers.Dense(bottleneck, activation="linear")(x)

    x = tf.keras.layers.Dense(128, activation="relu")(bott)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(input_dim, activation="linear")(x)

    ae = tf.keras.Model(inp, out)
    encoder = tf.keras.Model(inp, bott)
    ae.compile(optimizer="adam", loss="mse")
    return ae, encoder

ae_dims = [16, 32, 64]

for d in ae_dims:
    ae, encoder = build_autoencoder(X_train_emb.shape[1], d)
    ae.fit(
        X_train_emb, X_train_emb,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )

    Xtr_ae = encoder.predict(X_train_emb, verbose=0)
    Xte_ae = encoder.predict(X_test_emb,  verbose=0)

    svm = SVC(kernel="rbf", C=10, gamma="scale")
    acc = run_classifier(Xtr_ae, Xte_ae, y_train_int, y_test_int, svm)

    results.append({
        "Reduction": "Autoencoder",
        "Dim": d,
        "Classifier": "SVM",
        "Accuracy": acc
    })

# --------------- results summary ---------------
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Accuracy", ascending=False)

print(df_results)

'''
# ---------- reports and evaluation ----------
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

y_pred_prob = clf.predict(test_2d)
y_pred_int = np.argmax(y_pred_prob, axis=1)

test_loss, test_acc = clf.evaluate(test_2d, y_true, verbose=0)
print(f"\nAccuracy: {test_acc:.4f}")

report = classification_report(
    y_true, 
    y_pred_int, 
    labels=range(NUM_CLASSES), 
    target_names=le.classes_,
    zero_division=0
)
print("\nClassification Report:")
print(report) 

cm = confusion_matrix(y_true, y_pred_int)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show() '''