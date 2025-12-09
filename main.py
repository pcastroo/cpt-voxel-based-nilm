import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

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
y_true = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)

# ---------------- extract embeddings ----------------
embedding_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

emb_train = embedding_model.predict(X_train, batch_size=32, verbose=1)
emb_test  = embedding_model.predict(X_test,  batch_size=32, verbose=1)

print(f"Embeddings shape: {emb_train.shape}") 

# ---------------- classifier ----------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def evaluate_classifiers(train_reduced, test_reduced, y_train_int, y_true, dims):
    print("\n" + "="*60)
    print(f"TESTANDO CLASSIFICADORES NO ESPAÇO t-SNE {dims}D")
    print("="*60)

    # 1. Random Forest
    print("\n[1/5] Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(train_reduced, y_train_int)
    acc_rf = rf.score(test_reduced, y_true)
    print(f"  Acurácia: {acc_rf*100:.2f}%")

    # 2. SVM (RBF)
    print("\n[2/5] SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(train_reduced, y_train_int)
    acc_svm = svm.score(test_reduced, y_true)
    print(f"  Acurácia: {acc_svm*100:.2f}%")

    # 3. kNN
    print("\n[3/5] kNN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_reduced, y_train_int)
    acc_knn = knn.score(test_reduced, y_true)
    print(f"  Acurácia: {acc_knn*100:.2f}%")

    # 4. Logistic Regression (baseline fraco)
    print("\n[4/5] Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_reduced, y_train_int)
    acc_lr = lr.score(test_reduced, y_true)
    print(f"  Acurácia: {acc_lr*100:.2f}%")

    # 5. XGBoost
    print("\n[5/5] XGBoost Classifier...")
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb.fit(train_reduced, y_train_int)  
    acc_xgb = xgb.score(test_reduced, y_true)
    print(f"  Acurácia: {acc_xgb*100:.2f}%")

    # RESUMO
    print("\n" + "="*60)
    print(f"RESULTADOS - t-SNE {dims}D")
    print("="*60)
    print(f"Random Forest:        {acc_rf*100:.2f}%")
    print(f"SVM (RBF):            {acc_svm*100:.2f}%")
    print(f"kNN (k=5):            {acc_knn*100:.2f}%")
    print(f"Logistic Regression:  {acc_lr*100:.2f}%")
    print(f"XGBoost Classifier:   {acc_xgb*100:.2f}%")
    print(f"\nBaseline (ResNet3D + FocalLoss): 92.03%")
    print("="*60)

# ---------------- apply t-SNE ----------------
# t-SNE is non-parametric; train/test must be combined so the 2D space is consistent.
emb_all = np.concatenate([emb_train, emb_test], axis=0) # concatenate train + test

tsne_dims = [2, 5, 10, 20, 30]

for dims in tsne_dims:
    print(f"\nApplying t-SNE to reduce to {dims} dimensions...")

    # Use 'exact' method for dimensions > 3
    method = 'barnes_hut' if dims <= 3 else 'exact'
    
    tsne = TSNE(
        n_components=dims,
        perplexity=30,
        random_state=42,
        learning_rate='auto',
        init='pca',
        method=method,  # Added this parameter
        verbose=1  # Optional: show progress
    )

    all_reduced = tsne.fit_transform(emb_all)

    # split again
    train_reduced = all_reduced[:len(emb_train)]
    test_reduced  = all_reduced[len(emb_train):]

    # Evaluate classifiers on the reduced data
    evaluate_classifiers(train_reduced, test_reduced, y_train_int, y_true, dims)
'''
# ---------- reports and evaluation ----------
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