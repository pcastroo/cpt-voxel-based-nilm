import os
import numpy as np  

from data_processing.cpt_decomposition import CPT
from data_processing.signal_preprocessing import normalize, build_voxel_dataset
from loaders.plaid_loader import get_all_data

# Load or process data function
def load_or_process_data(x_path='X_steady.npy', y_path='y_steady.npy'):
    if os.path.exists(x_path) and os.path.exists(y_path):
        print("Loading preprocessed data from disk...")
        X = np.load(x_path)
        y = np.load(y_path)

    else:
        print("Processing data from scratch...")
        dados = get_all_data()

        tensors, labels = [], []

        for data in dados:
            cpt = CPT([data.voltage_segment], [data.current_segment])
            norm = normalize(cpt)
            tensor = build_voxel_dataset([norm])

            tensors.append(tensor)
            labels.extend([data.appliance_type] * tensor.shape[0])

        X = np.concatenate(tensors, axis=0)
        y = np.array(labels)

        np.save(x_path, X)
        np.save(y_path, y)
    return X, y 

def debug_load_data():
    print("Debug loading data...")
    dados = get_all_data()

    tensors, labels = [], []

    for data in dados:
        cpt = CPT([data.voltage_segment], [data.current_segment])
        norm = normalize(cpt)
        tensor = build_voxel_dataset([norm])

        tensors.append(tensor)
        labels.extend([data.appliance_type] * tensor.shape[0])

    X = np.concatenate(tensors, axis=0)
    y = np.array(labels)

    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    return X, y