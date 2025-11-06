import os
import numpy as np  

from loaders.plaid_loader import get_all_data

from data_processing.signal_preprocessing import normalize, build_voxel_dataset, NormalizedCurrents
from data_processing.cpt_decomposition import CPT

# process data from scratch, main function
def process_data(x_path, y_path):
    print("Processing data from scratch...")
    dados = get_all_data() # get all data from PLAID dataset

    tensors, labels = [], []

    for data in dados:
        cpt = CPT([data.voltage_segment], [data.current_segment]) # decompose current signal
        norm = normalize(cpt, data) # normalize current components

        if norm.is_underrepresented: # check if class is underrepresented (< 50 samples)
            print(f"[CROPPING] {norm.label} is underrepresented")
            norm.cropping_signal()
            print(f"Generated {norm.i_active.shape[0]} segments")

            for segment_idx in range(norm.i_active.shape[0]): # separete each segment to norm 
                segment_norm = NormalizedCurrents(
                    norm.current_segment,
                    norm.voltage_segment,
                    norm.label,
                    norm.sampling_frequency,
                    norm.f_mains,
                    norm.i_active[segment_idx],
                    norm.i_reactive[segment_idx],
                    norm.i_void[segment_idx]
                )

                tensor = build_voxel_dataset([segment_norm]) # voxelize each segment individually
                tensors.append(tensor)
                labels.extend([norm.label] * tensor.shape[0])
        else:
            print(f"[NORMAL] {norm.label} is well represented")
            tensor = build_voxel_dataset([norm])
            tensors.append(tensor)
            labels.extend([norm.label] * tensor.shape[0])

    X = np.concatenate(tensors, axis=0)
    y = np.array(labels)

    print(f"\nFinal dataset shape: {X.shape}, Labels shape: {y.shape}")

    np.save(x_path, X)
    np.save(y_path, y)

    return X, y 

# ---------- debug function ----------
def debug_load_data():
    print("Debug loading data...")
    dados = get_all_data()

    tensors, labels = [], []

    for data in dados:
        cpt = CPT([data.voltage_segment], [data.current_segment])
        norm = normalize(cpt, data)

        if norm.is_underrepresented:
            print(f"[CROPPING] {norm.label} is underrepresented")
            norm.cropping_signal()
            
            print(f" Generated {norm.i_active.shape[0]} segments")
            for segment_idx in range(norm.i_active.shape[0]):
                segment_norm = NormalizedCurrents(
                    norm.current_segment,
                    norm.voltage_segment,
                    norm.label,
                    norm.sampling_frequency,
                    norm.f_mains,
                    norm.i_active[segment_idx],
                    norm.i_reactive[segment_idx],
                    norm.i_void[segment_idx]
                )
                tensor = build_voxel_dataset([segment_norm])
                tensors.append(tensor)
                labels.extend([norm.label] * tensor.shape[0])
        else:
            print(f"[NORMAL] {norm.label} is well represented")
            tensor = build_voxel_dataset([norm])
            tensors.append(tensor)
            labels.extend([norm.label] * tensor.shape[0])

    X = np.concatenate(tensors, axis=0)
    y = np.array(labels)

    print(f"\nFinal dataset shape: {X.shape}, Labels shape: {y.shape}")

    return X, y