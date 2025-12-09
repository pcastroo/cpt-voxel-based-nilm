import sys, os
import numpy as np  
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loaders.plaid_loader import load_plaid
from loaders.whited_loader import load_whited

from data_processing.signal_preprocessing import build_voxel_dataset
from data_processing.normalization import normalize, data_augmentation, Currents
from data_processing.cpt_decomposition import CPT

def calculate_class_segments(samples):
    """Pre-calculate how many segments each class should have"""
    all_labels = [s.label for s in samples]
    unique_classes, class_counts = np.unique(all_labels, return_counts=True)
    
    segments_per_class = {}
    for cls, count in zip(unique_classes, class_counts):
        if count <= 10:
            segments_per_class[cls] = 16
            print(f"  ⚠️  [{cls}] Severely Underrepresented ({count} samples) - Will create 16 segments")
        elif 10 < count < 50:
            segments_per_class[cls] = 4
            print(f"  ⚠️  [{cls}] Underrepresented ({count} samples) - Will create 4 segments")
        else:
            segments_per_class[cls] = 1
    
    return segments_per_class

def process_data(x_path, y_path, save=False, dataset=''):
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset.upper()} DATASET")
    print(f"{'='*60}\n")
    
    # load dataset 
    if dataset.lower() == 'plaid':
        samples = load_plaid()
    elif dataset.lower() == 'whited':
        samples = load_whited()

    # PRE-CALCULATE segments per class
    print(f"\nAnalyzing class distribution...")
    segments_map = calculate_class_segments(samples)
    print(f"{'='*60}\n")

    tensors, labels = [], []
    total_samples = len(samples)
    
    print(f"\nProcessing {total_samples} samples...")
    print(f"{'='*60}\n")

    with tqdm(total=total_samples, desc="Overall Progress", unit="sample", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for sample in samples:
            pbar.set_description(f"Processing [{sample.label:30s}]")
            
            cpt_component = CPT(sample)
            normalized_sample = normalize(cpt_component, sample)
            
            num_segments = segments_map[sample.label]
            
            data_augmented_samples = data_augmentation(normalized_sample, num_segments)   

            for augmented_sample in data_augmented_samples:
                tensor = build_voxel_dataset(augmented_sample)
                tensors.append(tensor)
                labels.append(augmented_sample.label)
            
            pbar.update(1) # update overall progress bar

    # consolidate all tensors and labels
    print(f"\n{'='*60}")
    print("CONSOLIDATING TENSORS...")
    print(f"{'='*60}")
    
    X = np.concatenate(tensors, axis=0)
    y = np.array(labels)

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY - {dataset.upper()}")
    print(f"{'='*60}")
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    for class_name, count in zip(unique_classes, class_counts):
        percentage = (count / len(y)) * 100
        bar_length = int(percentage / 2)  # Escala para 50 caracteres
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{class_name:30s}: {count:4d} samples {bar} {percentage:5.1f}%")
    
    print(f"\n{'─'*60}")
    print(f"Total samples: {len(y)}")
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Final dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"{'='*60}")

    if save:
        print(f"\n💾 Saving processed data...")
        np.save(x_path, X)
        np.save(y_path, y)
        print(f"   ✓ Saved features to: {x_path}")
        print(f"   ✓ Saved labels to: {y_path}")
        print(f"{'='*60}")

    return X, y 

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CHOOSE DATASET TO PROCESS:")
    print("="*60)

    print("\n1 - PLAID")
    print("2 - WHITED")
    print("3 - PLAID + WHITED")
    print("4 - Exit\n")

    user_choice = input("Process which dataset?: ").strip()

    match user_choice:
        case '1' :
            X, y = process_data(f'X_plaid.npy', f'y_plaid.npy', save=True, dataset='plaid')

        case '2':
            X, y = process_data(f'X_whited.npy', f'y_whited.npy', save=True, dataset='whited')

        case '3' :
            X_whited, y_whited = process_data('X_whited.npy', 'y_whited.npy', save=False, dataset='whited')
            X_plaid, y_plaid = process_data('X_plaid.npy', 'y_plaid.npy', save=False, dataset='plaid')

            X_combined = np.concatenate((X_whited, X_plaid), axis=0)
            y_combined = np.concatenate((y_whited, y_plaid), axis=0)

            np.save('X_plaid_whited.npy', X_combined)
            np.save('y_plaid_whited.npy', y_combined) 

        case '4':
            print("Exiting...")
            sys.exit(0)

        case _:
            print("Invalid choice.")   