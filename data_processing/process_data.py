import sys, os
import numpy as np  
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loaders.plaid_loader import get_all_plaid_data
from loaders.whited_loader import get_all_whited_data

from data_processing.signal_preprocessing import normalize, build_voxel_dataset, Currents
from data_processing.cpt_decomposition import CPT

# minimum samples to not be considered underrepresented
MIN_SAMPLES_PLAID = 1
MIN_SAMPLES_WHITED = 1

# process data from scratch, main function
def process_data(x_path, y_path, save=False, dataset=''):
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset.upper()} DATASET")
    print(f"{'='*60}\n")
    
    # load dataset 
    if dataset.lower() == 'plaid':
        samples = get_all_plaid_data(min_samples=MIN_SAMPLES_PLAID)
    elif dataset.lower() == 'whited':
        samples = get_all_whited_data(min_samples=MIN_SAMPLES_WHITED)

    tensors, labels = [], []
    total_samples = len(samples)
    
    print(f"\nProcessing {total_samples} samples...")
    print(f"{'='*60}\n")

    # progress bar for overall processing
    with tqdm(total=total_samples, desc="Overall Progress", unit="sample", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for sample in samples:
            pbar.set_description(f"Processing [{sample.label:30s}]")
            
            cpt_component = CPT(sample)
            normalized_sample = normalize(cpt_component, sample)

            if normalized_sample.is_underrepresented:
                tqdm.write(f"  ⚠️  [{normalized_sample.label}] Underrepresented - Cropping signal...")
                normalized_sample.cropping_signal()
                num_segments = normalized_sample.i_active.shape[0]
                tqdm.write(f"      Generated {num_segments} segments")

                # sub-bar for segments 
                for segment_idx in range(num_segments):
                    segment_currents = Currents(
                        normalized_sample.current_segment,
                        normalized_sample.voltage_segment,
                        normalized_sample.label,
                        normalized_sample.sampling_frequency,
                        normalized_sample.f_mains,
                        normalized_sample.i_active[segment_idx],
                        normalized_sample.i_reactive[segment_idx],
                        normalized_sample.i_void[segment_idx]
                    )

                    tensor = build_voxel_dataset([segment_currents])
                    tensors.append(tensor)
                    labels.extend([normalized_sample.label] * tensor.shape[0])
            else:
                tensor = build_voxel_dataset([normalized_sample])
                tensors.append(tensor)
                labels.extend([normalized_sample.label] * tensor.shape[0])
            
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