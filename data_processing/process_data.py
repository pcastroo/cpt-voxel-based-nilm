import sys, os
import numpy as np  
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loaders.plaid_loader import get_all_plaid_data
from loaders.whited_loader import get_all_whited_data

from data_processing.signal_preprocessing import normalize, build_voxel_dataset, NormalizedCurrents
from data_processing.cpt_decomposition import CPT

# process data from scratch, main function
def process_data(x_path, y_path, save=False, dataset='whited'):
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset.upper()} DATASET")
    print(f"{'='*60}\n")
    
    # Carregar dataset apropriado
    if dataset.lower() == 'plaid':
        dados = get_all_plaid_data()
    elif dataset.lower() == 'whited':
        dados = get_all_whited_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'plaid' or 'whited'")

    tensors, labels = [], []
    total_samples = len(dados)
    
    print(f"\nProcessing {total_samples} samples...")
    print(f"{'='*60}\n")

    # Barra de progresso principal
    with tqdm(total=total_samples, desc="Overall Progress", unit="sample", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for idx, data in enumerate(dados, 1):
            # Atualizar descrição com classe atual
            pbar.set_description(f"Processing [{data.label:30s}]")
            
            # Decomposição CPT
            cpt = CPT(data)
            
            # Normalização
            norm = normalize(cpt, data)

            if norm.is_underrepresented:
                tqdm.write(f"  ⚠️  [{norm.label}] Underrepresented - Cropping signal...")
                norm.cropping_signal()
                num_segments = norm.i_active.shape[0]
                tqdm.write(f"      Generated {num_segments} segments")

                # Sub-barra para segmentos (opcional)
                for segment_idx in range(num_segments):
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
                tensor = build_voxel_dataset([norm])
                tensors.append(tensor)
                labels.extend([norm.label] * tensor.shape[0])
            
            # Atualizar barra de progresso
            pbar.update(1)
            
            # Mostrar estatística intermediária a cada 10%
            if idx % max(1, total_samples // 10) == 0:
                current_samples = sum(t.shape[0] for t in tensors)
                tqdm.write(f"  📊 Samples generated so far: {current_samples}")

    # Consolidar resultados
    print(f"\n{'='*60}")
    print("CONSOLIDATING TENSORS...")
    print(f"{'='*60}")
    
    X = np.concatenate(tensors, axis=0)
    y = np.array(labels)

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY - {dataset.upper()}")
    print(f"{'='*60}")
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    for cls, count in zip(unique_classes, class_counts):
        percentage = (count / len(y)) * 100
        bar_length = int(percentage / 2)  # Escala para 50 caracteres
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{cls:30s}: {count:4d} samples {bar} {percentage:5.1f}%")
    
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
    # Processar PLAID
    X, y = process_data('X_plaid.npy', 'y_plaid.npy', save=True, dataset='plaid')
    
    # Processar WHITED
    #X, y = process_data('X_whited.npy', 'y_whited.npy', save=True, dataset='whited'),