import os, sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.dataset_builder import debug_load_data
from loaders.plaid_loader import load_plaid
from loaders.lit_loader import load_lit
from loaders.whited_loader import load_whited
from data_processing.cpt_decomposition import CPT  

dados = load_plaid(max_workers=8)
data = dados[2]  # Select the i-th file

F_MAINS = 60 # mains frequency in Hz
POINTS_PER_CYCLE = 500 # number of samples per cycle
DT = 1 / (F_MAINS * POINTS_PER_CYCLE) # time step

cpt = CPT([data.voltage_segment], [data.current_segment])

t_clean = np.arange(len(cpt.i_active)) * DT
t = np.arange(len(data.current_segment)) * DT

# plots 2d
def plot_2d(cpt, data):
    fig = plt.figure(figsize=(15, 10))

    plots = [
        (t_clean, cpt.i_active, 'Time [s]', 'Active [Ia]'),
        (t_clean, cpt.i_reactive, 'Time [s]', 'Reactive [Ir]'),
        (t_clean, cpt.i_void, 'Time [s]', 'Void [Iv]'),
        (t, data.current_segment, 'Time [s]', 'Total [It]'),
    ]

    for idx, (x, y, xlabel, ylabel) in enumerate(plots, start=1):
        ax = fig.add_subplot(2, 2, idx) 
        ax.plot(x, y, color='blue')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{data.appliance_type} ({xlabel} x {ylabel}\n"
            f"{data.sampling_frequency/1000:.0f} kHz, 60 Hz, duration: {data.duration:.2f}s"
        )
    return fig, ax

# plots 3d
def plot_3d(cpt):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(cpt.i_active, cpt.i_reactive, cpt.i_void)

    ax.set_xlabel('Active [A]')
    ax.set_ylabel('Reactive [A]')
    ax.set_zlabel('Void [A]')
    ax.set_title(f"{data.appliance_type}\n")

    return fig, ax

#X, y = debug_load_data()

""" def visualize_voxel_3d(voxel_grid, threshold=0.01, title="Voxel Visualization", 
                       figsize=(10, 10), alpha=0.6, cmap='viridis'):
    # Remove channel dimension if present
    if voxel_grid.ndim == 4:
        voxel_grid = voxel_grid.squeeze()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of non-zero voxels above threshold
    filled = voxel_grid > threshold
    x, y, z = np.where(filled)
    
    # Get density values for coloring
    colors = voxel_grid[filled]
    
    # Normalize colors for colormap
    if colors.max() > 0:
        colors_normalized = colors / colors.max()
    else:
        colors_normalized = colors
    
    # Create scatter plot with density-based coloring
    scatter = ax.scatter(x, y, z, c=colors_normalized, 
                        cmap=cmap, marker='o', 
                        s=20, alpha=alpha, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Normalized Density', rotation=270, labelpad=15)
    
    # Labels
    ax.set_xlabel('Ia (Active)', fontsize=10)
    ax.set_ylabel('Ir (Reactive)', fontsize=10)
    ax.set_zlabel('Iv (Void)', fontsize=10)
    ax.set_title(title, fontsize=12, pad=20)
    
    # Set equal aspect ratio
    resolution = voxel_grid.shape[0]
    ax.set_xlim([0, resolution])
    ax.set_ylim([0, resolution])
    ax.set_zlim([0, resolution])
    
    plt.tight_layout()
    return fig, ax

def visualize_voxel_slices(voxel_grid, title="Voxel Slices", n_slices=8):
    # Remove channel dimension if present
    if voxel_grid.ndim == 4:
        voxel_grid = voxel_grid.squeeze()
    
    resolution = voxel_grid.shape[0]
    slice_indices = np.linspace(0, resolution-1, n_slices, dtype=int)
    
    fig, axes = plt.subplots(3, n_slices, figsize=(16, 6))
    fig.suptitle(title, fontsize=14)
    
    # Slices along Ia axis (YZ plane)
    for i, idx in enumerate(slice_indices):
        axes[0, i].imshow(voxel_grid[idx, :, :], cmap='hot', aspect='auto')
        axes[0, i].set_title(f'Ia={idx}', fontsize=8)
        axes[0, i].axis('off')
    
    # Slices along Ir axis (XZ plane)
    for i, idx in enumerate(slice_indices):
        axes[1, i].imshow(voxel_grid[:, idx, :], cmap='hot', aspect='auto')
        axes[1, i].set_title(f'Ir={idx}', fontsize=8)
        axes[1, i].axis('off')
    
    # Slices along Iv axis (XY plane)
    for i, idx in enumerate(slice_indices):
        axes[2, i].imshow(voxel_grid[:, :, idx], cmap='hot', aspect='auto')
        axes[2, i].set_title(f'Iv={idx}', fontsize=8)
        axes[2, i].axis('off')
    
    # Add axis labels
    axes[0, 0].set_ylabel('Slices\nalong Ia', fontsize=10, rotation=0, labelpad=40)
    axes[1, 0].set_ylabel('Slices\nalong Ir', fontsize=10, rotation=0, labelpad=40)
    axes[2, 0].set_ylabel('Slices\nalong Iv', fontsize=10, rotation=0, labelpad=40)
    
    plt.tight_layout()
    return fig


def visualize_multiple_samples(voxel_dataset, labels, indices=None, 
                               threshold=0.01, figsize=(15, 10)):
    if indices is None:
        indices = list(range(min(6, len(voxel_dataset))))
    
    n_samples = len(indices)
    cols = 3
    rows = (n_samples + cols - 1) // cols
    
    fig = plt.figure(figsize=figsize)
    
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        voxel = voxel_dataset[idx].squeeze()
        label = labels[idx]
        
        # Get non-zero voxels
        filled = voxel > threshold
        x, y, z = np.where(filled)
        colors = voxel[filled]
        
        if len(x) > 0 and colors.max() > 0:
            colors_normalized = colors / colors.max()
            ax.scatter(x, y, z, c=colors_normalized, cmap='viridis',
                      marker='o', s=10, alpha=0.6, edgecolors='none')
        
        ax.set_xlabel('Ia', fontsize=8)
        ax.set_ylabel('Ir', fontsize=8)
        ax.set_zlabel('Iv', fontsize=8)
        ax.set_title(f'Sample {idx} - Label: {label}', fontsize=10)
        
        resolution = voxel.shape[0]
        ax.set_xlim([0, resolution])
        ax.set_ylim([0, resolution])
        ax.set_zlim([0, resolution])
    
    plt.tight_layout()
    return fig

def compare_voxel_statistics(voxel_dataset, labels):
    print("=" * 60)
    print("VOXEL DATASET STATISTICS")
    print("=" * 60)
    
    # Overall statistics
    print(f"\nDataset shape: {voxel_dataset.shape}")
    print(f"Number of samples: {len(voxel_dataset)}")
    print(f"Voxel resolution: {voxel_dataset.shape[1]}³")
    print(f"Total voxels per sample: {voxel_dataset.shape[1]**3:,}")
    
    # Sparsity analysis
    sparsities = []
    occupied_voxels = []
    max_densities = []
    mean_densities = []
    
    for i in range(len(voxel_dataset)):
        voxel = voxel_dataset[i].squeeze()
        non_zero = np.count_nonzero(voxel)
        total = voxel.size
        sparsity = 1 - (non_zero / total)
        
        sparsities.append(sparsity)
        occupied_voxels.append(non_zero)
        max_densities.append(voxel.max())
        mean_densities.append(voxel[voxel > 0].mean() if non_zero > 0 else 0)
    
    print(f"\n--- SPARSITY ---")
    print(f"Average sparsity: {np.mean(sparsities)*100:.2f}%")
    print(f"Min sparsity: {np.min(sparsities)*100:.2f}%")
    print(f"Max sparsity: {np.max(sparsities)*100:.2f}%")
    
    print(f"\n--- OCCUPIED VOXELS ---")
    print(f"Average occupied voxels: {np.mean(occupied_voxels):.0f}")
    print(f"Min occupied voxels: {np.min(occupied_voxels)}")
    print(f"Max occupied voxels: {np.max(occupied_voxels)}")
    
    print(f"\n--- DENSITY VALUES ---")
    print(f"Average max density: {np.mean(max_densities):.4f}")
    print(f"Average mean density (non-zero): {np.mean(mean_densities):.4f}")
    
    # Label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n--- LABEL DISTRIBUTION ---")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    print("=" * 60)
    
    # Create visualization of statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Voxel Dataset Statistics', fontsize=14)
    
    # Sparsity distribution
    axes[0, 0].hist(sparsities, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Sparsity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Sparsity Distribution')
    axes[0, 0].axvline(np.mean(sparsities), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Occupied voxels
    axes[0, 1].hist(occupied_voxels, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Number of Occupied Voxels')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Occupied Voxels Distribution')
    
    # Max density per sample
    axes[1, 0].hist(max_densities, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Max Density Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Maximum Density Distribution')
    
    plt.tight_layout()
    return fig """

# visualize plot 2d
fig, ax = plot_2d(cpt, data)
plt.tight_layout()
plt.show()  

# visualize plot 3d
fig, ax = plot_3d(cpt)
plt.tight_layout()
plt.show()  

""" # visualize voxel 3d 
fig, ax = visualize_voxel_3d(X[0], threshold=0.01)
plt.tight_layout()
plt.show()

# visualize voxel slices
fig = visualize_voxel_slices(X[0])
plt.tight_layout()  
plt.show()

# visualize dataset statistics
fig = visualize_multiple_samples(X, y, indices=[0, 1, 2, 3, 4, 5], threshold=0.01)
plt.tight_layout()
plt.show() """