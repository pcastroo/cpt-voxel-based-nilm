import numpy as np
from scipy.ndimage import gaussian_filter

class NormalizedCurrents:
    def __init__(self, i_active, i_reactive, i_void):
        self.i_active = i_active
        self.i_reactive = i_reactive
        self.i_void = i_void

def normalize(cpt):
    print("Normalizing current components...")
    i_a = np.asarray(cpt.i_active)
    i_r = np.asarray(cpt.i_reactive)
    i_v = np.asarray(cpt.i_void)

    ia_min = np.min(i_a)
    ir_min = np.min(i_r)
    iv_min = np.min(i_v)
    ia_max = np.max(i_a)
    ir_max = np.max(i_r)
    iv_max = np.max(i_v)

    # Handle case where all values are constant
    if ia_max == ia_min and ir_max == ir_min and iv_max == iv_min:
        return NormalizedCurrents(
            np.zeros_like(i_a), 
            np.zeros_like(i_r), 
            np.zeros_like(i_v)
        )

    # Min-max normalization to [-1, 1]
    norm_ia = 2 * (i_a - ia_min) / (ia_max - ia_min) - 1
    norm_ir = 2 * (i_r - ir_min) / (ir_max - ir_min) - 1
    norm_iv = 2 * (i_v - iv_min) / (iv_max - iv_min) - 1

    print("Normalization completed.")
    return NormalizedCurrents(norm_ia, norm_ir, norm_iv)

def voxelize_3d_trajectory(ia, ir, iv, resolution=32, density_mode=True, smooth=True):
    # Initialize voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # Map from [-1, 1] to [0, resolution-1]
    ia_indices = ((ia + 1) / 2 * (resolution - 1)).astype(np.int32)
    ir_indices = ((ir + 1) / 2 * (resolution - 1)).astype(np.int32)
    iv_indices = ((iv + 1) / 2 * (resolution - 1)).astype(np.int32)
    
    # Clip to valid range (safety check)
    ia_indices = np.clip(ia_indices, 0, resolution - 1)
    ir_indices = np.clip(ir_indices, 0, resolution - 1)
    iv_indices = np.clip(iv_indices, 0, resolution - 1)
    
    # Fill voxel grid
    if density_mode:
        # Count how many points fall into each voxel (density)
        for i in range(len(ia)):
            voxel_grid[ia_indices[i], ir_indices[i], iv_indices[i]] += 1
        
        # Normalize density to [0, 1]
        if voxel_grid.max() > 0:
            voxel_grid = voxel_grid / voxel_grid.max()
    else:
        # Binary occupancy (0 or 1)
        for i in range(len(ia)):
            voxel_grid[ia_indices[i], ir_indices[i], iv_indices[i]] = 1
    
    # Optional: smooth the voxel grid to reduce sparsity
    if smooth and density_mode:
        voxel_grid = gaussian_filter(voxel_grid, sigma=1.0)
        # Renormalize after smoothing
        if voxel_grid.max() > 0:
            voxel_grid = voxel_grid / voxel_grid.max()
    
    return voxel_grid

def build_voxel_dataset(normalized_currents_list, resolution=32, density_mode=True, smooth=True):
    print(f"Building voxel dataset with resolution {resolution}^3...")
    
    voxel_tensors = []
    
    for idx, norm in enumerate(normalized_currents_list):
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(normalized_currents_list)} samples...")
        
        # Voxelize the 3D trajectory
        voxel_grid = voxelize_3d_trajectory(
            norm.i_active, 
            norm.i_reactive, 
            norm.i_void,
            resolution=resolution,
            density_mode=density_mode,
            smooth=smooth
        )
        
        # Add channel dimension for CNN3D (batch, depth, height, width, channels)
        voxel_grid = np.expand_dims(voxel_grid, axis=-1)
        voxel_tensors.append(voxel_grid)
    
    result = np.asarray(voxel_tensors, dtype=np.float32)
    print(f"Voxel dataset built with shape: {result.shape}")
    print("------------------------------")
    
    return result