import numpy as np
from scipy.ndimage import gaussian_filter

# parameters
RESOLUTION = 32
DENSITY_MODE = True
SMOOTH = True

def voxelize_3d_trajectory(ia, ir, iv): # voxelization of 3D trajectory
    # initialize voxel grid
    voxel_grid = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION), dtype=np.float32)

    # map from [-1, 1] to [0, RESOLUTION-1]
    ia_indices = ((ia + 1) / 2 * (RESOLUTION - 1)).astype(np.int32)
    ir_indices = ((ir + 1) / 2 * (RESOLUTION - 1)).astype(np.int32)
    iv_indices = ((iv + 1) / 2 * (RESOLUTION - 1)).astype(np.int32)
    
    # clip to valid range (safety check)
    ia_indices = np.clip(ia_indices, 0, RESOLUTION - 1)
    ir_indices = np.clip(ir_indices, 0, RESOLUTION - 1)
    iv_indices = np.clip(iv_indices, 0, RESOLUTION - 1)
    
    # fill voxel grid
    if DENSITY_MODE:
        # count how many points fall into each voxel (density)
        for i in range(len(ia)):
            voxel_grid[ia_indices[i], ir_indices[i], iv_indices[i]] += 1
        
        # normalize density to [0, 1]
        if voxel_grid.max() > 0:
            voxel_grid = voxel_grid / voxel_grid.max()
    else:
        # binary occupancy (0 or 1)
        for i in range(len(ia)):
            voxel_grid[ia_indices[i], ir_indices[i], iv_indices[i]] = 1
    
    # optional: smooth the voxel grid to reduce sparsity
    if SMOOTH and DENSITY_MODE:
        voxel_grid = gaussian_filter(voxel_grid, sigma=1.0)
        # renormalize after smoothing
        if voxel_grid.max() > 0:
            voxel_grid = voxel_grid / voxel_grid.max()
    
    return voxel_grid

def build_voxel_dataset(normalized_current): # build voxel for a SINGLE Currents object
    # voxelize the 3D trajectory
    voxel_grid = voxelize_3d_trajectory(
        normalized_current.i_active, 
        normalized_current.i_reactive, 
        normalized_current.i_void
    )
    
    # add channel dimension for CNN3D (depth, height, width, channels)
    voxel_grid = np.expand_dims(voxel_grid, axis=-1)
    
    # add batch dimension (1, depth, height, width, channels)
    voxel_tensor = np.expand_dims(voxel_grid, axis=0)
    
    return voxel_tensor