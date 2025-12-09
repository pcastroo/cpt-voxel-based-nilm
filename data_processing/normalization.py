import numpy as np
from scipy.ndimage import gaussian_filter

from data_processing.Data import Data

# parameters
RESOLUTION = 32
DENSITY_MODE = True
SMOOTH = True

class Currents(Data):
    def __init__(self, current_segment, voltage_segment, label, sampling_frequency, f_mains,
                 i_active, i_reactive, i_void):
        super().__init__(current_segment, voltage_segment, label, sampling_frequency, f_mains)
        
        # normalized current components
        self.i_active = i_active
        self.i_reactive = i_reactive
        self.i_void = i_void
    
def data_augmentation(data_obj, num_segments): # crop signal into segments based on mains frequency
    window_size = int(data_obj.sampling_frequency / data_obj.f_mains) 
    original_length = len(data_obj.i_active)

    # Calculate maximum possible segments based on available data
    max_possible_segments = original_length // window_size
    
    # Adjust num_segments if we don't have enough data
    if num_segments > max_possible_segments:
        print(f"\n  ⚠️  WARNING: Class [{data_obj.label}]")
        print(f"     Requested {num_segments} segments, but only enough data for {max_possible_segments}")
        print(f"     Original length: {original_length}, Window size: {window_size}")
        print(f"     → Reducing to {max_possible_segments} segments")
        num_segments = max_possible_segments
    
    # If we still don't have enough data for even 1 segment, skip
    if num_segments < 1:
        print(f"\n  ❌ ERROR: Class [{data_obj.label}] - Not enough data for even 1 segment!")
        print(f"     Original length: {original_length}, Window size: {window_size}")
        return []

    # truncate to exact multiple
    truncated_length = num_segments * window_size
        
    i_active = data_obj.i_active[:truncated_length].reshape((num_segments, window_size))
    i_reactive = data_obj.i_reactive[:truncated_length].reshape((num_segments, window_size))
    i_void = data_obj.i_void[:truncated_length].reshape((num_segments, window_size))

    segments = []
    for segment_idx in range(num_segments):
        segment_currents = Currents(
            data_obj.current_segment,
            data_obj.voltage_segment,
            data_obj.label,
            data_obj.sampling_frequency,
            data_obj.f_mains,
            i_active[segment_idx],
            i_reactive[segment_idx],
            i_void[segment_idx]
        )
        segments.append(segment_currents)

    return segments

def normalize(cpt, data): # normalize current components
    i_a = np.asarray(cpt.i_active)
    i_r = np.asarray(cpt.i_reactive)
    i_v = np.asarray(cpt.i_void)

    ia_min = np.min(i_a)
    ir_min = np.min(i_r)
    iv_min = np.min(i_v)
    ia_max = np.max(i_a)
    ir_max = np.max(i_r)
    iv_max = np.max(i_v)

    # handle case where all values are constant
    if ia_max == ia_min and ir_max == ir_min and iv_max == iv_min:
        norm_result = Currents(
            data.current_segment,
            data.voltage_segment,
            data.label,
            data.sampling_frequency,
            data.f_mains,
            np.zeros_like(i_a), 
            np.zeros_like(i_r), 
            np.zeros_like(i_v)
        )
        
        return norm_result

    # min-max normalization to [-1, 1]
    norm_ia = 2 * (i_a - ia_min) / (ia_max - ia_min) - 1
    norm_ir = 2 * (i_r - ir_min) / (ir_max - ir_min) - 1
    norm_iv = 2 * (i_v - iv_min) / (iv_max - iv_min) - 1

    norm_result = Currents(
        data.current_segment,
        data.voltage_segment,
        data.label,
        data.sampling_frequency,
        data.f_mains,
        np.asarray(norm_ia),
        np.asarray(norm_ir),
        np.asarray(norm_iv)
    )
    
    return norm_result