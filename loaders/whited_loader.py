import os
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from data_processing.Data import Data

class WhitedData(Data):
    def __init__(self, current_segment, voltage_segment, label, sampling_frequency, f_mains):
        super().__init__(current_segment, voltage_segment, label, sampling_frequency, f_mains)

def parse_whited_filename(filename):
    """Extract appliance type from WHITED filename"""
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    
    # First part is always the appliance type
    appliance_type = parts[0] if len(parts) > 0 else "Unknown"
    return appliance_type

def process_file(file_path):  # process a single file
    try:
        # Read FLAC file
        data, samplerate = sf.read(file_path)
        
        # WHITED has 2 channels: voltage and current
        if len(data.shape) > 1 and data.shape[1] == 2:
            voltage_segment = data[:, 0]
            current_segment = data[:, 1]
        else:
            # If mono or different structure, handle accordingly
            if len(data.shape) == 1:
                voltage_segment = data
                current_segment = np.zeros_like(data)
            else:
                voltage_segment = data[:, 0]
                current_segment = data[:, 1] if data.shape[1] > 1 else np.zeros_like(voltage_segment)
        
        # Parse filename for appliance type (label)
        filename = os.path.basename(file_path)
        label = parse_whited_filename(filename)
        
        # WHITED usa 50Hz (rede europeia)
        f_mains = 50
        
        return WhitedData(
            current_segment=current_segment,
            voltage_segment=voltage_segment,
            label=label,
            sampling_frequency=int(samplerate),
            f_mains=f_mains
        )
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_whited():  # load whole WHITED dataset
    print("------------------------------")
    print("Initiating WHITED dataset loading...")
    
    folder_path = './datasets/WHITED'
    
    file_list = []
    
    for root, _, files in os.walk(folder_path):  # walk through dataset directory
        for f in files:
            if f.endswith('.flac'):
                file_list.append(os.path.join(root, f))
    
    #file_list = file_list[:6]
    
    with ThreadPoolExecutor(max_workers=8) as executor:  # parallel loading utilizing threads and 8 workers
        results = list(executor.map(process_file, file_list))
    
    print(f"Loaded {len(results)} files from WHITED dataset.")
    print("------------------------------")
    
    WhitedData.check_underrepresented(results, min_samples=1)
    
    return results

# function to get all WHITED data, used in process_data.py
def get_all_whited_data():
    return load_whited()