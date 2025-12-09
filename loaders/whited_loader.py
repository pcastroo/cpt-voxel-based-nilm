import os
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from data_processing.Data import Data

def parse_whited_filename(filename):
    """Extract appliance type from WHITED filename"""
    name_without_ext = os.path.splitext(filename)[0]
    
    # remove the number in parentheses at the end
    if '(' in name_without_ext:
        appliance_type = name_without_ext.split('(')[0].strip()
    else:
        appliance_type = name_without_ext.strip()
    
    return appliance_type if appliance_type else "UNKNOWN"

def process_file(file_path):  # process a single file
    data, samplerate = sf.read(file_path)

    f_mains = 50
        
    # WHITED has 2 channels: voltage (channel 0) and current (channel 1)
    voltage_segment = data[:, 0]
    current_segment = data[:, 1]
        
    # parse filename for appliance type (label)
    filename = os.path.basename(file_path)
    label = parse_whited_filename(filename)
        
    return Data(
        current_segment=current_segment, 
        voltage_segment=voltage_segment,
        label=label,
        sampling_frequency=int(samplerate),
        f_mains=f_mains
    )

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
    
    return results

# function to get all WHITED data, used in process_data.py
def get_all_whited_data():
    return load_whited()