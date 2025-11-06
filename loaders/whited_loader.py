import os
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class WhitedData:
    def __init__(self, current_segment, voltage_segment,
                 appliance_type, sampling_frequency, duration):
        self.current_segment = current_segment
        self.voltage_segment = voltage_segment
        self.appliance_type = appliance_type
        self.sampling_frequency = sampling_frequency
        self.duration = duration

def parse_whited_filename(filename):
    """
    Parse WHITED filename format to extract appliance type
    Exemplo: GuitarAmp_Marshall8240_R3_MK2_20151115133402.flac
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')

    # First part is always the appliance type
    appliance_type = parts[0] if len(parts) > 0 else "Unknown"
    return appliance_type

def process_file(file_path):
    try:
        # Read FLAC file
        data, samplerate = sf.read(file_path)
        
        # WHITED has 2 channels: voltage and current
        # Check if file has 2 channels
        if len(data.shape) > 1 and data.shape[1] == 2:
            voltage_segment = data[:, 0]
            current_segment = data[:, 1]
        else:
            # If mono or different structure, handle accordingly
            print(f"Warning: {file_path} has unexpected channel structure")
            if len(data.shape) == 1:
                voltage_segment = data
                current_segment = np.zeros_like(data)
            else:
                voltage_segment = data[:, 0]
                current_segment = data[:, 1] if data.shape[1] > 1 else np.zeros_like(voltage_segment)
        
        # Parse filename for appliance type
        filename = os.path.basename(file_path)
        appliance_type = parse_whited_filename(filename)
        
        # Calculate duration
        duration = len(voltage_segment) / samplerate
        
        return WhitedData(
            current_segment=current_segment,
            voltage_segment=voltage_segment,
            appliance_type=appliance_type,
            sampling_frequency=samplerate,
            duration=duration
        )
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_whited(folder_path='./datasets/WHITED', max_workers=8, appliance_type_filter=None):
    print("------------------------------")
    print("Initiating WHITED dataset loading...")
    
    # Find all FLAC files
    file_list = []
    
    # Recursively find all FLAC files in the directory
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.flac'):
                file_path = os.path.join(root, f)
                
                # Apply filter if specified
                if appliance_type_filter is not None:
                    appliance_type = parse_whited_filename(f)
                    if appliance_type == appliance_type_filter:
                        file_list.append(file_path)
                else:
                    file_list.append(file_path)
    
    print(f"Found {len(file_list)} FLAC files to process...")
    
    # Uncomment to limit number of files for testing
    file_list = file_list[:10]
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resultados = list(executor.map(process_file, file_list))
    
    # Filter out None results (failed processing)
    resultados = [r for r in resultados if r is not None]
    
    print(f"Successfully loaded {len(resultados)} files from WHITED dataset.")
    print("------------------------------")
    
    return resultados

def get_all_data(folder_path='./datasets/WHITED'):
    return load_whited(folder_path=folder_path, max_workers=8, appliance_type_filter=None)

def get_appliance_types(folder_path='./datasets/WHITED'):
    appliance_types = set()
    
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.flac'):
                appliance_type = parse_whited_filename(f)
                appliance_types.add(appliance_type)
    
    return sorted(list(appliance_types))