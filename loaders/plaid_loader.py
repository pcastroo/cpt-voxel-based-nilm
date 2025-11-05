import os, json
import polars as pl
from concurrent.futures import ThreadPoolExecutor

class PlaidData:
    def __init__(self, current_segment, voltage_segment, 
                 appliance_type, sampling_frequency, duration, f_mains):
        self.current_segment = current_segment
        self.voltage_segment = voltage_segment
        self.appliance_type = appliance_type
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.f_mains = f_mains

def process_file(file_path, metadata):
    # extract file_id from filename
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    data = pl.read_csv(file_path, has_header=False, new_columns=["Current", "Voltage"])

    # extract metadata
    meta = metadata[file_id]
    appliance_type = meta["appliance"]["type"]
    sampling_frequency = int(meta["header"]["sampling_frequency"].replace("Hz", ""))
    f_mains = 60

    # extract segments
    current_segment = data["Current"].to_numpy()
    voltage_segment = data["Voltage"].to_numpy()

    return PlaidData(current_segment, voltage_segment, appliance_type, sampling_frequency, f_mains)

def load_plaid():
    print("------------------------------")
    print("Initiating PLAID dataset loading...")
    
    folder_path = './datasets/PLAID/submetered'
    metadata_path = './datasets/PLAID/metadata_submetered.json'

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    file_list = []

    # Recursively find all CSV files in the directory
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.csv'):
                file_list.append(os.path.join(root, f))

    #file_list = file_list[:6] # <-- Limit after filling the list!

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        resultados = list(executor.map(lambda fp: process_file(fp, metadata), file_list))

    print(f"Loaded {len(resultados)} files from PLAID dataset.")   
    print("------------------------------") 
    return resultados

def get_all_data():
    return load_plaid()