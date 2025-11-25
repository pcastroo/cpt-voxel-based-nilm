import os, json
import polars as pl
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from data_processing.Data import Data

# PLAID Data class: inherits from Data
class PlaidData(Data):
    def __init__(self, current_segment, voltage_segment, label, sampling_frequency, f_mains):
        super().__init__(current_segment, voltage_segment, label, sampling_frequency, f_mains)

def process_file(file_path, metadata): # process a single file
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    data = pl.read_csv(file_path, has_header=False, new_columns=["Current", "Voltage"])

    meta = metadata[file_id]
    label = meta["appliance"]["type"]
    sampling_frequency = int(meta["header"]["sampling_frequency"].replace("Hz", ""))
    f_mains = 60

    current_segment = data["Current"].to_numpy()
    voltage_segment = data["Voltage"].to_numpy()

    return PlaidData(current_segment, voltage_segment, label, sampling_frequency, f_mains)

def load_plaid(): # load whole PLAID dataset
    print("------------------------------")
    print("Initiating PLAID dataset loading...")
    
    folder_path = './datasets/PLAID/submetered'
    metadata_path = './datasets/PLAID/metadata_submetered.json'

    with open(metadata_path, "r") as f: # load metadata
        metadata = json.load(f)

    file_list = []

    for root, _, files in os.walk(folder_path): # walk through dataset directory
        for f in files:
            if f.endswith('.csv'):
                file_list.append(os.path.join(root, f))

    #file_list = file_list[:6]

    with ThreadPoolExecutor(max_workers=8) as executor: # parallel loading utilizing threads and 8 workers
        results = list(executor.map(lambda fp: process_file(fp, metadata), file_list))

    print(f"Loaded {len(results)} files from PLAID dataset.")
    print("------------------------------")

    PlaidData.check_underrepresented(results, min_samples=50)

    return results

# function to get all PLAID data, used in process_data.py
def get_all_plaid_data():
    return load_plaid()