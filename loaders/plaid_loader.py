import os, json
import polars as pl
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class PlaidData:
    def __init__(self, current_segment, voltage_segment, label, sampling_frequency, f_mains):
        self.current_segment = current_segment
        self.voltage_segment = voltage_segment
        self.label = label
        self.sampling_frequency = sampling_frequency
        self.f_mains = f_mains

        self.is_underrepresented = False # flag for underrepresented classes

    @staticmethod
    def check_underrepresented(data_list, min_samples=50):  # check which classes are underrepresented
        all_labels = [data.label for data in data_list]
        unique_classes, class_counts = np.unique(all_labels, return_counts=True)
        
        underrepresented_classes = set()
        for cls, count in zip(unique_classes, class_counts):
            if count < min_samples:
                underrepresented_classes.add(cls)
        
        if underrepresented_classes:
            for data in data_list:
                if data.label in underrepresented_classes:
                    data.is_underrepresented = True

        print(f"Underrepresented classes (less than {min_samples} samples): {underrepresented_classes}")

def process_file(file_path, metadata): # process a single file
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    data = pl.read_csv(file_path, has_header=False, new_columns=["Current", "Voltage"])

    meta = metadata[file_id]
    label = meta["appliance"]["type"]
    
    # PULAR BLENDER - FORMA MAIS SIMPLES
    if label == "Blender":
        return None
    
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

    with ThreadPoolExecutor(max_workers=8) as executor: # parallel loading utilizing threads and 8 workers
        results = list(executor.map(lambda fp: process_file(fp, metadata), file_list))

    # FILTRAR None (arquivos Blender pulados)
    results = [data for data in results if data is not None]

    print(f"Loaded {len(results)} files from PLAID dataset (Blender excluded).")
    print("------------------------------")

    PlaidData.check_underrepresented(results, min_samples=50)

    return results

def get_all_data():
    return load_plaid()