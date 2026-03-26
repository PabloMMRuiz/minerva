import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.loaders import load_dataset
from src.matrix_generation import PearsonCorrelationGenerator

def validate_batches(dataset_name, dataset_path):
    print(f"\n=== Validating Information Neighborhoods for {dataset_name} ===")
    
    # 1. Load Data
    data_dict = load_dataset(dataset_path, load_adj_matrix=False, verbose=False)
    data = data_dict['data']
    
    # 2. Compute "Ground Truth" Information Matrix (Pearson)
    # Use a larger chunk than 1000 for better validation (e.g., 5000)
    print("Computing ground truth information matrix (Pearson)...")
    val_data = np.array(data[:5000])
    gt_matrix = PearsonCorrelationGenerator().generate(val_data)
    # Ensure absolute values for correlation to represent "information"
    gt_matrix = np.abs(gt_matrix)
    
    batch_dir = os.path.join(project_root, 'scripts', 'traffic_experiment', 'batches')
    batch_files = [f for f in os.listdir(batch_dir) if f.startswith(dataset_name.lower()) and f.endswith('.json')]
    
    results = []
    
    for b_file in batch_files:
        path = os.path.join(batch_dir, b_file)
        with open(path, 'r') as f:
            batches = json.load(f)
        
        intra_infos = []
        inter_infos = []
        
        all_nodes = set(range(gt_matrix.shape[0]))
        
        for batch in batches:
            if len(batch) < 2:
                continue
            
            # Intra-batch: average info between pairs in the same batch
            batch_nodes = np.array(batch)
            sub_mtx = gt_matrix[np.ix_(batch_nodes, batch_nodes)]
            # Get upper triangle excluding diagonal
            indices = np.triu_indices(len(batch), k=1)
            intra_infos.extend(sub_mtx[indices])
            
            # Inter-batch: average info between nodes in batch and nodes NOT in batch
            other_nodes = list(all_nodes - set(batch))
            if other_nodes:
                inter_sub = gt_matrix[np.ix_(batch_nodes, np.array(other_nodes))]
                inter_infos.extend(inter_sub.flatten())
        
        if intra_infos and inter_infos:
            avg_intra = np.mean(intra_infos)
            avg_inter = np.mean(inter_infos)
            ratio = avg_intra / avg_inter if avg_inter > 0 else np.inf
            
            results.append({
                'File': b_file,
                'Intra-Avg': avg_intra,
                'Inter-Avg': avg_inter,
                'Ratio': ratio,
                'Valid': avg_intra > avg_inter
            })
    
    if results:
        df = pd.DataFrame(results)
        print("\nValidation Results:")
        print(df.to_string(index=False))
        
        valid_count = df['Valid'].sum()
        print(f"\n{valid_count}/{len(results)} configurations passed (Intra > Inter).")
    else:
        print("No valid batch configurations found for testing.")

if __name__ == "__main__":
    datasets = [
        ("PEMS-BAY", "data/PEMS-BAY/"),
        ("PEMS04", "data/PEMS04/")
    ]
    
    for name, path in datasets:
        if os.path.exists(path):
            validate_batches(name, path)
        else:
            print(f"Dataset path {path} not found. Skipping {name}.")
