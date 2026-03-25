import os
import sys
import json
import pickle
import numpy as np
import yaml

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.loaders import load_dataset, save_pkl
from src.matrix_generation import PearsonCorrelationGenerator, MutualInformationGenerator
from src.sparcification import MatrixConstructionPipeline, TopKRowSparsifier, RowL1Normalizer
from src.hybrid import SNFDiffusionFuser
from src.batching import LouvainBatcher, KHopBatcher, SpectralBatcher

def prepare_experiment(dataset_name, dataset_path):
    print(f"\n=== Preparing experiment for {dataset_name} ===")
    
    # 1. Load Data
    
    data_dict = load_dataset(dataset_path, load_adj_matrix=True, verbose=False)
    data = data_dict['data']
    adj_original = data_dict.get('adj_raw')
    
    # Limit data for matrix generation if too long (e.g. first 1000 steps) to speed up
    # but use enough to be statistically significant. Traffic data is often large.
    gen_data = np.array(data[:1000]) 
    
    output_base = os.path.join(project_root, 'scripts', 'traffic_experiment')
    matrix_dir = os.path.join(output_base, 'matrices')
    batch_dir = os.path.join(output_base, 'batches')
    
    # 2. Generate Matrices
    print("Generating matrices...")
    pearson_gen = PearsonCorrelationGenerator()
    mi_gen = MutualInformationGenerator(n_neighbors=3)
    
    m_pearson = pearson_gen.generate(gen_data)
    m_mi = mi_gen.generate(gen_data)
    
    # Fusion
    print("Fusing matrices (SNF)...")
    snf_fuser = SNFDiffusionFuser(k_neighbors=5, n_iterations=10)
    m_fused = snf_fuser.fuse([m_pearson, m_mi])
    
    # 3. Sparsification
    print("Sparsifying matrices...")
    pipeline = MatrixConstructionPipeline(
        sparsifiers=[TopKRowSparsifier(k_per_node=5)],
        normalizers=[RowL1Normalizer()]
    )
    
    adj_pearson = pipeline.run(m_pearson)
    adj_mi = pipeline.run(m_mi)
    adj_fused = pipeline.run(m_fused)
    
    # Process original if it exists
    if adj_original is not None:
        # The original might already be sparse/normalized, but let's ensure it fits our pipeline expectations
        # if it's not already.
        adj_original_proc = pipeline.run(adj_original)
    else:
        adj_original_proc = None

    # Save Matrices
    matrices = {
        'pearson': adj_pearson,
        'mi': adj_mi,
        'fused': adj_fused
    }
    if adj_original_proc is not None:
        matrices['original'] = adj_original_proc
        
    saved_matrix_paths = {}
    for name, mat in matrices.items():
        path = os.path.join(matrix_dir, f"{dataset_name.lower()}_{name}.pkl")
        save_pkl(mat, path)
        saved_matrix_paths[name] = os.path.relpath(path, project_root)

    # 4. Generate Batches
    print("Generating batches...")
    # We'll use the fused matrix as the base for batching as it combines perspectives
    # but we could also use others.
    batch_base_adj = adj_fused
    
    batchers = {
        'louvain': LouvainBatcher(),
        'khop': KHopBatcher(k=1, max_batch_size=32),
        'spectral': SpectralBatcher(n_clusters=8)
    }
    
    saved_batch_paths = {}
    for name, batcher in batchers.items():
        batches = batcher.batch(batch_base_adj)
        path = os.path.join(batch_dir, f"{dataset_name.lower()}_{name}.json")
        with open(path, 'w') as f:
            json.dump(batches, f)
        saved_batch_paths[name] = os.path.relpath(path, project_root)
        print(f"Generated {len(batches)} batches for {name}")

    return saved_matrix_paths, saved_batch_paths

def write_config(dataset_name, dataset_path, matrix_paths, batch_paths):
    config = {
        'dataset': dataset_path,
        'model': "amazon/chronos-2",
        'device': "cuda",
        'dtype': "bfloat16",
        'horizons': [3, 6, 12],
        'context_length': 288,
        'window_strategy': "absolute",
        'modes': ['single_node', 'whole_matrix', 'adj_neighbour', 'node_batches'],
        'adjacency_files': [matrix_paths[m] for m in matrix_paths],
        'node_batches_file': batch_paths['louvain'], # Default to louvain, user can change
        'num_runs': 1,
        'output_dir': "results/"
    }
    
    # We might want separate configs for different batching methods or just one comprehensive one.
    # The runner supports only one node_batches_file at a time in some versions, 
    # but let's check scripts/chronos_experiment/config.py to be sure.
    
    config_path = os.path.join(project_root, 'scripts', 'traffic_experiment', f"config_{dataset_name.lower()}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to {config_path}")

if __name__ == "__main__":
    datasets = [
        ("PEMS-BAY", "data/PEMS-BAY/"),
        ("PEMS04", "../data/PEMS04/")
    ]
    
    for name, path in datasets:
        m_paths, b_paths = prepare_experiment(name, path)
        write_config(name, path, m_paths, b_paths)
    
    print("\nPreparation complete!")
