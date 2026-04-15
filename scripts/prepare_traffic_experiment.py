import os
import sys
import json
import yaml
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.batching import LouvainBatcher, KHopBatcher, SpectralBatcher
from src.hybrid import SNFDiffusionFuser
from src.sparcification import MatrixConstructionPipeline, TopKRowSparsifier, RowL1Normalizer
from src.matrix_generation import PearsonCorrelationGenerator, MutualInformationGenerator
from src.data.loaders import load_dataset, save_pkl

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

    # ---------------------
    # --- Check for Zero Variance (Constant Columns) ---
    std_devs = np.std(gen_data, axis=0)
    zero_variance_indices = np.where(std_devs == 0)[0]
    if zero_variance_indices.size > 0:
        # Use a custom warning to make it stand out in the console
        msg = (f"\n[DATA QUALITY WARNING]: {len(zero_variance_indices)} nodes have zero variance "
               f"(constant values) and will cause NaNs in the correlation matrix.\n"
               f"Problematic Indices: {zero_variance_indices.tolist()}\n")
        print(msg)

    # -------------
    # FIX: Apply targeted noise to outlier columns for Pearson generation only
    # -------------
    if zero_variance_indices.size > 0:
        print('\nAdding noise to constant nodes to allow correlation calculations...')
        # Create a copy so we don't modify the original 'gen_data' used elsewhere
        stabilized_gen_data = gen_data.copy()

        # Calculate a safe noise scale (0.01% of the global mean)
        # We use nanmean just in case there are other issues, defaulting to 1e-6 if mean is 0
        data_mean = np.nanmean(gen_data) or 1e-6
        noise_scale = data_mean * 0.0001

        # Generate and add noise only to the columns with zero variance


        noise = np.random.normal(0, noise_scale, size=(
            gen_data.shape[0], zero_variance_indices.size))
        stabilized_gen_data[:, zero_variance_indices, 0] += noise

        # Generate the Pearson matrix using the stabilized data
        m_pearson = pearson_gen.generate(stabilized_gen_data)
    else:
        # No outliers found, proceed with original data
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

    adj_pearson = pipeline.run(m_pearson, k=5*325)
    adj_mi = pipeline.run(m_mi, k=5*325)
    adj_fused = pipeline.run(m_fused, k=5*325)

    # Process original if it exists
    if adj_original is not None:
        # The original might already be sparse/normalized, but let's ensure it fits our pipeline expectations
        # if it's not already.
        adj_original_proc = pipeline.run(adj_original, k=3*325)
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
    print("Generating batches for all matrices...")
    batchers = {
        'louvain': LouvainBatcher(),
        'khop': KHopBatcher(k=1, max_batch_size=32),
        'spectral': SpectralBatcher(n_clusters=8)
    }

    saved_batch_paths = []
    # Iterate over each generated matrix to create batches from different perspectives
    for m_name, base_adj in matrices.items():
        print(f"  Processing batches for {m_name} matrix...")
        for b_name, batcher in batchers.items():
            try:
                batches = batcher.batch(base_adj)
                # Filename: {dataset}_{matrix}_{batcher}.json
                path = os.path.join(
                    batch_dir, f"{dataset_name.lower()}_{m_name}_{b_name}.json")
                with open(path, 'w') as f:
                    json.dump(batches, f)
                saved_batch_paths.append(os.path.relpath(path, project_root))
                print(
                    f"    - Generated {len(batches)} batches for {m_name}/{b_name}")
            except Exception as e:
                print(f"    - Error generating {m_name}/{b_name} batches: {e}")

    return list(saved_matrix_paths.values()), saved_batch_paths


def write_config(dataset_name, dataset_path, matrix_paths, batch_paths):
    config = {
        'dataset': dataset_path,
        'model': "amazon/chronos-2",
        'device': "cuda",
        'dtype': "bfloat16",
        'horizons': [3, 6, 12],
        'context_length': 288,
        'window_strategy': "absolute",
        # 'modes': ['single_node', 'whole_matrix', 'adj_neighbour', 'node_batches'],
        'modes': ['whole_matrix', 'node_batches'],
        'adjacency_files': matrix_paths,
        'node_batches_files': batch_paths,
        'num_runs': 1,
        'output_dir': "results/"
    }

    config_path = os.path.join(
        project_root, 'scripts', 'traffic_experiment_2', f"config_{dataset_name.lower()}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    datasets = [
        # ("PEMS-BAY", "data/PEMS-BAY/"),
        ("PEMS07", "data/PEMS07/")
    ]

    for name, path in datasets:
        m_paths, b_paths = prepare_experiment(name, path)
        write_config(name, path, m_paths, b_paths)

    print("\nPreparation complete!")
