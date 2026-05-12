
import os
import sys
import json
import yaml
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.loaders import load_dataset, save_pkl, load_pkl
from src.matrix_generation import PearsonCorrelationGenerator, MutualInformationGenerator, GrangerCausalityGenerator
from src.sparcification import MatrixConstructionPipeline, RowL1Normalizer, ResistanceSpectralSparsifier
from src.hybrid import SNFDiffusionFuser
from src.batching import LouvainBatcher, SpectralBatcher
def get_matrix_with_cache(name, generator, data, cache_dir, dataset_name):
    path = os.path.join(cache_dir, f"{dataset_name.lower()}_{name}_raw.pkl")
    if os.path.exists(path):
        print(f"  Loading cached raw matrix: {name}")
        return load_pkl(path)

    print(f"  Generating matrix: {name}...")
    matrix = generator.generate(data)
    save_pkl(matrix, path)
    return matrix


def prepare_experiment(dataset_name, dataset_path):
    print(f"\n=== Preparing experiment for {dataset_name} ===")

    # Setup directories
    output_base = os.path.join(
        project_root, 'scripts', 'traffic_experiment_extended')
    matrix_dir = os.path.join(output_base, 'matrices')
    batch_dir = os.path.join(output_base, 'batches')
    cache_dir = os.path.join(output_base, 'cache')

    for d in [matrix_dir, batch_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Load Data
    data_dict = load_dataset(dataset_path, load_adj_matrix=True, verbose=False)
    data = data_dict['data']
    adj_original = data_dict.get('adj_raw')

    # Limit data for matrix generation
    gen_data = np.array(data[:1000])

    # 2. Generate/Load Base Matrices
    # Pearson
    pearson_gen = PearsonCorrelationGenerator()
    # Check for zero variance for Pearson
    std_devs = np.std(gen_data, axis=0)
    zero_variance_indices = np.where(std_devs == 0)[0]
    if zero_variance_indices.size > 0:
        print(
            f"  [WARNING] {len(zero_variance_indices)} nodes have zero variance. Adding noise for Pearson.")
        stabilized_gen_data = gen_data.copy()
        data_mean = np.nanmean(gen_data) or 1e-6
        noise_scale = data_mean * 0.0001
        noise = np.random.normal(0, noise_scale, size=(
            gen_data.shape[0], zero_variance_indices.size))
        stabilized_gen_data[:, zero_variance_indices, 0] += noise
        m_pearson = get_matrix_with_cache(
            'pearson', pearson_gen, stabilized_gen_data, cache_dir, dataset_name)
    else:
        m_pearson = get_matrix_with_cache(
            'pearson', pearson_gen, gen_data, cache_dir, dataset_name)

    # Mutual Information
    mi_gen = MutualInformationGenerator(n_neighbors=3)
    m_mi = get_matrix_with_cache(
        'mi', mi_gen, gen_data, cache_dir, dataset_name)

    # Granger Causality
    granger_gen = GrangerCausalityGenerator(maxlag=4)
    m_granger = get_matrix_with_cache(
        'granger', granger_gen, gen_data, cache_dir, dataset_name)

    # 3. Fusion
    snf_fuser = SNFDiffusionFuser(k_neighbors=5, n_iterations=10)

    # SNF(Pearson, MI)
    fused_p_mi_path = os.path.join(
        cache_dir, f"{dataset_name.lower()}_fused_p_mi_raw.pkl")
    if os.path.exists(fused_p_mi_path):
        print("  Loading cached SNF(Pearson, MI)")
        m_fused_p_mi = load_pkl(fused_p_mi_path)
    else:
        print("  Fusing Pearson and MI...")
        m_fused_p_mi = snf_fuser.fuse([m_pearson, m_mi])
        save_pkl(m_fused_p_mi, fused_p_mi_path)

    # SNF(Granger, MI)
    fused_g_mi_path = os.path.join(
        cache_dir, f"{dataset_name.lower()}_fused_g_mi_raw.pkl")
    if os.path.exists(fused_g_mi_path):
        print("  Loading cached SNF(Granger, MI)")
        m_fused_g_mi = load_pkl(fused_g_mi_path)
    else:
        print("  Fusing Granger and MI...")
        m_fused_g_mi = snf_fuser.fuse([m_granger, m_mi])
        save_pkl(m_fused_g_mi, fused_g_mi_path)

    # 4. Sparsification
    print("Sparsifying matrices...")
    # Adjust K based on number of nodes (approx 3 * num_nodes total edges)
    num_nodes = gen_data.shape[1]
    k_total = 4 * num_nodes

    pipeline = MatrixConstructionPipeline(
        sparsifiers=[ResistanceSpectralSparsifier()],
        normalizers=[RowL1Normalizer()]
    )

    all_raw_matrices = {
        'pearson': m_pearson,
        'mi': m_mi,
        'granger': m_granger,
        'fused_p_mi': m_fused_p_mi,
        'fused_g_mi': m_fused_g_mi
    }
    if adj_original is not None:
        all_raw_matrices['original'] = adj_original

    processed_matrices = {}
    saved_matrix_paths = {}

    for name, m_raw in all_raw_matrices.items():
        path = os.path.join(matrix_dir, f"{dataset_name.lower()}_{name}.pkl")
        if os.path.exists(path):
            print(f"  Loading sparsified matrix: {name}")
            mat = load_pkl(path)
        else:
            print(f"  Sparsifying {name}...")
            mat = pipeline.run(m_raw, k=k_total)
            save_pkl(mat, path)

        processed_matrices[name] = mat
        saved_matrix_paths[name] = os.path.relpath(path, project_root)

    # 5. Generate Batches
    print("Generating batches...")
    batchers = {
        'louvain': LouvainBatcher(),
        'spectral': SpectralBatcher(n_clusters=6)
    }

    saved_batch_paths = []
    for m_name, base_adj in processed_matrices.items():
        for b_name, batcher in batchers.items():
            path = os.path.join(
                batch_dir, f"{dataset_name.lower()}_{m_name}_{b_name}.json")
            if os.path.exists(path):
                print(f"  Skipping existing batch: {m_name}/{b_name}")
            else:
                try:
                    print(f"  Batching {m_name} with {b_name}...")
                    batches = batcher.batch(base_adj)
                    with open(path, 'w') as f:
                        json.dump(batches, f)
                    print(f"    - Generated {len(batches)} batches")
                except Exception as e:
                    print(f"    - Error batching {m_name}/{b_name}: {e}")
                    continue
            saved_batch_paths.append(os.path.relpath(path, project_root))

    return list(saved_matrix_paths.values()), saved_batch_paths


def write_configs(dataset_name, dataset_path, matrix_paths, batch_paths):
    window_sizes = [288, 2016]

    for window in window_sizes:
        config = {
            'dataset': dataset_path,
            'model': "amazon/chronos-2",
            'device': "cuda",
            'dtype': "bfloat16",
            'horizons': [3, 6, 12],
            'context_length': window,
            'window_strategy': "absolute",
            'modes': ['whole_matrix', 'node_batches'],
            'adjacency_files': matrix_paths,
            'node_batches_files': batch_paths,
            'num_runs': 1,
            'output_dir': "results/"
        }

        config_dir = os.path.join(
            project_root, 'scripts', 'traffic_experiment_extended', 'configs')
        os.makedirs(config_dir, exist_ok=True)

        config_path = os.path.join(
            config_dir, f"config_{dataset_name.lower()}_w{window}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Config saved to {config_path}")


if __name__ == "__main__":
    datasets = [
        ("PEMS08", "data/PEMS08/"),
        ("PEMS-BAY", "data/PEMS-BAY/"),
        ("PEMS04", "data/PEMS04/"),
        ("PEMS03", "data/PEMS03/"),
        ("PEMS07", "data/PEMS07/"),
        ("METR-LA", "data/METR-LA/"),
    ]

    for name, path in datasets:
        if not os.path.exists(os.path.join(project_root, path)):
            print(f"Skipping {name} (path not found: {path})")
            continue

        m_paths, b_paths = prepare_experiment(name, path)
        write_configs(name, path, m_paths, b_paths)

    print("\nPreparation complete!")
