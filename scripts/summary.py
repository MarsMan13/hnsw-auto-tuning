import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from .models.hnsw_config import HnswConfig
from .models.hnsw_result import HnswResult

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.results import hnsw_config_to_files, load_results
from ann_benchmarks.plotting.utils import compute_metrics, compute_metric
from .constraints import RESULTS_DIR, RESULTS_TEMP_DIR

def summary(hnsw_config: HnswConfig) -> list[HnswResult]:
    # Use cache if exists
    cache_filename = f"{hnsw_config.generate_signature()}.pkl"
    cache_dir = os.path.join(RESULTS_DIR, RESULTS_TEMP_DIR)
    cache_file = os.path.join(cache_dir, cache_filename) 
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    # Process
    dataset, _ = get_dataset(hnsw_config.dataset)
    distance = dataset.attrs["distance"]

    def calculate_metrics(properties, run):
        return {
            'recall': compute_metric(np.array(dataset["distances"]), (properties, run), "k-nn"),
            'qps': compute_metric(np.array(dataset["distances"]), (properties, run), "qps"),
            'index_size': properties["index_size"],
            'build_time': properties["build_time"],
            'avg_search_time': properties["avg_search_time"],
            'median_search_time': properties["median_search_time"]
        }

    results = []
    for config, files in hnsw_config_to_files(hnsw_config, distance):
        metrics_list = [calculate_metrics(properties, run) for properties, run in load_results(files)]

        recall = np.array([m['recall'] for m in metrics_list])
        qps = np.array([m['qps'] for m in metrics_list])
        index_size = np.array([m['index_size'] for m in metrics_list])
        build_time = np.array([m['build_time'] for m in metrics_list])
        avg_search_time = np.array([m['avg_search_time'] for m in metrics_list])
        median_search_time = np.array([m['median_search_time'] for m in metrics_list])
       
        results.append(
            HnswResult.from_config(config)(
                recall, qps, index_size, build_time, avg_search_time, median_search_time
            )
        )
    # Save cache
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    return results

def basic_plot(ax, x, y, z, xlabel="X", ylabel="Y", zlabel="Z", title="Result"):
    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    Z = np.zeros_like(X, dtype=float)
    for i in range(len(x)):
        xi = np.where(np.unique(x) == x[i])[0][0]
        yi = np.where(np.unique(y) == y[i])[0][0]
        Z[yi, xi] = z[i]
    ax.plot_surface(X, Y, Z, cmap='viridis')
    for i in range(len(x)):
        ax.scatter(x[i], y[i], 0, color='b', marker='o', alpha=0.5)
        ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], 'k--', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

def plot_metrics(hnsw_results: list):
    """ 저장된 점수를 이용해 scatter와 surface 두 개의 3D 그래프를 그립니다. """
    fig = plt.figure(figsize=(24, 12))

    # Set data up
    x = [hnsw_result.ef_construction for hnsw_result in hnsw_results]
    y = [hnsw_result.M for hnsw_result in hnsw_results]
    z = [hnsw_result.score for hnsw_result in hnsw_results]

    # Surface plot
    ax1 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    Z = np.zeros_like(X, dtype=float)

    for i in range(len(x)):
        xi = np.where(np.unique(x) == x[i])[0][0]
        yi = np.where(np.unique(y) == y[i])[0][0]
        Z[yi, xi] = z[i]

    ax1.plot_surface(X, Y, Z, cmap='viridis')

    # 각 점을 x-y 평면에 투영하고 점선으로 연결
    for i in range(len(x)):
        # 각 점을 x-y 평면에 투영한 위치에 표시
        ax1.scatter(x[i], y[i], 0, color='b', marker='o', alpha=0.5)  # x-y 평면의 점
        # 각 점과 투영된 점을 연결하는 점선 추가
        ax1.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], 'k--', alpha=0.5)  # 점선

    ax1.set_xlabel('efConstruction')
    ax1.set_ylabel('maxConnections')
    ax1.set_zlabel('Score')
    ax1.set_title('Surface Plot')

    plt.show()
    # plt.savefig('result.png')