from ann_benchmarks.results import load_results
from get_ground_truths import get_config_from_file
from ann_benchmarks.datasets import get_dataset
import numpy as np
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import compute_metric
from scripts.summary import summary, plot_metrics
from scripts.models.hnsw_result import HnswResult
import matplotlib.pyplot as plt

def compute_recall(true_nn_distances, res):
    properties, run = res
    print(list(run.keys()))
    run_distances = np.array(run["distances"])
    # times = run["times"]
    print(run_distances.shape)

# TEST : load_result
def test1():
    file = "./bench_setup/ground-truths/mnist-ground-truths.yml"
    hnsw_config = get_config_from_file(file)

    dataset, _ = get_dataset(hnsw_config.dataset)
    distance = dataset.attrs["distance"]

    result = load_results(hnsw_config, distance)
    ####
    try:
        metric = "k-nn"
        metric_value = compute_metric(dataset["distances"], result, metric)
        print(metric_value)
    finally:
        result[1].close()


def store_results(results: list[HnswResult]):
    import os
    import csv

    filename = "results_%s_%d.csv"%(results[0].dataset, results[0].k)
    csv_file = os.path.join("results", filename)

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["dataset", "k", "efConstruction", "M", "efSearch", "buildTime", "recall", "qps"])

        for result in results:
            for i in range(len(result.recall)):
                csvwriter.writerow([result.dataset, result.k, result.ef_construction, result.M, result.ef_search[i], np.mean(result.build_time), result.recall[i], result.qps[i]])
    print("Data successfully written to %s"%csv_file)

def store_hnsw_results(results: list[HnswResult]):
    import os
    import csv

    filename = "results_%s_%d.csv"%(results[0].dataset, results[0].k)
    csv_file = os.path.join("results", filename)

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["dataset", "k", "efConstruction", "M", "efSearchs", "buildTime", "indexSize", "score"])

        for result in results:
            csvwriter.writerow([result.dataset, result.k, result.ef_construction, result.M, result.ef_search, np.mean(result.build_time), np.mean(result.index_size), result.score])
    print("Data successfully written to %s"%csv_file)

def test3():
    file = "./bench_setup/ground-truths/nytimes-256.yml"
    hnsw_config = get_config_from_file(file)
    results = summary(hnsw_config)
    print(len(results))
    store_hnsw_results(results)

# TEST : Recall computation
def test2():
    def plot_results(results: list[HnswResult]):
        fig, ax = plt.subplots()
        for result in results:
            ax.plot(result.recall, result.qps, label="ef_construction: %d, M: %d"%(result.ef_construction, result.M))
        ax.set_xlabel("Recall")
        ax.set_ylabel("QPS")
        ax.set_title("Result")
        plt.legend()
        plt.show()

        # x-axis is recall, y-axis is qps
    file = "./bench_setup/ground-truths/mnist-ground-truths.yml"
    hnsw_config = get_config_from_file(file)
    results = summary(hnsw_config)
    results_ef_construction_is_16 = [result for result in results if result.ef_construction == 16]
    plot_results(results=results_ef_construction_is_16)

if __name__ == "__main__":
    test3()