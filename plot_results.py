import argparse
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from ast import arg
from get_ground_truths import get_config_from_file
from scripts.models.hnsw_result import HnswResult
from scripts.summary import summary, plot_metrics
from scripts.constraints import RESULTS_DIR, RESULTS_CSV_DIR

class PlotResults:
    def __init__(self, results: list[HnswResult]):
        self.results = results
        self.filtered_results = results

    def filter_results(self, params=None, Ms=None, ef_constructions=None, is_static=True):
        _filtered_results = self.results
        if params is not None:
            _filtered_results = list(filter(lambda x: (x.M, x.ef_construction) in params, _filtered_results))
        else:
            if Ms is not None:
                _filtered_results = list(filter(lambda x: x.M in Ms, _filtered_results))
            if ef_constructions is not None:
                _filtered_results = list(filter(lambda x: x.ef_construction in ef_constructions, _filtered_results))
        if is_static:
            self.filtered_results = _filtered_results
        return _filtered_results

    def plot_index_size(self, Ms=None, ef_constructions=None):
        filtered_results = self.filtered_results
        if Ms is not None or ef_constructions is not None:
            filtered_results = self.filter_results(Ms, ef_constructions, is_static=False)
        # END OF SETUP
        fig = plt.figure(figsize=(12, 12))

        # Set data up
        x = [result.ef_construction for result in filtered_results]
        y = [result.M for result in filtered_results]
        z = [result.index_size[0] for result in filtered_results]

        # Surface plot
        ax1 = fig.add_subplot(projection='3d')
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
        ax1.set_zlabel('IndexSize')
        ax1.set_title('IndexSize per (efConstruction, maxConnections)')

        plt.show()

    def plot_build_time(self, Ms=None, ef_constructions=None):
        filtered_results = self.filtered_results
        if Ms is not None or ef_constructions is not None:
            filtered_results = self.filter_results(Ms, ef_constructions, is_static=False)
        # END OF SETUP
        fig = plt.figure(figsize=(12, 12))

        # Set data up
        x = [result.ef_construction for result in filtered_results]
        y = [result.M for result in filtered_results]
        z = [result.build_time[0] for result in filtered_results]

        # Surface plot
        ax1 = fig.add_subplot(projection='3d')
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
        ax1.set_zlabel('BuildTime')
        ax1.set_title('BuildTime per (efConstruction, maxConnections)')

        plt.show()
        
    def plot_response_time(self, Ms=None, ef_constructions=None):
        filtered_results = self.filtered_results
        if Ms is not None or ef_constructions is not None:
            filtered_results = self.filter_results(Ms, ef_constructions, is_static=False)
        # END OF SETUP
        fig = plt.figure(figsize=(12, 12))

        # Set data up
        x = [result.ef_construction for result in filtered_results]
        y = [result.M for result in filtered_results]
        z1 = [np.mean(result.avg_search_time) for result in filtered_results]
        z2 = [np.median(result.avg_search_time) for result in filtered_results]
        z3 = [np.mean(result.median_search_time) for result in filtered_results]
        z4 = [np.median(result.median_search_time) for result in filtered_results]
        #
        zs = [z1, z2, z3, z4]
        subplots = [221, 222, 223, 224]
        zlabels = ['avg_avg_search_time', 'median_avg_search_time', 'avg_median_search_time', 'median_median_search_time']
        # 111 : np.mean(avg_search_time)
        for i, subplot in enumerate(subplots):
            ax = fig.add_subplot(subplot, projection='3d')
            X, Y = np.meshgrid(np.unique(x), np.unique(y))
            Z = np.zeros_like(X, dtype=float)

            for j in range(len(x)):
                xj = np.where(np.unique(x) == x[j])[0][0]
                yj = np.where(np.unique(y) == y[j])[0][0]
                Z[yj, xj] = zs[i][j]

            ax.plot_surface(X, Y, Z, cmap='viridis')

            # 각 점을 x-y 평면에 투영하고 점선으로 연결
            for j in range(len(x)):
                # 각 점을 x-y 평면에 투영한 위치에 표시
                ax.scatter(x[j], y[j], 0, color='b', marker='o', alpha=0.5)  # x-y 평면의 점
                # 각 점과 투영된 점을 연결하는 점선 추가
                ax.plot([x[j], x[j]], [y[j], y[j]], [0, zs[i][j]], 'k--', alpha=0.5)  # 점선

            ax.set_xlabel('efConstruction')
            ax.set_ylabel('maxConnections')
            ax.set_zlabel(zlabels[i])
            ax.set_title(f'{zlabels[i]}\nper (efConstruction, maxConnections)')

        plt.show()
        
    def plot_recall(self, params=None, Ms=None, ef_constructions=None):
        filtered_results = self.filtered_results
        if params is not None or Ms is not None or ef_constructions is not None:
            filtered_results = self.filter_results(params, Ms, ef_constructions, is_static=False)
        plt.figure(figsize=(16, 8))
        # END OF SETUP 
        for result in filtered_results:
            plt.plot(result.recall, result.qps, label="M: %d, efConstruction: %d, score: %f"%(result.M, result.ef_construction, result.score), marker='o')
        plt.xlabel("Recall")
        plt.ylabel("QPS")
        plt.title("Recall vs QPS")
        plt.legend()
        plt.show()

    def plot_score(self, Ms=None, ef_constructions=None):
        filtered_results = self.filtered_results
        if Ms is not None or ef_constructions is not None:
            filtered_results = self.filter_results(Ms, ef_constructions, is_static=False)
        # END OF SETUP
        fig = plt.figure(figsize=(12, 12))

        # Set data up
        x = [result.ef_construction for result in filtered_results]
        y = [result.M for result in filtered_results]
        z = [result.score for result in filtered_results]

        # Surface plot
        ax1 = fig.add_subplot(projection='3d')
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
        ax1.set_title('Score per (efConstruction, maxConnections)')

        plt.show()
        
    def save_csv(self, is_short=True):  # 여기서 results 인자를 제거하고 self.results 사용
        summary_filename = "summary_%s_%d.csv" % (self.results[0].dataset, self.results[0].k)
        results_filename = "results_%s_%d.csv" % (self.results[0].dataset, self.results[0].k)
        csv_path = os.path.join(RESULTS_DIR, RESULTS_CSV_DIR)
        summary_csv_file = os.path.join(csv_path, summary_filename)
        results_csv_file = os.path.join(csv_path, results_filename)
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        
        with open(summary_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["dataset", "k", "M", "efConstruction", "efSearchs", "buildTime", "indexSize", "score"])
            for result in self.results:
                csvwriter.writerow([result.dataset, result.k, result.M, result.ef_construction, result.ef_search, 
                                    np.mean(result.build_time), np.mean(result.index_size), result.score])

        print("Summary data successfully written to %s" % summary_csv_file)

        if not is_short:
            with open(results_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["dataset", "k", "M", "efConstruction", "efSearch", "buildTime", "indexSize", "recall", "qps"])
                for result in self.results:
                    for i in range(len(result.recall)):
                        csvwriter.writerow([result.dataset, result.k, result.M, result.ef_construction, result.ef_search[i], 
                                            np.mean(result.build_time), np.mean(result.index_size), result.recall[i], result.qps[i]])

            print("Results data successfully written to %s" % results_csv_file)


# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-f", "--f", default=None, type=str, help="Ground Truth Config File Path")
#     # ADD MORE ARGUMENTS
#     args = parser.parse_args()
#     if args.f is None:
#         parser.print_help()
#         exit()

if __name__ == "__main__":
    # Usage : Below variables can be set by user optionally 
    FILE = "./bench_setup/ground-truths/nytimes-256.yml"
    IS_SHORT = False
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    hnsw_config = get_config_from_file(FILE)
    results = summary(hnsw_config)
    plot_results = PlotResults(results)
    # Usage : User can call plot methods here
    # plot_results.save_csv(is_short=IS_SHORT)
    # plot_results.plot_index_size()
    # plot_results.plot_build_time()
    # plot_results.plot_response_time()
    # plot_results.plot_score()
    params = []
    Ms = [16]
    ef_constructions = list(range(64,512+1, 64))
    for M in Ms:
        for ef_construction in ef_constructions:
            params.append((M, ef_construction))
    plot_results.plot_recall(params=params)