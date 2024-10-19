from ast import arg
from get_ground_truths import get_config_from_file
from scripts.models.hnsw_result import HnswResult
from scripts.summary import summary, plot_metrics
import argparse


class PlotResults:
    def __init__(self, results: list[HnswResult]):
        self.results = results

    def plot_index_size():
        pass

    def plot_build_time():
        pass

    def plot_response_time():
        pass

    def plot_recall():
        pass

    def plot_score():
        pass

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--f", default=None, type=str, help="Ground Truth Config File Path")
    # ADD MORE ARGUMENTS
    args = parser.parse_args()
    if args.f is None:
        parser.print_help()
        exit()

if __name__ == "__main__":
    args = parse_arguments()
    hnsw_config = get_config_from_file(args.f)
    results = summary(hnsw_config)
    plot_results = PlotResults(results)
    # User can call plot methods here
    plot_results.plot_index_size()
