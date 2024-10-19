import argparse
from matplotlib import pyplot as plt
import yaml
import random
from typing import List
from ann_benchmarks.main import (filter_already_run_definitions, get_dataset, run_worker)
from ann_benchmarks.datasets import DATASETS
from ann_benchmarks.definitions import (Definition, InstantiationStatus, algorithm_status,
                                     get_definitions, list_algorithms)
from scripts.models.hnsw_config import HnswConfig
from scripts.models.hnsw_result import HnswResult
from scripts.utils import get_range, get_config_from_algo
import multiprocessing.pool
import logging
import logging.config
import psutil
from ann_benchmarks.runner import run, run_docker
import sys
import logging
from scripts.summary import basic_plot, summary, plot_metrics
import numpy as np
####

# logging.config.fileConfig("logging.conf")
# logger = logging.getLogger("annb")

# START OF INIT-FUNCTIONS <<<<<<<<<<
def setup_logging(debug: bool):
    if debug:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-f", "--file", default=None, type=str, help="Ground Truth Config File Path"
    )
    parser.add_argument(
        "-d", "--debug", default=False, type=bool, help="Enable debug logging"
    )
    parser.add_argument("--force", help="re-run algorithms even if their results already exist", action="store_true")
    args = parser.parse_args()
    if args.file is None:
        parser.print_help()
        exit()
    return args


def get_config_from_file(file: str) -> dict:
    with open(file, 'r') as f:
        _config = yaml.safe_load(f)
    # Validate ========
    if _config[HnswConfig.TARGET_OPTIONS]["dataset"] not in DATASETS.keys():
        raise ValueError("Dataset not found")
    if _config[HnswConfig.BENCHMARK_OPTIONS]["k"] <= 0:
        raise ValueError("k must be positive")
    # Parse ============
    config = HnswConfig.from_config(_config)
    return config

# END OF INIT-FUNCTIONS >>>>>>>>>>

def get_definitions(
    algorithm: str,
    Ms: List[int],
    efConstructions: List[int],
    distance_metric: str = "euclidean",
    query_argument_groups: List[str] = None,
) -> List[Definition]:
    definitions = []
    algo = get_config_from_algo(algorithm)
    ##
    for M in Ms:
        for efConstruction in efConstructions:
            definitions.append(
                Definition(
                    algorithm=algorithm,
                    docker_tag=algo["docker_tag"],
                    module=algo["module"],
                    constructor=algo["constructor"],
                    arguments=[distance_metric, {'M': M, 'efConstruction': efConstruction}],
                    query_argument_groups=query_argument_groups,
                    disabled=False
                )
            )
    ##
    return definitions

def create_workers_and_execute(definitions: List[Definition], config:HnswConfig):
    """
    Manages the creation, execution, and termination of worker processes based on provided arguments.

    Args:
        definitions (List[Definition]): List of algorithm definitions to be processed.
        config (HnswConfig): User provided arguments for running workers. 

    Raises:
        Exception: If the level of parallelism exceeds the available CPU count or if batch mode is on with more than 
                   one worker.
    """
    cpu_count = multiprocessing.cpu_count()
    if config.parallelism > cpu_count - 1:
        raise Exception(f"Parallelism larger than {cpu_count - 1}! (CPU count minus one)")

    if config.batch and config.parallelism > 1:
        raise Exception(
            f"Batch mode uses all available CPU resources, --parallelism should be set to 1. (Was: {config.parallelism})"
        )

    task_queue = multiprocessing.Queue()
    for definition in definitions:
        task_queue.put(definition)

    try:
        workers = [multiprocessing.Process(target=run_worker, args=(i + 1, config, task_queue)) for i in range(config.parallelism)]
        [worker.start() for worker in workers]
        [worker.join() for worker in workers]
    finally:
        logging.info("Terminating %d workers" % len(workers))
        [worker.terminate() for worker in workers]

def run_worker(cpu: int, config: HnswConfig, queue: multiprocessing.Queue) -> None:
    """
    Executes the algorithm based on the provided parameters.

    The algorithm is either executed directly or through a Docker container based on the `args.local`
     argument. The function runs until the queue is emptied. When running in a docker container, it 
    executes the algorithm in a Docker container.

    Args:
        cpu (int): The CPU number to be used in the execution.
        config (dict): User provided arguments for running workers. 
        queue (multiprocessing.Queue): The multiprocessing queue that contains the algorithm definitions.

    Returns:
        None
    """
    while not queue.empty():
        definition = queue.get()
        # if config.local:
        #     run(definition, config.dataset, config.count, config.runs, config.batch)
        # else:
        memory_margin = 500e6  # reserve some extra memory for misc stuff
        mem_limit = int((psutil.virtual_memory().available - memory_margin) / config.parallelism)
        cpu_limit = str(cpu) if not config.batch else f"0-{multiprocessing.cpu_count() - 1}"
        run_docker(definition, config.dataset, config.k, config.runs, config.timeout, config.batch, cpu_limit, mem_limit)

def plot_results(hnsw_results:list[HnswResult]):
    fig = plt.figure(figsize=(20, 20))
    x = [hnsw_result.ef_construction for hnsw_result in hnsw_results]
    y = [hnsw_result.M for hnsw_result in hnsw_results]
    # 1) Score
    ax = fig.add_subplot(221, projection='3d')
    z = [hnsw_result.score for hnsw_result in hnsw_results]
    basic_plot(ax, x, y, z, "ef_construction", "M", "Score", "Score")
    # 2) Build Time
    ax = fig.add_subplot(222, projection='3d')
    z = [np.mean(hnsw_result.build_time) for hnsw_result in hnsw_results]
    basic_plot(ax, x, y, z, "ef_construction", "M", "Build Time", "Build Time")
    # 3) Index Size
    ax = fig.add_subplot(223, projection='3d')
    z = [np.mean(hnsw_result.index_size) for hnsw_result in hnsw_results]
    basic_plot(ax, x, y, z, "ef_construction", "M", "Index Size", "Index Size")
    # 4) Search Time
    ax = fig.add_subplot(224, projection='3d')
    z = [np.mean(hnsw_result.avg_search_time) for hnsw_result in hnsw_results]
    basic_plot(ax, x, y, z, "ef_construction", "M", "Search Time", "Search Time")
    plt.show()


def main():
    args = parse_arguments()
    setup_logging(args.debug)
    # END OF INIT

    config = get_config_from_file(args.file)
    dataset, dimension = get_dataset(config.dataset)
    definitions = get_definitions(
        config.algorithm,
        config.M,
        config.ef_construction,
        dataset.attrs["distance"],
        config.ef_search,
    )
    random.shuffle(definitions)
    # END OF DEFINITIONS
    definitions = filter_already_run_definitions(
        definitions, config.dataset, config.k, config.batch, args.force
    )
    if len(definitions) != 0:
        create_workers_and_execute(definitions, config)
    # END OF EXECUTION

    results = summary(config)
    plot_results(results)

if __name__ == "__main__":
    main()