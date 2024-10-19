ALGORITHMS = {
    "hnswlib":{
        "constructor": "HnswLib",
        "disabled": False,
        "name": "hnswlib",
        "docker_tag": "ann-benchmarks-hnswlib",
        "module": "ann_benchmarks.algorithms.hnswlib",
    }
}

RESULTS_DIR = "results"
RESULTS_TEMP_DIR = ".tmp"
RESULTS_CSV_DIR = "csv"