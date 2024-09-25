from ann_benchmarks import data


class HnswConfig :
    def __init__(self, dataset, ef_construction, M, ef_search, k, runs, definitions, batch, parallelism, timeout, algorithm="hnswlib"):
        self.dataset = dataset
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.k = k; self.count = k
        self.algorithm = algorithm
        self.runs = runs
        self.definitions = definitions
        self.batch = batch
        self.parallelism = parallelism
        self.timeout = timeout

    def from_config(cls, config):
        TARGET_OPTIONS = "target_options"
        BENCHMARK_OPTIONS = "benchmark_options"
        return cls(
            algorithm=config[TARGET_OPTIONS]["algorithm"],
            dataset = config[TARGET_OPTIONS]["dataset"],
            ef_construction = config[TARGET_OPTIONS]["efConstruction"],
            M = config[TARGET_OPTIONS]["M"],
            ef_search = config[TARGET_OPTIONS]["efSearch"],
            k = config[BENCHMARK_OPTIONS]["k"],
            runs = config[BENCHMARK_OPTIONS]["runs"],
            definitions = config[BENCHMARK_OPTIONS]["definitions"],
            batch = config[BENCHMARK_OPTIONS]["batch"],
            parallelism = config[BENCHMARK_OPTIONS]["parallelism"],
            timeout = 2 * 3600
        )
