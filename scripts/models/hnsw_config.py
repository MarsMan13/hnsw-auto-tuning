from ann_benchmarks import data
from ..utils import get_range
import hashlib

class HnswConfig :

    TARGET_OPTIONS = "target-options"
    BENCHMARK_OPTIONS = "benchmark-options"

    def __init__(
            self, dataset:str, ef_construction:list, M:list,
            k:int, runs:int, definitions:str, batch:bool, parallelism:int, timeout:int,
            ef_search=[[16,24,32,48,64,96,128,256,512]], algorithm="hnswlib"):
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

    @classmethod
    def from_config(cls, config):
        TARGET_OPTIONS = cls.TARGET_OPTIONS
        BENCHMARK_OPTIONS = cls.BENCHMARK_OPTIONS
        return cls(
            algorithm=config[TARGET_OPTIONS]["algorithm"],
            dataset = config[TARGET_OPTIONS]["dataset"],
            ef_construction = get_range(config[TARGET_OPTIONS]["efConstruction"]),
            M = get_range(config[TARGET_OPTIONS]["M"]),
            ef_search = get_range(config[TARGET_OPTIONS]["efSearch"]),
            k = config[BENCHMARK_OPTIONS]["k"],
            runs = config[BENCHMARK_OPTIONS]["runs"],
            definitions = config[BENCHMARK_OPTIONS]["definitions"],
            batch = config[BENCHMARK_OPTIONS]["batch"],
            parallelism = config[BENCHMARK_OPTIONS]["parallelism"],
            timeout = 2 * 3600
        )

    def __str__(self):
        return f"dataset: {self.dataset}, ef_construction: {self.ef_construction}, M: {self.M}, ef_search: {self.ef_search}, k: {self.k}, algorithm: {self.algorithm}, runs: {self.runs}, definitions: {self.definitions}, batch: {self.batch}, parallelism: {self.parallelism}, timeout: {self.timeout}"

    def generate_signature(self):
        return hashlib.sha256(str(self).encode()).hexdigest()[:10]

class HnswSimpleConfig:
    def __init__(self, dataset:str, M:int, ef_construction:int, ef_search:list[int], k:int):
        self.algorithm = "hnswlib"
        self.dataset = dataset
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.k = k

    def __str__(self):
        return f"dataset: {self.dataset}, ef_construction: {self.ef_construction}, M: {self.M}, ef_search: {self.ef_search}, k: {self.k}"