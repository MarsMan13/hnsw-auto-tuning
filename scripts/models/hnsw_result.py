from .hnsw_config import HnswSimpleConfig
import numpy as np

class HnswResult(HnswSimpleConfig):

    DESIRED_RECALLS = np.arange(0.80, 1.00, 0.01)
    END_RECALL = 1.0 + 0.001
    __score = None

    def __init__(self, dataset: str, ef_construction:int, M:int, ef_search:list[int], k:int,
        recall:np.ndarray, qps:np.ndarray, index_size:np.ndarray, build_time:np.ndarray,
        avg_search_time:np.ndarray, median_search_time:np.ndarray):
        super().__init__(dataset, ef_construction, M, ef_search, k)
        self.recall = recall
        self.qps = qps
        self.index_size = index_size
        self.build_time = build_time
        self.avg_search_time = avg_search_time
        self.median_search_time = median_search_time

    @classmethod
    def from_config(cls, config:HnswSimpleConfig):
        return lambda recall, qps, index_size, build_time, avg_search_time, median_search_time: cls(
            config.dataset, config.ef_construction, config.M, config.ef_search, config.k,
            recall, qps, index_size, build_time, avg_search_time, median_search_time
        )

    @property
    def score(self):
        if self.__score != None:    # Caching
            return self.__score
        ####
        if self.END_RECALL not in self.recall:
            self.recall = np.concatenate((self.recall, np.array([self.END_RECALL])))
            self.qps = np.concatenate((self.qps, np.array([0.0])))
        interpolated_qps = np.interp(self.DESIRED_RECALLS, self.recall, self.qps)
        self.__score = np.sum(interpolated_qps)
        print("Dataset: %s, efConstruction: %d, M: %d, Score: %f"%(self.dataset, self.ef_construction, self.M, self.__score))
        return self.__score

    def __str__(self):
        return f"dataset: {self.dataset}, ef_construction: {self.ef_construction}, M: {self.M}, ef_search: {self.ef_search}, k: {self.k}"