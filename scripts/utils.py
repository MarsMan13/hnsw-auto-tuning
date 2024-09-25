from .constraints import ALGORITHMS

def get_config_from_algo(algorithm: str) -> dict:
    if algorithm not in ALGORITHMS.keys():
        raise ValueError("Algorithm not found")
    return ALGORITHMS[algorithm]

def get_range(s) -> list:
    if type(s) == list:
        return s
    parts = s.strip("[]").split(":")
    start = int(parts[0]); end = int(parts[1]); step = int(parts[2])
    return list(range(start, end+1, step))