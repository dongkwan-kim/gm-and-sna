from typing import Dict, List, Tuple

import numpy as np
# No external imports are allowed other than numpy


def tqdm_s(_iter):
    try:
        from tqdm import tqdm
        return tqdm(_iter)
    except ModuleNotFoundError:
        return _iter


def pbar_s(total):
    try:
        from tqdm import tqdm
        return tqdm(total=total)
    except ModuleNotFoundError:
        return None


def _remove_self_loops_and_parallel_edges(graph: List[Tuple[int, int]]):
    graph = [(u, v) for u, v in set(graph) if u != v]
    return graph


def load_email_dataset(preprocess=True):
    return load_dataset("../email-EuAll.txt", preprocess)


def load_dataset(path, preprocess) -> List[Tuple[int, int]]:
    _graph = []
    with open(path, "r") as f:
        for line in tqdm_s(f):
            if not line.startswith("#"):
                u, v = tuple(sorted([int(x) for x in line.strip().split()]))
                _graph.append((u, v))
    print("Loaded {}".format(path))
    if preprocess:
        _graph = _remove_self_loops_and_parallel_edges(_graph)
    return _graph


def _get_mapping(graph: List[Tuple[int, int]]):
    node_set = set()
    for u, v in graph:
        node_set.update([u, v])
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_set)}
    idx_to_node_id = {idx: node_id for idx, node_id in enumerate(node_set)}
    return node_set, node_id_to_idx, idx_to_node_id


def _get_to_neighbors(graph: List[Tuple[int, int]], node_id_to_idx, N) -> Dict[int, List[int]]:
    to_neighbors = {idx: [] for idx in range(N)}
    for u, v in graph:
        u, v = node_id_to_idx[u], node_id_to_idx[v]
        to_neighbors[u].append(v)
        to_neighbors[v].append(u)
    return to_neighbors


def _get_idx_to_degree(to_neighbors):
    return {idx: len(to_neighbors[idx]) for idx in range(len(to_neighbors))}


def count_influence(graph, seeds, threshold) -> int:
    """
    Implement the function that counts the ultimate number of active nodes.

    Inputs:
        graph: undirected input graph in the edge list format.
            That is, graph is a list of tuples of the form (u, v),
            which indicates that there is an edge between u and v.
            You can assume that both u and v are integers, while you cannot
            assume that the integers are within a specific range.
        seeds: a list of initial active nodes.
        threshold: the payoff threshold of the Linear Threshold Model.

    Output: the number of active nodes at time infinity.
    """
    pass


if __name__ == '__main__':

    MODE = "email"

    if MODE == "test":
        pass

    elif MODE == "email":
        email_dataset = load_email_dataset()

    else:
        raise ValueError
