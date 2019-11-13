from typing import Dict, List, Tuple

import numpy as np
# No external imports are allowed other than numpy
try:
    import networkx as nx
except ModuleNotFoundError as e:
    print("Not found!: {}".format(e))


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


def seed_selection(graph: List[Tuple[int, int]], policy, n) -> List[int]:
    """
    Implement the function that chooses n initial active nodes.

    Inputs:
        graph: undirected input graph in the edge list format.
            That is, graph is a list of tuples of the form (u, v),
            which indicates that there is an edge between u and v.
            You can assume that both u and v are integers, while you cannot
            assume that the integers are within a specific range.
        policy: policy for selecting initial active nodes. ('degree', 'random', or 'custom')
            if policy == 'degree', n nodes with highest degrees are chosen
            if policy == 'random', n nodes are randomly chosen
            if policy == 'custom', n nodes are chosen based on your own policy
        n: number of initial active nodes you should choose

    Outputs: a list of n integers, corresponding to n nodes.
    """
    node_set, node_id_to_idx, idx_to_node_id = _get_mapping(graph)

    if policy == "degree":
        to_neighbors = _get_to_neighbors(graph, node_id_to_idx, len(node_set))
        idx_to_degree = _get_idx_to_degree(to_neighbors)
        indices_sorted_by_degree = sorted(idx_to_degree, key=lambda idx: -idx_to_degree[idx])
        selected_indices = indices_sorted_by_degree[:n]
        return [idx_to_node_id[idx] for idx in selected_indices]

    elif policy == "random":
        selected_nodes = np.random.choice(list(node_set), size=n, replace=False)
        return list(selected_nodes)

    elif policy == "custom":
        raise NotImplementedError

    else:
        raise ValueError


if __name__ == '__main__':

    MODE = "test"

    if MODE == "test":
        G = nx.karate_club_graph()

        random_selected = seed_selection(graph=G.edges, policy="random", n=2)
        print(random_selected)

        degree_selected = seed_selection(graph=G.edges, policy="degree", n=2)
        print(degree_selected)

    elif MODE == "email":
        email_dataset = load_email_dataset()

    else:
        raise ValueError
