from typing import Dict, List

import numpy as np
from wcc import wcc
# No external imports are allowed other than numpy


def tqdm_s(_iter):
    try:
        from tqdm import tqdm
        return tqdm(_iter)
    except ModuleNotFoundError:
        return _iter


def _len(to_something_list, key):
    if key in to_something_list:
        return len(to_something_list[key])
    else:
        return 0


def _get_mapping(adjdict):
    node_set = set()
    for node_id, adjacent in adjdict.items():
        node_set.update([node_id] + adjacent)
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_set)}
    idx_to_node_id = {idx: node_id for idx, node_id in enumerate(node_set)}
    return node_set, node_id_to_idx, idx_to_node_id


def _get_followings_and_followers(adjdict, node_id_to_idx, N):
    to_followings = {}
    to_followers = {idx: [] for idx in range(N)}
    for node_id in adjdict:
        idx = node_id_to_idx[node_id]
        to_followings[idx] = np.asarray([node_id_to_idx[f_id] for f_id in adjdict[node_id]])
        for f_id in adjdict[node_id]:
            to_followers[node_id_to_idx[f_id]].append(idx)
    to_followers = {k: np.asarray(v) for k, v in to_followers.items()}
    return to_followings, to_followers


def slashburn(adjdict, k=1):

    # Index Mapping
    node_set, node_id_to_idx, idx_to_node_id = _get_mapping(adjdict)
    N = len(node_set)

    # to_followings, to_followers: Dict[int, np.ndarray[int]]
    to_followings, to_followers = _get_followings_and_followers(adjdict, node_id_to_idx, N)

    degree = np.asarray([_len(to_followings, idx) + _len(to_followers, idx) for idx in range(N)])

    ordered_node_indices = []
    wing_width_ratio = 0
    # TODO

    # Translation to ids.
    ordered_nodes = [idx_to_node_id[idx] for idx in ordered_node_indices]

    return ordered_nodes, wing_width_ratio


def _remove_self_loops_and_parallel_edges(adjdict: Dict[int, List[int]]):
    new_adjdict = {}
    for u, adj in adjdict.items():
        adj = list(set(adj))
        if u in adj:
            adj.remove(u)
        new_adjdict[u] = adj
    return new_adjdict


def load_stanford_dataset(preprocess=True):
    return load_dataset("../web-Stanford.txt", preprocess)


def load_email_dataset(preprocess=True):
    return load_dataset("../email-EuAll.txt", preprocess)


def load_dataset(path, preprocess) -> Dict[int, List[int]]:
    _adjdict = {}
    with open(path, "r") as f:
        for line in tqdm_s(f):
            if not line.startswith("#"):
                u, v = tuple([int(x) for x in line.strip().split()])
                if u in _adjdict:
                    _adjdict[u].append(v)
                else:
                    _adjdict[u] = [v]
    print("Loaded {}".format(path))
    if preprocess:
        _adjdict = _remove_self_loops_and_parallel_edges(_adjdict)
    return _adjdict


# You may add your own functions below to use them inside slashburn().
# However, remember that we will only do: from slashburn import slashburn
# to evaluate your implementation.
if __name__ == '__main__':

    MODE = "test"

    if MODE == "test":
        test_adjdict = {1: [2],
                        2: [3, 4],
                        3: [4, 6],
                        4: [1, 5],
                        5: [6],
                        6: []}
        test_result = slashburn(test_adjdict)
        assert set(test_result[0]) == {4, 2, 6, 3, 5, 1}
        assert test_result[1] - 0.5 < 0.0001

    elif MODE == "analysis":
        pass

    else:  # Other unit tests.
        test_adjdict = {1: [1],
                        2: [2, 3, 4, 3],
                        3: [1, 4],
                        4: []}
        test_result = _remove_self_loops_and_parallel_edges(test_adjdict)
        answer = {1: [], 2: [3, 4], 3: [1, 4], 4: []}
        for node_k in test_result.keys():
            assert set(answer[node_k]) == set(test_result[node_k])
