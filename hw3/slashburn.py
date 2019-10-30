from typing import Dict, List, Tuple, Set

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
        to_followings[idx] = [node_id_to_idx[f_id] for f_id in adjdict[node_id]]
        for f_id in adjdict[node_id]:
            to_followers[node_id_to_idx[f_id]].append(idx)
    return to_followings, to_followers


def _get_top_k_degree_nodes(idx_to_degree: Dict[int, int], k: int, idx_to_node_id: Dict[int, int]) -> List[int]:

    def cmp(idx_and_deg):
        idx, deg = idx_and_deg
        return -deg, idx_to_node_id[idx]

    if k == len(idx_to_degree):
        return list(idx_to_degree.keys())
    elif k < len(idx_to_degree):
        sorted_degree = sorted(idx_to_degree.items(), key=cmp)
        return [idx for idx, degree in sorted_degree[:k]]
    else:
        raise Exception("Something wrong")


def _get_ordered_wcc_list(wcc_list: List[Set[int]], idx_to_node_id: Dict[int, int]) -> List[Set[int]]:

    def cmp(w):
        return -len(w), min([idx_to_node_id[idx] for idx in w])

    return sorted(wcc_list, key=cmp)


def _remove_nodes(target_indices: List[int], idx_to_degree: Dict[int, int], to_followings: Dict[int, List[int]]) \
        -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    for node_idx in target_indices:
        del idx_to_degree[node_idx]
        if node_idx in to_followings:
            for following in to_followings[node_idx]:
                if following in idx_to_degree:
                    idx_to_degree[following] -= 1
            del to_followings[node_idx]

    for u, followings in to_followings.items():
        if any(target_idx in followings for target_idx in target_indices):
            to_followings[u] = [f for f in followings if f not in target_indices]
            idx_to_degree[u] -= 1

    return idx_to_degree, to_followings


def slashburn(adjdict, k=1):

    # Index Mapping
    node_set, node_id_to_idx, idx_to_node_id = _get_mapping(adjdict)
    N = len(node_set)

    # to_followings, to_followers: Dict[int, List[int]]
    to_followings, to_followers = _get_followings_and_followers(adjdict, node_id_to_idx, N)

    idx_to_degree = {idx: _len(to_followings, idx) + _len(to_followers, idx) for idx in range(N)}

    hub_indices, burned_indices, gcc = [], [], None
    i = 0
    while gcc is None or len(gcc) > k:

        i += 1

        # Find k-hubset.
        top_k_degree_nodes = _get_top_k_degree_nodes(idx_to_degree, k, idx_to_node_id)

        # Remove k-hubset from G to make the new graph G'.
        idx_to_degree, to_followings = _remove_nodes(top_k_degree_nodes, idx_to_degree, to_followings)

        # Add the removed k-hubset to the front of Γ.
        hub_indices += top_k_degree_nodes

        # Find connected components in G'.
        wcc_list = wcc(to_followings)
        gcc, *small_components = _get_ordered_wcc_list(wcc_list, idx_to_node_id)

        # Add nodes in non-giant connected components to the back of Γ, in
        # the decreasing order of sizes of connected components they belong to.
        _burned_indices_at_this_iter = sum([list(small_wcc) for small_wcc in reversed(small_components)], [])
        burned_indices = _burned_indices_at_this_iter + burned_indices

        # Set G to be the giant connected component(GCC) of G'.
        idx_to_degree, to_followings = _remove_nodes(_burned_indices_at_this_iter, idx_to_degree, to_followings)

    ordered_indices = hub_indices
    if gcc:
        ordered_indices += list(gcc)
    ordered_indices += burned_indices

    # Translation to ids.
    ordered_nodes = [idx_to_node_id[idx] for idx in ordered_indices]
    wing_width_ratio = (k * i) / N

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
        test_answer = [4, 2, 6, 3, 5, 1], 0.5
        for r, a in zip(test_result[0], test_answer[0]):
            assert r == a, "{} should be {}".format(r, a)
        assert abs(test_result[1] - test_answer[1]) < 0.00001, "{} should be {}".format(test_result[1], test_answer[1])
        print("Test passed")

        test_adjdict = {0: [1, 3],
                        1: [2],
                        2: [0],
                        3: [0, 4],
                        4: [5],
                        5: [3]}
        test_result = slashburn(test_adjdict)
        test_answer = [0, 3, 4, 5, 1, 2], 0.5
        for r, a in zip(test_result[0], test_answer[0]):
            assert r == a, "{} should be {}".format(r, a)
        assert abs(test_result[1] - test_answer[1]) < 0.00001, "{} should be {}".format(test_result[1], test_answer[1])
        print("Test passed")

        test_adjdict = {1: [2],
                        2: [3, 4],
                        3: [4, 6],
                        4: [1, 5],
                        5: [6],
                        6: []}
        test_result = slashburn(test_adjdict, k=2)
        test_answer = [4, 2, 6, 3, 5, 1], 2/3
        for r, a in zip(test_result[0], test_answer[0]):
            assert r == a, "{} should be {}".format(r, a)
        assert abs(test_result[1] - test_answer[1]) < 0.00001, "{} should be {}".format(test_result[1], test_answer[1])
        print("Test passed")

        test_adjdict = {0: [1, 3],
                        1: [2],
                        2: [0],
                        3: [0, 4],
                        4: [5],
                        5: [3]}
        test_result = slashburn(test_adjdict, k=2)
        test_answer = [0, 3, 1, 2, 4, 5], 1/3
        for r, a in zip(test_result[0], test_answer[0]):
            assert r == a, "{} should be {}".format(r, a)
        assert abs(test_result[1] - test_answer[1]) < 0.00001, "{} should be {}".format(test_result[1], test_answer[1])
        print("Test passed")

    elif MODE == "analysis":
        pass

    else:  # Other unit tests.
        test_adjdict = {1: [1],
                        2: [2, 3, 4, 3],
                        3: [1, 4],
                        4: []}
        test_result = _remove_self_loops_and_parallel_edges(test_adjdict)
        test_answer = {1: [], 2: [3, 4], 3: [1, 4], 4: []}
        for node_k in test_result.keys():
            assert set(test_answer[node_k]) == set(test_result[node_k])
