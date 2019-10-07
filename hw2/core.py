# No external imports are allowed other than numpy
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np


def _len(to_something_list, key):
    if key in to_something_list:
        return len(to_something_list[key])
    else:
        return 0


def tqdm_s(_iter):
    try:
        from tqdm import tqdm
        return tqdm(_iter)
    except ModuleNotFoundError:
        return _iter


def core(adjdict: Dict[int, List[int]]) -> Dict[int, int]:

    node_set = set()
    for node_id, adjacent in adjdict.items():
        node_set.update([node_id] + adjacent)
    N = len(node_set)

    # Index Mapping
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_set)}
    idx_to_node_id = {idx: node_id for idx, node_id in enumerate(node_set)}

    # to_followings, to_followers: Dict[int, List[int]]
    to_followings = {}
    to_followers = {idx: [] for idx in range(N)}
    for node_id in adjdict:
        idx = node_id_to_idx[node_id]
        to_followings[idx] = np.asarray([node_id_to_idx[f_id] for f_id in adjdict[node_id]])
        for f_id in adjdict[node_id]:
            to_followers[node_id_to_idx[f_id]].append(idx)
    to_followers = {k: np.asarray(v) for k, v in to_followers.items()}

    # Degree
    degree = np.asarray([_len(to_followings, idx) + _len(to_followers, idx) for idx in range(N)])
    max_d = np.max(degree)

    # Histogram of degree
    bin_table = np.histogram(degree, np.arange(max_d + 2))[0]

    # Cumulative histograms
    start = 1
    for d in range(max_d + 1):
        num = bin_table[d]
        bin_table[d] = start
        start += num

    pos = np.zeros(N).astype(np.int64)
    vert = dict()
    for idx in range(N):
        pos[idx] = bin_table[degree[idx]]
        vert[pos[idx]] = idx
        bin_table[degree[idx]] += 1

    bin_table[1:len(bin_table)] = bin_table[:len(bin_table) - 1]
    bin_table[0] = 1

    for order in tqdm_s(range(N)):
        v = vert[order + 1]
        neighbors = np.concatenate([to_followers.get(v, []), to_followings.get(v, [])]).astype(np.int64)

        for u in neighbors:
            if degree[u] > degree[v]:
                du = degree[u]
                pu = pos[u]
                pw = bin_table[du]
                w = vert[pw]
                if u != w:
                    pos[u] = pw
                    pos[w] = pu
                    vert[pu] = w
                    vert[pw] = u
                bin_table[du] += 1
                degree[u] -= 1

    coredict = {}
    for idx, c in enumerate(degree):
        coredict[idx_to_node_id[idx]] = c

    return coredict


# You may add your own functions below to use them inside core().
# However, remember that we will only do: from core import core
# to evaluate your implementation.
def load_stanford_dataset(path="../web-Stanford.txt") -> Dict[int, List[int]]:
    stanford_adjdict = {}
    with open(path, "r") as f:
        for line in tqdm_s(f):
            if not line.startswith("#"):
                u, v = tuple([int(x) for x in line.strip().split()])
                if u in stanford_adjdict:
                    stanford_adjdict[u].append(v)
                else:
                    stanford_adjdict[u] = [v]
    print("Loaded SNAP")
    return stanford_adjdict


if __name__ == '__main__':

    MODE = "TESTX"

    if MODE == "TEST":
        test_adjdict = {1: [2],
                        2: [3, 4],
                        3: [4, 6],
                        4: [1, 5],
                        5: [6],
                        6: []}

        test_coredict = core(test_adjdict)

        test_result = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}

        assert len(test_result) == len(test_coredict)
        for test_k in test_coredict:
            assert test_result[test_k] == test_coredict[test_k]

    else:
        from time import time
        data = load_stanford_dataset()
        start_time = time()
        core(data)
        print("Time for Stanford Dataset: {}s".format(time() - start_time))




