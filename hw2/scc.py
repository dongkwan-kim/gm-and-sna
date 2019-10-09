# No external imports are allowed other than numpy
import numpy as np
from typing import Dict, List, Set, Tuple

from collections import defaultdict


def tqdm_s(_iter):
    try:
        from tqdm import tqdm
        return tqdm(_iter)
    except ModuleNotFoundError:
        return _iter


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


def scc(adjdict: Dict[int, List[int]]) -> List[Set[int]]:
    """
    Return the list of strongly connected components of given graph.
    
    Inputs:
        adjdict: directed input graph in the adjacency dictionary format
    Outputs: a list of the strongly connected components.
       Each SCC is represented as a set of node indices.
    """

    # Index Mapping
    node_set, node_id_to_idx, idx_to_node_id = _get_mapping(adjdict)
    N = len(node_set)

    # to_followings, to_followers: Dict[int, np.ndarray[int]]
    to_followings, _ = _get_followings_and_followers(adjdict, node_id_to_idx, N)

    v_scc_list: List[Set[int]] = []

    # Each node v is assigned a unique integer v.index,
    # which numbers the nodes consecutively in the order in which they are discovered.
    v_index = np.zeros(N) - 1

    # It also maintains a value v.lowlink that represents the smallest index
    # of any node known to be reachable from v through v's DFS subtree, including v itself.
    v_lowlink = np.zeros(N) - 1

    stack = []
    v_on_stack = np.zeros(N) - 1

    visit_index = 0
    for vid in range(N):
        if v_index[vid] == -1 and vid in to_followings:
            _scc_list, visit_index, v_index, v_lowlink, stack, v_on_stack = _scc_one(
                vid,
                to_followings=to_followings,
                visit_index=visit_index,
                v_index=v_index,
                v_lowlink=v_lowlink,
                stack=stack,
                v_on_stack=v_on_stack,
            )
            v_scc_list += _scc_list

    scc_list: List[Set[int]] = []
    for v_scc in v_scc_list:
        scc_list.append({idx_to_node_id[vid] for vid in v_scc})

    return scc_list


def _scc_one(vid, to_followings, visit_index, v_index, v_lowlink, stack, v_on_stack) -> Tuple:

    _scc_list = list()

    _call_stack = []
    _v_to_num_called = defaultdict(lambda: -1)

    def _call(*args):
        for x in args:
            _call_stack.append(x)
            _v_to_num_called[x] += 1

    _call(vid)
    while len(_call_stack) != 0:

        vid = _call_stack.pop()

        # Set the depth index for v to the smallest unused index
        if _v_to_num_called[vid] == 0:
            v_index[vid] = visit_index
            v_lowlink[vid] = visit_index
            visit_index += 1
            stack.append(vid)
            v_on_stack[vid] = 1

        # Consider successors of vid
        for fid in to_followings[vid][_v_to_num_called[vid]:]:
            # Successor fid has not yet been visited; recurse on it.
            if v_index[fid] == -1 and fid in to_followings:
                _call(vid, fid)
                break

            # Successor w is in stack S and hence in the current SCC
            # If w is not on stack, then (v, w) is a cross-edge in the DFS tree and must be ignored
            elif v_on_stack[fid]:
                v_lowlink[vid] = min(v_lowlink[vid], v_index[fid])
        else:
            # If vid is a root node, pop the stack and generate an SCC
            if v_index[vid] == v_lowlink[vid]:
                wid, new_scc = -1, set()
                while wid != vid:
                    wid = stack.pop()
                    v_on_stack[wid] = 0
                    new_scc.add(wid)
                _scc_list.append(new_scc)

            if len(_call_stack) != 0:
                vid, fid = _call_stack[-1], vid
                v_lowlink[vid] = min(v_lowlink[vid], v_lowlink[fid])

    return _scc_list, visit_index, v_index, v_lowlink, stack, v_on_stack


def _scc_one_recursive(vid, to_followings, visit_index, v_index, v_lowlink, stack, v_on_stack) -> Tuple:

    _scc_list = list()

    # Set the depth index for v to the smallest unused index
    v_index[vid] = visit_index
    v_lowlink[vid] = visit_index
    visit_index += 1
    stack.append(vid)
    v_on_stack[vid] = 1

    # Consider successors of vid
    for fid in to_followings[vid]:
        # Successor fid has not yet been visited; recurse on it.
        if v_index[fid] == -1 and fid in to_followings:
            new_scc_list, visit_index, v_index, v_lowlink, stack, v_on_stack = _scc_one_recursive(
                fid, to_followings, visit_index, v_index, v_lowlink, stack, v_on_stack)
            _scc_list += new_scc_list
            v_lowlink[vid] = min(v_lowlink[vid], v_lowlink[fid])

        # Successor w is in stack S and hence in the current SCC
        # If w is not on stack, then (v, w) is a cross-edge in the DFS tree and must be ignored
        elif v_on_stack[fid]:
            v_lowlink[vid] = min(v_lowlink[vid], v_index[fid])

    # If vid is a root node, pop the stack and generate an SCC
    if v_index[vid] == v_lowlink[vid]:
        wid, new_scc = -1, set()
        while wid != vid:
            wid = stack.pop()
            v_on_stack[wid] = 0
            new_scc.add(wid)
        _scc_list.append(new_scc)

    return _scc_list, visit_index, v_index, v_lowlink, stack, v_on_stack


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

    MODE = "TEST"

    if MODE == "TEST":
        test_adjdict = {1: [2],
                        2: [3, 4],
                        3: [4, 6],
                        4: [1, 5],
                        5: [6],
                        6: []}
        test_scc = scc(test_adjdict)
        test_result = [{6}, {5}, {1, 2, 3, 4}]
        assert len(test_result) == len(test_scc)
        for ts, tr in zip(sorted(test_scc), sorted(test_result)):
            assert ts == tr, "{} != {}".format(ts, tr)
        print("TEST finished")

    else:
        from time import time
        data = load_stanford_dataset()
        start_time = time()
        scc(data)
        print("Time for Stanford Dataset: {}s".format(time() - start_time))
