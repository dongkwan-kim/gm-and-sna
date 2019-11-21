from pprint import pprint
from typing import Dict, List, Tuple, Any
from collections import deque, defaultdict
import numpy as np
from time import time
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
    return load_dataset("../email-Enron.txt", preprocess)


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
    node_set, node_id_to_idx, idx_to_node_id = _get_mapping(graph)
    to_neighbors = _get_to_neighbors(graph, node_id_to_idx, len(node_set))
    seed_indices = [node_id_to_idx[s] for s in seeds]

    active_indices = set(seed_indices)
    inactive_indices = set(to_neighbors.keys()) - active_indices

    num_actives_q = deque(maxlen=max(len(node_set) // 50, 1000))

    n_iter = 0
    while True:

        new_active_indices = set()

        for i in inactive_indices:
            nb = to_neighbors[i]
            active_nb = [n for n in nb if n in active_indices]

            if len(active_nb) / len(nb) > threshold:
                new_active_indices.add(i)

        active_indices.update(new_active_indices)
        inactive_indices -= new_active_indices

        # Update
        num_actives = len(active_indices)
        num_actives_q.append(num_actives)
        n_iter += 1

        # Stopping criteria
        changed_ratio = abs(np.mean(num_actives_q) - num_actives) / np.mean(num_actives_q)
        if len(num_actives_q) == num_actives_q.maxlen and changed_ratio < 1e-6:
            break
        if len(active_indices) == len(node_set):
            break

        # Print
        if n_iter % 100 == 0:
            print("{} iters (min: {})\n\t- mean(num_actives): {}\n\t- num_actives: {}\n\t- changed: {}%".format(
                n_iter, num_actives_q.maxlen, np.mean(num_actives_q), num_actives, round(changed_ratio * 100, 7)))

    print("Finished: {} iters\n\t- mean(num_actives): {}\n\t- num_actives: {}".format(
        n_iter, np.mean(num_actives_q), num_actives))
    return len(active_indices)


def simulate(graph, policy_list, ratio_list, threshold_list, trial=1, init_seed=42):

    node_set, _, _ = _get_mapping(graph)
    n = len(node_set)

    simulated_dict = defaultdict(lambda: {
        "number_of_active_nodes_at_infinity": [],
        "elapsed_time": [],
    })
    for policy in policy_list:
        for ratio in ratio_list:
            for threshold in threshold_list:

                for t in range(trial):

                    np.random.seed(init_seed + t)

                    selected_seeds = seed_selection(graph=graph, policy=policy, n=int(n * ratio))

                    _t = time()
                    number_of_active_nodes_at_infinity = count_influence(
                        graph=graph, seeds=selected_seeds, threshold=threshold)
                    elapsed_time = time() - _t
                    key = (policy, ratio, threshold)

                    simulated_dict[key]["number_of_active_nodes_at_infinity"].append(number_of_active_nodes_at_infinity)
                    simulated_dict[key]["elapsed_time"].append(elapsed_time)

    return simulated_dict


if __name__ == '__main__':

    MODE = "T"
    from seed_selection import seed_selection

    if MODE == "test":
        G = nx.karate_club_graph()
        selected = seed_selection(graph=G.edges, policy="degree", n=2)
        ret = count_influence(graph=G.edges, seeds=selected, threshold=0.5)
        print(ret)

    elif MODE == "email":  # 36692 nodes
        email_dataset = load_email_dataset()
        selected = seed_selection(graph=email_dataset, policy="random", n=int(36692 * 0.01))
        print("# of seed_selection: {}".format(len(selected)))

        t0 = time()
        ret = count_influence(graph=email_dataset, seeds=selected, threshold=0.5)
        print(ret)
        print("Elapsed: {}s".format(time() - t0))

    elif MODE == "N":
        n_trials = 10
        email_dataset = load_email_dataset()
        ret = simulate(email_dataset,
                       ["degree", "random"],
                       [r / 100 for r in [0.1, 0.25, 0.5, 1, 2]],
                       [0.5],
                       trial=n_trials)
        pprint(ret)
        print("\t".join(["Policy", "Initial number of active nodes", "Payoff threshold"] +
                        ["Average number of active nodes at infinity"] +
                        [str(t) for t in range(n_trials)]))
        for k, v in ret.items():
            n_an_at_inf = v["number_of_active_nodes_at_infinity"]
            line_list = list(k) + [float(np.mean(n_an_at_inf))] + list(n_an_at_inf)
            print("\t".join([str(x) for x in line_list]))

    elif MODE == "T":
        n_trials = 10
        email_dataset = load_email_dataset()
        ret = simulate(email_dataset,
                       ["degree", "random"],
                       [r / 100 for r in [0.25]],
                       [0.2, 0.4, 0.5, 0.6, 0.8],
                       trial=n_trials)
        pprint(ret)
        print("\t".join(["Policy", "Initial number of active nodes", "Payoff threshold"] +
                        ["Average number of active nodes at infinity"] +
                        [str(t) for t in range(n_trials)]))
        for k, v in ret.items():
            n_an_at_inf = v["number_of_active_nodes_at_infinity"]
            line_list = list(k) + [float(np.mean(n_an_at_inf))] + list(n_an_at_inf)
            print("\t".join([str(x) for x in line_list]))

    else:
        raise ValueError
