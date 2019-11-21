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


def pagerank(graph: List[Tuple[int, int]], damping_factor=0.85, to_sort=True) -> List[Tuple[int, float]]:
    """
    Implement the variant of pagerank described in Section 2.2.

    Inputs:
        graph: undirected input graph in the edge list format.

    Output: a list of tuples of the form (v, score), which indicates node v
        and its pagerank score. The list should be sorted in the decreasing
        order of pagerank scores. If multiple nodes have the same pagerank
        score, then return them in any order.
    """
    print("\nPAGERANK with DF of {}".format(damping_factor))

    node_set, node_id_to_idx, idx_to_node_id = _get_mapping(graph)
    num_nodes = len(node_set)
    to_neighbors = _get_to_neighbors(graph, node_id_to_idx, num_nodes)
    degree = [len(to_neighbors[node_idx]) for node_idx in range(num_nodes)]

    rank_old = {node_idx: 1 / num_nodes for node_idx in range(num_nodes)}

    diff, num_iter, eps = 100, 0, 0.0001
    while diff > eps:
        rank_new = {}

        diff = 0
        for node_idx in range(num_nodes):
            neighbors = to_neighbors[node_idx]
            # Follows a link from the current page (with probability c).
            if len(neighbors) > 0:
                _rank_neighbors = sum(rank_old[n_idx] / degree[n_idx] for n_idx in neighbors)
            # When the current web page does not have any links,
            # the surfer always jumps to a random page.
            else:
                _rank_neighbors = 0

            # Jumps to a random page (with prob. 1 - c)
            # The random surfer chooses a destination considering the degree of web pages:
            #   P(web page i is chosen) \prop 1 + d(i)
            _rank_random = (1 + degree[node_idx]) / (num_nodes + len(degree))

            rank_new[node_idx] = damping_factor * _rank_neighbors + (1 - damping_factor) * _rank_random
            diff += abs(rank_new[node_idx] - rank_old[node_idx])

        rank_old = rank_new
        num_iter += 1
        print("ITER: {}, DIFF: {}".format(num_iter, diff))

    nid_and_rank = [(idx_to_node_id[node_idx], rank) for node_idx, rank in rank_old.items()]
    if to_sort:
        return sorted(nid_and_rank, key=lambda emt: -emt[1])
    else:
        return nid_and_rank


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
        node_id_sorted_by_pr = [node_id for node_id, _ in pagerank(graph)]
        selected_node_ids = node_id_sorted_by_pr[:n]
        return selected_node_ids

    else:
        raise ValueError


if __name__ == '__main__':

    MODE = "custom"

    if MODE == "test":
        G = nx.karate_club_graph()

        random_selected = seed_selection(graph=G.edges, policy="random", n=10)
        degree_selected = seed_selection(graph=G.edges, policy="degree", n=10)
        custom_selected = seed_selection(graph=G.edges, policy="custom", n=10)

        print(random_selected)
        print(degree_selected)
        print(custom_selected)

    elif MODE == "pagerank":
        email_dataset = load_email_dataset()
        pr = pagerank(email_dataset)
        print(pr[:10])

    elif MODE == "custom":
        email_dataset = load_email_dataset()
        custom_selected = seed_selection(email_dataset, "custom", 10)
        print(custom_selected)

    else:
        raise ValueError
