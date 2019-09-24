from collections import defaultdict
from typing import Tuple, List, Dict

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError:
    pass


def tqdm_s(enumerable_instance):
    try:
        from tqdm import tqdm
        return tqdm(enumerable_instance)
    except ModuleNotFoundError:
        return enumerable_instance


def load_stanford_dataset(path="../web-Stanford.txt") -> List[Tuple[int, int]]:
    print("Load SNAP...")
    edges = []
    with open(path, "r") as f:
        for line in tqdm_s(f):
            if not line.startswith("#"):
                u, v = tuple([int(x) for x in line.strip().split()])
                edges.append((u, v))
    return edges


def get_node_data_structure(graph: List[Tuple[int, int]]) \
        -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int]]:
    node_idx_to_followers, node_idx_to_followings = defaultdict(list), defaultdict(list)

    node_set = set()
    for u, v in graph:
        node_set.update([u, v])
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_set)}

    # u --> v: u is follower of v, v is following of u.
    for u, v in graph:
        u_idx = node_id_to_index[u]
        v_idx = node_id_to_index[v]
        node_idx_to_followers[v_idx].append(u_idx)
        node_idx_to_followings[u_idx].append(v_idx)

    return node_idx_to_followers, node_idx_to_followings, node_id_to_index


def get_degree(node_idx_to_followers, node_idx_to_followings, node_id_to_index) -> Tuple[List[int], List[int]]:
    in_degree = [0 for _ in range(len(node_id_to_index))]
    out_degree = [0 for _ in range(len(node_id_to_index))]
    for node_idx, followers in node_idx_to_followers.items():
        in_degree[node_idx] = len(followers)
    for node_idx, followings in node_idx_to_followings.items():
        out_degree[node_idx] = len(followings)
    return in_degree, out_degree


def pagerank(graph: List[Tuple[int, int]], damping_factor=0.85) -> List[Tuple[int, float]]:
    """
    Implement the variant of pagerank described in Section 2.2.

    Inputs:
        graph: directed input graph in the edge list format.
        That is, graph is a list of tuples of the form (u, v),
        which indicates that there is a directed link u -> v.
        You can assume that both u and v are integers, while you cannot
        assume that the integers are within a specific range.
        You can assume that there is no isolated node.
        damping_factor: damping factor of the variant of pagerank

    Output: a list of tuples of the form (v, score), which indicates node v
        and its pagerank score. The list should be sorted in the decreasing
        order of pagerank scores. If multiple nodes have the same pagerank
        score, then return them in any order.
    """
    print("\nPAGERANK with DF of {}".format(damping_factor))

    node_idx_to_followers, node_idx_to_followings, node_id_to_index = get_node_data_structure(graph)
    index_to_node_id = {index: node_id for node_id, index in node_id_to_index.items()}
    in_degree, out_degree = get_degree(node_idx_to_followers, node_idx_to_followings, node_id_to_index)
    in_degree_sum = sum(in_degree)

    num_nodes = len(node_id_to_index)
    rank_old = {node_idx: 1 / num_nodes for node_idx in range(num_nodes)}

    diff, num_iter, eps = 100, 0, 0.0001
    while diff > eps:
        rank_new = {}

        diff = 0
        for node_idx in range(num_nodes):
            followers_of_node = node_idx_to_followers[node_idx]
            # Follows a link from the current page (with probability c).
            if len(followers_of_node) > 0:
                _rank_follows = sum(rank_old[f_idx] / out_degree[f_idx] for f_idx in followers_of_node)
            # When the current web page does not have any links,
            # the surfer always jumps to a random page.
            else:
                _rank_follows = 0

            # Jumps to a random page (with prob. 1 - c)
            # The random surfer chooses a destination considering the in-degree of web pages:
            #   P(web page i is chosen) \prop 1 + d_{in}(i)
            _rank_random = (1 + in_degree[node_idx]) / (num_nodes + in_degree_sum)

            rank_new[node_idx] = damping_factor * _rank_follows + (1 - damping_factor) * _rank_random
            diff += abs(rank_new[node_idx] - rank_old[node_idx])

        rank_old = rank_new
        num_iter += 1
        print("ITER: {}, DIFF: {}".format(num_iter, diff))

    return [(index_to_node_id[node_idx], rank) for node_idx, rank in rank_old.items()]


def sorted_pagerank(graph: List[Tuple[int, int]], damping_factor=0.85):
    pr = pagerank(graph, damping_factor)
    return sorted(pr, key=lambda v: -v[1])


def analyze_stanford_10(dataframe_path="../stanford_pagerank.pkl"):
    try:
        df = pd.read_pickle(dataframe_path)
        print("Load at {}".format(dataframe_path))
    except FileNotFoundError:
        stanford_graph = load_stanford_dataset()

        node_idx_to_followers, node_idx_to_followings, node_id_to_index = get_node_data_structure(stanford_graph)
        in_degree, out_degree = get_degree(node_idx_to_followers, node_idx_to_followings, node_id_to_index)

        damping_factor_list = [round(dfx10 * 0.1, 2) for dfx10 in range(1, 10, 2)]

        rank_list = []
        top10_node_set = set()
        for damping_factor in damping_factor_list:
            rank = sorted_pagerank(stanford_graph, damping_factor)
            rank_list.append(rank)
            top10_node_set.update([node_id for node_id, _ in rank[:10]])

        data = []
        for damping_factor, rank in zip(damping_factor_list, rank_list):
            checked = set()
            for i, (node_id, rank_value) in enumerate(rank):
                if node_id in top10_node_set:
                    checked.add(node_id)
                    node_idx = node_id_to_index[node_id]
                    data.append(
                        [damping_factor, node_id, rank_value, i < 10, in_degree[node_idx], out_degree[node_idx]])

                if len(checked) == len(top10_node_set):
                    break
        df = pd.DataFrame(
            data=data,
            columns=["damping factor", "node id", "pagerank", "highest 10 nodes", "in-degree", "out-degree"]
        )
        df.to_pickle(dataframe_path)
        print("Save at {}".format(dataframe_path))

    nodes_dec_by_in_degree = df.sort_values(["in-degree", "node id"], ascending=[False, False])
    unique_nodes_dec_by_in_degree = pd.unique(nodes_dec_by_in_degree["node id"])

    df_to_print = df.pivot(index="node id", columns="damping factor", values="pagerank")
    print("\t".join(["node id \\ damping factor", "0.1", "0.3", "0.5", "0.7", "0.9"]))
    for idx, r in df_to_print.iterrows():
        print("\t".join([str(idx)] + [str(round(emt, 6)) for emt in r]))

    sns.relplot(x="damping factor", y="pagerank", data=df,
                legend="full", kind="line", col="node id", col_wrap=3, col_order=unique_nodes_dec_by_in_degree,
                palette=sns.color_palette("tab20", len(pd.unique(df["node id"]))))
    plt.savefig("./tex/figs/stanford_df_vs_pagerank.png", bbox_inches='tight')
    plt.clf()

    sns.relplot(x="in-degree", y="pagerank", hue="node id", data=df,
                legend="full", col="damping factor", s=70, alpha=0.75,
                palette=sns.color_palette("tab20", len(pd.unique(df["node id"]))))
    plt.savefig("./tex/figs/stanford_indegree_vs_pagerank.png", bbox_inches='tight')
    plt.clf()

    sns.relplot(x="out-degree", y="pagerank", hue="node id", data=df,
                legend="full", col="damping factor", s=70, alpha=0.75,
                palette=sns.color_palette("tab20", len(pd.unique(df["node id"]))))
    plt.savefig("./tex/figs/stanford_outdegree_vs_pagerank.png", bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    MODE = "ANALYSIS"

    if MODE == "SMALL":
        """
        [(2, 0.425133489418432), (3, 0.37370009257188536), (5, 0.0759132065822711),
         (4, 0.033845368431195004), (6, 0.033845368431195004), ... ]
        """
        small_rank = sorted_pagerank([(2, 3), (3, 2), (4, 1), (4, 2), (5, 2), (5, 4),
                                      (5, 6), (6, 2), (6, 5), (7, 2), (7, 5), (8, 2),
                                      (8, 5), (9, 2), (9, 5), (10, 5), (11, 5)], 0.85)
        print(small_rank[:10])

    elif MODE == "STANFORD":
        stanford_dataset = load_stanford_dataset()
        stanford_rank = sorted_pagerank(stanford_dataset, 0.85)
        print(stanford_rank[:10])

    elif MODE == "ANALYSIS":
        analyze_stanford_10()
