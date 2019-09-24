from itertools import combinations
from copy import deepcopy
from pprint import pprint
from typing import List, Tuple, Set
from collections import Counter
import numpy as np
try:
    import networkx as nx
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


def draw_networkx(edges_or_g):
    if isinstance(edges_or_g, list):
        G = nx.Graph()
        G.add_edges_from(edges_or_g)
    else:
        G = edges_or_g
    pos = nx.layout.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=5)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def _attach_one_node(new_node: int, active_node_set: Set[int], u: float,
                     degree_array: np.ndarray, attached_nodes: List[int]) -> Tuple[np.ndarray, list]:

    # For each of the m edges of the new node, it is decided randomly (and independently) whether the
    # edge connects to a random node, or to an active node.

    # The former case happens with probability u, in which the random node is chosen based on
    # preferential attachment (i.e., the chance of a node being chosen is proportional to its degree).
    if np.random.random() < u:
        if degree_array.sum() != 0:
            possible_nodes = np.asarray([n for n in range(1, new_node) if n not in attached_nodes])
            probs = degree_array[possible_nodes - 1]
            probs /= probs.sum()
            selected_node, = np.random.choice(possible_nodes, 1, p=probs)
        else:  # Case of new_node = 2, active_node_set = {1}
            selected_node = 1

    # Of course, the latter case happens with probability 1 - u. In the latter case,
    # the chance that each active node is chosen is the same. In both cases, the same node cannot be chosen as a
    # destination multiple times. (i.e., there should be no duplicated edges).
    else:
        selected_node, = np.random.choice([n for n in active_node_set if n not in attached_nodes], 1)

    # Update degree_array
    degree_array[new_node - 1] += 1
    degree_array[selected_node - 1] += 1

    # Update attached_nodes
    attached_nodes.append(selected_node)

    return degree_array, attached_nodes


def _preferential_attachment_with_intermediates(m, u, N, interval=None):

    intermediates_edge_list = []

    active_node_set = set(range(1, m + 1))
    edge_list = list(combinations(range(1, m + 1), 2))
    degree_array = np.zeros(N)
    degree_array[:m] = m - 1

    for new_node in range(m + 1, N + 1):

        assert len(active_node_set) == m

        # A new node arrives and forms m edges.
        attached_nodes = []
        for _ in range(m):
            degree_array, attached_nodes = _attach_one_node(
                new_node=new_node,
                active_node_set=active_node_set,
                u=u,
                degree_array=degree_array,
                attached_nodes=attached_nodes)
        edge_list += [(selected_node, new_node) for selected_node in attached_nodes]

        # An active node is chosen and turned inactive.
        # The chance of a node being chosen is proportional to the inverse of its degree.
        active_nodes = np.asarray(list(active_node_set))
        inv_deg_of_active_nodes = (1 / degree_array[active_nodes - 1])
        inactive_node, = np.random.choice(active_nodes, 1,
                                          p=inv_deg_of_active_nodes / inv_deg_of_active_nodes.sum())
        active_node_set.remove(inactive_node)

        # The new node is turned active.
        active_node_set.add(new_node)

        if interval is not None and new_node % interval == 0:
            intermediates_edge_list.append(deepcopy(edge_list))

    return edge_list, intermediates_edge_list


def preferential_attachment(m=8, u=0.1, N=10000) -> List[Tuple[int, int]]:
    """
    Preferential attachment with long-range connections model.
    Inputs:
        m: number of initial active nodes
        u: the probability in our model
        N: the number of nodes

    Output: undirected output graph in the edge list format.
        That is, return a list of tuples of the form (u, v), which
        indicates that there is an edge between nodes u and v.
        Both u and v should be integers between 1 and N.
        For example, if the generated graph is a triangle,
        then the output should be [(1,2), (2,3), (3,1)].
        The tuples do not have to be sorted in any order.
    """
    edge_list, _ = _preferential_attachment_with_intermediates(m, u, N, interval=None)
    return edge_list


def get_g_from_edges(edges):
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def analyze_effect_of_u(subjects=None, debug=False, seed=4):

    np.random.seed(seed)
    sns.set(style="ticks")

    prefix = "large" if not debug else "small"
    subjects = subjects or ["clustering_coefficient",
                            "speed_of_increasing_of_average_path_length",
                            "degree_distribution"]

    u_list_11 = [round(0.1 * i, 1) for i in range(11)]

    graph_kwargs = dict(m=8, N=10000, interval=2000) if not debug else dict(m=4, N=1000, interval=200)

    g_list: List[nx.Graph] = []
    itm_g_lists: List[List[nx.Graph]] = []
    for i, u in enumerate(tqdm_s(u_list_11)):
        edges, itm_edges_list = _preferential_attachment_with_intermediates(u=u, **graph_kwargs)
        g = get_g_from_edges(edges)
        g_list.append(g)

        if i % 2 == 0:  # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            itm_g_lists.append([])
            for interval_i, itm_edges in enumerate(itm_edges_list):
                itm_g = get_g_from_edges(itm_edges)
                itm_g_lists[-1].append(itm_g)

    if "degree_distribution" in subjects:
        print("\ndegree_distribution")
        data = []
        for i, u in enumerate(tqdm_s(u_list_11)):
            if i % 2 == 0:
                g = g_list[i]
                degree_counter = Counter([d for n, d in g.degree])
                for degree, frequency in degree_counter.items():
                    data.append([u, np.log10(degree), np.log10(frequency)])
        df = pd.DataFrame(data, columns=["mu", "log10 of degree", "log10 of frequency"])
        sns.relplot(x="log10 of degree", y="log10 of frequency", col="mu", data=df, col_wrap=3)
        plt.savefig("./tex/figs/{}_degree_distribution.png".format(prefix), bbox_inches='tight')
        plt.clf()

    if "clustering_coefficient" in subjects:
        print("\nclustering_coefficient")
        data = []
        for i, u in enumerate(tqdm_s(u_list_11)):
            g = g_list[i]
            coefficient = nx.average_clustering(g)
            data.append([u, coefficient])
        df = pd.DataFrame(data, columns=["mu", "clustering coefficient"])
        sns.relplot(x="mu", y="clustering coefficient", data=df, kind="line")
        plt.savefig("./tex/figs/{}_clustering_coefficient.png".format(prefix), bbox_inches='tight')
        plt.clf()

    if "speed_of_increasing_of_average_path_length" in subjects:
        print("\nspeed_of_increasing_of_average_path_length")
        palette = sns.color_palette("Set1", 6)
        u_to_color = {}
        data = []
        for i, itm_g_list in enumerate(tqdm_s(itm_g_lists)):
            u = u_list_11[2 * i]
            u_to_color[u] = palette.pop(0)
            for interval_i, itm_g in enumerate(itm_g_list):
                num_nodes = (interval_i + 1) * graph_kwargs["interval"]
                average_path_length = nx.average_shortest_path_length(itm_g)
                data.append([u, num_nodes, average_path_length])

        df = pd.DataFrame(data, columns=["mu", "number of nodes", "average path length"])
        sns.lmplot(x="number of nodes", y="average path length", hue="mu", data=df,
                   legend="full", ci=None, scatter_kws={"s": 25, "alpha": 1}, palette=u_to_color)
        plt.savefig("./tex/figs/{}_speed_of_inc_of_avg_path_length.png".format(prefix), bbox_inches='tight')
        plt.clf()

        df_mu_non_negative = df[df.mu > 0.0]
        sns.lmplot(x="number of nodes", y="average path length", hue="mu", data=df_mu_non_negative,
                   legend="full", ci=None, scatter_kws={"s": 25, "alpha": 1}, palette=u_to_color)
        plt.savefig("./tex/figs/{}_speed_of_inc_of_avg_path_length_non_neg_mu.png".format(prefix), bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    analyze_effect_of_u(debug=False)
