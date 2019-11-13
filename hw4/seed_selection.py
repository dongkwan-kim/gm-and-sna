import numpy as np
# No external imports are allowed other than numpy


def seed_selection(graph, policy, n):
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