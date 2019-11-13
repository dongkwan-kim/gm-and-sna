import numpy as np
# No external imports are allowed other than numpy


def count_influence(graph, seeds, threshold):
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
