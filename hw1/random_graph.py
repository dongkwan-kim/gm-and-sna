import numpy as np
from typing import List, Tuple


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
    pass


if __name__ == '__main__':
    edge_list = preferential_attachment()
