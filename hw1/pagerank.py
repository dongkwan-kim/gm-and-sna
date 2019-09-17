import numpy as np
from typing import Tuple, List


def pagerank(graph, damping_factor=0.85) -> List[Tuple[int, float]]:
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
    pass


if __name__ == '__main__':
    """
    [(2, 0.425133489418432), (3, 0.37370009257188536), (5, 0.0759132065822711),
     (4, 0.033845368431195004), (6, 0.033845368431195004), ... ]
    """
    pagerank([(2, 3), (3, 2), (4, 1), (4, 2), (5, 2), (5, 4),
              (5, 6), (6, 2), (6, 5), (7, 2), (7, 5), (8, 2),
              (8, 5), (9, 2), (9, 5), (10, 5), (11, 5)], 0.85)
