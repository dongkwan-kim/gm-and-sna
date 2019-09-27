# No external imports are allowed other than numpy

from typing import Dict, List, Set


def wcc (adjdict: Dict[int, List[int]]) -> List[Set[int]]:
    """
    Return the list of weakly connected components of given graph.
    
    Inputs:
        adjdict: directed input graph in the adjacency dictionary format
    Outputs: a list of the weakly connected components.
       Each WCC is represented as a set of node indices.
    """
    wcc_list: List[Set[int]] = [] # List of WCC
    
    # Implement your code

    return wcc_list
