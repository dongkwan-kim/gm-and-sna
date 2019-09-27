# No external imports are allowed other than numpy

from typing import Dict, List, Set


def scc (adjdict: Dict[int, List[int]]) -> List[Set[int]]:
    """
    Return the list of strongly connected components of given graph.
    
    Inputs:
        adjdict: directed input graph in the adjacency dictionary format
    Outputs: a list of the strongly connected components.
       Each SCC is represented as a set of node indices.
    """
    scc_list: List[Set[int]] = [] # List of SCC

    # Implement your code
    
    return scc_list
