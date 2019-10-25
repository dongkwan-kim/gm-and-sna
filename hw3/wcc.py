import queue
from typing import Dict, List, Set, Iterator


# Class for graph
class Graph:
    def __init__(self):
        self.__nodes: Dict[int, List[int]] = {}

    # enddef

    def add_node(self, node: int):
        if node not in self.__nodes:
            self.__nodes[node] = []
        # endif

    # enddef

    def add_edge(self, u: int, v: int):
        if u not in self.__nodes:
            self.__nodes[u] = []
        # endif
        if v not in self.__nodes:
            self.__nodes[v] = []
        # endif
        self.__nodes[u].append(v)
        self.__nodes[v].append(u)

    # enddef

    def remove_node(self, node: int):
        try:
            nbrs: List[int] = self.__nodes[node]
            del self.__nodes[node]
        except KeyError:
            pass
        else:
            for n in nbrs:
                self.__nodes[n].remove(node)
            # endfor
        # endtry

    # enddef

    @property
    def nodes(self) -> List[int]:
        return list(self.__nodes.keys())

    # enddef

    def neighbors(self, node: int) -> Iterator[int]:
        return iter(self.__nodes[node])

    # enddef

    @property
    def degree(self) -> Dict[int, int]:
        return {k: len(v) for k, v in self.__nodes.items()}
    # enddef


# endclass


# Convert adjacency dictionary to Graph
def adjdict_to_G(adjdict: Dict[int, List[int]]) -> Graph:
    G = Graph()

    for node in adjdict:
        nbs = adjdict[node]
        if len(nbs) == 0:
            G.add_node(node)
        # endif
        for nb in nbs:
            G.add_edge(node, nb)
        # endfor
    # endfor

    return G


# enddef


# Implementation of the non-recursive WCC algorithm
def wcc(adjdict: Dict[int, List[int]]) -> List[Set[int]]:
    wcc_list: List[Set[int]] = []  # List of wcc

    G: Graph = adjdict_to_G(adjdict)
    nodeVisited: Dict[int, bool] = {i: False for i in G.nodes}

    for node in G.nodes:
        if not nodeVisited[node]:
            newWCC: Set[int] = set()  # Start a new WCC
            q = queue.Queue()
            q.put(node)
            nodeVisited[node] = True

            # Using BFS
            while not q.empty():
                n = q.get()
                newWCC.add(n)

                for v in G.neighbors(n):
                    if not nodeVisited[v]:
                        q.put(v)
                        nodeVisited[v] = True
                    # endif
                # endfor
            # endwhile

            wcc_list.append(newWCC)
        # endif
    # endfor

    return wcc_list
# enddef
