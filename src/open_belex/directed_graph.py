r"""
By Dylon Edwards
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, Sequence, Set, Tuple

import numpy

WHITE, GRAY, BLACK = 0, 1, 2

LOGGER = logging.getLogger()


class DirectedGraph:
    succ: Dict[int, Set[int]]
    pred: Dict[int, Set[int]]
    V: Set[int]

    def __init__(self: "DirectedGraph", vertices: Iterable[int]) -> None:
        self.succ = defaultdict(set)
        self.pred = defaultdict(set)
        self.V = set(vertices)

    def add_edge(self: "DirectedGraph", a: int, b: int) -> None:
        if a not in self.V or b not in self.V:
            raise "ERROR"
        self.succ[a].add(b)
        self.pred[b].add(a)

    def vertices(self: "DirectedGraph") -> Sequence[int]:
        return list(self.V)

    def edges(self: "DirectedGraph") -> Sequence[Tuple[int, int]]:
        return [(a, b) for a in self.succ.keys() for b in self.succ[a]]

    def get_src(self: "DirectedGraph") -> Sequence[int]:
        return [x for x in self.V if not self.pred[x]]

    def __eq__(self: "DirectedGraph", other: Any) -> bool:
        return isinstance(other, DirectedGraph) \
            and self.succ == other.succ \
            and self.pred == other.pred \
            and self.V == other.V

    def __ne__(self: "DirectedGraph", other: Any) -> bool:
        return not self.__eq__(other)

    def clone(self: "DirectedGraph") -> "DirectedGraph":
        newgraph = DirectedGraph(self.V)
        for src, tgt in self.edges():
            newgraph.add_edge(src, tgt)
        return newgraph

    def reachable(self: "DirectedGraph", src: int, tgt: int) -> bool:
        Q = deque([src])
        color = defaultdict(int)
        color[src] = BLACK

        while Q:
            v = Q.popleft()

            if v == tgt:
                return True

            for u in self.succ[v]:
                if color[u] == WHITE:
                    color[u] = BLACK
                    Q.append(u)

        return False

    def topological_sort2(self: "DirectedGraph") -> Sequence[tuple]:
        color = defaultdict(int)
        clock = {'t': 0}
        finished = defaultdict(int)

        src = self.get_src()

        for v in src:
            if color[v] == WHITE:
                self.dfs_visit(finished, color, clock, v)

        ordering = sorted([(-order, v) for v, order in finished.items()])
        return ordering #[(i,v) for i, v in ordering]

    def topological_sort(self: "DirectedGraph") -> Sequence[int]:
        ordering = self.topological_sort2()
        return [v for _, v in ordering]

    # undirected graph for graph-coloring-based register allocation
    def dfs_visit(self: "DirectedGraph",
                  finished: Dict[int, int],
                  color: Dict[int, int],
                  clock: Dict[str, int],
                  v: int) -> None:

        global WHITE, GRAY, BLACK

        clock['t'] += 1
        color[v] = GRAY

        for u in self.succ[v]:
            if color[u] == WHITE:
                self.dfs_visit(finished, color, clock, u)

        color[v] = BLACK
        clock['t'] += 1
        finished[v] = clock['t']

    def print_edges(self: "DirectedGraph") -> None:
        global LOGGER
        LOGGER.debug('-------- VERTICES OF DIRECTED GRAPH --------')
        LOGGER.debug(self.V)
        for src, dst in self.edges():
            LOGGER.debug(f"{src} -> {dst}")

    def adjacency_matrix(self) -> "numpy.ndarray(dtype=numpy.bool)":
        minV = min(self.V)
        maxV = max(self.V)
        index_count = maxV - minV + 1
        shape_tuple = (index_count, index_count)
        raw = numpy.zeros(shape_tuple)
        for src, tgt in self.edges():
            raw[src-minV, tgt-minV] = True
        result = numpy.array(raw, dtype=numpy.bool)
        return result
