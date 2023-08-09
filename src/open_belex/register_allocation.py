r"""
 By Dylon Edwards

 Copyright 2023 GSI Technology, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the “Software”), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Sequence, Set

from open_belex.intermediate_representation import IntermediateRepresentation


# undirected graph for graph-coloring-based register allocation
@dataclass
class UndirectedGraph:
    V: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_edge(self: "UndirectedGraph", a: str, b: str) -> None:
        self.V[a].add(b)
        self.V[b].add(a)

    def vertices(self: "UndirectedGraph") -> Sequence[str]:
        return list(self.V.keys())


# maximum cardinality search
def maximum_cardinality_search(G: UndirectedGraph) -> Sequence[str]:
    V = set(G.vertices())
    W = V
    wt = defaultdict(int)

    n = len(V)
    v_ordering = []
    for _ in range(n):
        v = max([(wt[w], w) for w in W])[1]
        v_ordering.append(v)
        for u in W - G.V[v]:
            wt[u] = wt[u] + 1
        W.remove(v)

    return v_ordering


def greedy_coloring(vs: Sequence[str],
                    G: UndirectedGraph,
                    nregisters: int) -> Dict[str, int]:
    colors = list(range(1, nregisters + 1))
    col = defaultdict(int)
    for v_i in vs:
        colors_in_Ni = set([col[n_i] + 1 for n_i in G.V[v_i]])
        c = min([color for color in colors if color not in colors_in_Ni])
        col[v_i] = c - 1
    return col


def build_register_graph(ir: IntermediateRepresentation) -> UndirectedGraph:
    """Builds the interference graph over pseudo-registers (i.e. those alive at
    the same time and thus cannot be allocated the same registers)."""
    G = UndirectedGraph()

    for line_ in ir.liveness:
        line = list(line_)
        for i in range(len(line)):
            for j in range(i+1, len(line)):
                G.add_edge(line[i], line[j])
    return G


def allocate_registers(
        ir: IntermediateRepresentation,
        num_registers: int) -> Dict[str, int]:
    G = build_register_graph(ir)
    vs = maximum_cardinality_search(G)
    register_map = greedy_coloring(vs, G, num_registers)
    return register_map
