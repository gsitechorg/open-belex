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

import inspect
import logging
from collections import OrderedDict, defaultdict
from itertools import chain
from typing import (Any, Callable, Dict, NamedTuple, Sequence, Set, TypeVar,
                    Union)

from open_belex.apl import APL_sb_from_src, Mask
from open_belex.apl_optimizations import (apl_liveness_analysis,
                                          apply_optimizations)
from open_belex.directed_graph import DirectedGraph
from open_belex.expressions import (AssignOperator, BelexRepresentation,
                                    Variable)
from open_belex.intermediate_representation import (APL_or_comment,
                                                    IntermediateRepresentation)
from open_belex.laning import lane_apl_v2
from open_belex.register_allocation import allocate_registers

logging.basicConfig(format='%(message)s')
LOGGER = logging.getLogger()


def reverse_map(groups: Dict[int, Set[int]]) -> Dict[int, int]:
    rev_mp = {}
    for g in groups:
        for inst in groups[g]:
            assert not (inst in rev_mp)
            rev_mp[inst] = g
    return rev_mp


# merge dependence graph based on formed groups
def merge_dependence_graph(dep_graph: DirectedGraph,
                           groups: Dict[int, Set[int]]) -> DirectedGraph:

    instr_grps = reverse_map(groups)

    group_graph = DirectedGraph([gr for gr in groups.keys()])
    for gr, insts in groups.items():
        for inst in insts:
            succs = dep_graph.succ[inst]
            for succ in succs:
                group_graph.add_edge(gr, instr_grps[succ])

    return group_graph


def go_together(ir: Sequence[AssignOperator],
                graph: DirectedGraph,
                i: int, j: int) -> bool:
    i__morphology = ir[i].morphology()
    j__morphology = ir[j].morphology()
    have_same_morphology = (i__morphology == j__morphology)
    max_is_reachable_from_min = graph.reachable(min(i, j), max(i, j))
    return have_same_morphology and not max_is_reachable_from_min


def assign_group(ir: Sequence[AssignOperator],
                 i: int,
                 groups: Dict[int, Set[int]],
                 graph: DirectedGraph) -> None:

    for j in groups.keys():
        if all(go_together(ir, graph, i, k) for k in groups[j]):
            groups[j].add(i)
            return

    groups[i].add(i)


def find_groups(ir: Sequence[AssignOperator],
                graph: DirectedGraph) -> Dict[int, Set[int]]:

    groups = defaultdict(set)

    for i, inst in enumerate(ir):
        assign_group(ir, i, groups, graph)

    return groups


def create_hir_dependence_graph(ir: Sequence[AssignOperator]) -> DirectedGraph:
    graph = DirectedGraph([i for i in range(len(ir))])

    for i in range(len(ir)):
        for j in range(i+1, len(ir)):
            if ir[i].is_dep(ir[j]):
                graph.add_edge(i, j)

    return graph


T = TypeVar("T")
TSS = Sequence[Union[Sequence[T], T]]


def is_std_tuple(xs: TSS) -> bool:
    # A NamedTuple will both be an instance of tuple and have a "_fields"
    # attribute; a standard tuple will not have the latter.
    return isinstance(xs, tuple) and not hasattr(xs, "_fields")


def flatten(xs: TSS) -> Sequence[T]:
    flattened = []
    for x in xs:
        if isinstance(x, list) or is_std_tuple(x):
            ys = x
            for y in flatten(ys):
                flattened.append(y)
        else:
            flattened.append(x)
    return flattened


def lhs_indices(instrs: Sequence[AssignOperator], i: int) -> Sequence[int]:
    return flatten([instr.lhs()[i].indices for instr in instrs])


def rhs_indices(instrs: Sequence[AssignOperator], i: int) -> Sequence[int]:
    return flatten([instr.rhs()[i].indices for instr in instrs])


def head(xs: Sequence[T]) -> T:
    return xs[0]


def vectorize(ir: Sequence[AssignOperator],
              group: Set[int]) -> AssignOperator:

    g_ir = [ir[i] for i in group]
    g_ir = sorted(g_ir, key=lambda ir: ir.lhs()[0].indices[0])

    t = head(g_ir).clone()

    for i in range(len(t.lhs())):
        t.lhs()[i].indices = lhs_indices(g_ir, i)

    for i in range(len(t.rhs())):
        t.rhs()[i].indices = rhs_indices(g_ir, i)

    return t


def is_subset(a,b):
    a = set(a)
    b = set(b)

    return a.issubset(b)


def copy_propagation_and_elimination(intrep):
    global LOGGER
    LOGGER.debug('after coallescing temps:')

    defs = {}
    reversed_defs = {}

    for x in intrep.intermediate_code:
        # if it's a valid definition, add to dictionary

        # get uses
        for rhs in x.rhs():
            if (rhs.var in defs) and is_subset(rhs.indices, defs[rhs.var].indices):
                # replace use with a valid definition
                # OBS: might be safer to clone VariableAccess object
                LOGGER.debug(f"replacing {rhs.var} -> {defs[rhs.var].var} in {x}")
                rhs.var = defs[rhs.var].var

        # if this statement constitutes a definition, add it to the set
        if len(x.rhs()) == 1 and x.rhs()[0].indices == x.lhs().indices: #check if rhs is instance of VariableAccess
            rhs = x.rhs()[0]
            lhs = x.lhs()

            defs[lhs.var] = rhs
            reversed_defs[rhs.var] = lhs.var
        # if assignment that is not a definition, render previous associated defs invalid
        else:
            lhs = x.lhs()

            #render invalid previous definitions or definitions relying on this variable
            defs.pop(lhs.var, None)
            reversed_defs.pop(lhs.var, None)


            if lhs.var in defs:
                defs[lhs.var] = None
            if lhs.var in reversed_defs:
                reversed_defs[lhs.var] = None


def is_dead(access, liveset):
    varname = str(access.var)
    return (not (varname in liveset)) \
        or (Mask(access.indices).mask & liveset[varname] == 0)


def is_dead2(mask, sb, liveset):
    not_live = (not (sb in liveset))
    result = not_live or (mask & liveset[sb] == 0)
    return result


def remove_copy_from_temps(intrep):
    #
    global LOGGER
    n = len(intrep.intermediate_code)

    # look for pattern:
    #   t[<indices>] := <expr>
    #   var[<indices>] := t[<indices>]
    # and when t[<indices>] is dead after the assignment to var

    to_eliminate = set()

    for i in range(n-1):
        instr1, instr2 = intrep.intermediate_code[i], intrep.intermediate_code[i+1]
        liveset = intrep.liveness2[i+2]


        if len(instr2.rhs()) == 1 and instr2.rhs()[0].indices == instr2.lhs().indices and str(instr1.lhs().var) == str(instr2.rhs()[0].var) and instr1.lhs().indices == instr2.lhs().indices and is_dead(instr2.rhs()[0], liveset):
            to_eliminate.add(i+1)

    for i in range(n):
        if i+1 in to_eliminate:
            intrep.intermediate_code[i].lhs().var = intrep.intermediate_code[i+1].lhs().var
        elif i in to_eliminate:
            intrep.intermediate_code[i] = None

    intrep.intermediate_code = [x for x in intrep.intermediate_code if x is not None]


# DEBUG BBECKMAN: add final parameter "param_nyms"
def belex_compile(
        belex: Sequence[AssignOperator],
        output_variables: Sequence[Variable],
        param_nyms: Sequence[str]) \
        -> IntermediateRepresentation:

    global LOGGER

    # NOTE: LOGGER init should be left to the application, it should not be
    # hard coded at the library level.
    # LOGGER.setLevel(logging.DEBUG)
    # LOGGER.addHandler(logging.StreamHandler())

    LOGGER.debug('high-level HIR:')
    for i, x in enumerate(belex):
        LOGGER.debug(f"{i} {x}")

    LOGGER.debug('finding groups:')
    graph = create_hir_dependence_graph(belex)
    LOGGER.debug('-------- DEP GRAPH BEFORE GROUPING ----')
    graph.print_edges()

    LOGGER.debug("-------- DEP GRAPH AFTER GROUPING ----:")
    groups = find_groups(belex, graph)
    graph.print_edges()
    vector_ir = []

    for gr_id, grp in groups.items():
        vector_ir.append(vectorize(belex, grp))

    LOGGER.debug('vectorized IR:')
    for x in vector_ir:
        LOGGER.debug(str(x))
    LOGGER.debug('-------- vectorized IR DONE --------')

    group_graph = merge_dependence_graph(graph, groups)

    vectorized_hir = []

    for gr_id, grp in enumerate(group_graph.topological_sort()):
        vectorized_hir.append(vectorize(belex, groups[grp]))

    LOGGER.debug('vectorized HIR:')
    LOGGER.debug('----')
    for i, x in enumerate(vectorized_hir):
        LOGGER.debug(f"{i} {x}")

    intrep = IntermediateRepresentation()
    # DEBUG -- original compiler
    # -----
    # for inst in vector_ir:
    for inst in vectorized_hir:
        inst.render(intrep)

    LOGGER.debug('generating MIR:')
    for x in intrep.intermediate_code:
        LOGGER.debug(str(x))
    LOGGER.debug('-------- MIR DONE --------')

    intrep.coalesce_temps()

    LOGGER.debug('----Before copy prop-----')
    for i,x in enumerate(intrep.intermediate_code):
        LOGGER.debug(f"({i}) {x}")

    intrep.perform_liveness_analysis2(output_variables)
    ## // remove_copy_from_temps(intrep)  [[bbeckman 30 Oct 2021 possible
    ## source of regressions.]]
    #copy_propagation_and_elimination(intrep)

    LOGGER.debug('----After copy prop-----')
    for i,x in enumerate(intrep.intermediate_code):
        LOGGER.debug(f"({i}) {x}")

    intrep.perform_liveness_analysis(output_variables)

    LOGGER.debug('------live slices:------')

    for x, y in zip(intrep.intermediate_code, intrep.liveness2):
        LOGGER.debug(f"{x} # vars alive = {len(y)}")
        LOGGER.debug(','.join(f'{a}:{hex(b)}' for a,b in y.items()))
        LOGGER.debug(' ')

    LOGGER.debug('-------------------------')

    # DEBUG
    liveness = []

    # for live_vars in intrep.liveness:
    #     for out_var in output_variables:
    #         if out_var.symbol not in live_vars:
    #             live_vars = list(live_vars)
    #             live_vars.append(out_var.symbol)
    #             live_vars = tuple(live_vars)
    #     liveness.append(live_vars)
    # intrep.liveness = liveness

    # DEBUG BBECKMAN: add all belex params to live list
    # TODO  BBECKMAN: make live_vars a set instead of a tuple
    for live_vars in intrep.liveness:
        live_vars = list(live_vars)
        for param_nym in param_nyms:
            if param_nym not in live_vars and param_nym != "IR":
                live_vars.append(param_nym)
        live_vars = tuple(live_vars)
        liveness.append(live_vars)
    intrep.liveness = liveness


    for x, y in zip(intrep.intermediate_code, intrep.liveness):
        LOGGER.debug(f"{x} {y} # vars alive = {len(y)}")

    return intrep


Kwargs = Dict[str, Any]
BelexFn = Callable[[Kwargs], Variable]
SB = int


class BelexCompilation(NamedTuple):
    apl: Sequence[APL_or_comment]
    register_map: Dict[str, int]
    args_by_reg_id: Dict[int, Sequence[str]]
    out_param: str


def allocate_missing_registers(register_map: Dict[str, SB],
                               belex_repr: BelexRepresentation) -> None:
    # Quick way to ensure all registers are allocated
    # print('************************************************')
    # from pprint import pprint
    # pprint(register_map)
    # print('************************************************')
    max_reg_id = -1
    if len(register_map) > 0:
        max_reg_id = max(register_map.values())

    for param_nym in belex_repr.symbol_table.symbols.keys():
        if param_nym not in register_map:
            register_map[param_nym] = 1 + max_reg_id
            max_reg_id += 1


def is_local_var_nym(nym: str) -> bool:
    return nym.startswith("_INTERNAL")


def is_tmp_var_nym(nym: str) -> bool:
    return nym.startswith("t_")


def is_var_nym(nym: str) -> bool:
    return is_local_var_nym(nym) or is_tmp_var_nym(nym)


def is_param_nym(nym: str) -> bool:
    return not is_var_nym(nym)


def group_by_reg_categories(unsorted_register_map: Dict[str, int]) -> Sequence[str]:
    IRs = []
    params = []
    local_vars = []
    tmp_vars = []
    for nym in unsorted_register_map:
        if nym == "IR":
            IRs.append(nym)
        elif is_param_nym(nym):
            params.append(nym)
        elif is_local_var_nym(nym):
            local_vars.append(nym)
        elif is_tmp_var_nym(nym):
            tmp_vars.append(nym)
        else:
            raise ValueError(f"Unsupported nym type: {nym}")
    return IRs, params, local_vars, tmp_vars


def sort_register_map(unsorted_register_map: Dict[str, int]) -> Dict[str, int]:
    IRs, params, local_vars, tmp_vars = group_by_reg_categories(unsorted_register_map.keys())
    sorted_nyms = chain(IRs, params, sorted(local_vars), sorted(tmp_vars))
    sorted_register_map = OrderedDict()
    for nym in sorted_nyms:
        sorted_register_map[nym] = unsorted_register_map[nym]
    return sorted_register_map


def compile_apl_from_belex(
        function_under_test: BelexFn,
        optimizations: Sequence[Callable] = []) \
        -> BelexCompilation:

    global LOGGER

    sig = inspect.signature(function_under_test)
    kwnames = list(sig.parameters.keys())

    belex_repr = BelexRepresentation()

    compiler_symbols = [belex_repr.install_symbol(k) for k in kwnames]

    new_kwargs = dict(zip(kwnames, compiler_symbols))
    new_kwargs['IR'] = belex_repr

    the_retval = function_under_test(**new_kwargs)

    # DEBUG BBECKMAN: add kwnames final argument
    intrep = belex_compile(belex_repr.ir, [the_retval], kwnames)

    LOGGER.debug('IR before allocation')
    for var_set in intrep.liveness:
        LOGGER.debug(var_set)

    LOGGER.debug('------end----')

    register_map = allocate_registers(intrep, 24)
    allocate_missing_registers(register_map, belex_repr)
    register_map = sort_register_map(register_map)

    LOGGER.debug("Allocated registers:")
    for a, b in register_map.items():
        LOGGER.debug(f"{a} : {b}")

    apl = []
    for ic in intrep.intermediate_code:
        rep = ic.generate_apl(register_map)
        apl.append(rep)

    apl = flatten(apl)
    LOGGER.debug('Original APL:')
    for inst in apl:
        LOGGER.debug(str(inst))

    # DEBUG BBECKMAN HACK
    inverse_reg_map = {
        v: [q for q, w in register_map.items() if v == w]
        for k, v in register_map.items()}
    context = {"keep_alives": [register_map[the_retval.symbol]],
               "alive_symbol": the_retval.symbol}
    apl = apply_optimizations(apl, optimizations, context)
    apl_liveness = apl_liveness_analysis(apl, context)

    dead_writes = []
    n = len(apl)
    for i in range(n - 1):  # HACK: the only output is in the last command.
        mask = apl[i].msk.mask
        if type(apl[i].stmt) == APL_sb_from_src:
            sbs = apl[i].stmt.sbs
            for sb in sbs:
                liveness_key = f'liveness[{i + 1}]'
                inspectable = {
                    'i': i, 'vars': inverse_reg_map[sb],
                    'sb': sb, 'mask': hex(mask),
                    liveness_key: {k: (inverse_reg_map[k], hex(v))
                                   for k, v in apl_liveness[i + 1].items()},
                    'is_dead2': is_dead2(mask, sb, apl_liveness[i + 1])}
                # pprint(inspectable)
                if is_dead2(mask, sb, apl_liveness[i + 1]):
                    dead_writes.append(i)
                    LOGGER.debug(f"*** {sb} is dead in {apl[i]}")

    LOGGER.debug('------APL and liveness:-----')

    for inst, liveness in zip(apl, apl_liveness):
        LOGGER.debug(f"{inst}")
        LOGGER.debug(','.join(f'{a}:{hex(b)}' for a,b in liveness.items()))
        LOGGER.debug(" ")

    LOGGER.debug('---------------------------')
    if dead_writes:
        dead_writes.pop()
    ## apl = [cmd for i, cmd in enumerate(apl) if not i in dead_writes]
    ## [[bbeckman: the line above causes 17 regressions]]

    LOGGER.debug('APL before laning: ----------')
    for inst in apl:
        LOGGER.debug(inst)
        LOGGER.debug(inst.defs())
        LOGGER.debug(inst.uses())

    LOGGER.debug('-----------------------------')
    m = len(apl)

    LOGGER.debug('Pairs of conflicts: ----------')

    ################# Laning V2 starts here ###############
    apl_dag = DirectedGraph(range(m))
    roots_set = set()
    conflict_list = []
    dont_conflict_list = []
    for i in range(m):
        conflict = False
        for j in range(i):
            if apl[i].conflict(apl[j]):
                apl_dag.add_edge(j, i)
                conflict = True
                conflict_list.append(
                    f'{apl[j]}\n{apl[i]} NOT REVERSIBLE\n')
            else:
                dont_conflict_list.append(
                    f"{apl[j]}\n{apl[i]} REVERSIBLE\n")
        if not conflict:
            roots_set.add(i)

    # pprint treats newline \n characters poorly. Use print.
    # print('')
    # for c in conflict_list:
    #     print(c)
    # for d in dont_conflict_list:
    #     print(d)

    # pprint('')
    # pprint('\n'.join(conflict_list))
    # pprint('\n'.join(dont_conflict_list))

    # am = apl_dag.adjacency_matrix()
    # fig = pyplot.figure(figsize=(5,5))
    # pyplot.imshow(am, cmap='Greys', interpolation='none')

    # size_of_biggest_roots_set = 0
    # alive_set = set(range(m))
    # i = 0
    # valid_to_schedule = []
    # while len(roots_set) > 0:
    #     print(f"iteration {i} : roots = {[str(apl[t]) for t in roots_set]}")
    #     # Picks a vertex unpredictably.
    #     curr_vertex = roots_set.pop()
    #     valid_to_schedule.append(curr_vertex)
    #     print(f"   selected {apl[curr_vertex]}")
    #     alive_set.remove(curr_vertex)
    #     for v in apl_dag.succ[curr_vertex]:
    #         if not (apl_dag.pred[v] & alive_set):
    #             roots_set.add(v)
    #             size_of_biggest_roots_set = max(size_of_biggest_roots_set,
    #                                            len(roots_set))
    #     i += 1

    # needs to be continued...

    # 1) take a few tests and run the loop above - check to see size
    #    of the root set at each time

    # 2) look at the def and use sets manually and verify they are correct

    # 3) Finish the loop above by making bundles of commands available in
    #    root at each cycle into instructions according to the laning rules.
    #    Special cases: read after write, broadcast after read;
    #    equivalently,  write before read, read before broadcast

    # 4) replace lane_apl_v1 with the new laning strategy

    ##################### laning v2 ends here ############
    # TODO: Sample trajectories & evaluate; explore stochastic optimization strategy.
    #  Partition roots into commands compatible with current, and not. If
    #  incompatible roots is empty, emit. Else, recurse sampling.

    # TODO: Distinguish library generator from compiler. Lib gen can do
    #   overnight optimization runs.

    # TODO: Better to consider the list "valid_to_schedule" than to sort the
    #  graph topologically.

    # node_order = apl_dag.topological_sort2()
    # print ('node ordering:', node_order)
    # print ('graph roots: ', roots_set)

    # apl = [apl[i] for i in valid_to_schedule]

    # NOISY!
    LOGGER.debug('APL:')
    # for inst in apl:
    #     print(str(inst))
    #     print('defs:', inst.defs())
    #     print('uses:', inst.uses())

    # apl = lane_apl_v0(apl)

    apl = lane_apl_v2(apl, roots_set, apl_dag)
    # apl = lane_apl_v1(apl)

    LOGGER.debug(str(register_map))

    args_by_reg_id = defaultdict(list)
    out_nym = the_retval.symbol
    reg_id = register_map[out_nym]
    args_by_reg_id[reg_id].append(out_nym)
    for arg in kwnames:
        if arg in register_map and arg != out_nym:
            reg_id = register_map[arg]
            args_by_reg_id[reg_id].append(arg)
    args_by_reg_id = dict(args_by_reg_id)

    return BelexCompilation(
        apl=apl,
        register_map=register_map,
        args_by_reg_id=args_by_reg_id,
        out_param=the_retval.symbol)
