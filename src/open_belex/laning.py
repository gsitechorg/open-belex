r"""
By Dylon Edwards
"""

from typing import List, Set

from open_belex.apl import APL_multi_statement
from open_belex.directed_graph import DirectedGraph
from open_belex.intermediate_representation import APL_or_comment


def lane_apl_v0(apl):
    return [APL_multi_statement(stmts=[x]) for x in apl]


def lane_apl_v1(apl):
    apl2 = [APL_multi_statement(stmts=[apl[0]])]
    for candidate_cmd in apl[1:]:
        current_instruction = apl2[-1]

        if current_instruction.is_laning_v1_compatible(candidate_cmd):
            current_instruction.stmts.append(candidate_cmd)
        else:
            apl2.append(APL_multi_statement(stmts=[candidate_cmd]))

    return apl2


def select_compatible(curr_instr, roots_set, apl):
    r"""Given curr_instr and root_set, find all cmds in apl that are
    compatible with curr_instr. Select one by oracle (say greedy).

    Precondition: only called on non-empty roots.

    Invariant: select_compatible will select any element in roots_set
    if the curr_instr is empty.
    """
    result = None

    compatible_set = set(
        [cmd_number
         for cmd_number in roots_set
         if curr_instr.is_laning_v2_compatible(apl[cmd_number])])

    if len(compatible_set) > 0:
        result = next(iter(compatible_set))  # Pick arbitrary element
        # TODO: Sophisticated Combinatorial Constraint Optimization :)

    return result;


def lane_apl_v2(apl: List[APL_or_comment],
                roots_set: Set[int],
                apl_dag: DirectedGraph) \
        -> List[APL_multi_statement]:

    r"""Empty roots implied by empty apl, and produces empty apl2 result."""

    assert len(apl) > 0
    assert len(roots_set) > 0

    # apl2 = [APL_multi_statement(stmts=[apl[0]])]
    apl2 = []
    m = len(apl)
    assert len(apl_dag.V) == m
    alive_set = set(range(m))

    # curr_instr = APL_multi_statement(stmts=[apl[0]])
    curr_instr = APL_multi_statement(stmts=[])
    # TODO: Do fresh compuatation of the graph inside this function.

    while len(roots_set) > 0:
        # Prevent double add of first candidate.
        candidate_cmd_number = select_compatible(curr_instr, roots_set, apl)
        if candidate_cmd_number is not None:
            curr_instr.stmts.append(apl[candidate_cmd_number])
            roots_set.remove(candidate_cmd_number)
            alive_set.remove(candidate_cmd_number)
            for succ_number in apl_dag.succ[candidate_cmd_number]:
                if not (apl_dag.pred[succ_number] & alive_set):
                    roots_set.add(succ_number)
        else:  # candidate_cmd_number is None
            apl2.append(curr_instr)
            curr_instr = APL_multi_statement(stmts=[])

    if len(curr_instr.stmts) > 0:
        # Emit instruction by adding curr_instr to apl2:
        apl2.append(curr_instr)

    return apl2
