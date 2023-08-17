r"""open_belex.intermediate_representation: Intermediate representations for High-Level Belex.

By Brian Beckman and Dylon Edwards

********************************************************

This module consists of classes and functions for intermediate
representations for communicating amongst phases of the compiler
for high-level Belex.

Classes
    * APL_or_comment
    * AssignmentType
    * IntermediateRepresentation
    * IrAssignOperator
    * IrBinaryOperator
    * IrNibbleAndFoldOperator
    * IrReduceGGLOperator

Functions
    * access_overlap
    * find_stride
    * get_first_viable

Global Data
    * LOGGER

**Detailed Docs**

"""

import logging
from abc import ABC
from dataclasses import field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from open_belex.apl import (APL_comment, APL_masked_stmt, APL_rl_from_sb,
                            APL_rl_from_src, APL_sb_from_src, APL_src_from_rl,
                            APL_statement, Mask)
from open_belex.bleir.types import bleir_dataclass
from open_belex.expressions import SymbolTable, Variable, VariableAccess
from open_belex.renderable import Renderer, Rendition

LOGGER = logging.getLogger()


class AssignmentType(Enum):
    NRL = 0
    SRL = 1
    GL  = 2


def validate_assignment(
        left_indices: Sequence[int],
        right_indices: Sequence[int]) -> \
        Tuple[int, AssignmentType]:
    r"""Validate an assignment with multiple indices on the left and
    right to ensure the cardinalities are equal and that strides
    are consistent with NRL or SRL, that is, with section shifts.

    Parameters
    ----------
    left_indices: Sequence[int]
        input: section indices on the left-hand side of an
        assignment statement.

    right_indices: Sequence[int]
        input: section indices on the right-hand side of an
        assignment statement.

    Returns
    -------
    Tuple[int, AssignmentType]
        a tuple of the absolute value of the stride and an
        AssignmentType: GL if the stride is zero, NRL if the
        stride is positive, and SRL if the stride is negative.

    """
    if len(right_indices) == 1 and len(left_indices) > 1:
        return 0, AssignmentType.GL

    if len(right_indices) == len(left_indices):
        stride = find_stride(left_indices, right_indices)
        nsteps = abs(stride)
        if stride > 0:
            assignment_type = AssignmentType.NRL
        else:
            assignment_type = AssignmentType.SRL
        return nsteps, assignment_type

    raise ValueError(
        f"ERROR: Invalid stride or broadcast assignment: "
        f"left_indices={left_indices}, right_indices={right_indices}")


def find_stride(left_indices: Sequence[int],
                right_indices: Sequence[int]) -> int:
    r"""Calculate the stride for two sequences of section indices,
    ensuring that the stride is equal for each pair, in order.
    Return 0 if the sequences are empty.

    Parameters
    ----------
    left_indices: Sequence[int]
        input: section indices on the left-hand side of an
        assignment statement.

    right_indices: Sequence[int]
        input: section indices on the right-hand side of an
        assignment statement.

    Returns
    -------
    int
        the stride, that is, the difference between a
        corresponding pair of indices. The strides must be equal
        for all pair of indices in the sequences.

    """
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0

    stride = left_indices[0] - right_indices[0]

    # validate stride is valid
    if not all(left_idx - right_idx == stride
               for left_idx, right_idx
               in zip(left_indices, right_indices)):
        raise ValueError(
            f"ERROR: Stride must be constant across indices: "
            f"left_indices={left_indices}, right_indices={right_indices}")

    # TODO: Determine if this is the correct behavior. The indices will not
    #  necessarily be ordered.
    return stride


class Operator(ABC):
    """Base class for intermediate code. Subclasses are
    IrAssignOperator, IrBinaryOperator, IrNibbleAndFoldOperator,
    and IrReduceGGLOperator.
    """
    stmt: APL_statement
    msk: Mask
    attr_map: Callable[..., Dict[str, Any]]


APL_or_comment = Union[APL_masked_stmt, APL_comment]


@bleir_dataclass
class IrReduceGGLOperator(Operator, Rendition):
    a: VariableAccess
    r: VariableAccess

    def lhs(self: "IrBinaryOperator") -> VariableAccess:
        return self.r

    def rhs(self: "IrBinaryOperator") -> List[VariableAccess]:
        return [self.a]

    def op(self: "IrBinaryOperator") -> str:
        return "reduce_ggl"

    def __str__(self: "IrBinaryOperator") -> str:
        return "{r} := reduce_ggl({a});".format(
            r=self.r,
            a=self.a)

    def generate_apl(
            self: "IrBinaryOperator",
            register_map: Dict[str, int]) \
            -> Sequence[APL_or_comment]:

        def append_stmt_(apl_, indices, stmt):
            apl_.append(APL_masked_stmt(msk=Mask(indices), stmt=stmt))
            return

        apl = [APL_comment(self)]

        sbs_a = [register_map[str(self.a.var)]]
        sbs_r = [register_map[str(self.r.var)]]

        append_stmt_(apl,
                    self.a.indices,
                    APL_rl_from_sb(sbs=sbs_a, assign_op=''))

        append_stmt_(apl,
                    self.a.indices,
                    APL_src_from_rl(src='GGL', assign_op=''))

        append_stmt_(apl,
                    list(range(16)),
                    APL_sb_from_src(sbs=sbs_r, src='GGL'))

        return apl


@bleir_dataclass
class IrNibbleAndFoldOperator(Operator, Rendition):
    a: VariableAccess
    r: VariableAccess

    def lhs(self: "IrBinaryOperator") -> VariableAccess:
        return self.r

    def rhs(self: "IrBinaryOperator") -> List[VariableAccess]:
        return [self.a]

    def op(self: "IrBinaryOperator") -> str:
        return "reduce_ggl"

    def __str__(self: "IrBinaryOperator") -> str:
        return "{r} := reduce_ggl({a});".format(
            r=self.r,
            a=self.a)

    def generate_apl(
            self: "IrBinaryOperator",
            register_map: Dict[str, int]) \
            -> Sequence[APL_or_comment]:
        def append_stmt_(apl_, indices, stmt):
            apl_.append(APL_masked_stmt(msk=Mask(indices), stmt=stmt))
            return

        apl = [APL_comment(self)]

        sbs_a = [register_map[str(self.a.var)]]
        sbs_r = [register_map[str(self.r.var)]]

        append_stmt_(apl,
                     list(range(16)),  # assert 0xFFFF
                     APL_rl_from_sb(sbs=sbs_a, assign_op=''))

        append_stmt_(apl,
                     [1, 3, 5, 7, 9, 11, 13, 15],  # assert 0xAAAA
                     APL_rl_from_src(src='NRL', assign_op='&'))

        append_stmt_(apl,
                     [1, 5, 9, 13],  # assert 0x2222
                     APL_src_from_rl(src='GGL', assign_op=''))

        append_stmt_(apl,
                     [2, 3, 6, 7, 10, 11, 14, 15],  # assert 0xCCCC
                     APL_rl_from_src(src='GGL', assign_op='&'))

        append_stmt_(apl,
                     list(range(16)),
                     APL_sb_from_src(sbs=sbs_r, src='RL'))

        return apl


@bleir_dataclass
class IrBinaryOperator(Operator, Rendition):
    operator: str
    a: VariableAccess
    b: VariableAccess
    r: VariableAccess

    def lhs(self: "IrBinaryOperator") -> VariableAccess:
        return self.r

    def rhs(self: "IrBinaryOperator") -> List[VariableAccess]:
        return [self.a, self.b]

    def op(self: "IrBinaryOperator") -> str:
        return self.operator

    def __str__(self: "IrBinaryOperator") -> str:
        return "{r} := {a} {op} {b};".format(
            r=self.r,
            a=self.a,
            op=self.operator,
            b=self.b)

    def generate_apl(
            self: "IrBinaryOperator",
            register_map: Dict[str, int]) \
            -> Sequence[APL_or_comment]:

        stride = find_stride(self.a.indices, self.b.indices)
        nsteps = abs(stride)
        src_to_use = 'NRL' if stride > 0 else 'SRL'

        apl = []
        apl.append(APL_comment(self))

        sbs_b = [register_map[str(self.b.var)]]
        rl_from_sb = APL_rl_from_sb(sbs=sbs_b, assign_op='')
        apl.append(APL_masked_stmt(msk=Mask(self.b.indices), stmt=rl_from_sb))

        for _ in range(nsteps):
            mask = Mask.from_hex(0xffff)
            rl_from_src = APL_rl_from_src(src=src_to_use, assign_op='')
            masked_stmt = APL_masked_stmt(msk=mask, stmt=rl_from_src)
            apl.append(masked_stmt)

        sbs_a = [register_map[str(self.a.var)]]
        rl_from_sb = APL_rl_from_sb(sbs=sbs_a, assign_op=self.operator)
        apl.append(APL_masked_stmt(msk=Mask(self.a.indices), stmt=rl_from_sb))

        sbs_r = [register_map[str(self.r.var)]]
        sb_from_src = APL_sb_from_src(sbs=sbs_r, src='RL')
        apl.append(APL_masked_stmt(msk=Mask(self.r.indices), stmt=sb_from_src))

        return apl


@bleir_dataclass
class IrAssignOperator(Operator, Rendition):
    lvalue: Rendition
    rvalue: Rendition
    operator: str = ":="

    def __str__(self: "IrAssignOperator"):
        return f"{self.lvalue} := {self.rvalue}"

    def generate_apl(
            self: "IrAssignOperator",
            register_map: Dict[str, int]) \
            -> Sequence[APL_or_comment]:

        apl = []
        apl.append(APL_comment(self))

        if self.lvalue.var.symbol not in register_map:
            raise RuntimeError(
                f"ERROR: Symbol not allocated to register: "
                f"{self.lvalue.var.symbol}")

        # validate stride
        # we support two cases:
        # 1) len(lhs) = len(rhs) and stride = d, where d is an integer
        #    e.g. a([a,2,3]) = b([4,5,6]), all indices differ by |d|=3
        #
        # 2) 0 < len(lhs) < 16 and len(rhs) = 1
        #    e.g. a([1,2,3]) = b(4), the value b(4) is broadcast to a(1), a(2), a(3)
        #
        #    (mask "0xffff") RL = SB[b]
        #    (mask "4") GL = RL
        #    (mask "1,2,3") SB[a] = GL
        #
        #    loop #1:
        #    t2(4) <= b(4)  (1) <-> t2([4]) <= b([4])
        #    t() <= b(4)    (2) <-> t([0]) <= b([4]) ... t([15]) <= b([4])
        #    for i in range(16):
        #           d(i) = c(i) | t(i)
        #
        #    a(4) <=> a([4]) , and a() <=> a([0,1,2,3,4,5,6,7,8,9,10,11,112,13,14,15])
        #
        # Note: Shall we enable this below?
        #    c([4,7,13]) = a() ^ b() | d() (???)

        nsteps, assignment_type = validate_assignment(self.lvalue.indices, self.rvalue.indices)

        # stride = find_stride(self.lvalue.indices, self.rvalue.indices)
        # nsteps = abs(stride)
        # src_to_use = 'NRL' if stride > 0 else 'SRL'

        # (mask) RL = SB[rhs]
        mask_rhs = Mask(self.rvalue.indices)
        sbs_rhs = [register_map[self.rvalue.var.symbol]]
        rl_from_sb = APL_rl_from_sb(sbs=sbs_rhs, assign_op='')
        masked_stmt = APL_masked_stmt(msk=mask_rhs, stmt=rl_from_sb)
        apl.append(masked_stmt)

        if assignment_type == AssignmentType.GL:
            # (mask_rhs) GL = RL
            src_from_rl = APL_src_from_rl(src='GL', assign_op='')
            masked_stmt = APL_masked_stmt(msk=mask_rhs, stmt=src_from_rl)
            apl.append(masked_stmt)

            # (0xffff) RL = GL
            mask = Mask.from_hex(0xffff)
            rl_from_src = APL_rl_from_src(src='GL', assign_op='')
            masked_stmt = APL_masked_stmt(msk=mask, stmt=rl_from_src)
            apl.append(masked_stmt)

        else:
            assert assignment_type in [AssignmentType.NRL, AssignmentType.SRL]

            src_to_use = assignment_type.name
            for _ in range(nsteps):
                mask = Mask.from_hex(0xffff)
                rl_from_src = APL_rl_from_src(src=src_to_use, assign_op='')
                masked_stmt = APL_masked_stmt(msk=mask, stmt=rl_from_src)
                apl.append(masked_stmt)

        #
        mask = Mask(self.lvalue.indices)
        sbs_lhs = [register_map[self.lvalue.var.symbol]]
        sb_from_src = APL_sb_from_src(sbs=sbs_lhs, src='RL')
        masked_stmt = APL_masked_stmt(msk=mask, stmt=sb_from_src)
        apl.append(masked_stmt)

        return apl

    def lhs(self: "IrAssignOperator") -> VariableAccess:
        return self.lvalue

    def rhs(self: "IrAssignOperator") -> List[VariableAccess]:
        return [self.rvalue]

    def op(self: "IrAssignOperator") -> str:
        return self.operator


def access_overlap(t1: VariableAccess, t2: VariableAccess) -> bool:
    return bool(set(t1.indices) & set(t2.indices))


def get_first_viable(col_temps: Sequence[VariableAccess],
                     access: VariableAccess) -> Optional[VariableAccess]:

    global LOGGER

    for col_temp in col_temps:
        if not access_overlap(col_temp, access):
            LOGGER.debug(f"{access} -> {col_temp}")
            return col_temp

    return None


@bleir_dataclass
class IntermediateRepresentation(Renderer):
    r"""The central class for the intermediate representation.
    Has methods for rendering APL code (apl.py).

    Attributes
    ----------

    intermediate_representation
        foo bar baz quux

    liveness
        foo bar baz quux

    liveness2
        foo bar baz quux

    symbol_table
        foo bar baz quux

    """
    intermediate_code: Sequence[Operator] = field(default_factory=list)
    liveness: Sequence[Sequence[str]] = field(default_factory=list)
    symbol_table: SymbolTable = field(default_factory=SymbolTable)

    def BinOp(self: "IntermediateRepresentation",
              a: VariableAccess,
              b: VariableAccess,
              op: str) -> VariableAccess:
        r"""Foo bar baz quux."""

        global LOGGER

        r = self.symbol_table.get_temp()
        r.ir = self
        LOGGER.debug(f"{type(a)}, {a}, {a.var}, {a.indices}")
        op = IrBinaryOperator(op, a, b, r(a.indices))

        # the following works because all binary
        # operators commute
        if self.intermediate_code \
           and (self.intermediate_code[-1].lhs() == str(b)):
            op.a, op.b = op.b, op.a

        self.intermediate_code.append(op)
        return r(op.a.indices)

    def ReduceGGL(self: "IntermediateRepresentation",
                  a: VariableAccess) -> IrAssignOperator:
        r"""Foo bar baz quux."""

        global LOGGER

        r = self.symbol_table.get_temp()
        r.ir = self
        LOGGER.debug(f"{type(a)}, {a}, {a.var}, {a.indices}")
        op = IrReduceGGLOperator(a, r())

        self.intermediate_code.append(op)

        return r()

    def NibbleAndFold(self: "IntermediateRepresentation",
                 a: VariableAccess) -> IrAssignOperator:
        r"""Foo bar baz quux."""

        global LOGGER

        r = self.symbol_table.get_temp()
        r.ir = self
        LOGGER.debug(f"{type(a)}, {a}, {a.var}, {a.indices}")
        op = IrNibbleAndFoldOperator(a, r())

        self.intermediate_code.append(op)

        return r()

    def AssignOp(self: "IntermediateRepresentation",
                 lhs: VariableAccess,
                 rhs: VariableAccess) -> IrAssignOperator:
        r"""Foo bar baz quux."""

        op = IrAssignOperator(lhs, rhs)
        self.intermediate_code.append(op)
        return op

    def perform_liveness_analysis2(self: "IntermediateRepresentation",
                                  return_values: Sequence[Variable]) -> None:
        r"""Foo bar baz quux."""

        global LOGGER
        hex_index_mask = lambda indices : Mask(indices).mask

        alive = [{str(var):0xffff for var in return_values}]
        LOGGER.debug(f"alive: {alive}")

        for inst in reversed(self.intermediate_code):
            curr_alive = alive[-1].copy()

            lhs_var = str(inst.lhs().var)

            if lhs_var in curr_alive:
                curr_alive[lhs_var] &= ~hex_index_mask(inst.lhs().indices)
                if curr_alive[lhs_var] == 0x0:
                    curr_alive.pop(lhs_var)

            for rhs in inst.rhs():
                rhs_var = str(rhs.var)
                rhs_indices = rhs.indices

                if not rhs_var in curr_alive:
                    curr_alive[rhs_var] = hex_index_mask(rhs_indices)
                else:
                    curr_alive[rhs_var] |= hex_index_mask(rhs_indices)

            alive.append(curr_alive)

        self.liveness2 = [x for x in reversed(alive)]

    def perform_liveness_analysis(self: "IntermediateRepresentation",
                                  return_values: Sequence[Variable]) -> None:
        r"""Foo bar baz quux."""

        global LOGGER

        # this version of liveness analysis operates over string representation
        # of the VariableAccess objects
        alive = [{str(var) for var in return_values}]
        LOGGER.debug(f"alive: {alive}")
        for inst in reversed(self.intermediate_code):
            curr_alive = alive[-1].copy()

            lhs_var = str(inst.lhs().var)
            rhs_vars = [str(x.var) for x in inst.rhs()]

            if lhs_var in curr_alive:
                curr_alive.remove(lhs_var)

            curr_alive |= set(rhs_vars)
            alive.append(curr_alive)

        self.liveness = [tuple(x) for x in reversed(alive)]

    def coalesce_temps(self: "IntermediateRepresentation") -> None:
        r"""Foo bar baz quux."""

        aliases = {}
        coalesced_temps = []

        # iterate over IR once to
        # * find which temps can be merged
        # * create map to merged temps
        for instr in self.intermediate_code:
            if instr.lhs().var.is_tmp():
                # find first viable
                t = get_first_viable(coalesced_temps, instr.lhs())
                if not t:
                    coalesced_temps.append(instr.lhs().clone())
                else:
                    for i in instr.lhs().indices:
                        t.indices.append(i)
                    aliases[instr.lhs().var] = t.var

        for instr in self.intermediate_code:
            if instr.lhs().var in aliases:
                instr.lhs().var = aliases[instr.lhs().var]
            for op in instr.rhs():
                if op.var in aliases:
                    op.var = aliases[op.var]

    def REDUCE_GGL(self: "IntermediateRepresentation",
            variable: VariableAccess)-> VariableAccess:
        r"""Foo bar baz quux."""
        return self.ReduceGGL(variable)

    def NIBBLE_AND_FOLD(self: "IntermediateRepresentation",
            variable: VariableAccess)-> VariableAccess:
        r"""Foo bar baz quux."""
        return self.NibbleAndFold(variable)

    def AND(self: "IntermediateRepresentation",
            a: VariableAccess,
            b: VariableAccess) -> VariableAccess:
        r"""Foo bar baz quux."""
        return self.BinOp(a, b, "&")

    def OR(self: "IntermediateRepresentation",
           a: VariableAccess,
           b: VariableAccess) -> VariableAccess:
        r"""Foo bar baz quux."""
        return self.BinOp(a, b, "|")

    def XOR(self: "IntermediateRepresentation",
            a: VariableAccess,
            b: VariableAccess) -> VariableAccess:
        r"""Foo bar baz quux."""
        return self.BinOp(a, b, "^")

    def ASSIGN(self: "IntermediateRepresentation",
               a: VariableAccess,
               b: VariableAccess) -> Rendition:
        r"""Foo bar baz quux."""
        return self.AssignOp(a, b)
