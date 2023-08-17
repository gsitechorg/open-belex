r"""
By Dylon Edwards
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import wraps
from hashlib import shake_128
from heapq import heapify, heappop, heappush
from itertools import islice
from typing import (Callable, ClassVar, Deque, Dict, Iterator, NamedTuple,
                    Optional, Sequence, Set, Tuple, Type, Union)

import numpy as np

from open_belex.bleir.analyzers import (MAX_FRAGMENT_INSTRUCTIONS,
                                        LiveSectionScanner,
                                        NumFragmentInstructionsAnalyzer,
                                        RegisterGrouper,
                                        RegisterParameterFinder,
                                        RegisterScanner, UserParameterScanner)
from open_belex.bleir.syntactic_validators import MAX_C_ID_LEN
from open_belex.bleir.types import (BROADCAST, BROADCAST_EXPR, EWE_REG,
                                    GGL_ASSIGNMENT, L1_REG, L2_REG,
                                    LGL_ASSIGNMENT, LX_ASSIGNMENT, MASK,
                                    MASKED, NOOP, RE_REG, READ, RN_REG,
                                    SHIFTED_SM_REG, SM_REG, SPECIAL, SRC_EXPR,
                                    STATEMENT, UNARY_OP, UNARY_SRC, WRITE,
                                    ActualParameter, FormalParameter, Fragment,
                                    FragmentCaller, FragmentCallerCall,
                                    FragmentMetadata, GlassStatement,
                                    LineComment, LXRegWithOffsets,
                                    MultiLineComment, MultiStatement,
                                    Operation, Operation_or_LineComment,
                                    ReadWriteInhibit, RegisterParameter,
                                    SingleLineComment, Snippet,
                                    STATEMENT_or_LineComment, TrailingComment,
                                    ValueParameter, statement)
from open_belex.bleir.walkables import (BLEIRListener, BLEIRTransformer,
                                        BLEIRWalker)
from open_belex.common.constants import NSB, NSECTIONS
from open_belex.common.register_arenas import NUM_RN_REGS
from open_belex.kernel_libs.common import cpy_16, cpy_imm_16, cpy_vr
from open_belex.kernel_libs.memory import (NUM_VM_REGS, copy_l1_to_l2_byte,
                                           copy_l2_to_l1_byte, l2_end,
                                           load_16_t0, restore_ggl, restore_rl,
                                           restore_vr, spill_ggl, spill_rl,
                                           spill_vr, store_16_t0, vmr_to_row)
from open_belex.literal import (SM_0X000F, SM_0X0F0F, SM_0X00FF, SM_0X0001,
                                SM_0X001F, SM_0X003F, SM_0X0101, SM_0X0707,
                                SM_0X1111, SM_0X3333, SM_0X5555, SM_0XFFFF,
                                flatten, inline_bleir)
from open_belex.utils.section_utils import parse_sections

LOGGER = logging.getLogger()

L1_REGS_PER_RN_REG = 4

SRC_SET = set(map(str, SRC_EXPR))


class RewriteSingletonMultiStatement(BLEIRListener, BLEIRTransformer):
    r"""Wraps all individual STATEMENTs within a singleton MultiStatement (for safety). Existing
    MultiStatements are left alone."""

    in_multi_statement: bool = False

    def enter_multi_statement(self: "RewriteSingletonMultiStatement",
                              multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True

    def exit_multi_statement(self: "RewriteSingletonMultiStatement",
                             multi_statement: MultiStatement) -> None:
        self.in_multi_statement = False

    def transform_statement(self: "RewriteSingletonMultiStatement",
                            statement: STATEMENT) -> MultiStatement:

        if self.in_multi_statement:
            return statement

        return MultiStatement(statements=[statement])


Parameter_and_Value = Tuple[RegisterParameter, int]


@dataclass
class EnsureNoEmptyFragBodies(BLEIRTransformer):
    num_fragment_instructions_analyzer: NumFragmentInstructionsAnalyzer

    def transform_fragment(self: "EnsureNoEmptyFragBodies",
                           fragment: Fragment) -> Fragment:

        if fragment.children is not None and len(fragment.children) > 0:
            return fragment

        num_instructions = \
            self.num_fragment_instructions_analyzer \
                .num_instructions_by_fragment[fragment.identifier]

        if num_instructions > 0:
            return fragment

        operations = list(fragment.operations)  # in case they are all comments
        operations.append(statement(NOOP))
        return fragment.having(operations=operations)


# TODO: Determine whether we need to pause after operating on RSP16 and GGL
# NOTE: At least two clock cycles must follow operations on RSP16 before it
# may be operated on again, whether those are NOOPs or otherwise.
EXPRESSIONS_REQUIRING_PAUSE: Set[Union[SRC_EXPR, BROADCAST_EXPR]] = set([

    SRC_EXPR.GL,
    # SRC_EXPR.GGL,
    # SRC_EXPR.RSP16,

    SRC_EXPR.INV_GL,
    # SRC_EXPR.INV_GGL,
    # SRC_EXPR.INV_RSP16,

    BROADCAST_EXPR.GL,
    # BROADCAST_EXPR.GGL,
    # BROADCAST_EXPR.RSP16,
])


@dataclass
class InjectMissingNOOPs(BLEIRTransformer, BLEIRListener):
    inject_noop: bool = False
    is_pause: bool = False
    prev_inject_noop: bool = False
    prev_operation: Optional[Operation] = None
    operations_requiring_noops: Optional[Set[Operation]] = None
    in_multi_statement: bool = False

    def enter_fragment(self: "InjectMissingNOOPs", fragment: Fragment) -> None:
        self.prev_inject_noop = False
        self.operations_requiring_noops = set()

    def transform_fragment(self: "InjectMissingNOOPs", fragment: Fragment) -> Fragment:
        operations = []

        for operation in fragment.operations:
            operations.append(operation)
            if operation in self.operations_requiring_noops:
                operations.append(statement(NOOP))

        if self.prev_inject_noop:
            operations.append(statement(NOOP))

        return fragment.having(operations=operations)

    def enter_multi_statement(self: "InjectMissingNOOPs", multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True
        self.inject_noop = False
        self.is_pause = False

    def exit_multi_statement(self: "InjectMissingNOOPs", multi_statement: MultiStatement) -> None:
        self.in_multi_statement = False
        if self.prev_inject_noop and not self.is_pause:
            self.inject_noop = False
            self.operations_requiring_noops.add(self.prev_operation)
        self.prev_operation = multi_statement
        self.prev_inject_noop = self.inject_noop

    def enter_statement(self: "InjectMissingNOOPs", statement: STATEMENT) -> None:
        if not self.in_multi_statement:
            self.inject_noop = False
            self.is_pause = False

    def exit_statement(self: "InjectMissingNOOPs", statement: STATEMENT) -> None:
        if not self.in_multi_statement:
            if self.prev_inject_noop and not self.is_pause:
                self.inject_noop = False
                self.operations_requiring_noops.add(self.prev_operation)
            self.prev_operation = statement
            self.prev_inject_noop = self.inject_noop

    def enter_unary_src(self: "InjectMissingNOOPs", unary_src: UNARY_SRC) -> None:
        self.inject_noop = unary_src.expression in EXPRESSIONS_REQUIRING_PAUSE

    def enter_broadcast(self: "InjectMissingNOOPs", broadcast: BROADCAST) -> None:
        self.inject_noop = broadcast.lvalue in EXPRESSIONS_REQUIRING_PAUSE

    def enter_special(self: "InjectMissingNOOPs", special: SPECIAL) -> None:
        if special is SPECIAL.NOOP:
            self.inject_noop = False
            self.is_pause = True


@dataclass
class PartitionFragmentsIntoDigestibleChunks(BLEIRTransformer):
    num_fragment_instructions_analyzer: NumFragmentInstructionsAnalyzer

    def transform_fragment(
            self: "PartitionFragmentsIntoDigestibleChunks",
            fragment: Fragment) -> Fragment:

        num_instructions = \
            self.num_fragment_instructions_analyzer \
                .num_instructions_by_fragment[fragment.identifier]

        if num_instructions <= MAX_FRAGMENT_INSTRUCTIONS:
            return fragment

        if fragment.has_metadata():
            metadata = dict(fragment.metadata)
        else:
            metadata = {}

        metadata[FragmentMetadata.ORIGINAL_IDENTIFIER] = fragment.identifier

        children = []
        for index, partition in enumerate(self.partition(fragment.operations)):
            frag_nym = f"__{fragment.identifier}__{index}"
            child = fragment.having(
                identifier=frag_nym,
                operations=partition,
                metadata=metadata)
            children.append(child)

        return fragment.having(operations=[], children=children)

    @staticmethod
    def is_instruction(operation_or_comment: Operation_or_LineComment) -> bool:
        return isinstance(operation_or_comment, MultiStatement) \
            or isinstance(operation_or_comment, STATEMENT) \
            and not isinstance(operation_or_comment.operation,
                               GlassStatement)

    @classmethod
    def is_comment(cls: Type["PartitionFragmentsIntoDigestibleChunks"],
                   operation_or_comment: Operation_or_LineComment) -> bool:
        return not cls.is_instruction(operation_or_comment)

    @classmethod
    def partition(cls: Type["PartitionFragmentsIntoDigestibleChunks"],
                  operations: Sequence[Operation]) \
            -> Iterator[Sequence[Operation]]:
        """Partitions the operations of a fragment into sub-sequences of <= 255
        operations. Comments do not count as operations for the purpose of
        partitioning."""

        start = 0
        stop = len(operations)
        step = MAX_FRAGMENT_INSTRUCTIONS

        partition_size = -1
        partition = None

        cursor = start
        while cursor < stop:
            partition_size = 0
            partition = []
            while cursor < stop:
                operation_or_comment = operations[cursor]
                cursor += 1

                partition.append(operation_or_comment)

                if cls.is_instruction(operation_or_comment):
                    partition_size += 1

                # If the rest of the lines are comments, add them to the
                # current partition
                if all(map(cls.is_comment, islice(operations, cursor, stop))):
                    partition.extend(islice(operations, cursor, stop))
                    cursor = stop

                if partition_size == step:
                    yield partition
                    break

        if 0 < len(partition) < step:
            yield partition


@dataclass
class LongNymObfuscator(BLEIRTransformer, BLEIRListener):

    def transform_fragment(self: "LongNymObfuscator",
                           fragment: Fragment) -> Fragment:
        max_len = MAX_C_ID_LEN - len("_caller")
        if len(fragment.identifier) > max_len:
            prefix_len = max_len - 11
            nym_prefix = fragment.identifier[:prefix_len]
            nym_suffix = \
                shake_128(fragment.identifier.encode()).hexdigest(5)
            comment_lines = (f"Original nym: {fragment.identifier}",)
            if fragment.doc_comment is not None:
                comment_lines = fragment.doc_comment.lines + comment_lines
            if fragment.has_metadata():
                metadata = dict(fragment.metadata)
            else:
                metadata = {}
            if FragmentMetadata.ORIGINAL_IDENTIFIER not in metadata:
                metadata[FragmentMetadata.ORIGINAL_IDENTIFIER] = \
                    fragment.identifier
            return fragment.having(
                identifier=f"{nym_prefix}_{nym_suffix}",
                doc_comment=MultiLineComment(lines=comment_lines),
                metadata=metadata)
        return fragment

    def transform_rn_reg(self: "LongNymObfuscator",
                         rn_reg: RN_REG) -> RN_REG:
        # max_len = MAX_C_ID_LEN - len("_rp")
        max_len = MAX_C_ID_LEN
        if len(rn_reg.identifier) > max_len:
            prefix_len = max_len - 11
            nym_prefix = rn_reg.identifier[:prefix_len]
            nym_suffix = \
                shake_128(rn_reg.identifier.encode()).hexdigest(5)
            return rn_reg.having(identifier=f"{nym_prefix}_{nym_suffix}")
        return rn_reg

    def transform_sm_reg(self: "LongNymObfuscator",
                         sm_reg: SM_REG) -> SM_REG:
        # max_len = MAX_C_ID_LEN - len("_rp")
        max_len = MAX_C_ID_LEN
        if len(sm_reg.identifier) > max_len:
            prefix_len = max_len - 11
            nym_prefix = sm_reg.identifier[:prefix_len]
            nym_suffix = \
                shake_128(sm_reg.identifier.encode()).hexdigest(5)
            return sm_reg.having(identifier=f"{nym_prefix}_{nym_suffix}")
        return sm_reg

    def transform_value_parameter(
            self: "LongNymObfuscator",
            value_parameter: ValueParameter) -> ValueParameter:
        max_len = MAX_C_ID_LEN - len("")
        if len(value_parameter.identifier) > max_len:
            prefix_len = max_len - 11
            nym_prefix = value_parameter.identifier[:prefix_len]
            nym_suffix = \
                shake_128(value_parameter.identifier.encode()).hexdigest(5)
            return value_parameter.having(identifier=f"{nym_prefix}_{nym_suffix}")
        return value_parameter


@dataclass
class L1RegNormalizer(BLEIRTransformer):
    minimized_l1_regs_by_id: Dict[str, L1_REG]

    # Non-standard transform
    def transform_l1_reg(self: "L1RegNormalizer",
                         l1_reg: L1_REG) -> Union[L1_REG, LXRegWithOffsets]:

        if l1_reg.identifier in self.minimized_l1_regs_by_id:
            min_l1_reg = self.minimized_l1_regs_by_id[l1_reg.identifier]
            bank_offset = l1_reg.bank_id - min_l1_reg.bank_id
            group_offset = l1_reg.group_id - min_l1_reg.group_id
            row_offset = l1_reg.row_id - min_l1_reg.row_id
            return min_l1_reg + (bank_offset, group_offset, row_offset)

        return l1_reg


@dataclass
class SpillRestoreScheduler(BLEIRListener, BLEIRTransformer):

    register_scanner: RegisterScanner

    register_grouper: RegisterGrouper

    max_rn_regs: int = NUM_RN_REGS

    # Example: reservations = {"sm_regs": set(range(4, 15 + 1))}
    reservations: Dict[str, Set[int]] = field(default_factory=dict)

    frag_ops: Optional[Deque[Operation]] = None

    active_regs: Optional[Set[RN_REG]] = None

    frag_regs: Optional[Set[RN_REG]] = None

    user_regs: Optional[Set[RN_REG]] = None

    temp_regs: Optional[Set[RN_REG]] = None

    rn_regs: Optional[Deque[RN_REG]] = None

    reg_idx: int = -1

    in_multi_statement: bool = False

    in_statement: bool = False

    stmt_regs: Optional[Set[RN_REG]] = None

    l1_regs_by_rn_reg: Optional[Dict[RN_REG, L1_REG]] = None

    spilled_rn_regs: Optional[Dict[RN_REG, L1_REG]] = None

    groups_by_reg: Optional[Dict[RN_REG, Sequence[RN_REG]]] = None

    active_reg_by_group: Optional[Dict[Sequence[RN_REG], RN_REG]] = None

    l1_regs_by_rn_reg_by_frag: Dict[str, Dict[RN_REG, L1_REG]] = \
        field(default_factory=dict)

    user_regs_by_frag: Dict[str, Set[RN_REG]] = field(default_factory=dict)

    initial_active_regs_by_frag: Dict[str, Set[RN_REG]] = \
        field(default_factory=dict)

    final_active_regs_by_frag: Dict[str, Set[RN_REG]] = \
        field(default_factory=dict)

    initial_spilled_rn_regs_by_frag: Dict[str, Dict[RN_REG, L1_REG]] = \
        field(default_factory=dict)

    final_spilled_rn_regs_by_frag: Dict[str, Dict[RN_REG, L1_REG]] = \
        field(default_factory=dict)

    available_l1_regs: Optional[Sequence[Tuple[int, L1_REG]]] = None

    l1_reg_for_ggl: Optional[L1_REG] = None

    def reset_available_l1_regs(self: "SpillRestoreScheduler") -> None:
        self.available_l1_regs = []

        if "l1_rows" in self.reservations:
            l1_reservations = self.reservations["l1_rows"]

            l1_rows_by_group_size = defaultdict(list)

            vmr = -2
            while vmr < NUM_VM_REGS - 2:
                vmr += 2
                group = []

                l1_row = vmr_to_row(vmr)
                l1_range = set(range(l1_row, l1_row + 4))
                if len(l1_range & l1_reservations) == 0:
                    group.append(l1_row)

                l1_row += 4
                l1_range = set(range(l1_row, l1_row + 4))
                if len(l1_range & l1_reservations) == 0:
                    group.append(l1_row)

                l1_rows_by_group_size[len(group)].append(group)

                l1_row += 4
                if l1_row not in l1_reservations \
                   and self.l1_reg_for_ggl is None:
                    l1_reg = L1_REG(
                        identifier=f"_INTERNAL_L1_{vmr // 2}",
                        bank_group_row=l1_row)
                    self.l1_reg_for_ggl = l1_reg

                # If we fill 4 groups of size=2 of L1 rows and assign an L1_REG
                # to GGL, we've finished.
                if len(l1_rows_by_group_size[2]) == 4 \
                   and self.l1_reg_for_ggl is not None:
                    break

            groups = l1_rows_by_group_size[2]
            if len(groups) == 4:
                # 4 groups of size=2 == 4 L1_REGs (full)
                pass
            elif len(l1_rows_by_group_size[1]) >= 4 - len(groups):
                # There are only 4 L1_REGs (L1_REG_0 through L1_REG_3), each
                # group of size=2 will occupy one L1_REG, and each group of
                # size=1 will occupy one L1_REG, so we need to occupy (4 -
                # len(groups)) L1_REGs with groups of size=1.
                groups = groups + l1_rows_by_group_size[1][:4 - len(groups)]
            else:
                # There aren't enough groups to occupy all L1_REGs so occupy as
                # many as possible.
                groups = groups + l1_rows_by_group_size[1]

            for vmr, l1_row in enumerate(flatten(groups)):
                l1_reg = L1_REG(
                    identifier=f"_INTERNAL_L1_{vmr // 2}",
                    bank_group_row=l1_row)
                self.available_l1_regs.append((l1_row, l1_reg))

        else:  # No reservations, use defaults ...
            l1_reg = L1_REG(
                identifier="_INTERNAL_L1_0",
                bank_group_row=8)
            self.l1_reg_for_ggl = l1_reg

            for vmr in range(8):
                l1_row = vmr_to_row(vmr)
                l1_reg = L1_REG(
                    identifier=f"_INTERNAL_L1_{vmr // 2}",
                    bank_group_row=l1_row)
                self.available_l1_regs.append((l1_row, l1_reg))

        heapify(self.available_l1_regs)

    @property
    def is_spillable(self: "SpillRestoreScheduler") -> bool:
        return self.reg_idx >= 0

    def enter_fragment(self: "SpillRestoreScheduler",
                       fragment: Fragment) -> None:

        if fragment.children is not None:
            return

        frag_id = fragment.original_identifier

        if frag_id not in self.register_scanner.regs_by_frag:
            return

        rn_regs = self.register_scanner.regs_by_frag[frag_id]

        frag_regs = set(rn_regs)
        if len(frag_regs) > self.max_rn_regs:
            # Only 8 params are supported (by hardware) for auto spilling/restoring
            assert len(frag_regs) - self.max_rn_regs <= 8

            self.reset_available_l1_regs()

            self.l1_regs_by_rn_reg = {}
            # for vmr, rn_reg in enumerate(frag_regs):
            #     l1_reg = L1_REG(
            #         identifier=f"_INTERNAL_L1_{vmr // 2}",
            #         bank_group_row=vmr_to_row(vmr))
            #     self.l1_regs_by_rn_reg[rn_reg] = l1_reg

            self.l1_regs_by_rn_reg_by_frag[frag_id] = self.l1_regs_by_rn_reg

            self.frag_ops = deque()

            self.frag_regs = frag_regs

            self.temp_regs = {reg for reg in frag_regs
                              if reg.is_temporary
                              or reg.identifier.startswith("_INTERNAL")}

            self.user_regs = frag_regs - self.temp_regs
            self.user_regs_by_frag[frag_id] = self.user_regs

            groups = self.register_grouper.groups_by_frag[frag_id]

            groups_by_reg = {}
            for group in groups:
                for reg in group:
                    groups_by_reg[reg] = group
            self.groups_by_reg = groups_by_reg

            active_reg_by_group = {}
            first_active_by_group_reg = {}
            for reg in rn_regs:
                if reg not in first_active_by_group_reg:
                    active_reg = reg
                    group = groups_by_reg[active_reg]
                    active_reg_by_group[group] = active_reg
                    first_active = active_reg
                    for group_reg in group:
                        first_active_by_group_reg[group_reg] = first_active
            self.active_reg_by_group = active_reg_by_group

            self.active_regs = set(first_active_by_group_reg.values())
            self.initial_active_regs_by_frag[frag_id] = set(self.active_regs)

            self.spilled_rn_regs = {}

            for user_reg in (self.user_regs - self.active_regs):
                _, l1_reg_to_spill = heappop(self.available_l1_regs)
                self.spilled_rn_regs[user_reg] = l1_reg_to_spill

            for temp_reg in (self.temp_regs - self.active_regs):
                _, l1_reg_to_spill = heappop(self.available_l1_regs)
                self.spilled_rn_regs[temp_reg] = l1_reg_to_spill

            self.initial_spilled_rn_regs_by_frag[frag_id] = \
                dict(self.spilled_rn_regs)

            self.rn_regs = rn_regs
            self.reg_idx = 0

            self.frag_id = frag_id

    def exit_fragment(self: "SpillRestoreScheduler",
                      fragment: Fragment) -> None:

        if fragment.children is not None or not self.is_spillable:
            return

        frag_id = fragment.original_identifier

        self.l1_regs_by_rn_reg = None

        self.frag_regs = None
        self.user_regs = None
        self.temp_regs = None

        self.groups_by_reg = None
        self.active_reg_by_group = None

        self.final_active_regs_by_frag[frag_id] = self.active_regs
        self.active_regs = None

        self.final_spilled_rn_regs_by_frag[frag_id] = \
            self.spilled_rn_regs
        self.spilled_rn_regs = None

        self.rn_regs = None
        self.reg_idx = -1

    def transform_fragment(self: "SpillRestoreScheduler",
                           fragment: Fragment) -> Fragment:

        if not self.is_spillable:
            return fragment

        register_parameter_finder = RegisterParameterFinder()
        walker = BLEIRWalker()

        fragment = fragment.having(operations=tuple(self.frag_ops))
        walker.walk(register_parameter_finder, fragment)

        # "identifier" -> (bank, group, row)
        l1_regs_by_id: Dict[str, Deque[L1_REG]] = defaultdict(deque)

        for parameter in register_parameter_finder.register_parameters:
            if isinstance(parameter, L1_REG):
                if parameter.bank_group_row is not None:
                    l1_regs_by_id[parameter.identifier].append(parameter)

        # "identifier" -> (min_bank, min_group, min_row)
        minimized_l1_regs_by_id = {}
        for identifier, l1_regs in l1_regs_by_id.items():
            l1_reg = l1_regs.popleft()
            min_bank, min_group, min_row = \
                (l1_reg.bank_id, l1_reg.group_id, l1_reg.row_id)
            for l1_reg in l1_regs:
                bank, group, row = (l1_reg.bank_id, l1_reg.group_id, l1_reg.row_id)
                min_bank = min(min_bank, bank)
                min_group = min(min_group, group)
                min_row = min(min_row, row)
            minimized_l1_regs_by_id[identifier] = l1_reg.having(
                bank_group_row=(min_bank, min_group, min_row))

        l1_reg_normalizer = L1RegNormalizer(minimized_l1_regs_by_id)
        fragment = walker.walk(l1_reg_normalizer, fragment)

        register_parameter_finder = RegisterParameterFinder()
        walker.walk(register_parameter_finder, fragment)

        parameters = list(fragment.parameters)
        for parameter in register_parameter_finder.register_parameters:
            if parameter not in parameters:
                parameters.append(parameter)

        return fragment.having(parameters=tuple(parameters))

    def exit_single_line_comment(
            self: "SpillRestoreScheduler",
            single_line_comment: SingleLineComment) -> None:
        if self.is_spillable:
            self.frag_ops.append(single_line_comment)

    def exit_multi_line_comment(
            self: "SpillRestoreScheduler",
            multi_line_comment: MultiLineComment) -> None:
        if self.is_spillable:
            self.frag_ops.append(multi_line_comment)

    def enter_multi_statement(self: "SpillRestoreScheduler",
                              multi_statement: MultiStatement) -> None:
        if not self.is_spillable:
            return
        self.stmt_regs = self.register_scanner.regs_by_stmt[multi_statement]
        self.in_multi_statement = True

    def exit_multi_statement(self: "SpillRestoreScheduler",
                             multi_statement: MultiStatement) -> None:
        if self.is_spillable:
            self.frag_ops.append(multi_statement)
        self.stmt_regs = None
        self.in_multi_statement = False

    def enter_statement(self: "SpillRestoreScheduler",
                        statement: STATEMENT) -> None:
        if self.is_spillable:
            self.in_statement = True

    def exit_statement(self: "SpillRestoreScheduler",
                       statement: STATEMENT) -> None:
        if not self.is_spillable:
            return
        if not self.in_multi_statement:
            self.frag_ops.append(statement)
        self.in_statement = False

    def find_reg_to_spill(self: "SpillRestoreScheduler",
                          reg_to_restore: RN_REG) -> RN_REG:

        # available_regs = set(self.active_regs)

        # if self.in_multi_statement:
        #     # Do not spill regs in the current multi-statement
        #     available_regs -= self.stmt_regs

        # assert len(available_regs) > 0
        # reg_to_spill = None

        # visited = set()
        # for reg_idx in range(self.reg_idx + 1, len(self.rn_regs)):
        #     rn_reg = self.rn_regs[reg_idx]
        #     if rn_reg in available_regs and rn_reg not in visited:
        #         reg_to_spill = rn_reg
        #         visited.add(rn_reg)

        # if reg_to_spill is None:
        #     reg_to_spill = next(iter(available_regs))

        group = self.groups_by_reg[reg_to_restore]
        active_reg = self.active_reg_by_group[group]
        reg_to_spill = active_reg
        return reg_to_spill

    def swap(self: "SpillRestoreScheduler",
             rn_reg_to_spill: RN_REG,
             rn_reg_to_restore: RN_REG) -> None:

        _, l1_reg_to_spill = heappop(self.available_l1_regs)
        l1_reg_to_restore = self.spilled_rn_regs[rn_reg_to_restore]

        l1_reg_for_ggl = self.l1_reg_for_ggl
        _, l1_reg_for_rl = heappop(self.available_l1_regs)

        # self.frag_ops.append(MultiLineComment(lines=[
        #     f"Spilling {rn_reg_to_spill.identifier} to restore "
        #     f"{rn_reg_to_restore.identifier}."
        # ]))
        with inline_bleir():
            # self.frag_ops.append(SingleLineComment(
            #     line=f"Spilling GGL to L1={l1_reg_for_ggl.bank_group_row}"))
            self.frag_ops.extend(spill_ggl(l1_reg_for_ggl, debug=False))
        with inline_bleir():
            # self.frag_ops.append(SingleLineComment(
            #     line=f"Spilling RL to L1=[{l1_reg_for_rl.bank_group_row},"
            #          f"{4 + l1_reg_for_rl.bank_group_row})"))
            self.frag_ops.extend(spill_rl(l1_reg_for_rl, debug=False))
        with inline_bleir():
            # self.frag_ops.append(SingleLineComment(
            #     line=f"Spilling {rn_reg_to_spill.identifier} "
            #          f"to L1=[{l1_reg_to_spill.bank_group_row},"
            #          f"{4 + l1_reg_to_spill.bank_group_row})"))
            self.frag_ops.extend(spill_vr(l1_reg_to_spill, rn_reg_to_spill, debug=False))
        with inline_bleir():
            # self.frag_ops.append(SingleLineComment(
            #     line=f"Restoring RL from L1=[{l1_reg_for_rl.bank_group_row},"
            #          f"{4 + l1_reg_for_rl.bank_group_row})"))
            self.frag_ops.extend(restore_rl(l1_reg_for_rl, debug=False))
        with inline_bleir():
            # self.frag_ops.append(SingleLineComment(
            #     line=f"Restoring {rn_reg_to_restore.identifier} "
            #          f"from L1=[{l1_reg_to_restore.bank_group_row},"
            #          f"{4 + l1_reg_to_restore.bank_group_row})"))
            self.frag_ops.extend(restore_vr(rn_reg_to_restore, l1_reg_to_restore, debug=False))
        with inline_bleir():
            # self.frag_ops.append(SingleLineComment(
            #     line=f"Restoring GGL from L1={l1_reg_for_ggl.bank_group_row}"))
            self.frag_ops.extend(restore_ggl(l1_reg_for_ggl, debug=False))

        # self.frag_ops.append(SingleLineComment(
        #     line=f"Finished restoring {rn_reg_to_restore.identifier} "
        #          f"from L1=[{l1_reg_to_restore.bank_group_row},"
        #          f"{4 + l1_reg_to_restore.bank_group_row})."))

        heappush(self.available_l1_regs, (l1_reg_to_restore.bank_group_row,
                                          l1_reg_to_restore))
        heappush(self.available_l1_regs, (l1_reg_for_rl.bank_group_row,
                                          l1_reg_for_rl))

        self.spilled_rn_regs[rn_reg_to_spill] = l1_reg_to_spill
        self.active_regs.remove(rn_reg_to_spill)

        del self.spilled_rn_regs[rn_reg_to_restore]
        self.active_regs.add(rn_reg_to_restore)

        group = self.groups_by_reg[rn_reg_to_restore]
        self.active_reg_by_group[group] = rn_reg_to_restore

    def enter_rn_reg(self: "SpillRestoreScheduler", rn_reg: RN_REG) -> None:
        if self.is_spillable and self.in_statement:
            if len(self.active_regs) < self.max_rn_regs:
                self.active_regs.add(rn_reg)

            elif rn_reg not in self.active_regs:
                rn_reg_to_spill = self.find_reg_to_spill(rn_reg)
                rn_reg_to_restore = rn_reg
                self.swap(rn_reg_to_spill, rn_reg_to_restore)

            self.reg_idx += 1


@dataclass
class InjectKernelLibs(BLEIRTransformer):
    KERNEL_CALLERS: ClassVar[Set[FragmentCaller]] = {
        belex_caller.__caller__
        for belex_caller in [
            copy_l1_to_l2_byte,
            copy_l2_to_l1_byte,
            cpy_16,
            cpy_imm_16,
            cpy_vr,
            l2_end,
            load_16_t0,
            restore_vr,
            spill_vr,
            store_16_t0,
        ]
    }

    @staticmethod
    def build_callers(callers: Set[FragmentCaller]) -> Set[FragmentCaller]:
        built_callers = set()
        for caller in callers:
            if not isinstance(caller, FragmentCaller) \
               and callable(caller):
                caller = caller()
            if isinstance(caller, FragmentCaller):
                built_callers.add(caller)
            else:
                raise ValueError(
                    f"Unsupported caller type ({caller.__class__.__name__}): "
                    f"{caller}")
        return built_callers

    def transform_snippet(self: "InjectKernelLibs",
                          snippet: Snippet) -> Snippet:
        library_callers = set(self.KERNEL_CALLERS)
        if snippet.library_callers is not None:
            library_callers |= set(snippet.library_callers)
        library_callers = self.build_callers(library_callers)
        fragment_caller_ids = {call.caller.identifier
                               for call in snippet.fragment_caller_calls}
        for library_caller in list(library_callers):
            if library_caller.identifier in fragment_caller_ids:
                library_callers.remove(library_caller)
        library_callers = tuple(library_callers)
        return snippet.having(library_callers=library_callers)


@dataclass
class AllocateTemporaries(BLEIRListener, BLEIRTransformer):
    user_parameter_scanner: UserParameterScanner

    # Example: reservations = {"sm_regs": set(range(4, 15 + 1))}
    reservations: Dict[str, Set[int]] = field(default_factory=dict)

    row_numbers_by_rn_reg_by_frag: Dict[str, Dict[str, int]] \
        = field(default_factory=dict)

    _available_row_numbers: Optional[Deque[int]] = None
    row_numbers_by_rn_reg: Optional[Dict[str, int]] = None
    row_masks_by_re_reg: Optional[Dict[str, int]] = None
    temporaries: Optional[Set[str]] = None

    _reserved_row_numbers: Optional[Set[int]] = None

    @property
    def reserved_row_numbers(self: "AllocateTemporaries") -> Set[int]:
        if self._reserved_row_numbers is None:
            self._reserved_row_numbers = set()

            if "row_numbers" in self.reservations:
                self._reserved_row_numbers.update(
                    self.reservations["row_numbers"])

            self._reserved_row_numbers.update(
                self.user_parameter_scanner.parameters_by_type[RN_REG])

        return self._reserved_row_numbers

    @property
    def available_row_numbers(self: "AllocateTemporaries") -> Deque[int]:
        if self._available_row_numbers is None:
            available_row_numbers = set(range(NSB))
            available_row_numbers -= self.reserved_row_numbers
            self._available_row_numbers = deque(sorted(available_row_numbers))
        return self._available_row_numbers

    def enter_fragment_caller(self: "AllocateTemporaries",
                              fragment_caller: FragmentCaller) -> None:
        fragment = fragment_caller.fragment
        self.enter_fragment(fragment)  # for AllocatedRegister

    def exit_fragment_caller(self: "AllocateTemporaries",
                             fragment_caller: FragmentCaller) -> None:
        fragment = fragment_caller.fragment
        self.exit_fragment(fragment)  # for AllocatedRegister

    def enter_fragment(self: "AllocateTemporaries", fragment: Fragment) -> None:
        if fragment.children is not None:
            return
        self.row_numbers_by_rn_reg = dict()
        self.row_masks_by_re_reg = dict()
        self.temporaries = {parameter.identifier
                            for parameter in fragment.temporaries}

    def exit_fragment(self: "AllocateTemporaries", fragment: Fragment) -> None:
        if fragment.children is not None:
            return
        frag_id = fragment.original_identifier
        if self.row_numbers_by_rn_reg is not None:
            self.row_numbers_by_rn_reg_by_frag[frag_id] = self.row_numbers_by_rn_reg
        self._available_row_numbers = None
        self.row_numbers_by_rn_reg = None
        self.row_masks_by_re_reg = None
        self.temporaries = None

    def transform_rn_reg(self: "AllocateTemporaries", rn_reg: RN_REG) -> RN_REG:
        if rn_reg.identifier not in self.temporaries:
            return rn_reg

        if rn_reg.row_number is not None \
           and rn_reg.row_number not in self.reserved_row_numbers:
            return rn_reg

        if rn_reg.identifier in self.row_numbers_by_rn_reg:
            row_number = self.row_numbers_by_rn_reg[rn_reg.identifier]
        else:
            row_number = self.available_row_numbers.popleft()
            self.row_numbers_by_rn_reg[rn_reg.identifier] = row_number

        return rn_reg.having(row_number=row_number)

    def transform_re_reg(self: "AllocateTemporaries", re_reg: RE_REG) -> RE_REG:
        if re_reg.identifier not in self.temporaries:
            return re_reg

        if re_reg.row_mask is not None:
            return re_reg

        if re_reg.identifier in self.row_masks_by_re_reg:
            row_mask = self.row_masks_by_re_reg[re_reg.identifier]
        elif re_reg.rows is not None:
            row_mask = 0x000000
            for row in re_reg.rows:
                if isinstance(row, RN_REG):
                    row_mask |= (1 << row.row_number)
                elif isinstance(row, RE_REG):
                    row_mask |= row.row_mask
                else:
                    raise ValueError(
                        f"Unsupported row type ({row.__class__.__name__}): {row}")
            self.row_masks_by_re_reg[re_reg.identifier] = row_mask

        return re_reg.having(row_mask=row_mask)


def only_temporaries(fn):

    @wraps(fn)
    def wrapper(self: "ParameterLowerizer",
                formal_parameter: FormalParameter) -> FormalParameter:
        # if not formal_parameter.identifier.startswith("_INTERNAL_"):
        if not formal_parameter.identifier.startswith("_INTERNAL"):
            return formal_parameter
        return fn(self, formal_parameter)

    return wrapper


@dataclass
class ParameterLowerizer(BLEIRListener, BLEIRTransformer):
    lowered_parameters_by_value_by_type: Dict[Type, Dict[int, FormalParameter]] \
        = field(default_factory=dict)

    parameter_map: Optional[Dict[FormalParameter, ActualParameter]] = None

    def enter_fragment_caller_call(
            self: "ParameterLowerizer",
            fragment_caller_call: FragmentCallerCall) -> None:
        self.parameter_map = fragment_caller_call.parameter_map

    def transform_fragment_caller_call(
            self: "ParameterLowerizer",
            fragment_caller_call: FragmentCallerCall) -> FragmentCallerCall:

        formal_parameters = set(fragment_caller_call.formal_parameters)

        actual_parameters = []
        for formal_parameter, actual_parameter in self.parameter_map.items():
            if formal_parameter in formal_parameters:
                actual_parameters.append(actual_parameter)

        return fragment_caller_call.having(parameters=actual_parameters)

    def exit_fragment_caller_call(
            self: "ParameterLowerizer",
            fragment_caller_call: FragmentCallerCall) -> None:
        self.parameter_map = None

    def transform_fragment_caller(
            self: "ParameterLowerizer",
            fragment_caller: FragmentCaller) -> FragmentCaller:

        if fragment_caller.registers is None:
            return fragment_caller

        registers = []
        for formal_parameter, allocated_register \
                in fragment_caller.register_map.items():
            if not formal_parameter.is_lowered:
                registers.append(allocated_register)

        return fragment_caller.having(registers=registers)

    def transform_fragment(self: "ParameterLowerizer",
                           fragment: Fragment) -> Fragment:

        parameters = []
        for parameter in fragment.parameters:
            if not parameter.is_lowered:
                parameters.append(parameter)

        return fragment.having(parameters=parameters)

    def lowered_parameters_by_value(
            self: "ParameterLowerizer",
            parameter_type: Type) -> Dict[int, FormalParameter]:

        if parameter_type in self.lowered_parameters_by_value_by_type:
            return self.lowered_parameters_by_value_by_type[parameter_type]

        lowered_parameters_by_value = {}
        self.lowered_parameters_by_value_by_type[parameter_type] \
            = lowered_parameters_by_value
        return lowered_parameters_by_value

    @only_temporaries
    def transform_rn_reg(self: "ParameterLowerizer",
                         rn_reg: RN_REG) -> RN_REG:

        lowered_parameters_by_value \
            = self.lowered_parameters_by_value(RN_REG)

        if rn_reg.row_number in lowered_parameters_by_value:
            lowered_parameter = lowered_parameters_by_value[rn_reg.row_number]
        else:
            num_lowered_parameters = len(lowered_parameters_by_value)
            lowered_parameter = RN_REG(
                identifier=f"BELEX_RN_REG_T{num_lowered_parameters}",
                is_lowered=True,
                row_number=rn_reg.row_number)
            lowered_parameters_by_value[rn_reg.row_number] = lowered_parameter

        # Keep all the RN_REG's attributes except those that are specific to the
        # lowered parameter
        return rn_reg.having(
            identifier=lowered_parameter.identifier,
            is_lowered=lowered_parameter.is_lowered,
            row_number=lowered_parameter.row_number)

    @only_temporaries
    def transform_re_reg(self: "ParameterLowerizer",
                         re_reg: RE_REG) -> RE_REG:

        lowered_parameters_by_value \
            = self.lowered_parameters_by_value(RE_REG)

        if re_reg.row_mask in lowered_parameters_by_value:
            lowered_parameter = lowered_parameters_by_value[re_reg.row_mask]
        else:
            num_lowered_parameters = len(lowered_parameters_by_value)
            lowered_parameter = RE_REG(
                identifier=f"BELEX_RE_REG_T{num_lowered_parameters}",
                is_lowered=True,
                row_mask=re_reg.row_mask)
            lowered_parameters_by_value[re_reg.row_mask] = lowered_parameter

        # Keep all the RN_REG's attributes except those that are specific to the
        # lowered parameter
        return re_reg.having(
            identifier=lowered_parameter.identifier,
            is_lowered=lowered_parameter.is_lowered,
            row_mask=lowered_parameter.row_mask)

    @only_temporaries
    def transform_l1_reg(self: "ParameterLowerizer",
                         l1_reg: L1_REG) -> L1_REG:

        lowered_parameters_by_value \
            = self.lowered_parameters_by_value(L1_REG)

        if l1_reg.bank_group_row in lowered_parameters_by_value:
            lowered_parameter = lowered_parameters_by_value[l1_reg.bank_group_row]
        else:
            num_lowered_parameters = len(lowered_parameters_by_value)
            lowered_parameter = L1_REG(
                identifier=f"BELEX_L1_REG_T{num_lowered_parameters}",
                is_lowered=True,
                bank_group_row=l1_reg.bank_group_row)
            lowered_parameters_by_value[l1_reg.bank_group_row] = lowered_parameter

        # Keep all the L1_REG's attributes except those that are specific to the
        # lowered parameter
        return l1_reg.having(
            identifier=lowered_parameter.identifier,
            is_lowered=lowered_parameter.is_lowered,
            bank_group_row=lowered_parameter.bank_group_row)

    @only_temporaries
    def transform_sm_reg(self: "ParameterLowerizer",
                         sm_reg: SM_REG) -> SM_REG:

        lowered_parameters_by_value \
            = self.lowered_parameters_by_value(SM_REG)

        if sm_reg.constant_value in lowered_parameters_by_value:
            lowered_parameter = lowered_parameters_by_value[sm_reg.constant_value]
        else:
            lowered_parameter = SM_REG(
                identifier=f"SM_0X{sm_reg.constant_value:04X}",
                is_lowered=True,
                constant_value=sm_reg.constant_value)
            lowered_parameters_by_value[sm_reg.constant_value] = lowered_parameter

        # Keep all the SM_REG's attributes except those that are specific to the
        # lowered parameter
        return sm_reg.having(
            identifier=lowered_parameter.identifier,
            is_lowered=lowered_parameter.is_lowered,
            constant_value=lowered_parameter.constant_value)


@dataclass
class EnumerateInstructions(BLEIRTransformer):
    num_instructions: int = -1

    in_multi_statement: bool = False

    def transform_fragment(self: "EnumerateInstructions",
                           fragment: Fragment) -> Fragment:
        self.num_instructions = 0
        enumerated_operations = \
            flatten(map(self.enumerate_operation, fragment.operations))
        self.num_instructions = -1
        return fragment.having(operations=enumerated_operations)

    def enumerate_operation(
                self: "EnumerateInstructions",
                operation_or_line_comment: Operation_or_LineComment) \
            -> Union[LineComment, Tuple[LineComment, Operation]]:

        if isinstance(operation_or_line_comment, LineComment.__args__):
            line_comment = operation_or_line_comment
            return line_comment

        operation = operation_or_line_comment

        if isinstance(operation, MultiStatement):
            self.in_multi_statement = True
            commands = \
                flatten(map(self.enumerate_operation, operation.statements))
            self.in_multi_statement = False
            self.num_instructions += 1
            return operation.having(statements=commands)

        text = f"instruction {1 + self.num_instructions}"

        if not self.in_multi_statement:
            self.num_instructions += 1

        if operation.comment is None:
            comment = TrailingComment(value=text)
            return operation.having(comment=comment)
        else:
            comment = SingleLineComment(line=text)
            return [comment, operation]


@dataclass
class NormalizeSectionMasks(BLEIRTransformer, BLEIRListener):
    GVML_MASKS: ClassVar[Dict[int, SM_REG]] = {
        mask.constant_value: mask
        for mask in map(lambda reg: reg.as_bleir().expression, [
            SM_0XFFFF,
            SM_0X0001,
            SM_0X1111,
            SM_0X0101,
            SM_0X000F,
            SM_0X0F0F,
            SM_0X0707,
            SM_0X5555,
            SM_0X3333,
            SM_0X00FF,
            SM_0X001F,
            SM_0X003F,
        ])
    }

    reservations: Set[int]

    in_statement: bool = False

    @staticmethod
    def get_shift(mask: int) -> int:
        for i in range(NSECTIONS):
            if mask & 0x0001 == 1:
                return i
            mask = mask >> 1
        return 0

    # def transform_fragment_caller(
    #         self: "NormalizeSectionMasks",
    #         fragment_caller: FragmentCaller) -> FragmentCaller:
    #     fragment = fragment_caller.fragment
    #     parameters = set(fragment.parameters)
    #     registers = []
    #     for register in fragment_caller.registers:
    #         if register.parameter in parameters:
    #             registers.append(register)
    #     return fragment_caller.having(registers=registers)

    def transform_fragment(self: "NormalizeSectionMasks",
                           fragment: Fragment) -> Fragment:
        parameters = []
        for parameter in fragment.parameters:
            if not parameter.is_lowered:
                parameters.append(parameter)
        return fragment.having(parameters=parameters)

    def enter_statement(self: "NormalizeSectionMasks",
                        statement: STATEMENT) -> None:
        self.in_statement = True

    def exit_statement(self: "NormalizeSectionMasks",
                       statement: STATEMENT) -> None:
        self.in_statement = False

    def transform_sm_reg(self: "NormalizeSectionMasks",
                         sm_reg: SM_REG) -> Union[SM_REG, SHIFTED_SM_REG]:

        if sm_reg.constant_value is None:
            return sm_reg

        mask = sm_reg.constant_value
        shift_width = self.get_shift(mask)

        if (mask >> shift_width) == (0xFFFF >> shift_width):
            normalized_value = 0xFFFF
        else:
            normalized_value = mask >> shift_width

        if normalized_value in self.GVML_MASKS:
            gvml_mask = self.GVML_MASKS[normalized_value]
            normalized_mask = sm_reg.having(
                identifier=gvml_mask.identifier,
                constant_value=gvml_mask.constant_value,
                register=gvml_mask.register,
                is_lowered=gvml_mask.is_lowered)
            self.reservations.add(gvml_mask.register)
        else:
            normalized_mask = sm_reg.having(constant_value=normalized_value)

        if shift_width == 0 or not self.in_statement:
            return normalized_mask

        return normalized_mask << shift_width


@dataclass
class InitializeTemporaries(BLEIRTransformer, BLEIRListener):
    frag_nym: Optional[str] = None
    operations: Optional[Deque[Operation_or_LineComment]] = None
    initial_values_by_register: Optional[Dict[str, int]] = None

    in_multi_statement: bool = False
    in_statement: bool = False
    instruction_number: int = 0
    has_init: bool = False

    @property
    def in_instruction(self: "InitializeTemporaries") -> bool:
        return self.in_multi_statement or self.in_statement

    def enter_fragment(self: "InitializeTemporaries",
                       fragment: Fragment) -> None:
        self.frag_nym = fragment.original_identifier
        self.operations = deque()
        self.initial_values_by_register = {}

    def transform_fragment(self: "InitializeTemporaries",
                           fragment: Fragment) -> Fragment:
        parameters = fragment.parameters
        return fragment.having(
            parameters=parameters,
            operations=tuple(self.operations))

    def exit_fragment(self: "InitializeTemporaries",
                      fragment: Fragment) -> None:
        self.frag_nym = None
        self.operations = None
        self.initial_values_by_register = None
        self.has_init = False

    def enter_multi_statement(self: "InitializeTemporaries",
                              multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True
        self.instruction_number += 1

    def exit_multi_statement(self: "InitializeTemporaries",
                             multi_statement: MultiStatement) -> None:
        self.operations.append(multi_statement)
        self.in_multi_statement = False

    def enter_statement(self: "InitializeTemporaries",
                        statement: STATEMENT) -> None:
        self.in_statement = True
        if not self.in_multi_statement:
            self.instruction_number += 1

    def exit_statement(self: "InitializeTemporaries",
                       statement: STATEMENT) -> None:
        self.in_statement = False
        if not self.in_multi_statement:
            self.operations.append(statement)

    def exit_multi_line_comment(self: "InitializeTemporaries",
                                multi_line_comment: MultiLineComment) -> None:
        if not self.in_multi_statement:
            self.operations.append(multi_line_comment)

    def exit_single_line_comment(
            self: "InitializeTemporaries",
            single_line_comment: SingleLineComment) -> None:
        if not self.in_multi_statement:
            self.operations.append(single_line_comment)

    def enter_rn_reg(self: "InitializeTemporaries",
                     rn_reg: RN_REG) -> None:

        if not self.in_instruction:
            # Skip the fragment parameter list
            return

        if rn_reg.identifier in self.initial_values_by_register:
            initial_value = self.initial_values_by_register[rn_reg.identifier]
            if rn_reg.initial_value is not None and \
               rn_reg.initial_value != initial_value:
                raise ValueError(
                    f"Conflicting initial values for {rn_reg.identifier} in "
                    f"fragment {self.frag_nym}: {rn_reg.initial_value} != "
                    f"{initial_value}")

        elif rn_reg.initial_value is not None:
            initial_value = rn_reg.initial_value
            with inline_bleir():
                self.operations.extend(
                    cpy_imm_16(rn_reg, initial_value))
            self.initial_values_by_register[rn_reg.identifier] = initial_value
            self.has_init = True


@dataclass
class CoalesceGroupedTemporaries(BLEIRTransformer, BLEIRListener):
    local_to_lowered_groups: Callable[[Fragment],
                                      Optional[Sequence[Sequence[RN_REG]]]]

    coalesced_temporaries: Optional[Dict[str, RN_REG]] = None

    coalesced_temporaries_by_frag: Dict[str, Dict[str, RN_REG]] = \
        field(default_factory=dict)

    @staticmethod
    def collect_temporaries(group: Sequence[RN_REG]) -> Sequence[RN_REG]:
        return [rn_reg for rn_reg in group if rn_reg.is_lowered]

    @staticmethod
    def find_min_attrs(group: Sequence[RN_REG]) \
            -> Tuple[Optional[str], Optional[str], Optional[int]]:
        min_nym = min_reg = min_row = None
        for rn_reg in group:
            if min_nym is None or rn_reg.identifier < min_nym:
                min_nym = rn_reg.identifier
            if min_reg is None or \
               rn_reg.register is not None and rn_reg.register < min_reg:
                min_reg = rn_reg.register
            if min_row is None or \
               rn_reg.row_number is not None and rn_reg.row_number < min_row:
                min_row = rn_reg.row_number
        return min_nym, min_reg, min_row

    def coalesce_temporaries(self: "CoalesceGroupedTemporaries",
                             group: Sequence[RN_REG]) -> None:

        temporaries = self.collect_temporaries(group)
        min_nym, min_reg, min_row = self.find_min_attrs(temporaries)
        coalesced_temporary = RN_REG(
            identifier=min_nym,
            register=min_reg,
            row_number=min_row,
            is_lowered=True)
        for temporary in temporaries:
            temp_nym = temporary.identifier
            LOGGER.debug("Coalescing temporary %s with temporary %s",
                         temp_nym,
                         coalesced_temporary.identifier)
            self.coalesced_temporaries[temp_nym] = coalesced_temporary

    def enter_fragment(self: "CoalesceGroupedTemporaries",
                       fragment: Fragment) -> None:
        groups = self.local_to_lowered_groups(fragment)
        if groups is not None:
            self.coalesced_temporaries = {}
            for group in groups:
                self.coalesce_temporaries(group)

    def exit_fragment(self: "CoalesceGroupedTemporaries",
                      fragment: Fragment) -> None:
        frag_id = fragment.original_identifier
        self.coalesced_temporaries_by_frag[frag_id] = self.coalesced_temporaries
        self.row_numbers_by_rn_reg = None
        self.lowered_parameters_by_value = None
        self.coalesced_temporaries = None

    def transform_rn_reg(self: "CoalesceGroupedTemporaries",
                         rn_reg: RN_REG) -> RN_REG:

        if self.coalesced_temporaries is None:
            return rn_reg

        if rn_reg.identifier in self.coalesced_temporaries:
            return self.coalesced_temporaries[rn_reg.identifier]

        return rn_reg


@dataclass
class ResetDebugValues(BLEIRTransformer, BLEIRListener):
    """Resets attributes for debugging fragments. These attributes interfere with
    the VM pipeline and since they are no longer useful, they need to be
    removed."""

    parameter_ids: Optional[Set[FormalParameter]] = None

    def is_parameterized(self: "ResetDebugValues",
                         register_parameter: RegisterParameter) -> bool:
        return self.parameter_ids is not None \
            and register_parameter.identifier in self.parameter_ids

    def enter_fragment(self: "ResetDebugValues",
                       fragment: Fragment) -> None:
        self.parameter_ids = {parameter.identifier
                              for parameter in fragment.parameters}

    def exit_fragment(self: "ResetDebugValues",
                      fragment: Fragment) -> None:
        self.parameter_ids = None

    def transform_rn_reg(self: "ResetDebugValues",
                         rn_reg: RN_REG) -> RN_REG:
        if self.is_parameterized(rn_reg) \
           or rn_reg.is_temporary \
           or rn_reg.identifier.startswith("_INTERNAL"):
            return rn_reg.having(row_number=None)
        return rn_reg

    def transform_re_reg(self: "ResetDebugValues",
                         re_reg: RE_REG) -> RE_REG:
        if self.is_parameterized(re_reg) \
           and not re_reg.identifier.startswith("_INTERNAL"):
            return re_reg.having(row_mask=None)
        return re_reg

    def transform_ewe_reg(self: "ResetDebugValues",
                          ewe_reg: EWE_REG) -> EWE_REG:
        if self.is_parameterized(ewe_reg) \
           and not ewe_reg.identifier.startswith("_INTERNAL"):
            return ewe_reg.having(wordline_mask=None)
        return ewe_reg

    def transform_l1_reg(self: "ResetDebugValues",
                         l1_reg: L1_REG) -> L1_REG:
        if self.is_parameterized(l1_reg) \
           and not l1_reg.identifier.startswith("_INTERNAL"):
            return l1_reg.having(bank_group_row=None)
        return l1_reg

    def transform_l2_reg(self: "ResetDebugValues",
                         l2_reg: L2_REG) -> L2_REG:
        if self.is_parameterized(l2_reg) \
           and not l2_reg.identifier.startswith("_INTERNAL"):
            return l2_reg.having(value=None)
        return l2_reg

    def transform_sm_reg(self: "ResetDebugValues",
                         sm_reg: SM_REG) -> SM_REG:
        if self.is_parameterized(sm_reg) \
           and not sm_reg.identifier.startswith("_INTERNAL"):
            return sm_reg.having(constant_value=None)
        return sm_reg


@dataclass
class CoalesceCompatibleTemporaries(BLEIRListener, BLEIRTransformer):
    live_section_scanner: LiveSectionScanner

    live_sections_by_param: Optional[Dict[str, Dict[int, Set[int]]]] = None
    sections_by_vr: Optional[Dict[str, Set[Union[int, Tuple[str, bool]]]]] = None

    aliases: Optional[Dict[str, RN_REG]] = None
    coalesced_temps: Optional[Sequence[Sequence[RN_REG]]] = None

    in_fragment: bool = False

    def are_compatible_wrt_param(self: "CoalesceCompatibleTemporaries",
                                 coalesced_temp: RN_REG, temp: RN_REG) -> bool:

        coalesced_sections = \
            list(self.sections_by_vr[coalesced_temp.identifier])
        temp_sections = list(self.sections_by_vr[temp.identifier])

        return len(coalesced_sections) == 1 and len(temp_sections) == 1 \
            and not isinstance(coalesced_sections[0], int) \
            and not isinstance(temp_sections[0], int) \
            and coalesced_sections[0][0] == temp_sections[0][0] \
            and coalesced_sections[0][1] != temp_sections[0][1]

    def access_overlap(self: "CoalesceCompatibleTemporaries",
                       coalesced_temps: Sequence[RN_REG],
                       temp: RN_REG) -> bool:

        temp_liveness = self.live_sections_by_param[temp.identifier]

        # Track each coalesced temporary, individually, in case there is an
        # acceptable overlap in section liveness. If the temps are coalesced too
        # soon, we will lose information on which may have acceptable overlaps
        # with the current temporary.
        for coalesced_temp in coalesced_temps:
            coalesced_liveness = \
                self.live_sections_by_param[coalesced_temp.identifier]
            shared_sections = \
                set(coalesced_liveness.keys()) & set(temp_liveness.keys())

            for section in shared_sections:
                coalesced_sec_liveness = coalesced_liveness[section]
                temp_sec_liveness = temp_liveness[section]
                if len(coalesced_sec_liveness & temp_sec_liveness) > 0 \
                   and not self.are_compatible_wrt_param(coalesced_temp, temp):
                    return True

        return False

    def get_first_viable(self: "CoalesceCompatibleTemporaries",
                         temp: RN_REG) -> Optional[Sequence[RN_REG]]:
        for coalesced_temp in self.coalesced_temps:
            if not self.access_overlap(coalesced_temp, temp):
                return coalesced_temp

    def enter_fragment(self: "CoalesceCompatibleTemporaries",
                       fragment: Fragment) -> None:
        self.in_fragment = True

        frag_id = fragment.original_identifier

        self.live_sections_by_param = \
            self.live_section_scanner \
                .live_sections_by_param_by_frag[frag_id]

        self.sections_by_vr = \
            self.live_section_scanner \
                .sections_by_vr_by_frag[frag_id]

        self.aliases = {}
        self.coalesced_temps = []

    def exit_fragment(self: "CoalesceCompatibleTemporaries",
                      fragment: Fragment) -> None:
        self.live_sections_by_param = None
        self.sections_by_vr = None
        self.aliases = None
        self.coalesced_temps = None
        self.in_fragment = False

    def enter_rn_reg(self: "CoalesceCompatibleTemporaries",
                     rn_reg: RN_REG) -> None:
        if not self.in_fragment \
           or not rn_reg.is_temporary \
           and not rn_reg.identifier.startswith("_INTERNAL") \
           or rn_reg.identifier in self.aliases:
            return
        temp = rn_reg
        coalesced_temps = self.get_first_viable(temp)
        if coalesced_temps is None:
            coalesced_temps = [temp]
            self.coalesced_temps.append(coalesced_temps)
        else:
            coalesced_temps.append(temp)
            LOGGER.debug("Coalescing %s into %s ...",
                         temp.identifier,
                         coalesced_temps[0].identifier)
        self.aliases[temp.identifier] = coalesced_temps[0]

    def transform_rn_reg(self: "CoalesceCompatibleTemporaries",
                         rn_reg: RN_REG) -> RN_REG:
        if self.in_fragment and rn_reg.identifier in self.aliases:
            return self.aliases[rn_reg.identifier]
        return rn_reg


class NegatableSectionMask(NamedTuple):
    identifier: str
    num_shifted_bits: int
    is_negated: bool

    def is_lane_compatible(self: "NegatableSectionMask",
                           other: Union[Sequence[int],
                                        "NegatableSectionMask"]) -> bool:
        return isinstance(other, NegatableSectionMask) \
            and self.identifier == other.identifier \
            and self.num_shifted_bits == other.num_shifted_bits \
            and self.is_negated != other.is_negated


Lane = Sequence[Operation_or_LineComment]
Statement_or_MultiStatement = Union[STATEMENT, MultiStatement]


@dataclass
class AutomaticLaner(BLEIRListener, BLEIRTransformer):

    # List of lanes in order of creation
    lanes: Optional[Sequence[Lane]] = None

    in_multi_statement: bool = False

    in_rwinh_on_exit_multi_statement: Optional[bool] = None
    in_rwinh: bool = False

    @staticmethod
    def lemmatize(target: str) -> str:
        if target.startswith("INV_"):
            target = target[len("INV_"):]
        if target in ["NRL", "ERL", "WRL", "SRL"]:
            target = "RL"
        return target

    @staticmethod
    def decompose_mask(mask: MASK) -> Tuple[SM_REG, int, bool]:
        if isinstance(mask.expression, SHIFTED_SM_REG):
            shifted_sm_reg = mask.expression
            sm_reg = shifted_sm_reg.register
            num_shifted_bits = shifted_sm_reg.num_bits
        else:
            sm_reg = mask.expression
            num_shifted_bits = 0

        is_negated = (mask.operator is UNARY_OP.NEGATE)
        return sm_reg, num_shifted_bits, is_negated

    @classmethod
    def lemmatized_sections(
            cls: Type["AutomaticLaner"],
            mask: MASK,
            target: str) -> Tuple[str, Union[Sequence[int],
                                             NegatableSectionMask]]:

        lemma = cls.lemmatize(target)
        sm_reg, num_shifted_bits, is_negated = cls.decompose_mask(mask)

        if sm_reg.constant_value is not None:
            sections = (sm_reg.constant_value << num_shifted_bits) & 0xFFFF

            if is_negated:
                sections = 0xFFFF - sections

            # For the purpose of dependency analysis, even though the union
            # should be incorrect, C-sim seems to consider the orignal mask and
            # not the consumed sections so we need this for laning to be
            # compatible with C-sim.
            # ^^^ This won't affect correctness since it is more conservative.
            # TODO: Determine if hardware has the same constraints.
            if target in ["NRL", "INV_NRL"]:
                sections |= (sections >> 1) & 0xFFFF
            elif target in ["SRL", "INV_SRL"]:
                sections |= (sections << 1) & 0xFFFF

            sections = parse_sections(f"0x{sections:04X}")

            if lemma == "GL":
                # sections will be either [0] or empty
                sections = sorted(set(section // 16 for section in sections))
            elif lemma == "GGL":
                sections = sorted(set(section // 4 for section in sections))

            return lemma, sections

        # Else, SM_REG is a parameter so assume all sections
        return lemma, NegatableSectionMask(
            identifier=sm_reg.identifier,
            num_shifted_bits=num_shifted_bits,
            is_negated=is_negated)

    @classmethod
    def masked_sections(cls: Type["AutomaticLaner"],
                        mask: MASK,
                        target: str) -> Tuple[str, Union[Sequence[int],
                                                         NegatableSectionMask]]:

        sm_reg, num_shifted_bits, is_negated = cls.decompose_mask(mask)

        if sm_reg.constant_value is not None:
            sections = (sm_reg.constant_value << num_shifted_bits) & 0xFFFF

            if is_negated:
                sections = 0xFFFF - sections

            sections = parse_sections(f"0x{sections:04X}")
            return target, sections

        # Else, SM_REG is a parameter so assume all sections
        return target, NegatableSectionMask(
            identifier=sm_reg.identifier,
            num_shifted_bits=num_shifted_bits,
            is_negated=is_negated)

    @classmethod
    def source_sections(
            cls: Type["AutomaticLaner"],
            statement: STATEMENT) -> Sequence[Tuple[str, Union[Sequence[int],
                                                               NegatableSectionMask]]]:

        operation = statement.operation

        if hasattr(operation, "right_operands") and isinstance(operation, MASKED):
            # For the purpose of this analysis, it seems only the right-most
            # source matters even in statements like "RL ^= GL", which has two
            # right operands (RL and GL).
            source = operation.right_operands[-1]
            if source in SRC_SET:
                mask = operation.mask
                return [cls.masked_sections(mask, source)]

        return []

    @classmethod
    def consumes(cls: Type["AutomaticLaner"],
                 statement: STATEMENT) -> Sequence[Tuple[str, Union[Sequence[int],
                                                                    NegatableSectionMask]]]:

        # FIXME: Update consumes(.) and provides(.) to resolve the parameters
        # like "sb1"->"SB:1" or "l1_for_ggl"->"L1:8"

        operation = statement.operation

        if hasattr(operation, "right_operands"):
            targets = operation.right_operands

            if isinstance(operation, MASKED):
                mask = operation.mask
                return [cls.lemmatized_sections(mask, target)
                        for target in targets]

            if isinstance(operation, (GGL_ASSIGNMENT, LGL_ASSIGNMENT, LX_ASSIGNMENT)):
                sections = list(range(4))
            else:
                sections = list(range(NSECTIONS))
            return [(target, sections) for target in targets]

        return []

    @classmethod
    def provides(cls: Type["AutomaticLaner"],
                 statement: STATEMENT) -> Sequence[Tuple[str, Union[Sequence[int],
                                                                    NegatableSectionMask]]]:

        operation = statement.operation

        if hasattr(operation, "left_operands"):
            dependencies = operation.left_operands

            if isinstance(operation, MASKED):
                mask = operation.mask
                return [cls.lemmatized_sections(mask, dependency)
                        for dependency in dependencies]

            if isinstance(operation, (GGL_ASSIGNMENT, LGL_ASSIGNMENT, LX_ASSIGNMENT)):
                sections = list(range(4))
            else:
                sections = list(range(NSECTIONS))
            return [(dependency, sections) for dependency in dependencies]

        return []

    @classmethod
    def depends_on(cls: Type["AutomaticLaner"],
                   statement_or_multi_statement: Statement_or_MultiStatement,
                   query: Lane) -> bool:
        if isinstance(statement_or_multi_statement, MultiStatement):
            multi_statement = statement_or_multi_statement
            for statement in multi_statement:
                if cls.depends_on(statement, query):
                    return True
            return False
        statement = statement_or_multi_statement
        consumptions = dict(cls.consumes(statement))
        for candidate in query:
            if isinstance(candidate, STATEMENT):
                provisions = dict(cls.provides(candidate))
                for src in consumptions.keys() & provisions.keys():
                    consumed_sections = consumptions[src]
                    provided_sections = provisions[src]
                    if set(consumed_sections) & set(provided_sections):
                        return True
        return False

    @staticmethod
    def has_rwinh_set_and_write(lane: Lane) -> bool:
        write = None
        rwinh = None
        for statement in lane:
            command = statement.operation
            if isinstance(command, MASKED):
                if command.read_write_inhibit is ReadWriteInhibit.RWINH_SET:
                    rwinh = command
                elif command.assignment is not None \
                     and isinstance(command.assignment.operation, WRITE):
                    write = command
        return (write is not None) and (rwinh is not None)

    def has_conflict_with_operation(
            self: "AutomaticLaner",
            prev_operation: Operation,
            curr_operation: STATEMENT,
            prev_lane: Optional[Lane],
            curr_lane: Optional[Lane]) -> bool:

        if isinstance(prev_operation, MultiStatement):
            for command in prev_operation:
                if self.has_conflict_with_operation(
                        command, curr_operation, prev_lane, curr_lane):
                    return True
            return False

        prev = prev_operation.operation
        curr = curr_operation.operation

        # Leave SPECIAL instructions where they are (like a memory fence)
        if isinstance(prev, SPECIAL) or isinstance(curr, SPECIAL):
            return True

        prev_rwinh = None
        prev_mask = None
        if isinstance(prev, MASKED):
            prev_rwinh = prev.read_write_inhibit
            prev_mask = prev.mask

        curr_rwinh = None
        curr_mask = None
        if isinstance(curr, MASKED):
            curr_rwinh = curr.read_write_inhibit
            curr_mask = curr.mask

        if isinstance(prev, MASKED) and prev.assignment is not None:
            prev = prev.assignment.operation

        if isinstance(curr, MASKED) and curr.assignment is not None:
            curr = curr.assignment.operation

        # Do not merge a WRITE-before-READ within the context of RWINH, per a
        # note in the design doc: Write operations may not be initiated in the
        # same cycle as Read operations with rwinhset=1, because the Write result
        # would be indeterminate - it may or may not be inhibited.
        if isinstance(prev, WRITE) \
           and isinstance(curr, READ) \
           and curr_rwinh is ReadWriteInhibit.RWINH_SET \
           and (not prev_mask.is_constant
                or not curr_mask.is_constant
                or (prev_mask.resolve() & curr_mask.resolve()) != 0x0000):
            return True

        # In the following situation, we may not merge the READ into the
        # instruction as the RWINH_SET because of the WRITE in the latter
        # instruction. WRITEs and READs may not appear within the same
        # instruction within the context of RWINH, but the latter is a special
        # case in which the WRITE appears in the same instruction as RWINH_SET
        # without the corresponding READ:
        # --------------------------------------------------------------------
        # {
        #     RL[msk1] <= mrk()
        # }
        # {
        #     RWINH_SET[msk2]
        #     vr[msk3] op= src()
        # }
        if curr_lane is not None \
           and isinstance(prev, READ) \
           and self.has_rwinh_set_and_write(curr_lane):
            return True

        if (prev_rwinh is ReadWriteInhibit.RWINH_SET
            and isinstance(curr, WRITE)) \
           or (curr_rwinh is ReadWriteInhibit.RWINH_SET
               and isinstance(prev, WRITE)):
            return True

        if (prev_rwinh is ReadWriteInhibit.RWINH_RST
            and isinstance(curr, (WRITE, READ))) \
           or (curr_rwinh is ReadWriteInhibit.RWINH_RST
               and isinstance(prev, (WRITE, READ))):
            return True

        # Two instances of RWINH_(SET|RST)
        if prev_rwinh is not None and curr_rwinh is not None:
            return True

        # Do not move RWINH_SET outside of the instruction containing the read
        # marker
        if prev_lane is not None \
           and curr_rwinh is ReadWriteInhibit.RWINH_SET:
            for compatible_command in prev_lane:
                if not isinstance(compatible_command, STATEMENT):
                    continue
                compat = compatible_command.operation
                if isinstance(compat, MASKED) \
                   and compat.assignment is not None:
                    compat = compat.assignment.operation
                if isinstance(compat, READ):
                    return True

        # Do not allow subsequent READs to be merged into an instruction with
        # RWINH_SET
        if prev_rwinh is ReadWriteInhibit.RWINH_SET and isinstance(curr, READ):
            return True

        # TODO: C-sim fails when READs are combined with BROADCASTs to RSP16; see
        # if HW has the same error.
        # TODO: Determine if we just needed one or two NOOPs after the broadcast ...
        if isinstance(prev, READ) and isinstance(curr, BROADCAST) \
           and curr.lvalue is BROADCAST_EXPR.RSP16 \
           or isinstance(prev, BROADCAST) and isinstance(curr, READ) \
           and prev.lvalue is BROADCAST_EXPR.RSP16:
            return True

        # Cannot consume different sources from same sections
        # if isinstance(prev, READ) or isinstance(curr, READ):
        if isinstance(prev, WRITE) or isinstance(curr, WRITE):
            prev_sources = dict(self.source_sections(prev_operation))
            curr_sources = dict(self.source_sections(curr_operation))
            shared_sources = set(prev_sources.keys()) | set(curr_sources.keys())
            if len(shared_sources) > 1:
                prev_indices = next(iter(prev_sources.values()))
                curr_indices = next(iter(curr_sources.values()))
                if isinstance(prev_indices, NegatableSectionMask):
                    if not prev_indices.is_lane_compatible(curr_indices):
                        return True
                elif isinstance(curr_sources, NegatableSectionMask):
                    return True
                else:
                    shared_indices = set(prev_indices) & set(curr_indices)
                    if len(shared_indices) > 0:
                        return True

        # These are always lane compatible:
        # 1. Read-after-Write (conversely, Write-before-Read)
        # 2. Broadcast-after-Read (conversely, Read-before-Broadcast)
        # 3. By the transitive property, it follows that Write-before-Broadcast
        #    is also lane-compatible:
        # if isinstance(prev, WRITE) and isinstance(curr, (READ, BROADCAST)) \
        #    or isinstance(prev, READ) and isinstance(curr, BROADCAST):
        #     return False
        # ---------------------------------------------------------------------
        # ^^^ this rule broke some instructions that wrote-to and read-from the
        # same SB, same sections.

        curr_consumptions = dict(self.consumes(curr_operation))
        curr_provisions = dict(self.provides(curr_operation))

        # If the current command is a BROADCAST and the previously determined
        # compatible lane (not the current one) contained a READ that produced a
        # dependency for the current BROADCAST, then the current lane is
        # incompatible regardless its content (the current command must be laned
        # with the previously determined compatible lane).
        if isinstance(curr, (READ, BROADCAST)) and prev_lane is not None:
            for command_or_comment in prev_lane:
                if not isinstance(command_or_comment, STATEMENT):
                    continue

                command = command_or_comment
                operation = command.operation

                if isinstance(operation, MASKED) \
                   and operation.assignment is not None:
                    operation = operation.assignment.operation

                if not (isinstance(operation, READ) and isinstance(curr, BROADCAST)
                        or isinstance(operation, WRITE) and isinstance(curr, READ)):
                    continue

                cmd_provisions = dict(self.provides(command))
                potential_dependencies = \
                    set(cmd_provisions.keys()) & set(curr_consumptions.keys())

                if len(potential_dependencies) > 0:
                    for potential_dependency in potential_dependencies:
                        provided_sections = cmd_provisions[potential_dependency]
                        consumed_sections = curr_consumptions[potential_dependency]

                        if not isinstance(provided_sections, NegatableSectionMask):
                            shared_sections = \
                                set(provided_sections) & set(consumed_sections)
                            if len(shared_sections) > 0:
                                return True

                cmd_consumptions = dict(self.consumes(command))
                potential_anti_dependencies = \
                    set(cmd_consumptions.keys()) & set(curr_provisions.keys())

                if len(potential_anti_dependencies) > 0:
                    for potential_anti_dependency in potential_anti_dependencies:
                        provided_sections = cmd_consumptions[potential_anti_dependency]
                        consumed_sections = curr_provisions[potential_anti_dependency]

                        if not isinstance(provided_sections, NegatableSectionMask):
                            shared_sections = \
                                set(provided_sections) & set(consumed_sections)
                            if len(shared_sections) > 0:
                                return True

        prev_provisions = dict(self.provides(prev_operation))
        potential_dependencies = \
            set(prev_provisions.keys()) & set(curr_consumptions.keys())

        if len(potential_dependencies) > 0:
            for potential_dependency in potential_dependencies:
                provided_sections = prev_provisions[potential_dependency]
                consumed_sections = curr_consumptions[potential_dependency]

                if isinstance(provided_sections, NegatableSectionMask):
                    if not provided_sections.is_lane_compatible(consumed_sections):
                        return True

                elif isinstance(consumed_sections, NegatableSectionMask):
                    return True

                else:
                    shared_sections = set(provided_sections) & set(consumed_sections)
                    if len(shared_sections) > 0 and (
                            (isinstance(prev, READ)
                             and isinstance(curr, WRITE))
                            or (isinstance(prev, WRITE)
                                and isinstance(curr, BROADCAST)
                                and curr.lvalue is BROADCAST_EXPR.GL)
                            or not isinstance(prev, (READ, WRITE))
                            or not isinstance(curr, (WRITE, BROADCAST))):
                        return True

        prev_consumptions = dict(self.consumes(prev_operation))
        potential_anti_dependencies = \
            set(prev_consumptions.keys()) & set(curr_provisions.keys())

        if len(potential_anti_dependencies) > 0:
            for potential_anti_dependency in potential_anti_dependencies:
                provided_sections = prev_consumptions[potential_anti_dependency]
                consumed_sections = curr_provisions[potential_anti_dependency]

                if isinstance(provided_sections, NegatableSectionMask):
                    if not provided_sections.is_lane_compatible(consumed_sections):
                        return True

                elif isinstance(consumed_sections, NegatableSectionMask):
                    return True

                else:
                    shared_sections = \
                        set(provided_sections) & set(consumed_sections)
                    if len(shared_sections) > 0 and not (
                            type(prev) is type(curr)
                            or (isinstance(prev, WRITE)
                                and isinstance(curr, (READ, BROADCAST)))
                            or (isinstance(prev, READ)
                                and isinstance(curr, BROADCAST))):
                        return True

        shared_provisions = \
            set(prev_provisions.keys()) & set(curr_provisions.keys())

        if len(shared_provisions) > 0:
            for shared_provision in shared_provisions:
                prev_sections = prev_provisions[shared_provision]
                curr_sections = curr_provisions[shared_provision]

                if isinstance(prev_sections, NegatableSectionMask):
                    if not prev_sections.is_lane_compatible(curr_sections):
                        return True

                elif isinstance(curr_sections, NegatableSectionMask):
                    return True

                else:
                    shared_sections = set(prev_sections) & set(curr_sections)
                    if len(shared_sections) > 0:
                        return True

        if isinstance(prev, WRITE) and isinstance(curr, WRITE) \
           and prev.left_operands != curr.left_operands:
            # Two conditions must be satisfied to combine WRITEs:
            # 1. All SBs within the WRITEs must be pseudo-constants.
            # 2. All SBs within the WRITEs must have with row numbers in the same
            #    group.
            sb_group = None
            for rn_reg_or_xe in set(prev.lvalue) | set(curr.lvalue):
                if not isinstance(rn_reg_or_xe, RN_REG):
                    # TODO: Determine the laning conditions for XE regs. Until
                    # then, leave their WRITEs where they are.
                    return True

                rn_reg = rn_reg_or_xe
                if rn_reg.row_number is None:
                    # TODO: Once we control the entire application, perform
                    # additional analysis over all fragment caller calls. Until
                    # then, we can only perform the analysis, reliably, against
                    # pseudo-constants.
                    return True

                if sb_group is None:
                    sb_group = rn_reg.row_number // 8
                elif sb_group != rn_reg.row_number // 8:
                    return True

        return False

    def has_conflict_with_lane(
            self: "AutomaticLaner",
            lane: Lane,
            statement: STATEMENT,
            prev_lane: Optional[Lane],
            curr_lane: Optional[Lane]) -> bool:
        for operation_or_line_comment in lane:
            if isinstance(operation_or_line_comment, Operation.__args__):
                operation = operation_or_line_comment
                if self.has_conflict_with_operation(
                        operation, statement, prev_lane, curr_lane):
                    return True
        return False

    @staticmethod
    def count_commands(
            statements_and_line_comments: Sequence[STATEMENT_or_LineComment]) -> int:
        num_commands = 0
        for statement_or_line_comment in statements_and_line_comments:
            num_commands += isinstance(statement_or_line_comment, STATEMENT)
        return num_commands

    def merge_multi_statement(self: "AutomaticLaner",
                              multi_statement: MultiStatement) -> None:

        compatible_lane = None

        num_commands = self.count_commands(multi_statement)

        for lane in reversed(self.lanes):
            if not isinstance(lane, list):
                continue

            has_conflict = False

            for statement_or_line_comment in multi_statement:
                if isinstance(statement_or_line_comment, STATEMENT):
                    statement = statement_or_line_comment
                    has_conflict = self.has_conflict_with_lane(
                        lane, statement, compatible_lane, multi_statement)
                    if has_conflict:
                        break

            if has_conflict:
                break

            if num_commands + self.count_commands(lane) > 4:
                if self.depends_on(multi_statement, lane):
                    break
                continue

            compatible_lane = lane

        if compatible_lane is None:
            compatible_lane = []
            self.lanes.append(compatible_lane)

        if multi_statement.comment is not None:
            # Keep any multi-statement comments
            compatible_lane.append(multi_statement.comment)
        compatible_lane.extend(multi_statement)

    def merge_statement(self: "AutomaticLaner",
                        statement: STATEMENT) -> None:

        compatible_lane = None

        # Iterate as far up the lanes as possible
        for lane in reversed(self.lanes):
            if not isinstance(lane, list):
                # It's a comment
                continue

            if self.has_conflict_with_lane(lane, statement, compatible_lane, None):
                break

            if 1 + self.count_commands(lane) > 4:
                if self.depends_on(statement, lane):
                    break
                continue

            compatible_lane = lane

        if compatible_lane is None:
            compatible_lane = []
            self.lanes.append(compatible_lane)

        compatible_lane.append(statement)

    def enter_fragment(self: "AutomaticLaner",
                       fragment: Fragment) -> None:
        self.lanes = []

    def transform_fragment(self: "AutomaticLaner",
                           fragment: Fragment) -> Fragment:
        operations_and_comments = []
        for lane in self.lanes:
            if isinstance(lane, list):
                if len(lane) == 1:
                    operation_or_comment = lane[0]
                else:
                    operation_or_comment = MultiStatement(statements=lane)
            else:
                operation_or_comment = lane  # likely a comment
            operations_and_comments.append(operation_or_comment)
        return fragment.having(operations=operations_and_comments)

    def exit_fragment(self: "AutomaticLaner",
                      fragment: Fragment) -> None:
        self.lanes = None

    def enter_multi_statement(self: "AutomaticLaner",
                              multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True
        self.merge_multi_statement(multi_statement)

    def exit_multi_statement(self: "AutomaticLaner",
                             multi_statement: MultiStatement) -> None:
        self.in_multi_statement = False
        if self.in_rwinh_on_exit_multi_statement is not None:
            self.in_rwinh = self.in_rwinh_on_exit_multi_statement
            self.in_rwinh_on_exit_multi_statement = None

    def enter_multi_line_comment(self: "AutomaticLaner",
                                 multi_line_comment: MultiLineComment) -> None:
        if not self.in_multi_statement:
            self.lanes.append(multi_line_comment)

    def enter_single_line_comment(self: "AutomaticLaner",
                                  single_line_comment: SingleLineComment) -> None:
        if not self.in_multi_statement:
            self.lanes.append(single_line_comment)

    def enter_statement(self: "AutomaticLaner",
                        statement: STATEMENT) -> None:
        if not self.in_multi_statement:
            self.merge_statement(statement)

    def exit_statement(self: "AutomaticLaner",
                       statement: STATEMENT) -> None:
        operation = statement.operation
        if isinstance(operation, MASKED):
            rwinh = operation.read_write_inhibit
            if rwinh is not None:
                in_rwinh = rwinh is ReadWriteInhibit.RWINH_SET
                if self.in_multi_statement:
                    self.in_rwinh_on_exit_multi_statement = in_rwinh
                else:
                    self.in_rwinh = in_rwinh


@dataclass
class GroupMultipleBroadcasts(BLEIRListener, BLEIRTransformer):
    param_to_value: Callable[[RegisterParameter], Union[int, np.ndarray]]

    broadcasts_by_lvalue: Optional[Dict[BROADCAST_EXPR, Sequence[STATEMENT]]] \
        = None
    non_broadcasts: Optional[Sequence[STATEMENT]] = None
    in_multi_statement: bool = False
    serial_mask_id: int = -1

    def next_mask_id(self: "GroupMultipleBroadcasts") -> int:
        self.serial_mask_id += 1
        return self.serial_mask_id

    def build_temp_mask(self: "GroupMultipleBroadcasts",
                        constant_value: int) -> SM_REG:
        return SM_REG(
            identifier=f"_INTERNAL_GROUP_{self.next_mask_id():03d}",
            constant_value=constant_value)

    def enter_multi_statement(
            self: "GroupMultipleBroadcasts",
            multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True
        self.broadcasts_by_lvalue = defaultdict(list)
        self.non_broadcasts = []

    def unify_masks(
            self: "GroupMultipleBroadcasts",
            broadcasts: Sequence[STATEMENT]) -> MASK:
        unification = 0x0000
        for broadcast in broadcasts:
            mask = broadcast.operation.mask
            value = self.param_to_value(mask.sm_reg)
            unification |= mask.resolve(value)
        sm_reg = self.build_temp_mask(unification)
        return MASK(expression=sm_reg)

    def unify_broadcasts(
            self: "GroupMultipleBroadcasts",
            broadcasts: Sequence[STATEMENT]) -> STATEMENT:
        mask = self.unify_masks(broadcasts)
        masked = broadcasts[0].operation.having(mask=mask)
        return broadcasts[0].having(operation=masked)

    def transform_multi_statement(
            self: "GroupMultipleBroadcasts",
            multi_statement: MultiStatement) -> MultiStatement:

        if len(self.broadcasts_by_lvalue) > 0:
            statements = self.non_broadcasts
            for broadcasts in self.broadcasts_by_lvalue.values():
                if len(broadcasts) > 1:
                    broadcast = self.unify_broadcasts(broadcasts)
                    statements.append(broadcast)
                else:
                    statements.extend(broadcasts)
            if len(statements) < len(multi_statement.statements):
                multi_statement = multi_statement.having(statements=statements)

        return multi_statement

    def exit_multi_statement(
            self: "GroupMultipleBroadcasts",
            multi_statement: MultiStatement) -> None:
        self.in_multi_statement = False
        self.broadcasts_by_lvalue = None
        self.non_broadcasts = None

    def unifiable_broadcast(
            self: "GroupMultipleBroadcasts",
            statement: STATEMENT) -> Optional[BROADCAST]:

        if not self.in_multi_statement:
            return None

        operation = statement.operation
        if not isinstance(operation, MASKED):
            return None

        if operation.assignment is None:
            return None

        assignment = operation.assignment.operation
        if not isinstance(assignment, BROADCAST):
            return None

        if assignment.lvalue in [BROADCAST_EXPR.GL, BROADCAST_EXPR.GGL]:
            return assignment

    def enter_statement(
            self: "GroupMultipleBroadcasts",
            statement: STATEMENT) -> None:

        broadcast = self.unifiable_broadcast(statement)
        if broadcast is not None:
            self.broadcasts_by_lvalue[broadcast.lvalue].append(statement)
        else:
            self.non_broadcasts.append(statement)
