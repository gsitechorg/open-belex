r"""
By Dylon Edwards
"""

import logging
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from heapq import heapify, heappop, heappush
from itertools import chain, groupby
from typing import (Any, Callable, Deque, Dict, Optional, Sequence, Set, Tuple,
                    Type, Union)

from open_belex.bleir.commands import (ApplyPatch, Command, LoadRegister,
                                       LoadRegisters, LoadSrc, MimicFSelNoOp,
                                       MimicGGLFromL1, MimicGGLFromRL,
                                       MimicGGLFromRLAndL1, MimicGLFromRL,
                                       MimicL1FromGGL, MimicL1FromLGL,
                                       MimicL2End, MimicL2FromLGL,
                                       MimicLGLFromL1, MimicLGLFromL2,
                                       MimicNoOp, MimicRLAndEqInvSB,
                                       MimicRLAndEqSB, MimicRLAndEqSBAndInvSrc,
                                       MimicRLAndEqSBAndSrc, MimicRLAndEqSrc,
                                       MimicRLFromInvSB,
                                       MimicRLFromInvSBAndInvSrc,
                                       MimicRLFromInvSBAndSrc,
                                       MimicRLFromInvSrc, MimicRLFromSB,
                                       MimicRLFromSBAndInvSrc,
                                       MimicRLFromSBAndSrc,
                                       MimicRLFromSBOrInvSrc,
                                       MimicRLFromSBOrSrc,
                                       MimicRLFromSBXorInvSrc,
                                       MimicRLFromSBXorSrc, MimicRLFromSrc,
                                       MimicRLOrEqInvSrc, MimicRLOrEqSB,
                                       MimicRLOrEqSBAndInvSrc,
                                       MimicRLOrEqSBAndSrc, MimicRLOrEqSrc,
                                       MimicRLXorEqInvSrc, MimicRLXorEqSB,
                                       MimicRLXorEqSBAndInvSrc,
                                       MimicRLXorEqSBAndSrc, MimicRLXorEqSrc,
                                       MimicRSP2KFromRSP32K,
                                       MimicRSP2KFromRSP256, MimicRSP16FromRL,
                                       MimicRSP16FromRSP256,
                                       MimicRSP32KFromRSP2K,
                                       MimicRSP256FromRSP2K,
                                       MimicRSP256FromRSP16, MimicRSPEnd,
                                       MimicRSPStartRet, MimicRWInhRst,
                                       MimicRWInhSet, MimicSBCondEqInvSrc,
                                       MimicSBCondEqSrc, MimicSBFromInvSrc,
                                       MimicSBFromSrc, MimicSetRL, UnifySMRegs)
from open_belex.bleir.types import (ASSIGN_OP, BINARY_EXPR, EWE_REG, L1_REG,
                                    L2_REG, MASK, MASKED, RE_REG, READ, RN_REG,
                                    SHIFTED_SM_REG, SM_REG, SRC_EXPR,
                                    STATEMENT, UNARY_EXPR, UNARY_OP, UNARY_SRC,
                                    WRITE, AllocatedRegister, FormalParameter,
                                    Fragment, FragmentCaller,
                                    FragmentCallerCall, GlassStatement,
                                    MultiStatement, RegisterParameter, Snippet,
                                    ValueParameter)
from open_belex.bleir.walkables import (BLEIRListener, BLEIRVisitor,
                                        BLEIRWalker, camel_case_to_underscore)
from open_belex.common.constants import NSECTIONS
from open_belex.common.gvml import (GVML_NYMS_BY_SM_REG_VAL, GVML_SM_REG_VALS,
                                    GVML_VALUES_BY_SM_REG_VAL)
from open_belex.common.register_arenas import NUM_RN_REGS
from open_belex.common.types import Integer

LOGGER = logging.getLogger()

MAX_FRAGMENT_INSTRUCTIONS = 255


@dataclass
class NumFragmentInstructionsAnalyzer(BLEIRListener):
    num_instructions: Optional[int] = None
    num_commands: Optional[int] = None
    in_multi_statement: bool = False

    num_instructions_by_fragment: Dict[str, int] = \
        field(default_factory=lambda: defaultdict(int))

    num_commands_by_fragment: Dict[str, int] = \
        field(default_factory=lambda: defaultdict(int))

    def enter_fragment(self: "NumFragmentInstructionsValidator",
                       fragment: Fragment) -> None:
        if fragment.children is not None:
            assert len(fragment.operations) == 0
        self.num_instructions = 0
        self.num_commands = 0

    def exit_fragment(self: "NumFragmentInstructionsValidator",
                      fragment: Fragment) -> None:
        # global LOGGER

        # if fragment.children is not None:
        #     LOGGER.debug("%s ::= (num_instructions=%d, num_commands=%d)",
        #                  fragment.identifier,
        #                  self.num_instructions,
        #                  self.num_commands)

        # if self.num_instructions > MAX_FRAGMENT_INSTRUCTIONS:
        #     warn(f"No more than {MAX_FRAGMENT_INSTRUCTIONS} instructions are "
        #          f"supported by the APU, but {fragment.identifier} has "
        #          f"{self.num_instructions}")

        if fragment.children is None:
            self.num_instructions_by_fragment[fragment.identifier] = \
                self.num_instructions
            self.num_commands_by_fragment[fragment.identifier] = \
                self.num_commands
        else:
            self.num_instructions_by_fragment[fragment.identifier] = 0
            self.num_commands_by_fragment[fragment.identifier] = 0

        self.num_instructions = None
        self.num_commands = None

    def enter_multi_statement(self: "NumFragmentInstructionsValidator",
                              multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True
        self.num_instructions += 1

    def exit_multi_statement(self: "NumFragmentInstructionsValidator",
                             multi_statement: MultiStatement) -> None:
        self.in_multi_statement = False

    def enter_statement(self: "NumFragmentInstructionsValidator",
                        statement: STATEMENT) -> None:
        if isinstance(statement.operation, GlassStatement):
            return
        if not self.in_multi_statement:
            self.num_instructions += 1
        self.num_commands += 1


@dataclass
class LiveParameterMarker(BLEIRListener):
    parameter_liveness: Dict[str, Sequence[bool]] = field(default_factory=dict)
    live_parameters: Optional[Set[RegisterParameter]] = None
    in_statement: bool = False

    def enter_fragment(self: "LiveParameterMarker",
                       fragment: Fragment) -> None:
        if fragment.children is None:
            self.live_parameters = set()

    def exit_fragment(self: "LiveParameterMarker",
                      fragment: Fragment) -> None:
        if fragment.children is None:
            parameter_is_live = [parameter in self.live_parameters
                                for parameter in fragment.parameters]
            self.parameter_liveness[fragment.identifier] = parameter_is_live
            self.live_parameters = None

    def enter_statement(self: "LiveParameterMarker",
                        statement: STATEMENT) -> None:
        self.in_statement = True

    def exit_statement(self: "LiveParameterMarker",
                       statement: STATEMENT) -> None:
        self.in_statement = False

    def enter_rn_reg(self: "LiveParameterMarker",
                     rn_reg: RN_REG) -> RN_REG:
        if self.in_statement and self.live_parameters is not None:
            self.live_parameters.add(rn_reg)
        return rn_reg

    def enter_re_reg(self: "LiveParameterMarker",
                     re_reg: RE_REG) -> RE_REG:
        if self.in_statement and self.live_parameters is not None:
            self.live_parameters.add(re_reg)
        return re_reg

    def enter_ewe_reg(self: "LiveParameterMarker",
                      ewe_reg: EWE_REG) -> EWE_REG:
        if self.in_statement and self.live_parameters is not None:
            self.live_parameters.add(ewe_reg)
        return ewe_reg

    def enter_l1_reg(self: "LiveParameterMarker",
                     l1_reg: L1_REG) -> L1_REG:
        if self.in_statement and self.live_parameters is not None:
            self.live_parameters.add(l1_reg)
        return l1_reg

    def enter_l2_reg(self: "LiveParameterMarker",
                     l2_reg: L2_REG) -> L2_REG:
        if self.in_statement and self.live_parameters is not None:
            self.live_parameters.add(l2_reg)
        return l2_reg

    def enter_sm_reg(self: "LiveParameterMarker",
                     sm_reg: SM_REG) -> SM_REG:
        if self.in_statement and self.live_parameters is not None:
            self.live_parameters.add(sm_reg)
        return sm_reg


@dataclass
class SBGroupAnalyzer(BLEIRListener):
    """Groups RN_REGs in the following fashion:
        1. One group per RN_REG parameter
        2. One group per statement involving a WRITE
        3. One group per multi-statement involving one-or-more WRITEs
    """

    groupings: Optional[Set[Set[RN_REG]]] = field(default_factory=lambda: None)
    grouping: Optional[Set[RN_REG]] = field(default_factory=lambda: None)
    is_in_multi_statement: bool = field(default_factory=lambda: False)
    is_in_statement: bool = field(default_factory=lambda: False)

    def enter_fragment(self: "SBGroupAnalyzer",
                       fragment: Fragment) -> None:
        self.groupings = set()
        for formal_parameter in fragment.parameters:
            if isinstance(formal_parameter, RN_REG):
                self.groupings.add((formal_parameter,))

    def enter_multi_statement(self: "SBGroupAnalyzer",
                              multi_statement: MultiStatement) -> None:
        self.is_in_multi_statement = True
        self.grouping = set()

    def exit_multi_statement(self: "SBGroupAnalyzer",
                             multi_statement: MultiStatement) -> None:
        self.groupings.add(tuple(self.grouping))
        self.grouping = None
        self.is_in_multi_statement = False

    def enter_statement(self: "SBGroupAnalyzer", statement: STATEMENT) -> None:
        self.is_in_statement = True
        if not self.is_in_multi_statement:
            self.grouping = set()

    def exit_statement(self: "SBGroupAnalyzer", statement: STATEMENT) -> None:
        self.is_in_statement = False
        if not self.is_in_multi_statement:
            self.groupings.add(tuple(self.grouping))
            self.grouping = None

    def enter_rn_reg(self: "SBGroupAnalyzer", rn_reg: RN_REG) -> None:
        if self.is_in_statement:
            self.grouping.add(rn_reg)


@dataclass
class RegisterParameterFinder(BLEIRListener):
    register_parameters: Set[RegisterParameter] = \
        field(default_factory=set)
    lowered_registers: Set[RegisterParameter] = \
        field(default_factory=set)

    def lowered_registers_by_type(self: "RegisterParameterFinder") \
            -> Dict[Type, Sequence[FormalParameter]]:
        # Sometimes a lowered register will appear twice because one instance of
        # it is flagged as a temporary and the other is not.
        lowered_registers = {reg.identifier: reg
                             for reg in self.lowered_registers}
        lowered_registers = sorted(lowered_registers.values(),
                                   key=lambda reg: reg.identifier)
        lowered_registers = groupby(lowered_registers, key=type)
        lowered_registers = {key: tuple(val) for key, val in lowered_registers}
        return lowered_registers

    def enter_register_parameter(self: "RegisterParameterFinder",
                                 register_parameter: RegisterParameter) -> None:
        if register_parameter.is_literal:  # do not record
            pass
        elif register_parameter.is_lowered:
            self.lowered_registers.add(register_parameter)
        else:
            self.register_parameters.add(register_parameter)

    def enter_rn_reg(self: "RegisterParameterFinder", rn_reg: RN_REG) -> None:
        self.enter_register_parameter(rn_reg)

    def enter_re_reg(self: "RegisterParameterFinder", re_reg: RE_REG) -> None:
        self.enter_register_parameter(re_reg)

    def enter_ewe_reg(self: "RegisterParameterFinder", ewe_reg: EWE_REG) -> None:
        self.enter_register_parameter(ewe_reg)

    def enter_l1_reg(self: "RegisterParameterFinder", l1_reg: L1_REG) -> None:
        self.enter_register_parameter(l1_reg)

    def enter_l2_reg(self: "RegisterParameterFinder", l2_reg: L2_REG) -> None:
        self.enter_register_parameter(l2_reg)

    def enter_sm_reg(self: "RegisterParameterFinder", sm_reg: SM_REG) -> None:
        self.enter_register_parameter(sm_reg)

    def enter_allocated_register(self: "RegisterParameterFinder",
                                 allocated_register: AllocatedRegister) -> None:
        if allocated_register.isa(SM_REG):
            sm_reg = allocated_register.parameter
            sm_reg_val = allocated_register.register
            if sm_reg_val in GVML_SM_REG_VALS and not sm_reg.is_lowered:
                lowered_sm_reg = SM_REG(
                    identifier=GVML_NYMS_BY_SM_REG_VAL[sm_reg_val],
                    register=allocated_register.reg_id,
                    constant_value=GVML_VALUES_BY_SM_REG_VAL[sm_reg_val],
                    is_lowered=True)
                self.lowered_registers.add(lowered_sm_reg)


@dataclass
class RegisterScanner(BLEIRListener):
    regs_by_frag: Dict[str, Deque[RN_REG]] \
        = field(default_factory=dict)

    regs_by_stmt: Dict[MultiStatement, Set[RN_REG]] \
        = field(default_factory=dict)

    frag_regs: Optional[Deque[RN_REG]] = None
    stmt_regs: Optional[Set[RN_REG]] = None

    in_multi_statement: bool = False
    in_statement: bool = False

    max_rn_regs: int = NUM_RN_REGS

    @property
    def is_spillable(self: "RegisterScanner") -> bool:
        return self.frag_regs is not None

    def enter_fragment(self: "RegisterScanner", fragment: Fragment) -> None:
        if fragment.children is not None:
            return

        num_rn_regs = sum(isinstance(param, RN_REG)
                          for param in fragment.parameters)

        if num_rn_regs > self.max_rn_regs:
            self.frag_regs = deque()
            self.regs_by_frag[fragment.original_identifier] = self.frag_regs

    def exit_fragment(self: "RegisterScanner", fragment: Fragment) -> None:
        if fragment.children is None:
            self.frag_regs = None

    def enter_multi_statement(self: "RegisterScanner",
                              multi_statement: MultiStatement) -> None:
        if self.is_spillable:
            self.stmt_regs = set()
            self.regs_by_stmt[multi_statement] = self.stmt_regs
            self.in_multi_statement = True

    def exit_multi_statement(self: "RegisterScanner",
                             multi_statement: MultiStatement) -> None:
        self.stmt_regs = None
        self.in_multi_statement = False

    def enter_statement(self: "RegisterScanner", statement: STATEMENT) -> None:
        if self.is_spillable:
            self.in_statement = True

    def exit_statement(self: "RegisterScanner", statement: STATEMENT) -> None:
        self.in_statement = False

    def enter_rn_reg(self: "RegisterScanner", rn_reg: RN_REG) -> None:
        if not self.is_spillable:
            return
        if self.in_multi_statement:
            self.stmt_regs.add(rn_reg)
        if self.in_statement:
            self.frag_regs.append(rn_reg)


@dataclass
class RegisterCoOcurrenceScanner(BLEIRListener):

    in_statement: bool = False
    in_multi_statement: bool = False

    co_occurrences_by_reg_by_frag: Dict[str, Dict[RN_REG, Set[RN_REG]]] \
        = field(default_factory=dict)

    co_occurrences_by_reg: Optional[Dict[RN_REG, Set[RN_REG]]] = None

    co_occurrences: Optional[Set[RN_REG]] = None

    def enter_fragment(self: "RegisterCoOcurrenceScanner",
                       fragment: Fragment) -> None:
        if fragment.children is None:
            self.co_occurrences_by_reg = defaultdict(set)

    def exit_fragment(self: "RegisterCoOcurrenceScanner",
                      fragment: Fragment) -> None:
        if fragment.children is None:
            frag_id = fragment.original_identifier
            self.co_occurrences_by_reg_by_frag[frag_id] \
                = dict(self.co_occurrences_by_reg)
            self.co_occurrences_by_reg = None

    def map_co_ocurrences(self: "RegisterCoOcurrenceScanner") -> None:
        for rn_reg in self.co_occurrences:
            self.co_occurrences_by_reg[rn_reg].update(self.co_occurrences)
            self.co_occurrences_by_reg[rn_reg].remove(rn_reg)

    def enter_multi_statement(
            self: "RegisterCoOcurrenceScanner",
            multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True
        self.co_occurrences = set()

    def exit_multi_statement(
            self: "RegisterCoOcurrenceScanner",
            multi_statement: MultiStatement) -> None:
        self.in_multi_statement = False
        self.map_co_ocurrences()
        self.co_occurrences = None

    def enter_statement(self: "RegisterCoOcurrenceScanner",
                        statement: STATEMENT) -> None:
        self.in_statement = True
        if not self.in_multi_statement:
            self.co_occurrences = set()

    def exit_statement(self: "RegisterCoOcurrenceScanner",
                       statement: STATEMENT) -> None:

        self.in_statement = False

        if not self.in_multi_statement:
            self.map_co_ocurrences()
            self.co_occurrences = None

    def enter_rn_reg(self: "RegisterCoOcurrenceScanner",
                     rn_reg: RN_REG) -> None:
        if self.in_statement:
            self.co_occurrences.add(rn_reg)


@dataclass
class UserParameterScanner(BLEIRListener):

    parameters_by_type: Dict[Type, Set[Integer]] \
        = field(default_factory=lambda: defaultdict(set))

    def enter_value_parameter(
            self: "UserParameterScanner",
            value_parameter: ValueParameter) -> None:
        row_number = value_parameter.row_number
        self.parameters_by_type[RN_REG].add(row_number)

    def enter_fragment_caller_call(
            self: "UserParameterScanner",
            fragment_caller_call: FragmentCallerCall) -> None:

        fragment = fragment_caller_call.caller.fragment
        temporaries = set(fragment.temporaries)

        for formal_parameter, actual_parameter \
                in fragment_caller_call.parameter_map.items():

            if formal_parameter not in temporaries \
               and isinstance(actual_parameter, Integer.__args__):

                formal_type = formal_parameter.__class__
                self.parameters_by_type[formal_type].add(actual_parameter)


@dataclass
class RegisterGrouper(BLEIRListener):
    register_scanner: RegisterScanner
    register_co_ocurrence_scanner: RegisterCoOcurrenceScanner

    max_rn_regs: int = NUM_RN_REGS

    groups_by_frag: Dict[str, Sequence[Sequence[RN_REG]]] = field(default_factory=dict)

    @staticmethod
    def build_histogram(rn_regs: Sequence[RN_REG]) -> Dict[RN_REG, int]:
        hist = defaultdict(int)
        for rn_reg in rn_regs:
            hist[rn_reg] += 1
        return hist

    def group_by_least_usage(
        self: "RegisterGrouper",
        fragment: Fragment,
        hist: Dict[RN_REG, int]) \
            -> Sequence[Tuple[int, Sequence[RN_REG]]]:

        frag_id = fragment.original_identifier
        co_occurrences_by_reg = self.register_co_ocurrence_scanner \
                                    .co_occurrences_by_reg_by_frag[frag_id]

        co_occurrences_by_group = {}
        for rn_reg in hist.keys():
            group = (rn_reg,)
            co_occurrences_by_group[group] = co_occurrences_by_reg[rn_reg]

        groups = [(freq, (rn_reg,)) for rn_reg, freq in hist.items()]
        heapify(groups)

        num_groups = self.max_rn_regs
        while len(groups) > num_groups:
            lhs_freq, lhs_group = heappop(groups)
            lhs_co_occurrences = co_occurrences_by_group[lhs_group]
            incompatibilities = []

            # Do not group registers that co-occur in the same statement (they
            # will share the same MMB space):
            rhs_freq, rhs_group = heappop(groups)
            while len(lhs_co_occurrences & set(rhs_group)) > 0:
                if len(groups) == 0:
                    raise RuntimeError(
                        f"Failed to find compatible grouping of {num_groups} "
                        f"registers")
                incompatibilities.append((rhs_freq, rhs_group))
                rhs_freq, rhs_group = heappop(groups)

            group = lhs_group + rhs_group
            freq = lhs_freq + rhs_freq
            heappush(groups, (freq, group))

            rhs_co_occurrences = co_occurrences_by_group[rhs_group]
            group_co_occurrences = lhs_co_occurrences | rhs_co_occurrences
            co_occurrences_by_group[group] = group_co_occurrences

            for incompatibility in incompatibilities:
                heappush(groups, incompatibility)

        return groups

    def enter_fragment_caller(
            self: "RegisterGrouper",
            fragment_caller: FragmentCaller) -> None:

        # Scan for lowered parameters
        walker = BLEIRWalker()
        register_parameter_finder = RegisterParameterFinder()
        walker.walk(register_parameter_finder, fragment_caller)

        fragment = fragment_caller.fragment

        frag_id = fragment.identifier
        if frag_id in self.register_scanner.regs_by_frag:
            frag_regs = self.register_scanner.regs_by_frag[frag_id]
            histogram = self.build_histogram(frag_regs)
            groups = self.group_by_least_usage(fragment, histogram)
            groups = tuple(group for freq, group in groups)
            self.groups_by_frag[frag_id] = groups


@dataclass
class CoalesceGroupedRegisters(BLEIRListener):
    local_to_lowered_groups: Callable[[Fragment],
                                      Optional[Sequence[Sequence[RN_REG]]]]

    shared_registers_by_frag: Dict[str, Dict[RN_REG, RN_REG]] = \
        field(default_factory=dict)

    @staticmethod
    def find_temporary(group: Sequence[RN_REG]) -> Optional[RN_REG]:
        for rn_reg in sorted(group, key=lambda reg: reg.identifier):
            if rn_reg.is_lowered:
                return rn_reg
        return None

    def enter_fragment_caller(
            self: "CoalesceGroupedRegisters",
            fragment_caller: FragmentCaller) -> None:
        fragment = fragment_caller.fragment
        frag_id = fragment.original_identifier
        groups = self.local_to_lowered_groups(fragment)
        if groups is not None:
            shared_registers = {}
            for group in groups:
                temporary = self.find_temporary(group)
                if temporary is not None:
                    shared_register = temporary
                else:
                    shared_register = group[0]
                for rn_reg in group:
                    if rn_reg != shared_register:
                        LOGGER.debug(
                            "Coalescing register for %s with register for %s",
                            rn_reg.identifier,
                            shared_register.identifier)
                        shared_registers[rn_reg] = shared_register
            self.shared_registers_by_frag[frag_id] = shared_registers
        return fragment_caller


@dataclass
class UserRegisterScanner(BLEIRListener):
    local_to_lowered_groups: Callable[[Fragment],
                                      Optional[Sequence[Sequence[RN_REG]]]]
    shared_registers_by_frag: Dict[Type, Set[int]]

    registers_by_type: Dict[Type, Set[int]] \
        = field(default_factory=lambda: defaultdict(set))

    @staticmethod
    def has_temporary(group: Sequence[RN_REG]) -> bool:
        for rn_reg in group:
            if rn_reg.is_lowered:
                return True
        return False

    def enter_fragment_caller(
            self: "UserRegisterScanner",
            fragment_caller: FragmentCaller) -> None:

        grouped_with_temporaries = set()

        fragment = fragment_caller.fragment
        groups = self.local_to_lowered_groups(fragment)
        if groups is not None:
            for group in groups:
                if self.has_temporary(group):
                    for rn_reg in group:
                        grouped_with_temporaries.add(rn_reg.identifier)

        frag_id = fragment.original_identifier
        if frag_id in self.shared_registers_by_frag:
            shared_registers = self.shared_registers_by_frag[frag_id]
            for rn_reg in shared_registers.keys():
                grouped_with_temporaries.add(rn_reg.identifier)

        for formal_parameter, allocated_register in fragment_caller.register_map.items():
            if formal_parameter.identifier not in grouped_with_temporaries \
               and allocated_register.register != "<PLACEHOLDER>":
                formal_type = formal_parameter.__class__
                register = allocated_register.reg_id
                self.registers_by_type[formal_type].add(register)


def thunked(*kinds: Sequence[Type]) -> Callable[[], Any]:
    kind, *rest = kinds
    if len(rest) > 0:
        inner_thunk = thunked(*rest)
        outer_thunk = lambda: kind(inner_thunk)
        return outer_thunk
    thunk = kind
    return thunk


FragNym = str
VarNym = str
SectionLiteral = int
IsNegated = bool
OptNegSectMask = Tuple[VarNym, IsNegated]
Literal_or_Mask = Union[SectionLiteral, OptNegSectMask]
InstructionNum = int


@dataclass
class LiveSectionScanner(BLEIRVisitor):

    # Identifier x SectionNumber -> Sequence[InstructionNumber]
    # ---------------------------------------------------------
    # Specifies the instructions (numbers) at which each section for each
    # parameter in a fragment is written to or read from. This is an intermediate
    # step to computing all instructions over which the parameter is "alive".
    writes_by_section: Optional[Dict[VarNym,
                                     Dict[Literal_or_Mask,
                                          Sequence[InstructionNum]]]] = None
    reads_by_section: Optional[Dict[VarNym,
                                    Dict[Literal_or_Mask,
                                         Sequence[InstructionNum]]]] = None

    writes_by_section_by_frag: Dict[FragNym,
                                    Dict[VarNym,
                                         Dict[Literal_or_Mask,
                                              Sequence[InstructionNum]]]] = \
        field(default_factory=dict)

    reads_by_section_by_frag: Dict[FragNym,
                                   Dict[VarNym,
                                        Dict[Literal_or_Mask,
                                             Sequence[InstructionNum]]]] = \
        field(default_factory=dict)

    # Fragment x Parameter x SectionNumber -> Set[InstructionNumber]
    # --------------------------------------------------------------
    # Contains a set of all instructions (numbers) over which a parameter in a
    # fragment is "alive"
    live_sections_by_param_by_frag: Dict[str, Dict[str, Dict[int, Set[int]]]] = \
        field(default_factory=thunked(defaultdict, defaultdict, defaultdict, set))

    # Fragment x Parameter -> Set[SectionNumber]
    # ------------------------------------------
    # Contains a set of all sections used, including parameterized section masks,
    # for each parameter in a fragment.
    sections_by_vr_by_frag: Dict[str, Dict[str, Set[Union[int, Tuple[str, bool]]]]] = \
        field(default_factory=thunked(defaultdict, defaultdict, set))

    in_multi_statement: bool = False
    instruction_number: int = 0
    sections: Optional[Union[Sequence[int], Tuple[Tuple[str, bool]]]] = None

    def visit_fragment(self: "LiveSectionScanner", fragment: Fragment) -> None:
        self.writes_by_section = defaultdict(lambda: defaultdict(deque))
        self.reads_by_section = defaultdict(lambda: defaultdict(deque))

        for operation_or_comment in fragment.operations:
            if isinstance(operation_or_comment, MultiStatement):
                multi_statement = operation_or_comment
                self.visit_multi_statement(multi_statement)
            elif isinstance(operation_or_comment, STATEMENT):
                statement = operation_or_comment
                self.visit_statement(statement)

        frag_id = fragment.original_identifier

        self.writes_by_section_by_frag[frag_id] = deepcopy(self.writes_by_section)
        self.reads_by_section_by_frag[frag_id] = deepcopy(self.reads_by_section)

        # var_nym -> section -> set[instr_num]
        live_sections_by_param = self.live_sections_by_param_by_frag[frag_id]
        sections_by_vr = self.sections_by_vr_by_frag[frag_id]
        for var_nym, var_writes in self.writes_by_section.items():
            var_reads = self.reads_by_section[var_nym]
            var_liveness = live_sections_by_param[var_nym]

            vr_sections = set(chain(var_writes.keys(), var_reads.keys()))
            sections_by_vr[var_nym] = vr_sections

            # FIXME: If there is a literal write with 0xFFFF, it should clear all
            # parameterized writes above its location. It's okay without this fix
            # because coalescing will be more conservative.

            for section in vr_sections:
                # If there is a parameterized read after a literal write,
                # consider the parameterized read a read for the literal in the
                # write.
                if isinstance(section, int):
                    literal_reads = var_reads[section]
                    literal_writes = var_writes[section]
                    if len(literal_writes) == 0:
                        continue
                    min_write = literal_writes[0]
                    for other_section in (vr_sections - {section}):
                        if isinstance(other_section, int):
                            continue
                        other_reads = var_reads[other_section]
                        literal_reads.extend(
                            filter(lambda read: read >= min_write,
                                   other_reads))

                # If there is a parameterized write, consider all the reads
                # following it, for all section masks, reads from it.
                else:
                    parameterized_reads = var_reads[section]
                    for other_section in (vr_sections - {section}):
                        other_writes = var_writes[other_section]
                        other_reads = var_reads[other_section]
                        if len(other_writes) == 0:
                            continue
                        min_write = other_writes[0]
                        other_reads.extend(
                            filter(lambda read: read >= min_write,
                                   parameterized_reads))

            # Expand the instruction numbers over which each section of the
            # parameter is alive.
            for section, sec_writes in var_writes.items():
                if isinstance(section, int):
                    sec_livenesses = [var_liveness[section]]
                else:
                    # Consider parameterized section masks alive over all
                    # sections for the remaining life of the temporary
                    sec_livenesses = [var_liveness[i]
                                      for i in range(NSECTIONS)]

                sec_reads = deque(sorted(set(var_reads[section])))

                curr_write = None
                while len(sec_writes) > 0:
                    curr_write = sec_writes.popleft()
                    for sec_liveness in sec_livenesses:
                        sec_liveness.add(curr_write)

                    if len(sec_writes) > 0:
                        next_write = sec_writes[0]
                    else:
                        next_write = None

                    while len(sec_reads) > 0:
                        curr_read = sec_reads[0]

                        if next_write is not None \
                           and curr_read > next_write \
                           and isinstance(section, int):
                            break

                        sec_reads.popleft()
                        for sec_liveness in sec_livenesses:
                            sec_liveness.update(range(curr_write, curr_read + 1))
                        curr_write = curr_read

        self.writes_by_section = None
        self.reads_by_section = None

    def visit_multi_statement(self: "LiveSectionScanner",
                              multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True
        self.instruction_number += 1
        for statement_or_comment in multi_statement:
            if isinstance(statement_or_comment, STATEMENT):
                statement = statement_or_comment
                self.visit_statement(statement)
        self.in_multi_statement = False

    def visit_statement(self: "LiveSectionScanner",
                        statement: STATEMENT) -> None:
        if not self.in_multi_statement:
            self.instruction_number += 1
        if isinstance(statement.operation, MASKED):
            masked = statement.operation
            self.visit_masked(masked)

    def visit_masked(self: "LiveSectionScanner", masked: MASKED) -> None:
        sections = self.visit_mask(masked.mask)

        if len(sections) == 0:
            return

        if len(sections) > 0 and isinstance(sections[0], str):
            identifier, negated, num_shifted_bits = sections
            self.sections = [(f"{identifier}<<{num_shifted_bits}", negated)]

        else:
            self.sections = sections

        self.visit_mask(masked.mask)

        assignment = masked.assignment
        if assignment is not None:
            operation = assignment.operation
            if isinstance(operation, READ):
                read = operation
                self.visit_read(read)
            elif isinstance(operation, WRITE):
                write = operation
                self.visit_write(write)

        self.sections = None

    def visit_mask(self: "LiveSectionScanner", mask: MASK) \
            -> Optional[Union[Sequence[int], Tuple[str, bool, int]]]:

        if isinstance(mask.expression, SHIFTED_SM_REG):
            shifted_sm_reg = mask.expression
            sections = self.visit_shifted_sm_reg(shifted_sm_reg)

        elif isinstance(mask.expression, SM_REG):
            sm_reg = mask.expression
            sections = self.visit_sm_reg(sm_reg)

        else:
            raise ValueError(
                f"Unsupported expression type "
                f"({mask.expression.__class__.__name__}): {mask.expression}")

        if mask.operator is UNARY_OP.NEGATE:
            if len(sections) > 0 and isinstance(sections[0], str):
                identifier, _, num_shifted_bits = sections
                return identifier, True, num_shifted_bits
            else:
                sections = set(sections)
                all_sections = set(range(NSECTIONS))
                sections = sorted(all_sections - sections)

        return sections

    def visit_shifted_sm_reg(self: "LiveSectionScanner",
                             shifted_sm_reg: SHIFTED_SM_REG) \
            -> Optional[Union[Sequence[int], Tuple[str, bool, int]]]:

        sections = self.visit_sm_reg(shifted_sm_reg.register)
        num_shifted_bits = shifted_sm_reg.num_bits

        if len(sections) > 0 and isinstance(sections[0], str):
            identifier, negated, _ = sections
            return identifier, negated, num_shifted_bits

        sections = [(section + num_shifted_bits) for section in sections
                    if (section + num_shifted_bits) < NSECTIONS]

        return sections

    def visit_sm_reg(self: "LiveSectionScanner", sm_reg: SM_REG) \
            -> Optional[Union[Sequence[int], Tuple[str, bool, int]]]:

        if sm_reg.constant_value is not None:
            section_mask = sm_reg.constant_value
            if sm_reg.is_section:
                num_shifted_bits = section_mask
                section_mask = (0x0001 << num_shifted_bits)
            sections = [section for section in range(NSECTIONS)
                        if (section_mask & (0x0001 << section)) > 0]

            return sections

        return (sm_reg.identifier, sm_reg.negated, 0)

    def visit_read(self: "LiveSectionScanner", read: READ) -> None:
        if self.sections is not None:
            for operand in read.right_operands:
                for section in self.sections:
                    self.reads_by_section[operand][section].append(
                        self.instruction_number)

    def visit_write(self: "LiveSectionScanner", write: WRITE) -> None:
        if self.sections is not None:
            for operand in write.left_operands:
                for section in self.sections:
                    self.writes_by_section[operand][section].append(
                        self.instruction_number)


@dataclass
class CountNumInstructionsAndCommands(BLEIRListener):
    num_fragment_instructions_analyzer: NumFragmentInstructionsAnalyzer

    num_instructions_and_commands_by_snippet: Dict[str, Tuple[int, int]] = \
        field(default_factory=dict)

    def enter_snippet(self: "CountNumInstructionsAndCommands",
                      snippet: Snippet) -> None:

        total_num_instructions = 0
        total_num_commands = 0

        num_instructions_by_fragment \
            = self.num_fragment_instructions_analyzer \
                  .num_instructions_by_fragment

        num_commands_by_fragment \
            = self.num_fragment_instructions_analyzer \
                  .num_commands_by_fragment

        for fragment_caller_call in snippet.fragment_caller_calls:
            fragment = fragment_caller_call.fragment
            if fragment.children is not None:
                for child in fragment.children:
                    num_instructions = \
                        num_instructions_by_fragment[child.identifier]
                    num_commands = \
                        num_commands_by_fragment[child.identifier]
                    total_num_instructions += num_instructions
                    total_num_commands += num_commands
            else:
                num_instructions = \
                    num_instructions_by_fragment[fragment.identifier]
                num_commands = \
                    num_commands_by_fragment[fragment.identifier]
                total_num_instructions += num_instructions
                total_num_commands += num_commands

        self.num_instructions_and_commands_by_snippet[snippet.name] = \
            (total_num_instructions, total_num_commands)

    def __call__(self: "CountNumInstructionsAndCommands",
                 snippet: Snippet) -> Tuple[int, int]:
        total_num_instructions, total_num_commands = \
            self.num_instructions_and_commands_by_snippet[snippet.name]
        return total_num_instructions, total_num_commands


def get_or_set_command_declaration(fn: Callable) -> Callable:

    @wraps(fn)
    def wrapper(self: "CommandDeclarationScanner",
                command: Command) -> Optional[Tuple[str, str]]:
        if command in self.command_declarations:
            declaration = self.command_declarations[command]
        else:
            declaration = fn(self, command)
            if declaration is not None:
                self.command_declarations[command] = declaration
        return declaration

    return wrapper


@dataclass
class CommandDeclarationScanner:
    command_declarations: Dict[Command, Tuple[str, str]] = \
        field(default_factory=dict)

    nums_by_kind: Dict[str, int] = \
        field(default_factory=lambda: defaultdict(int))

    def next_kind_nym(self: "CommandDeclarationScanner",
                      kind: str) -> Tuple[str, str]:
        self.nums_by_kind[kind] += 1
        kind_num = self.nums_by_kind[kind]
        kind_nym = f"__baryon_{kind}_{kind_num:04d}"
        return kind, kind_nym

    def visit_commands(self: "CommandDeclarationScanner",
                       commands: Sequence[Command]) -> None:
        for command in commands:
            self.visit_command(command)

    def visit_command(self: "CommandDeclarationScanner",
                      command: Command) -> None:
        underscore_nym = camel_case_to_underscore(command.__class__)
        visit_nym = f"visit_{underscore_nym}"
        if hasattr(self, visit_nym):
            visit_fn = getattr(self, visit_nym)
            visit_fn(command)

    def visit_apply_patch(
            self: "CommandDeclarationScanner",
            apply_patch: ApplyPatch) -> Tuple[str, str]:
        self.visit_command(apply_patch.mimic)

    @get_or_set_command_declaration
    def visit_mimic_rsp16_from_rsp256(
            self: "CommandDeclarationScanner",
            mimic_rsp16_from_rsp256: MimicRSP16FromRSP256) -> Tuple[str, str]:
        return self.next_kind_nym("rsp16_from_rsp256")

    @get_or_set_command_declaration
    def visit_mimic_rsp256_from_rsp16(
            self: "CommandDeclarationScanner",
            mimic_rsp256_from_rsp16: MimicRSP256FromRSP16) -> Tuple[str, str]:
        return self.next_kind_nym("rsp256_from_rsp16")

    @get_or_set_command_declaration
    def visit_mimic_rsp256_from_rsp2k(
            self: "CommandDeclarationScanner",
            mimic_rsp256_from_rsp2k: MimicRSP256FromRSP2K) -> Tuple[str, str]:
        return self.next_kind_nym("rsp256_from_rsp2k")

    @get_or_set_command_declaration
    def visit_mimic_rsp2k_from_rsp256(
            self: "CommandDeclarationScanner",
            mimic_rsp2k_from_rsp256: MimicRSP2KFromRSP256) -> Tuple[str, str]:
        return self.next_kind_nym("rsp2k_from_rsp256")

    @get_or_set_command_declaration
    def visit_mimic_rsp2k_from_rsp32k(
            self: "CommandDeclarationScanner",
            mimic_rsp2k_from_rsp32k: MimicRSP2KFromRSP32K) -> Tuple[str, str]:
        return self.next_kind_nym("rsp2k_from_rsp32k")

    @get_or_set_command_declaration
    def visit_mimic_rsp32k_from_rsp2k(
            self: "CommandDeclarationScanner",
            mimic_rsp32k_from_rsp2k: MimicRSP32KFromRSP2K) -> Tuple[str, str]:
        return self.next_kind_nym("rsp32k_from_rsp2k")

    @get_or_set_command_declaration
    def visit_mimic_no_op(
            self: "CommandDeclarationScanner",
            mimic_no_op: MimicNoOp) -> Tuple[str, str]:
        return self.next_kind_nym("no_op")

    @get_or_set_command_declaration
    def visit_mimic_f_sel_no_op(
            self: "CommandDeclarationScanner",
            mimic_f_sel_no_op: MimicFSelNoOp) -> Tuple[str, str]:
        return self.next_kind_nym("f_sel_no_op")

    @get_or_set_command_declaration
    def visit_mimic_rsp_end(
            self: "CommandDeclarationScanner",
            mimic_rsp_end: MimicRSPEnd) -> Tuple[str, str]:
        return self.next_kind_nym("rsp_end")

    @get_or_set_command_declaration
    def visit_mimic_rsp_start_ret(
            self: "CommandDeclarationScanner",
            mimic_rsp_start_ret: MimicRSPStartRet) -> Tuple[str, str]:
        return self.next_kind_nym("rsp_start_ret")

    @get_or_set_command_declaration
    def visit_mimic_l2_end(
            self: "CommandDeclarationScanner",
            mimic_l2_end: MimicL2End) -> Tuple[str, str]:
        return self.next_kind_nym("l2_end")

    @get_or_set_command_declaration
    def visit_mimic_sb_from_src(
            self: "CommandDeclarationScanner",
            mimic_sb_from_src: MimicSBFromSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_sb_from_src.mask)
        self.visit_load_registers(mimic_sb_from_src.vrs)
        self.visit_load_src(mimic_sb_from_src.src)
        src, _ = self.command_declarations[mimic_sb_from_src.src]
        return self.next_kind_nym(f"sb_from_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_sb_from_inv_src(
            self: "CommandDeclarationScanner",
            mimic_sb_from_inv_src: MimicSBFromInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_sb_from_inv_src.mask)
        self.visit_load_registers(mimic_sb_from_inv_src.vrs)
        self.visit_load_src(mimic_sb_from_inv_src.src)
        src, _ = self.command_declarations[mimic_sb_from_inv_src.src]
        return self.next_kind_nym(f"sb_from_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_sb_cond_eq_src(
            self: "CommandDeclarationScanner",
            mimic_sb_cond_eq_src: MimicSBCondEqSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_sb_cond_eq_src.mask)
        self.visit_load_registers(mimic_sb_cond_eq_src.vrs)
        self.visit_load_src(mimic_sb_cond_eq_src.src)
        src, _ = self.command_declarations[mimic_sb_cond_eq_src.src]
        return self.next_kind_nym(f"sb_cond_eq_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_sb_cond_eq_inv_src(
            self: "CommandDeclarationScanner",
            mimic_sb_cond_eq_inv_src: MimicSBCondEqInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_sb_cond_eq_inv_src.mask)
        self.visit_load_registers(mimic_sb_cond_eq_inv_src.vrs)
        self.visit_load_src(mimic_sb_cond_eq_inv_src.src)
        src, _ = self.command_declarations[mimic_sb_cond_eq_inv_src.src]
        return self.next_kind_nym(f"sb_cond_eq_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_set_rl(
            self: "CommandDeclarationScanner",
            mimic_set_rl: MimicSetRL) -> Tuple[str, str]:
        self.visit_load_register(mimic_set_rl.mask)
        return self.next_kind_nym("set_rl")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_src: MimicRLFromSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_src.mask)
        self.visit_load_src(mimic_rl_from_src.src)
        src, _ = self.command_declarations[mimic_rl_from_src.src]
        return self.next_kind_nym(f"rl_from_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_inv_src: MimicRLFromInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_inv_src.mask)
        self.visit_load_src(mimic_rl_from_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_from_inv_src.src]
        return self.next_kind_nym(f"rl_from_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_or_eq_src(
            self: "CommandDeclarationScanner",
            mimic_rl_or_eq_src: MimicRLOrEqSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_or_eq_src.mask)
        self.visit_load_src(mimic_rl_or_eq_src.src)
        src, _ = self.command_declarations[mimic_rl_or_eq_src.src]
        return self.next_kind_nym(f"rl_or_eq_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_or_eq_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_or_eq_inv_src: MimicRLOrEqInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_or_eq_inv_src.mask)
        self.visit_load_src(mimic_rl_or_eq_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_or_eq_inv_src.src]
        return self.next_kind_nym(f"rl_or_eq_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_and_eq_src(
            self: "CommandDeclarationScanner",
            mimic_rl_and_eq_src: MimicRLAndEqSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_and_eq_src.mask)
        self.visit_load_src(mimic_rl_and_eq_src.src)
        src, _ = self.command_declarations[mimic_rl_and_eq_src.src]
        return self.next_kind_nym(f"rl_and_eq_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_xor_eq_src(
            self: "CommandDeclarationScanner",
            mimic_rl_xor_eq_src: MimicRLXorEqSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_xor_eq_src.mask)
        self.visit_load_src(mimic_rl_xor_eq_src.src)
        src, _ = self.command_declarations[mimic_rl_xor_eq_src.src]
        return self.next_kind_nym(f"rl_xor_eq_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_xor_eq_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_xor_eq_inv_src: MimicRLXorEqInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_xor_eq_inv_src.mask)
        self.visit_load_src(mimic_rl_xor_eq_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_xor_eq_inv_src.src]
        return self.next_kind_nym(f"rl_xor_eq_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_sb(
            self: "CommandDeclarationScanner",
            mimic_rl_from_sb: MimicRLFromSB) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_sb.mask)
        self.visit_load_registers(mimic_rl_from_sb.vrs)
        return self.next_kind_nym("rl_from_sb")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_inv_sb(
            self: "CommandDeclarationScanner",
            mimic_rl_from_inv_sb: MimicRLFromInvSB) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_inv_sb.mask)
        self.visit_load_registers(mimic_rl_from_inv_sb.vrs)
        return self.next_kind_nym("rl_from_inv_sb")

    @get_or_set_command_declaration
    def visit_mimic_rl_or_eq_sb(
            self: "CommandDeclarationScanner",
            mimic_rl_or_eq_sb: MimicRLOrEqSB) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_or_eq_sb.mask)
        self.visit_load_registers(mimic_rl_or_eq_sb.vrs)
        return self.next_kind_nym("rl_or_eq_sb")

    @get_or_set_command_declaration
    def visit_mimic_rl_and_eq_sb(
            self: "CommandDeclarationScanner",
            mimic_rl_and_eq_sb: MimicRLAndEqSB) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_and_eq_sb.mask)
        self.visit_load_registers(mimic_rl_and_eq_sb.vrs)
        return self.next_kind_nym("rl_and_eq_sb")

    @get_or_set_command_declaration
    def visit_mimic_rl_and_eq_inv_sb(
            self: "CommandDeclarationScanner",
            mimic_rl_and_eq_inv_sb: MimicRLAndEqInvSB) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_and_eq_inv_sb.mask)
        self.visit_load_registers(mimic_rl_and_eq_inv_sb.vrs)
        return self.next_kind_nym("rl_and_eq_inv_sb")

    @get_or_set_command_declaration
    def visit_mimic_rl_xor_eq_sb(
            self: "CommandDeclarationScanner",
            mimic_rl_xor_eq_sb: MimicRLXorEqSB) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_xor_eq_sb.mask)
        self.visit_load_registers(mimic_rl_xor_eq_sb.vrs)
        return self.next_kind_nym("rl_xor_eq_sb")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_sb_and_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_sb_and_src: MimicRLFromSBAndSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_sb_and_src.mask)
        self.visit_load_registers(mimic_rl_from_sb_and_src.vrs)
        self.visit_load_src(mimic_rl_from_sb_and_src.src)
        src, _ = self.command_declarations[mimic_rl_from_sb_and_src.src]
        return self.next_kind_nym(f"rl_from_sb_and_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_or_eq_sb_and_src(
            self: "CommandDeclarationScanner",
            mimic_rl_or_eq_sb_and_src: MimicRLOrEqSBAndSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_or_eq_sb_and_src.mask)
        self.visit_load_registers(mimic_rl_or_eq_sb_and_src.vrs)
        self.visit_load_src(mimic_rl_or_eq_sb_and_src.src)
        src, _ = self.command_declarations[mimic_rl_or_eq_sb_and_src.src]
        return self.next_kind_nym(f"rl_or_eq_sb_and_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_or_eq_sb_and_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_or_eq_sb_and_inv_src: MimicRLOrEqSBAndInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_or_eq_sb_and_inv_src.mask)
        self.visit_load_registers(mimic_rl_or_eq_sb_and_inv_src.vrs)
        self.visit_load_src(mimic_rl_or_eq_sb_and_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_or_eq_sb_and_inv_src.src]
        return self.next_kind_nym(f"rl_or_eq_sb_and_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_and_eq_sb_and_src(
            self: "CommandDeclarationScanner",
            mimic_rl_and_eq_sb_and_src: MimicRLAndEqSBAndSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_and_eq_sb_and_src.mask)
        self.visit_load_registers(mimic_rl_and_eq_sb_and_src.vrs)
        self.visit_load_src(mimic_rl_and_eq_sb_and_src.src)
        src, _ = self.command_declarations[mimic_rl_and_eq_sb_and_src.src]
        return self.next_kind_nym(f"rl_and_eq_sb_and_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_and_eq_sb_and_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_and_eq_sb_and_inv_src: MimicRLAndEqSBAndInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_and_eq_sb_and_inv_src.mask)
        self.visit_load_registers(mimic_rl_and_eq_sb_and_inv_src.vrs)
        self.visit_load_src(mimic_rl_and_eq_sb_and_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_and_eq_sb_and_inv_src.src]
        return self.next_kind_nym(f"rl_and_eq_sb_and_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_xor_eq_sb_and_src(
            self: "CommandDeclarationScanner",
            mimic_rl_xor_eq_sb_and_src: MimicRLXorEqSBAndSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_xor_eq_sb_and_src.mask)
        self.visit_load_registers(mimic_rl_xor_eq_sb_and_src.vrs)
        self.visit_load_src(mimic_rl_xor_eq_sb_and_src.src)
        src, _ = self.command_declarations[mimic_rl_xor_eq_sb_and_src.src]
        return self.next_kind_nym(f"rl_xor_eq_sb_and_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_xor_eq_sb_and_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_xor_eq_sb_and_inv_src: MimicRLXorEqSBAndInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_xor_eq_sb_and_inv_src.mask)
        self.visit_load_registers(mimic_rl_xor_eq_sb_and_inv_src.vrs)
        self.visit_load_src(mimic_rl_xor_eq_sb_and_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_xor_eq_sb_and_inv_src.src]
        return self.next_kind_nym(f"rl_xor_eq_sb_and_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_sb_or_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_sb_or_src: MimicRLFromSBOrSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_sb_or_src.mask)
        self.visit_load_registers(mimic_rl_from_sb_or_src.vrs)
        self.visit_load_src(mimic_rl_from_sb_or_src.src)
        src, _ = self.command_declarations[mimic_rl_from_sb_or_src.src]
        return self.next_kind_nym(f"rl_from_sb_or_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_sb_or_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_sb_or_inv_src: MimicRLFromSBOrInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_sb_or_inv_src.mask)
        self.visit_load_registers(mimic_rl_from_sb_or_inv_src.vrs)
        self.visit_load_src(mimic_rl_from_sb_or_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_from_sb_or_inv_src.src]
        return self.next_kind_nym(f"rl_from_sb_or_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_sb_xor_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_sb_xor_src: MimicRLFromSBXorSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_sb_xor_src.mask)
        self.visit_load_registers(mimic_rl_from_sb_xor_src.vrs)
        self.visit_load_src(mimic_rl_from_sb_xor_src.src)
        src, _ = self.command_declarations[mimic_rl_from_sb_xor_src.src]
        return self.next_kind_nym(f"rl_from_sb_xor_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_sb_xor_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_sb_xor_inv_src: MimicRLFromSBXorInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_sb_xor_inv_src.mask)
        self.visit_load_registers(mimic_rl_from_sb_xor_inv_src.vrs)
        self.visit_load_src(mimic_rl_from_sb_xor_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_from_sb_xor_inv_src.src]
        return self.next_kind_nym(f"rl_from_sb_xor_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_inv_sb_and_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_inv_sb_and_src: MimicRLFromInvSBAndSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_inv_sb_and_src.mask)
        self.visit_load_registers(mimic_rl_from_inv_sb_and_src.vrs)
        self.visit_load_src(mimic_rl_from_inv_sb_and_src.src)
        src, _ = self.command_declarations[mimic_rl_from_inv_sb_and_src.src]
        return self.next_kind_nym(f"rl_from_inv_sb_and_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_inv_sb_and_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_inv_sb_and_inv_src: MimicRLFromInvSBAndInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_inv_sb_and_inv_src.mask)
        self.visit_load_registers(mimic_rl_from_inv_sb_and_inv_src.vrs)
        self.visit_load_src(mimic_rl_from_inv_sb_and_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_from_inv_sb_and_inv_src.src]
        return self.next_kind_nym(f"rl_from_inv_sb_and_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rl_from_sb_and_inv_src(
            self: "CommandDeclarationScanner",
            mimic_rl_from_sb_and_inv_src: MimicRLFromSBAndInvSrc) -> Tuple[str, str]:
        self.visit_load_register(mimic_rl_from_sb_and_inv_src.mask)
        self.visit_load_registers(mimic_rl_from_sb_and_inv_src.vrs)
        self.visit_load_src(mimic_rl_from_sb_and_inv_src.src)
        src, _ = self.command_declarations[mimic_rl_from_sb_and_inv_src.src]
        return self.next_kind_nym(f"rl_from_sb_and_inv_src_{src}")

    @get_or_set_command_declaration
    def visit_mimic_rsp16_from_rl(
            self: "CommandDeclarationScanner",
            mimic_rsp16_from_rl: MimicRSP16FromRL) -> Tuple[str, str]:
        if isinstance(mimic_rsp16_from_rl.mask, LoadRegister):
            self.visit_load_register(mimic_rsp16_from_rl.mask)
        elif isinstance(mimic_rsp16_from_rl.mask, UnifySMRegs):
            self.visit_unify_sm_regs(mimic_rsp16_from_rl.mask)
        else:
            raise ValueError(
                f"Unsupported mask type "
                f"({mimic_rsp16_from_rl.mask.__class__.__name__}): "
                f"{mimic_rsp16_from_rl.mask}")
        return self.next_kind_nym("rsp16_from_rl")

    @get_or_set_command_declaration
    def visit_mimic_gl_from_rl(
            self: "CommandDeclarationScanner",
            mimic_gl_from_rl: MimicGLFromRL) -> Tuple[str, str]:
        if isinstance(mimic_gl_from_rl.mask, LoadRegister):
            self.visit_load_register(mimic_gl_from_rl.mask)
        elif isinstance(mimic_gl_from_rl.mask, UnifySMRegs):
            self.visit_unify_sm_regs(mimic_gl_from_rl.mask)
        else:
            raise ValueError(
                f"Unsupported mask type "
                f"({mimic_gl_from_rl.mask.__class__.__name__}): "
                f"{mimic_gl_from_rl.mask}")
        return self.next_kind_nym("gl_from_rl")

    @get_or_set_command_declaration
    def visit_mimic_ggl_from_rl(
            self: "CommandDeclarationScanner",
            mimic_ggl_from_rl: MimicGGLFromRL) -> Tuple[str, str]:
        if isinstance(mimic_ggl_from_rl.mask, LoadRegister):
            self.visit_load_register(mimic_ggl_from_rl.mask)
        elif isinstance(mimic_ggl_from_rl.mask, UnifySMRegs):
            self.visit_unify_sm_regs(mimic_ggl_from_rl.mask)
        else:
            raise ValueError(
                f"Unsupported mask type "
                f"({mimic_ggl_from_rl.mask.__class__.__name__}): "
                f"{mimic_ggl_from_rl.mask}")
        return self.next_kind_nym("ggl_from_rl")

    @get_or_set_command_declaration
    def visit_mimic_l1_from_ggl(
            self: "CommandDeclarationScanner",
            mimic_l1_from_ggl: MimicL1FromGGL) -> Tuple[str, str]:
        self.visit_load_register(mimic_l1_from_ggl.l1_addr)
        return self.next_kind_nym("l1_from_ggl")

    @get_or_set_command_declaration
    def visit_mimic_lgl_from_l1(
            self: "CommandDeclarationScanner",
            mimic_lgl_from_l1: MimicLGLFromL1) -> Tuple[str, str]:
        self.visit_load_register(mimic_lgl_from_l1.l1_addr)
        return self.next_kind_nym("lgl_from_l1")

    @get_or_set_command_declaration
    def visit_mimic_l2_from_lgl(
            self: "CommandDeclarationScanner",
            mimic_l2_from_lgl: MimicL2FromLGL) -> Tuple[str, str]:
        self.visit_load_register(mimic_l2_from_lgl.l2_addr)
        return self.next_kind_nym("l2_from_lgl")

    @get_or_set_command_declaration
    def visit_mimic_lgl_from_l2(
            self: "CommandDeclarationScanner",
            mimic_lgl_from_l2: MimicLGLFromL2) -> Tuple[str, str]:
        self.visit_load_register(mimic_lgl_from_l2.l2_addr)
        return self.next_kind_nym("lgl_from_l2")

    @get_or_set_command_declaration
    def visit_mimic_l1_from_lgl(
            self: "CommandDeclarationScanner",
            mimic_l1_from_lgl: MimicL1FromLGL) -> Tuple[str, str]:
        self.visit_load_register(mimic_l1_from_lgl.l1_addr)
        return self.next_kind_nym("l1_from_lgl")

    @get_or_set_command_declaration
    def visit_mimic_ggl_from_l1(
            self: "CommandDeclarationScanner",
            mimic_ggl_from_l1: MimicGGLFromL1) -> Tuple[str, str]:
        self.visit_load_register(mimic_ggl_from_l1.l1_addr)
        return self.next_kind_nym("ggl_from_l1")

    @get_or_set_command_declaration
    def visit_mimic_ggl_from_rl_and_l1(
            self: "CommandDeclarationScanner",
            mimic_ggl_from_rl_and_l1: MimicGGLFromRLAndL1) -> Tuple[str, str]:
        if isinstance(mimic_ggl_from_rl_and_l1.mask, LoadRegister):
            self.visit_load_register(mimic_ggl_from_rl_and_l1.mask)
        elif isinstance(mimic_ggl_from_rl_and_l1.mask, UnifySMRegs):
            self.visit_unify_sm_regs(mimic_ggl_from_rl_and_l1.mask)
        else:
            raise ValueError(
                f"Unsupported mask type "
                f"({mimic_ggl_from_rl_and_l1.mask.__class__.__name__}): "
                f"{mimic_ggl_from_rl_and_l1.mask}")
        self.visit_load_register(mimic_ggl_from_rl_and_l1.l1_addr)
        return self.next_kind_nym("ggl_from_rl_and_l1")

    @get_or_set_command_declaration
    def visit_mimic_rw_inh_set(
            self: "CommandDeclarationScanner",
            mimic_rw_inh_set: MimicRWInhSet) -> Tuple[str, str]:
        self.visit_load_register(mimic_rw_inh_set.mask)
        return self.next_kind_nym("rw_inh_set")

    @get_or_set_command_declaration
    def visit_mimic_rw_inh_rst(
            self: "CommandDeclarationScanner",
            mimic_rw_inh_rst: MimicRWInhRst) -> Tuple[str, str]:
        self.visit_load_register(mimic_rw_inh_rst.mask)
        return self.next_kind_nym("rw_inh_rst")

    @get_or_set_command_declaration
    def visit_unify_sm_regs(self: "CommandDeclarationScanner",
                            unify_sm_regs: UnifySMRegs) -> Tuple[str, str]:
        nyms = []
        for load_register in unify_sm_regs:
            _, nym = self.visit_load_register(load_register)
            nyms.append(nym[len("__baryon_sm_"):])
        nyms = "_".join(nyms)
        return "sm", f"__baryon_sm_{nyms}"

    @get_or_set_command_declaration
    def visit_load_registers(
            self: "CommandDeclarationScanner",
            load_registers: LoadRegisters) -> Optional[Tuple[str, str]]:
        for load_register in load_registers:
            self.visit_load_register(load_register)
        return None

    @get_or_set_command_declaration
    def visit_load_register(self: "CommandDeclarationScanner",
                            load_register: LoadRegister) -> Tuple[str, str]:
        kind = load_register.kind.name[:-len("_REG")].lower()
        return self.next_kind_nym(kind)

    @get_or_set_command_declaration
    def visit_load_src(self: "CommandDeclarationScanner",
                       load_src: LoadSrc) -> Optional[Tuple[str, str]]:
        src = load_src.src.name.lower()
        return self.next_kind_nym(src)


@dataclass
class DoubleNegativeAnalyzer(BLEIRVisitor):
    negate_mask: bool = False

    in_multi_statement: bool = False
    in_read: bool = False
    in_write: bool = False

    secs: Optional[Sequence[int]] = None

    read_secs: Optional[Set[int]] = None
    write_secs: Optional[Set[int]] = None

    resolve_multi_statement: bool = True

    def visit_multi_statement(self: "DoubleNegativeAnalyzer",
                              multi_statement: MultiStatement) -> None:

        self.in_multi_statement = True
        self.read_secs = set()
        self.write_secs = set()

        for statement_or_comment in multi_statement:
            if isinstance(statement_or_comment, STATEMENT):
                statement = statement_or_comment
                self.visit_statement(statement)

        # NOTE: If there is an intersection over READs and WRITEs in the same
        # MultiStatement, then do not resolve double negatives over the SRCs in
        # the MultiStatement. An overlap indicates the SRCs must be identical
        # (including any shifts or inversions, e.g. INV_GL).
        shared_secs = self.read_secs & self.write_secs
        self.resolve_multi_statement = (len(shared_secs) == 0)

        self.write_secs = None
        self.read_secs = None
        self.in_multi_statement = False

    def visit_statement(self: "DoubleNegativeAnalyzer",
                        statement: STATEMENT) -> None:
        operation = statement.operation
        if isinstance(operation, MASKED):
            self.visit_masked(operation)

    def visit_masked(self: "DoubleNegativeAnalyzer", masked: MASKED) -> None:
        self.secs = self.visit_mask(masked.mask)

        assignment = masked.assignment
        if assignment is not None:
            operation = assignment.operation
            if isinstance(operation, READ):
                self.visit_read(operation)
            elif isinstance(operation, WRITE):
                self.visit_write(operation)

        self.secs = None

    def visit_mask(self: "DoubleNegativeAnalyzer", mask: MASK) -> Sequence[int]:
        secs = mask.constant_value
        if secs is None:
            secs = 0xFFFF
        return [bit for bit in range(16)
                if (secs & (1 << bit)) > 0]

    def visit_read(self: "DoubleNegativeAnalyzer", read: READ) -> None:
        self.in_read = True
        if read.operator is ASSIGN_OP.EQ:
            rvalue = read.rvalue
            if isinstance(rvalue, UNARY_EXPR):
                self.visit_unary_expr(rvalue)
            elif isinstance(rvalue, BINARY_EXPR):
                self.visit_binary_expr(rvalue)
            else:
                raise ValueError(
                    f"Unsupported rvalue type ({rvalue.__class__.__name__}): "
                    f"{rvalue}")
        self.in_read = False

    def visit_write(self: "DoubleNegativeAnalyzer", write: WRITE) -> None:
        self.in_write = True
        if write.operator is ASSIGN_OP.EQ:
            lvalue = write.lvalue
            rvalue = write.rvalue
            self.visit_unary_src(rvalue)
        self.in_write = False

    def visit_binary_expr(self: "DoubleNegativeAnalyzer",
                          binary_expr: BINARY_EXPR) -> None:
        right_operand = binary_expr.right_operand
        if isinstance(right_operand, UNARY_SRC):
            self.visit_unary_src(right_operand)

    def visit_unary_expr(self: "DoubleNegativeAnalyzer",
                         unary_expr: UNARY_EXPR) -> None:
        expression = unary_expr.expression
        if isinstance(expression, UNARY_SRC):
            self.visit_unary_src(expression)

    def visit_unary_src(self: "DoubleNegativeAnalyzer",
                        unary_src: UNARY_SRC) -> None:
        expression = unary_src.expression
        self.visit_src_expr(expression)

    def visit_src_expr(self: "DoubleNegativeAnalyzer",
                       src_expr: SRC_EXPR) -> None:
        if self.in_multi_statement:
            if self.in_read:
                self.read_secs.update(self.secs)
            elif self.in_write:
                self.write_secs.update(self.secs)
            else:
                raise RuntimeError(
                    "Unsupported multi-statement state "
                    "(neither in_read nor in_write)")
