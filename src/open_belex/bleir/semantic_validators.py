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

import re
from collections import defaultdict
from dataclasses import dataclass, field
from re import Pattern
from typing import ClassVar, Dict, Optional, Sequence, Set, Tuple, Union
from warnings import warn

from open_belex.bleir.analyzers import (MAX_FRAGMENT_INSTRUCTIONS,
                                        LiveSectionScanner,
                                        NumFragmentInstructionsAnalyzer,
                                        RegisterParameterFinder)
from open_belex.bleir.types import (ASSIGN_OP, BINARY_EXPR, EWE_REG, L1_REG,
                                    L2_REG, MASK, MASKED, RE_REG, READ, RN_REG,
                                    SB_EXPR, SM_REG, SRC_EXPR, STATEMENT,
                                    UNARY_EXPR, UNARY_SB, UNARY_SRC, WRITE,
                                    ActualParameter, AllocatedRegister,
                                    CallMetadata, FormalParameter, Fragment,
                                    FragmentCaller, FragmentCallerCall,
                                    GlassStatement, MultiStatement,
                                    ReadWriteInhibit, RegisterParameter,
                                    SemanticError, Snippet)
from open_belex.bleir.walkables import BLEIRListener, BLEIRVisitor, BLEIRWalker
from open_belex.common.mask import Mask
from open_belex.common.types import Integer

RE_VALID_SNIPPET_NAME: ClassVar[Pattern] = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

RL_SRCS: Set[SRC_EXPR] = {
    SRC_EXPR.RL,
    SRC_EXPR.NRL,
    SRC_EXPR.ERL,
    SRC_EXPR.WRL,
    SRC_EXPR.SRL,

    SRC_EXPR.INV_RL,
    SRC_EXPR.INV_NRL,
    SRC_EXPR.INV_ERL,
    SRC_EXPR.INV_WRL,
    SRC_EXPR.INV_SRL,
}


class SnippetNameValidator(BLEIRListener):
    """Ensures snippet names are valid for use in function identifiers."""

    def enter_snippet(self: "SnippetNameValidator", snippet: Snippet) -> None:
        global RE_VALID_SNIPPET_NAME
        if not RE_VALID_SNIPPET_NAME.fullmatch(snippet.name):
            raise SemanticError(
                f"Snippet name (\"{snippet.name}\") does not match pattern: "
                f"{RE_VALID_SNIPPET_NAME}")


class ParameterIDValidator(BLEIRListener):

    # Internal state variables
    parameter_ids: Optional[Set[str]] = None
    in_statement: bool = False

    def enter_fragment(self: "ParameterIDValidator", fragment: Fragment) -> None:
        self.parameter_ids = set()

    def exit_fragment(self: "ParameterIDValidator", fragment: Fragment) -> None:
        self.parameter_ids = None

    def observe_formal_parameter(
            self: "ParameterIDValidator",
            formal_parameter: FormalParameter) -> None:
        if self.parameter_ids is not None:
            if formal_parameter.identifier in self.parameter_ids:
                raise SemanticError(
                    f"Conflicting parameter ids: {formal_parameter.identifier} "
                    f"already declared")
            self.parameter_ids.add(formal_parameter.identifier)

    def observe_register_parameter(
            self: "ParameterIDValidator",
            register_parameter: RegisterParameter) -> None:
        if self.parameter_ids is not None \
           and register_parameter.identifier not in self.parameter_ids \
           and not register_parameter.is_lowered \
           and not (isinstance(register_parameter, RN_REG) \
                    and register_parameter.initial_value is not None \
                    or isinstance(register_parameter, SM_REG) \
                    and register_parameter.constant_value is not None):
            raise SemanticError(
                f"Undefined {register_parameter.__class__.__name__}: "
                f"{register_parameter.identifier}")

    def enter_statement(self: "ParameterIDValidator",
                        statement: STATEMENT) -> None:
        self.in_statement = True

    def exit_statement(self: "ParameterIDValidator",
                       statement: STATEMENT) -> None:
        self.in_statement = False

    def enter_rn_reg(self: "ParameterIDValidator", rn_reg: RN_REG) -> None:
        if self.in_statement:
            self.observe_register_parameter(rn_reg)
        else:
            self.observe_formal_parameter(rn_reg)

    def enter_sm_reg(self: "ParameterIDValidator", sm_reg: SM_REG) -> None:
        if self.in_statement:
            self.observe_register_parameter(sm_reg)
        else:
            self.observe_formal_parameter(sm_reg)


ParameterPair = Tuple[FormalParameter, ActualParameter]


class RegisterValidator(BLEIRListener):
    parameters: Optional[Set[FormalParameter]] = None

    def enter_fragment_caller(self: "RegisterValidator",
                              fragment_caller: FragmentCaller) -> None:
        self.parameters = set()

    def exit_fragment_caller(self: "RegisterValidator",
                             fragment_caller: FragmentCaller) -> None:
        self.parameters = None

    def enter_allocated_register(self: "RegisterValidator",
                                 allocated_register: AllocatedRegister) -> None:
        if self.parameters is not None:
            if allocated_register.parameter in self.parameters:
                raise SemanticError(
                    f"RegisterParameter already allocated for "
                    f"{allocated_register.parameter}")
            self.parameters.add(allocated_register.parameter)

    def enter_rn_reg(self: "RegisterValidator", rn_reg: RN_REG) -> None:
        if self.parameters is not None \
           and rn_reg not in self.parameters \
           and not rn_reg.is_lowered:
            raise SemanticError(
                f"No rn_reg allocated for parameter: {rn_reg}")

    def enter_sm_reg(self: "RegisterValidator", sm_reg: SM_REG) -> None:
        if self.parameters is not None \
           and sm_reg not in self.parameters \
           and not sm_reg.is_lowered:
            raise SemanticError(
                f"No sm_reg allocated for parameter: {sm_reg}")


class FragmentCallerCallValidator(BLEIRListener):

    def enter_fragment_caller_call(
            self: "FragmentCallerCallValidator",
            fragment_caller_call: FragmentCallerCall) -> None:

        fragment_caller = fragment_caller_call.caller
        fragment = fragment_caller.fragment
        actual_parameters = fragment_caller_call.parameters

        temporaries = set(fragment.temporaries)

        formal_parameters = []
        for formal_parameter in fragment.parameters:
            if isinstance(formal_parameter, RN_REG) and (
                    formal_parameter.initial_value is not None
                    or formal_parameter in temporaries):
                continue

            if isinstance(formal_parameter, SM_REG) \
               and formal_parameter.constant_value is not None:
                continue

            # if isinstance(formal_parameter, RE_REG) \
            #    and formal_parameter.row_mask is not None:
            #     continue

            # if isinstance(formal_parameter, EWE_REG) \
            #    and formal_parameter.wordline_mask is not None:
            #     continue

            if isinstance(formal_parameter, L1_REG) \
               and formal_parameter.bank_group_row is not None:
                continue

            if isinstance(formal_parameter, L2_REG) \
               and formal_parameter.value is not None:
                continue

            formal_parameters.append(formal_parameter)

        if len(formal_parameters) != len(actual_parameters):
            raise SemanticError(
                f"Expected {len(formal_parameters)} actual parameters but "
                f"received {len(actual_parameters)}: {fragment_caller_call}")

        for formal_parameter, actual_parameter in zip(formal_parameters, actual_parameters):
            if isinstance(actual_parameter, Integer.__args__):
                value = actual_parameter
            elif isinstance(formal_parameter, SM_REG):
                value = Mask(actual_parameter).full_integer
            else:
                raise SemanticError(
                    f"Unsupported actual_parameter type "
                    f"({actual_parameter.__class__.__name__}) for "
                    f"formal_parameter type "
                    f"({formal_parameter.__class__.__name__})")
            if isinstance(formal_parameter, RN_REG) and not 0 <= value < 24 \
               or isinstance(formal_parameter, SM_REG) and not 0x0000 <= value <= 0xFFFF \
               or isinstance(formal_parameter, RE_REG) and not 0x000000 <= value <= 0xFFFFFF \
               or isinstance(formal_parameter, EWE_REG) and not 0x000 <= value < 0x300:
                raise SemanticError(
                    f"Expected value of {formal_parameter.identifier} to be an "
                    f"unsigned 16-bit or 24-bit int in the respective range: {value}")


class MultiStatementValidator(BLEIRVisitor):

    # Internal state variables
    in_multi_statement: bool = False
    num_commands: int = -1

    read_regs_by_mask: Optional[Dict[MASK, Sequence[RN_REG]]] = None
    write_regs_by_mask: Optional[Dict[MASK, Sequence[RN_REG]]] = None

    read_srcs_by_sec: Optional[Dict[MASK, Sequence[SRC_EXPR]]] = None
    write_srcs_by_sec: Optional[Dict[MASK, Sequence[SRC_EXPR]]] = None

    read_masks: Optional[Set[MASK]] = None
    write_masks: Optional[Set[MASK]] = None

    frag_nym: Optional[str] = None

    mask: Optional[MASK] = None
    in_read: bool = False
    in_write: bool = False

    @staticmethod
    def is_constant_mask(mask: MASK) -> bool:
        return mask.is_constant

    def rw_masks_are_var_compatible(self: "MultiStatementValidator") -> bool:
        if any(map(self.is_constant_mask, self.read_masks)):
            return False

        if any(map(self.is_constant_mask, self.write_masks)):
            return False

        if len(self.read_masks) != 1 or len(self.write_masks) != 1:
            return False

        read_mask = next(iter(self.read_masks))
        write_mask = next(iter(self.write_masks))
        return read_mask == ~write_mask

    def visit_fragment(self: "MultiStatementValidator",
                       fragment: Fragment) -> None:
        self.frag_nym = fragment.original_identifier
        for operation_or_comment in fragment.operations:
            if isinstance(operation_or_comment, MultiStatement):
                multi_statement = operation_or_comment
                self.visit_multi_statement(multi_statement)
            elif isinstance(operation_or_comment, STATEMENT):
                statement = operation_or_comment
                self.visit_statement(statement)
        self.frag_nym = None

    def visit_multi_statement(self: "MultiStatementValidator",
                              multi_statement: MultiStatement) -> None:

        self.in_multi_statement = True
        self.num_commands = 0
        self.read_regs_by_mask = defaultdict(set)
        self.write_regs_by_mask = defaultdict(set)
        self.read_srcs_by_sec = defaultdict(set)
        self.write_srcs_by_sec = defaultdict(set)
        self.read_masks = set()
        self.write_masks = set()

        for statement_or_comment in multi_statement:
            if isinstance(statement_or_comment, STATEMENT):
                statement = statement_or_comment
                self.visit_statement(statement)

        read_masks = set(self.read_regs_by_mask.keys())
        write_masks = set(self.write_regs_by_mask.keys())
        shared_masks = read_masks & write_masks

        # Adjust self.num_commands for R2W1
        for mask in shared_masks:
            reads = self.read_regs_by_mask[mask]
            writes = self.write_regs_by_mask[mask]
            reads, writes = (reads - writes), (writes - reads)
            num_reads = len(reads)
            num_writes = len(writes)
            while num_reads > 0 and num_writes > 0:
                self.num_commands -= 1
                num_reads -= 2
                num_writes -= 1

        if self.num_commands > 4:
            raise SemanticError(
                f"Too many laned commands ({self.num_commands} > 4) in "
                f"multi-statement of fragment {self.frag_nym}: "
                f"{multi_statement}")

        if not self.rw_masks_are_var_compatible():
            read_secs = set(self.read_srcs_by_sec.keys())
            write_secs = set(self.write_srcs_by_sec.keys())
            shared_secs = read_secs & write_secs

            for sec in shared_secs:
                reads = self.read_srcs_by_sec[sec]
                writes = self.write_srcs_by_sec[sec]

                if len(reads) > 0 and len(writes) > 0:
                    reads_or_writes = reads | writes
                    if len(reads_or_writes) > 1:
                        raise SemanticError(
                            f"Only 1 SRC allowed when READs and WRITEs share "
                            f"sections, but found {len(reads_or_writes)} in "
                            f"multi-statement of fragment {self.frag_nym}: "
                            f"{multi_statement}")

            for secs, masks in \
                (read_secs, self.write_masks), (write_secs, self.read_masks):
                if len(secs) > 0 and len(masks) > 0 and \
                   not all(map(self.is_constant_mask, masks)):
                    sec_str = f"[{','.join(map(str, secs))}]"
                    for mask in masks:
                        if not mask.is_constant:
                            identifier = mask.sm_reg.identifier
                            warn(f"Be careful to assign a value to SM_REG "
                                 f"{identifier} of fragment {self.frag_nym} "
                                 f"that will not evaluate to one of the "
                                 f"sections {sec_str} in expression: {mask}")

        self.in_multi_statement = False
        self.num_commands = -1
        self.read_regs_by_mask = None
        self.write_regs_by_mask = None
        self.read_srcs_by_sec = None
        self.write_srcs_by_sec = None
        self.read_masks = None
        self.write_masks = None

    def visit_statement(self: "MultiStatementValidator",
                        statement: STATEMENT) -> None:
        if self.in_multi_statement \
           and not isinstance(statement.operation, GlassStatement):
            self.num_commands += 1
            operation = statement.operation
            if isinstance(operation, MASKED):
                self.visit_masked(operation)

    def visit_masked(self: "MultiStatementValidator", masked: MASKED) -> None:
        self.mask = masked.mask
        self.secs = self.visit_mask(masked.mask)
        assignment = masked.assignment
        if assignment is not None:
            operation = assignment.operation
            if isinstance(operation, READ):
                self.visit_read(operation)
            elif isinstance(operation, WRITE):
                self.visit_write(operation)
        self.secs = None
        self.mask = None

    def visit_mask(self: "MultiStatementValidator",
                   mask: MASK) -> Optional[Sequence[int]]:
        secs = mask.constant_value
        if secs is None:
            return None
        return [bit for bit in range(16)
                if (secs & (1 << bit)) > 0]

    def visit_read(self: "MultiStatementValidator", read: READ) -> None:
        self.in_read = True
        if read.operator is ASSIGN_OP.EQ:
            self.read_masks.add(self.mask)
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

    def visit_write(self: "MultiStatementValidator", write: WRITE) -> None:
        self.in_write = True
        if write.operator is ASSIGN_OP.EQ:
            self.write_masks.add(self.mask)
            lvalue = write.lvalue
            rvalue = write.rvalue
            self.visit_sb_expr(lvalue)
            self.visit_unary_src(rvalue)
        self.in_write = False

    def visit_binary_expr(self: "MultiStatementValidator",
                          binary_expr: BINARY_EXPR) -> None:
        left_operand = binary_expr.left_operand
        if isinstance(left_operand, UNARY_SB):
            self.visit_unary_sb(left_operand)
        right_operand = binary_expr.right_operand
        if isinstance(right_operand, UNARY_SRC):
            self.visit_unary_src(right_operand)

    def visit_unary_expr(self: "MultiStatementValidator",
                         unary_expr: UNARY_EXPR) -> None:
        expression = unary_expr.expression
        if isinstance(expression, UNARY_SB):
            self.visit_unary_sb(expression)
        elif isinstance(expression, UNARY_SRC):
            self.visit_unary_src(expression)

    def visit_unary_sb(self: "MultiStatementValidator",
                       unary_sb: UNARY_SB) -> None:
        expression = unary_sb.expression
        self.visit_sb_expr(expression)

    def visit_unary_src(self: "MultiStatementValidator",
                        unary_src: UNARY_SRC) -> None:
        expression = unary_src.expression
        self.visit_src_expr(expression)

    def visit_sb_expr(self: "MultiStatementValidator",
                      sb_expr: SB_EXPR) -> None:
        parameters = sb_expr.parameters
        if self.in_multi_statement \
           and len(parameters) > 0 \
           and isinstance(parameters[0], RN_REG) \
           and self.mask is not None:
            if self.in_read:
                self.read_regs_by_mask[self.mask].update(parameters)
            elif self.in_write:
                self.write_regs_by_mask[self.mask].update(parameters)

    def visit_src_expr(self: "MultiStatementValidator",
                       src_expr: SRC_EXPR) -> None:
        if self.in_multi_statement and self.secs is not None:
            if self.in_read:
                srcs_by_sec = self.read_srcs_by_sec
            elif self.in_write:
                srcs_by_sec = self.write_srcs_by_sec
            else:
                raise RuntimeError(
                    "Unsupported multi-statement state "
                    "(neither in_read nor in_write)")
            for sec in self.secs:
                srcs_by_sec[sec].add(src_expr)


@dataclass
class FragmentSignatureUniquenessValidator(BLEIRListener):
    frags_by_nym: Dict[str, Fragment] = field(default_factory=dict)

    def enter_fragment(self: "FragmentSignatureUniquenessValidator", fragment: Fragment) -> None:
        if fragment.identifier not in self.frags_by_nym:
            self.frags_by_nym[fragment.identifier] = fragment
        elif fragment != self.frags_by_nym[fragment.identifier]:
            raise SemanticError(f"Fragment already defined for name: {fragment.identifier}")


class ReservedRegisterValidator(BLEIRListener):
    reserved_registers: Dict[str, str] = {
        # "RN_REG_0": "full-VR I/O"
    }

    fragment_caller_id: Optional[str] = None
    is_initializer: bool = False

    def enter_fragment_caller_call(
            self: "ReservedRegisterValidator",
            fragment_caller_call: FragmentCallerCall) -> None:
        self.is_initializer = \
            fragment_caller_call.has_metadata(CallMetadata.IS_INITIALIZER, True)

    def exit_fragment_caller_call(
            self: "ReservedRegisterValidator",
            fragment_caller_call: FragmentCallerCall) -> None:
        self.is_initializer = False

    def enter_fragment_caller(
            self: "ReservedRegisterValidator",
            fragment_caller: FragmentCaller) -> None:
        self.fragment_caller_id = fragment_caller.identifier

    def exit_fragment_caller(
            self: "ReservedRegisterValidator",
            fragment_caller: FragmentCaller) -> None:
        self.fragment_caller_id = None

    def enter_allocated_register(
            self: "ReservedRegisterValidator",
            allocated_register: AllocatedRegister) -> None:
        if not self.is_initializer:
            identifier = allocated_register.parameter.identifier
            register = allocated_register.register
            if register in self.reserved_registers:
                reason = self.reserved_registers[register]
                raise SemanticError(
                    f"{register}, allocated for {identifier} in "
                    f"{self.fragment_caller_id}, is reserved for the "
                    f"following reason: {reason}")


@dataclass
class MultiStatementSBGroupingEnforcer(BLEIRListener):
    register_map: Optional[Dict[RN_REG, int]] = field(default_factory=lambda: None)
    color: int = field(default_factory=lambda: -1)
    is_in_multi_statement: bool = field(default_factory=lambda: False)
    is_in_write: bool = field(default_factory=lambda: False)

    def enter_fragment_caller_call(self: "MultiStatementSBGroupingEnforcer",
                                   fragment_caller_call: FragmentCallerCall) -> None:

        actual_parameters = fragment_caller_call.parameters
        fragment_caller = fragment_caller_call.caller
        allocated_registers = fragment_caller.registers

        self.register_map = {}
        zipped_params = zip(actual_parameters, allocated_registers)
        for actual_parameter, allocated_register in zipped_params:
            formal_parameter = allocated_register.parameter
            self.register_map[formal_parameter] = actual_parameter // 8

        # Generated params
        for index in range(len(actual_parameters), len(allocated_registers)):
            allocated_register = allocated_registers[index]
            formal_parameter = allocated_register.parameter
            if isinstance(formal_parameter, RN_REG):
                actual_parameter = allocated_register.reg_id
                self.register_map[formal_parameter] = actual_parameter // 8

    def exit_fragment_caller_call(self: "MultiStatementSBGroupingEnforcer",
                                  fragment_caller_call: FragmentCallerCall) -> None:
        self.register_map = None

    def enter_multi_statement(self: "MultiStatementSBGroupingEnforcer",
                              multi_statement: MultiStatement) -> None:
        self.is_in_multi_statement = True
        self.color = -1

    def exit_multi_statement(self: "MultiStatementSBGroupingEnforcer",
                             multi_statement: MultiStatement) -> None:
        self.is_in_multi_statement = False
        self.color = -1

    def enter_write(self: "MultiStatementSBGroupingEnforcer", write: WRITE) -> None:
        self.is_in_write = True
        if not self.is_in_multi_statement:
            self.color = -1

    def exit_write(self: "MultiStatementSBGroupingEnforcer", write: WRITE) -> None:
        self.is_in_write = False
        if not self.is_in_multi_statement:
            self.color = -1

    def enter_rn_reg(self: "MultiStatementSBGroupingEnforcer", rn_reg: RN_REG) -> None:
        # NOTE: It seems that only WRITE SBs need to be in the same group
        # within multi-statements
        if self.is_in_write \
           and self.register_map is not None \
           and not rn_reg.is_lowered:
            color = self.register_map[rn_reg]
            if self.color == -1:
                self.color = color
            elif self.color != color:
                raise SemanticError(
                    f"All written SBs in a multi-statement must be in the same group")


class AllocatedRegisterUniquenessValidator(BLEIRListener):

    def enter_fragment_caller(self: "AllocatedRegisterUniquenessValidator",
                              fragment_caller: FragmentCaller) -> None:

        if fragment_caller.registers is None:
            return

        allocated = {}
        for allocated_register in fragment_caller.registers:
            register = allocated_register.register
            if register in allocated:
                raise SemanticError(
                    f"Already allocated register {register} for {allocated[register]}")
            allocated[register] = allocated_register.parameter


@dataclass
class NumFragmentInstructionsValidator(BLEIRListener):
    num_fragment_instructions_analyzer: NumFragmentInstructionsAnalyzer

    in_snippet: bool = False

    def enter_snippet(self: "NumFragmentInstructionsValidator",
                      snippet: Snippet) -> None:
        self.in_snippet = True

    def exit_snippet(self: "NumFragmentInstructionsValidator",
                     snippet: Snippet) -> None:
        self.in_snippet = False

    def enter_fragment(self: "NumFragmentInstructionsValidator",
                       fragment: Fragment) -> None:

        # Workaround for when fragments are compiled before snippets (the
        # partitioning will occur when the snippet is compiled)
        if not self.in_snippet:
            return

        num_instructions = \
            self.num_fragment_instructions_analyzer \
                .num_instructions_by_fragment[fragment.identifier]

        if num_instructions > MAX_FRAGMENT_INSTRUCTIONS:
            raise SemanticError(
                f"No more than {MAX_FRAGMENT_INSTRUCTIONS} instructions are "
                f"supported by the APU, but {fragment.identifier} has "
                f"{num_instructions}")


@dataclass
class EnsureWriteBeforeRead(BLEIRListener):
    live_section_scanner: LiveSectionScanner

    walker: BLEIRWalker = field(default_factory=BLEIRWalker)

    @staticmethod
    def is_parameterized(section: Union[int, Tuple[str, bool]]) -> bool:
        return not isinstance(section, int)

    def enter_fragment(self: "EnsureWriteBeforeRead",
                       fragment: Fragment) -> None:
        # Analyze from the top-level fragment
        if fragment.identifier == fragment.original_identifier:
            frag_id = fragment.original_identifier

            writes_by_section_by_operand = \
                self.live_section_scanner \
                    .writes_by_section_by_frag[frag_id]

            reads_by_section_by_operand = \
                self.live_section_scanner \
                     .reads_by_section_by_frag[frag_id]

            register_parameter_finder = RegisterParameterFinder()
            self.walker.walk(register_parameter_finder, fragment)
            register_parameters = register_parameter_finder.register_parameters

            for register_parameter in register_parameters:
                if isinstance(register_parameter, RN_REG) \
                   and (register_parameter.is_temporary
                        or register_parameter.identifier.startswith("_INTERNAL")):
                    operand = register_parameter.identifier
                    writes_by_section = writes_by_section_by_operand[operand]
                    reads_by_section = reads_by_section_by_operand[operand]
                    for section, reads in reads_by_section.items():
                        # Only consider section literals; we do not have the
                        # ability to reliably analyze user parameters right now.
                        # ------------------------------------------------
                        # FIXME: Once the entire application is written in
                        # Copperhead/Belex, perform liveness analysis at the
                        # FragmentCallerCall level to capture section values of
                        # user parameters.
                        if not isinstance(section, int) \
                           or any(map(self.is_parameterized, writes_by_section)):
                            continue

                        if section not in writes_by_section \
                           or len(writes_by_section[section]) == 0:
                            raise SemanticError(
                                f"No write for read section {section} on "
                                f"instruction {reads[0]} for fragment "
                                f"{fragment.identifier}, operand {operand}")

                        writes = writes_by_section[section]
                        if reads[0] < writes[0]:
                            raise SemanticError(
                                f"Section {section} read on instruction "
                                f"{reads[0]} before being written on "
                                f"instruction {writes[0]} for fragment "
                                f"{fragment.identifier}, operand {operand}")


@dataclass
class ReadWriteInhibitValidator(BLEIRListener):
    in_multi_statement: bool = False

    read_mask: Optional[int] = None
    write_mask: Optional[int] = None

    rwinh: Optional[ReadWriteInhibit] = None
    has_read: bool = False
    has_write: bool = False

    has_write_from_rl: bool = False

    operation: Optional[Union[STATEMENT, MultiStatement]] = None

    def validate_rwinh_set(
            self: "ReadWriteInhibitValidator",
            statement_or_multi_statement: Union[STATEMENT, MultiStatement]) \
            -> None:
        if self.rwinh is ReadWriteInhibit.RWINH_SET \
           and not (self.has_read or self.has_write_from_rl):
            raise SemanticError(
                f"RWINH_SET must appear in an instruction containing a READ "
                f"to or WRITE from RL: {statement_or_multi_statement}")

    def cleanup(self: "ReadWriteInhibitValidator") -> None:
        self.has_read = False
        self.has_write = False
        self.has_write_from_rl = False

        self.read_mask = None
        self.write_mask = None
        self.rwinh = None
        self.operation = None

    def enter_multi_statement(self: "ReadWriteInhibitValidator",
                              multi_statement: MultiStatement) -> None:
        self.operation = multi_statement
        self.in_multi_statement = True
        self.read_mask = 0x0000
        self.write_mask = 0x0000

    def exit_multi_statement(self: "ReadWriteInhibitValidator",
                             multi_statement: MultiStatement) -> None:

        self.in_multi_statement = False

        if self.rwinh is ReadWriteInhibit.RWINH_SET \
           and self.has_read \
           and self.has_write \
           and (self.read_mask is None
                or self.write_mask is None
                or (self.read_mask & self.write_mask) != 0x0000):
            # Note: Write operations may not be initiated in the same cycle as
            # Read operations with rwinhset=1, because the Write result would
            # be indeterminate - it may or may not be inhibited.
            raise SemanticError(
                f"May not combine WRITE and READ within the same instruction "
                f"within the context of Read/Write-Inhibit: {multi_statement}")

        self.validate_rwinh_set(multi_statement)
        self.cleanup()

    def enter_statement(self: "ReadWriteInhibitValidator",
                        statement: STATEMENT) -> None:
        if not self.in_multi_statement:
            self.operation = statement

    def exit_statement(self: "ReadWriteInhibitValidator",
                       statement: STATEMENT) -> None:
        if not self.in_multi_statement:
            self.validate_rwinh_set(statement)
            self.cleanup()

    def enter_masked(self: "ReadWriteInhibitValidator",
                     masked: MASKED) -> None:
        if masked.read_write_inhibit is not None \
           and masked.assignment is not None \
           and not isinstance(masked.assignment.operation, READ):
            raise SemanticError(
                f"READ operation required to set or reset Read/Write-Inhibit: "
                f"{masked}")

    def exit_masked(self: "ReadWriteInhibitValidator",
                    masked: MASKED) -> None:

        rwinh = masked.read_write_inhibit
        if rwinh is not None:
            if self.rwinh is None:
                self.rwinh = rwinh
            elif self.rwinh != rwinh:
                SemanticError(
                    f"When mixing RWINH_SET and RWINH_RST, the RWINH_SET "
                    f"will have no effect (only the RWINH_RST will take "
                    f"effect): {self.operation}")

        if masked.assignment is not None \
           and isinstance(masked.assignment.operation, READ):
            if masked.mask.is_constant and self.read_mask is not None:
                self.read_mask |= masked.mask.resolve()
            else:
                self.read_mask = None

        elif masked.assignment is not None \
             and isinstance(masked.assignment.operation, WRITE):
            if masked.mask.is_constant and self.write_mask is not None:
                self.write_mask |= masked.mask.resolve()
            else:
                self.write_mask = None

    def enter_read(self: "ReadWriteInhibitValidator", read: READ) -> None:
        self.has_read = True

    def enter_write(self: "ReadWriteInhibitValidator", write: WRITE) -> None:
        self.has_write = True
        if write.rvalue.expression in RL_SRCS:
            self.has_write_from_rl = True
