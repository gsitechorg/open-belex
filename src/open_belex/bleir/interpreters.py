r"""
By Dylon Edwards
"""

import logging
from dataclasses import dataclass, field
from functools import partial
from itertools import repeat
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
from open_belex.bleir.analyzers import (CoalesceGroupedRegisters,
                                        RegisterParameterFinder)
from open_belex.bleir.rewriters import GroupMultipleBroadcasts
from open_belex.bleir.types import (ASSIGN_OP, ASSIGNMENT, BINARY_EXPR, BINOP,
                                    BIT_EXPR, BROADCAST, BROADCAST_EXPR,
                                    EWE_REG, GGL_ASSIGNMENT, GGL_EXPR, L1_REG,
                                    L2_REG, LGL_ASSIGNMENT, LGL_EXPR, LX_ADDR,
                                    LX_ASSIGNMENT, MASK, MASKED, RE_REG, READ,
                                    RL_EXPR, RN_REG, RSP2K_ASSIGNMENT,
                                    RSP2K_EXPR, RSP2K_RVALUE, RSP16_ASSIGNMENT,
                                    RSP16_RVALUE, RSP32K_ASSIGNMENT,
                                    RSP32K_EXPR, RSP32K_RVALUE,
                                    RSP256_ASSIGNMENT, RSP256_EXPR,
                                    RSP256_RVALUE, SB_EXPR, SHIFTED_SM_REG,
                                    SM_REG, SPECIAL, SRC_EXPR, STATEMENT,
                                    UNARY_EXPR, UNARY_OP, UNARY_SB, UNARY_SRC,
                                    WRITE, ActualParameter, CFunction,
                                    CFunctionCall, CFunctionMetadata, Example,
                                    ExtendedRegister, FormalParameter,
                                    Fragment, FragmentCaller,
                                    FragmentCallerCall, GlassFormat, Glassible,
                                    GlassOrder, GlassStatement, LineComment,
                                    LXParameter, LXRegWithOffsets,
                                    MultiStatement, Operation,
                                    Operation_or_LineComment, ReadWriteInhibit,
                                    RegisterParameter, Snippet)
from open_belex.bleir.walkables import BLEIRVisitor, BLEIRWalker
from open_belex.common.constants import NSB, NSECTIONS, NUM_PLATS_PER_APUC
from open_belex.common.mask import Mask
from open_belex.common.stack_manager import contextual
from open_belex.common.types import Integer
from open_belex.diri.half_bank import DIRI, PatchFn, PatchPair
from reactivex.subject import Subject

LOGGER = logging.getLogger()

INVERT_SRC_EXPR: Dict[SRC_EXPR, SRC_EXPR] = {
    SRC_EXPR.INV_RL: SRC_EXPR.RL,
    SRC_EXPR.INV_NRL: SRC_EXPR.NRL,
    SRC_EXPR.INV_ERL: SRC_EXPR.ERL,
    SRC_EXPR.INV_WRL: SRC_EXPR.WRL,
    SRC_EXPR.INV_SRL: SRC_EXPR.SRL,
    SRC_EXPR.INV_GL: SRC_EXPR.GL,
    SRC_EXPR.INV_GGL: SRC_EXPR.GGL,
    SRC_EXPR.INV_RSP16: SRC_EXPR.RSP16,
}


READ_PATTERNS: Dict = {
    ASSIGN_OP.EQ: {
        UNARY_EXPR: {
            BIT_EXPR: DIRI.set_rl,
            UNARY_SB: {
                None: DIRI.rl_from_sb,
                UNARY_OP.NEGATE: DIRI.rl_from_inv_sb,
            },
            UNARY_SRC: {
                None: DIRI.rl_from_src,
                UNARY_OP.NEGATE: DIRI.rl_from_inv_src,
            },
        },
        BINARY_EXPR: {
            BINOP.AND: {
                (None, None): DIRI.rl_from_sb_and_src,
                (UNARY_OP.NEGATE, None): DIRI.rl_from_inv_sb_and_src,
                (None, UNARY_OP.NEGATE): DIRI.rl_from_sb_and_inv_src,
                (UNARY_OP.NEGATE, UNARY_OP.NEGATE): DIRI.rl_from_inv_sb_and_inv_src,
            },
            BINOP.OR: {
                (None, None): DIRI.rl_from_sb_or_src,
                (None, UNARY_OP.NEGATE): DIRI.rl_from_sb_or_inv_src,
            },
            BINOP.XOR: {
                (None, None): DIRI.rl_from_sb_xor_src,
                (None, UNARY_OP.NEGATE): DIRI.rl_from_sb_xor_inv_src,
            }
        },
    },
    ASSIGN_OP.AND_EQ: {
        UNARY_EXPR: {
            UNARY_SB: {
                None: DIRI.rl_and_equals_sb,
                UNARY_OP.NEGATE: DIRI.rl_and_equals_inv_sb,
            },
            UNARY_SRC: {
                None: DIRI.rl_and_equals_src,
                UNARY_OP.NEGATE: DIRI.rl_and_equals_inv_src,
            },
        },
        BINARY_EXPR: {
            BINOP.AND: {
                (None, None): DIRI.rl_and_equals_sb_and_src,
                (None, UNARY_OP.NEGATE): DIRI.rl_and_equals_sb_and_inv_src,
            },
        },
    },
    ASSIGN_OP.OR_EQ: {
        UNARY_EXPR: {
            UNARY_SB: DIRI.rl_or_equals_sb,
            UNARY_SRC: {
                None: DIRI.rl_or_equals_src,
                UNARY_OP.NEGATE: DIRI.rl_or_equals_inv_src,
            },
        },
        BINARY_EXPR: {
            BINOP.AND: {
                (None, None): DIRI.rl_or_equals_sb_and_src,
                (None, UNARY_OP.NEGATE): DIRI.rl_or_equals_sb_and_inv_src,
            },
        },
    },
    ASSIGN_OP.XOR_EQ: {
        UNARY_EXPR: {
            UNARY_SB: DIRI.rl_xor_equals_sb,
            UNARY_SRC: {
                None: DIRI.rl_xor_equals_src,
                UNARY_OP.NEGATE: DIRI.rl_xor_equals_inv_src,
            },
        },
        BINARY_EXPR: {
            BINOP.AND: {
                (None, None): DIRI.rl_xor_equals_sb_and_src,
                (None, UNARY_OP.NEGATE): DIRI.rl_xor_equals_sb_and_inv_src,
            },
        },
    },
}


def is_statement(statement_or_comment: Operation_or_LineComment) -> bool:
    return isinstance(statement_or_comment, STATEMENT)


# Highest-to-lowest precedence
STATEMENT_WEIGHTS: Dict[Union[Type, ReadWriteInhibit], int] = {
    statement_type: weight
    for weight, statement_type
    in enumerate([
        WRITE,
        READ,
        ReadWriteInhibit.RWINH_SET,
        ReadWriteInhibit.RWINH_RST,
        LX_ASSIGNMENT,
        LGL_ASSIGNMENT,
        GGL_ASSIGNMENT,
        BROADCAST,
        RSP16_ASSIGNMENT,
        RSP256_ASSIGNMENT,
        RSP2K_ASSIGNMENT,
        RSP32K_ASSIGNMENT,
        SPECIAL,
        GlassStatement,
    ])
}


def statement_type_of(statement: STATEMENT) -> Union[Type, ReadWriteInhibit]:
    operation = statement.operation
    if isinstance(operation, MASKED):
        if operation.read_write_inhibit is not None:
            return operation.read_write_inhibit
        return type(operation.assignment.operation)
    return type(operation)


def statement_weight(statement: STATEMENT) -> int:
    statement_type = statement_type_of(statement)
    if statement_type in STATEMENT_WEIGHTS:
        return STATEMENT_WEIGHTS[statement_type]
    raise ValueError(
        f"Unsupported statement type ({statement.__class__.__name__}): "
        f"{statement}")


@contextual
@dataclass
class BLEIRInterpreter(BLEIRVisitor):
    coalesce_grouped_temporaries: Optional[CoalesceGroupedRegisters] = None

    diri: Optional[DIRI] = None

    walker: BLEIRWalker = field(default_factory=BLEIRWalker)
    group_multiple_broadcasts: Optional[GroupMultipleBroadcasts] = None

    # internal state variables
    values_by_parameter_id: Optional[Dict[str, int]] = None
    data_by_parameter_id: Optional[Dict[str, Union[int, np.ndarray]]] = None
    # row_numbers_by_rn_reg: Optional[Dict[str, int]] = None
    mask: Optional[Mask] = None
    in_multi_statement: bool = False

    # Failure ϵ (message, Example, example_idx, plat)
    failures: Sequence[Tuple[str, Example, int, int]] = \
        field(default_factory=list)

    example: Optional[Example] = None

    formal_actual_parameters: \
        Optional[Sequence[Tuple[FormalParameter, ActualParameter]]] = None

    frag_nym: Optional[str] = None
    num_instructions: int = 0

    has_read: bool = False

    subject: Subject = field(default_factory=Subject)

    def __post_init__(self: "BLEIRInterpreter") -> None:
        if self.diri is None:
            self.diri = DIRI()
        self.group_multiple_broadcasts = \
            GroupMultipleBroadcasts(self.visit_register_parameter)

    @property
    def instruction_num(self: "BLEIRInterpreter") -> int:
        return 1 + self.num_instructions

    @staticmethod
    def to_dec_hex(value: int) -> str:
        return f"({value}, 0x{value:04X})"

    def subscribe(self: "BLEIRInterpreter",
                  subscriber: Callable[[Any], None]) -> None:
        self.subject.subscribe(subscriber)

    def visit_snippet(self: "BLEIRInterpreter", snippet: Snippet) -> None:
        if len(snippet.examples) == 0:
            for fragment_caller_call in snippet.initializers:
                self.visit_fragment_caller_call(fragment_caller_call)
            for fragment_caller_call in snippet.body:
                self.visit_fragment_caller_call(fragment_caller_call)

        for index, example in enumerate(snippet.examples):
            self.example = example

            for call in snippet.initializers:
                if isinstance(call, FragmentCallerCall):
                    self.visit_fragment_caller_call(call)
                elif isinstance(call, CFunctionCall):
                    self.visit_c_function_call(call)
                else:
                    raise ValueError(
                        f"Unsupported call type ({call.__class__.__name__}: "
                        f"{call}")

            value_parameter = example.expected_value
            self.diri.hb[value_parameter.row_number] = False

            for value_parameter in example.parameters:
                vr = [Mask(f"{plat:04X}").little_endian_numpy_bool_array
                      for plat in value_parameter.value]
                vr = np.array(vr, dtype=bool)
                self.diri.hb[value_parameter.row_number] = vr

            for call in snippet.body:
                if isinstance(call, FragmentCallerCall):
                    self.visit_fragment_caller_call(call)
                elif isinstance(call, CFunctionCall):
                    self.visit_c_function_call(call)
                else:
                    raise ValueError(
                        f"Unsupported call type ({call.__class__.__name__}: "
                        f"{call}")

            actual_value = self.diri.hb[example.expected_value.row_number]
            actual_value = [Mask(plat).full_integer for plat in actual_value]
            actual_value = np.array(actual_value, dtype=np.uint16)
            expected_value = example.expected_value.value

            if not np.array_equal(actual_value, expected_value):
                for plat_index in range(len(expected_value)):
                    if actual_value[plat_index] != expected_value[plat_index]:
                        message = (f"Assertion failed (example {index}): "
                                   f"Expected plat {plat_index} to be "
                                   f"0x{expected_value[plat_index]:04X} but "
                                   f"was 0x{actual_value[plat_index]:04X}")
                        record = (message, example, index, plat_index)
                        self.failures.append(record)
                        break

    def visit_c_function_call(
            self: "BLEIRInterpreter",
            c_function_call: CFunctionCall) -> None:
        c_function = c_function_call.c_function
        actual_parameters = c_function_call.actual_parameters
        py_fn = self.visit_c_function(c_function)
        py_fn(*actual_parameters)

    def visit_c_function(
            self: "BLEIRInterpreter",
            c_function: CFunction) -> Callable:
        py_fn = c_function.get_metadata(CFunctionMetadata.PYTHON_FUNCTION)
        return py_fn

    def visit_fragment_caller_call(
            self: "BLEIRInterpreter",
            fragment_caller_call: FragmentCallerCall) -> None:

        fragment_caller = fragment_caller_call.caller
        fragment = fragment_caller.fragment
        frag_id = fragment.original_identifier

        values_by_parameter_id = {}

        register_parameter_finder = RegisterParameterFinder()
        self.walker.walk(register_parameter_finder, fragment)

        for lowered_register in register_parameter_finder.lowered_registers:
            if isinstance(lowered_register, RN_REG):
                rn_nym = lowered_register.identifier
                sb = lowered_register.row_number
                values_by_parameter_id[rn_nym] = sb
                if lowered_register.initial_value is not None:
                    value = lowered_register.initial_value
                    for section in range(NSECTIONS):
                        self.diri.hb[sb, ::, section] = (value & 0b0001)
                        value = (value >> 1)

            elif isinstance(lowered_register, RE_REG):
                re_nym = lowered_register.identifier
                row_mask = lowered_register.row_mask
                values_by_parameter_id[re_nym] = row_mask

            elif isinstance(lowered_register, SM_REG):
                sm_nym = lowered_register.identifier
                value = lowered_register.constant_value
                if lowered_register.is_section:
                    # Coerce section value to mask value
                    value = (0x0001 << value)
                values_by_parameter_id[sm_nym] = value

            elif isinstance(lowered_register, L1_REG):
                l1_nym = lowered_register.identifier
                l1_addr = lowered_register.bank_group_row
                values_by_parameter_id[l1_nym] = l1_addr

            elif isinstance(lowered_register, L2_REG):
                l2_nym = lowered_register.identifier
                l2_addr = lowered_register.value
                values_by_parameter_id[l2_nym] = l2_addr

            else:
                raise ValueError(
                    f"Unsupported lowered_register type "
                    f"({lowered_register.__class__.__name__}): "
                    f"{lowered_register}")

        formal_parameters = list(fragment_caller_call.formal_parameters)
        actual_parameters = list(fragment_caller_call.actual_parameters)
        formal_actual_parameters = list(zip(formal_parameters, actual_parameters))

        for formal_parameter, actual_parameter in formal_actual_parameters:
            identifier = formal_parameter.identifier
            value = self.visit_actual_parameter(actual_parameter)
            if isinstance(formal_parameter, SM_REG) \
               and formal_parameter.is_section:
                # Coerce section value to mask value
                value = (0x0001 << value)
            values_by_parameter_id[identifier] = value

        replace_refs_with_vals = \
            partial(self.replace_refs_with_vals, values_by_parameter_id)

        spill_calls = fragment_caller.initializers
        if spill_calls is None:
            spill_calls = []
        spill_calls = list(map(replace_refs_with_vals, spill_calls))

        restore_calls = fragment_caller.finalizers
        if restore_calls is None:
            restore_calls = []
        restore_calls = list(map(replace_refs_with_vals, restore_calls))

        for spill_call in spill_calls:
            self.visit_fragment_caller_call(spill_call)

        if self.coalesce_grouped_temporaries is not None:
            shared_registers_by_frag = \
                self.coalesce_grouped_temporaries \
                    .shared_registers_by_frag
        else:
            shared_registers_by_frag = {}

        if frag_id in shared_registers_by_frag:
            shared_registers = shared_registers_by_frag[frag_id]
            for src, dest in shared_registers.items():
                values_by_parameter_id[src.identifier] = \
                    values_by_parameter_id[dest.identifier]

        self.formal_actual_parameters = formal_actual_parameters
        self.values_by_parameter_id = values_by_parameter_id
        self.visit_fragment_caller(fragment_caller)

        for restore_call in restore_calls:
            self.visit_fragment_caller_call(restore_call)

    @staticmethod
    def replace_refs_with_vals(
            values_by_parameter_id: Dict[str, int],
            fragment_caller_call: FragmentCallerCall) -> FragmentCallerCall:
        actual_parameters = []
        for formal_parameter, actual_parameter \
                in fragment_caller_call.parameter_map.items():
            if isinstance(actual_parameter, str):
                actual_parameter = values_by_parameter_id[actual_parameter]
            actual_parameters.append(actual_parameter)
        return fragment_caller_call.having(parameters=actual_parameters)

    def visit_actual_parameter(self: "BLEIRInterpreter",
                               actual_parameter: ActualParameter) -> int:
        return actual_parameter

    def visit_fragment_caller(self: "BLEIRInterpreter",
                              fragment_caller: FragmentCaller) -> None:
        self.visit_fragment(fragment_caller.fragment)

    def visit_fragment(self: "BLEIRInterpreter", fragment: Fragment) -> None:
        if fragment.children is not None:
            for child in fragment.children:
                self.visit_fragment(child)
        else:
            self.frag_nym = fragment.identifier
            if self.values_by_parameter_id is not None:
                for operation in fragment.operations:
                    self.visit_operation(operation)

    def visit_operation(self: "BLEIRInterpreter",
                        operation: Operation) -> None:
        if isinstance(operation, MultiStatement):
            if len(self.subject.observers) > 0:
                self.subject.on_next(("multi_statement::enter", operation))
            self.visit_multi_statement(operation)
            if len(self.subject.observers) > 0:
                self.subject.on_next(("multi_statement::exit", operation))
        elif isinstance(operation, STATEMENT):
            if len(self.subject.observers) > 0:
                self.subject.on_next(("statement::enter", operation))
            self.visit_statement(operation)
            if len(self.subject.observers) > 0:
                self.subject.on_next(("statement::exit", operation))
        elif isinstance(operation, LineComment.__args__):
            # Do not record the comment as an instruction
            self.num_instructions -= 1
        else:
            raise NotImplementedError(
                f"Unsupported operation type: {type(operation)}")
        self.num_instructions += 1

    def apply_patch(self: "BLEIRInterpreter",
                    patch_fn: PatchFn,
                    patch: Any,
                    statement: STATEMENT) -> None:
        patch_fn(self.diri, patch)

    def visit_multi_statement(self: "BLEIRInterpreter",
                              multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True

        patch_pairs = []

        # Group multiple GL broadcasts within a multi-statement into a single
        # broadcast unified over their section masks
        multi_statement = self.walker.walk(self.group_multiple_broadcasts,
                                           multi_statement)

        # Re-order in half-clock friendly manner
        half_clock_friendly_statements = \
            sorted(filter(is_statement, multi_statement.statements),
                   key=statement_weight)

        # TODO: Verify the following sequence of actions for Read/Write Inhibit

        has_pending_write = False
        has_pending_read = False
        for statement in half_clock_friendly_statements:
            # simulate half-clock operations
            statement_type = statement_type_of(statement)

            if statement_type is WRITE:
                has_pending_write = True

            elif statement_type is READ:
                has_pending_read = True
                if has_pending_write:
                    # flush the buffer to simulate half-clock operations
                    # write-before-read is a valid half-clock operation
                    for (patch_fn, patch), patch_statement in patch_pairs:
                        self.apply_patch(patch_fn, patch, patch_statement)
                    patch_pairs = []
                    has_pending_write = False

            elif statement_type is BROADCAST:
                if has_pending_read:
                    # flush the buffer to simulate half-clock operations
                    # read-before-broadcast is a valid half-clock operation
                    for (patch_fn, patch), patch_statement in patch_pairs:
                        self.apply_patch(patch_fn, patch, patch_statement)
                    patch_pairs = []
                    has_pending_read = False

            statement_patches = self.visit_statement(statement)
            if statement_patches is not None:
                for patch_pair in statement_patches:
                    patch_pairs.append((patch_pair, statement))

        for (patch_fn, patch), patch_statement in patch_pairs:
            self.apply_patch(patch_fn, patch, patch_statement)

        self.in_multi_statement = False
        self.has_read = False

    def visit_statement(self: "BLEIRInterpreter",
                        statement: STATEMENT) -> Sequence[PatchPair]:

        if self.in_multi_statement:
            prev_in_place = self.diri.in_place
            self.diri.in_place = False

        patch_pairs = None

        if isinstance(statement.operation, MASKED):
            patch_pairs = self.visit_masked(statement.operation)
        elif isinstance(statement.operation, GGL_ASSIGNMENT):
            patch_pairs = self.visit_ggl_assignment(statement.operation)
        elif isinstance(statement.operation, LX_ASSIGNMENT):
            patch_pairs = self.visit_lx_assignment(statement.operation)
        elif isinstance(statement.operation, LGL_ASSIGNMENT):
            patch_pairs = self.visit_lgl_assignment(statement.operation)
        elif isinstance(statement.operation, SPECIAL):
            patch_pairs = self.visit_special(statement.operation)
        elif isinstance(statement.operation, RSP16_ASSIGNMENT):
            patch_pairs = self.visit_rsp16_assignment(statement.operation)
        elif isinstance(statement.operation, RSP256_ASSIGNMENT):
            patch_pairs = self.visit_rsp256_assignment(statement.operation)
        elif isinstance(statement.operation, RSP2K_ASSIGNMENT):
            patch_pairs = self.visit_rsp2k_assignment(statement.operation)
        elif isinstance(statement.operation, RSP32K_ASSIGNMENT):
            patch_pairs = self.visit_rsp32k_assignment(statement.operation)
        elif isinstance(statement.operation, GlassStatement):
            self.visit_glass_statement(statement.operation)
            if not self.in_multi_statement:
                # [WORKAROUND] Decrement num_instructions so when it is
                # incremented at the end of visit_operation, it will not have
                # changed.
                self.num_instructions -= 1
        else:
            raise NotImplementedError(
                f"Unsupported operation type: {type(statement.operation)}")

        if patch_pairs is None:
            return []

        if self.in_multi_statement:
            self.diri.in_place = prev_in_place
            return patch_pairs

        self.has_read = False

    def subject_from_diri(self: "BLEIRInterpreter",
                          subject: Glassible) -> np.ndarray:

        if isinstance(subject, RN_REG):
            vr_or_row_num = self.visit_register_parameter(subject)
            if isinstance(vr_or_row_num, Integer.__args__):
                row_num = vr_or_row_num
                return self.diri.hb[row_num]
            vr = vr_or_row_num
            return vr

        if isinstance(subject, L1_REG):
            l1_addr = self.visit_register_parameter(subject)
            return self.diri.L1[l1_addr]

        if isinstance(subject, L2_REG):
            l2_addr = self.visit_register_parameter(subject)
            return self.diri.L2[l2_addr]

        if isinstance(subject, LXRegWithOffsets):
            lx_addr = self.visit_register_parameter(subject) + subject.offset

            if isinstance(subject.parameter, L1_REG):
                return self.diri.L1[lx_addr]

            if isinstance(subject.parameter, L2_REG):
                return self.diri.L2[lx_addr]

            raise ValueError(
                f"Unsupported LX type "
                f"({subject.parameter.__class__.__name__}): "
                f"{subject.parameter}")

        if subject is SRC_EXPR.RL:
            return self.diri.RL()

        if subject is SRC_EXPR.INV_RL:
            return ~self.diri.RL()

        if subject is SRC_EXPR.NRL:
            return self.diri.NRL()

        if subject is SRC_EXPR.INV_NRL:
            return ~self.diri.NRL()

        if subject is SRC_EXPR.ERL:
            return self.diri.ERL()

        if subject is SRC_EXPR.INV_ERL:
            return ~self.diri.ERL()

        if subject is SRC_EXPR.WRL:
            return self.diri.WRL()

        if subject is SRC_EXPR.INV_WRL:
            return ~self.diri.WRL()

        if subject is SRC_EXPR.SRL:
            return self.diri.SRL()

        if subject is SRC_EXPR.INV_SRL:
            return ~self.diri.SRL()

        if subject is SRC_EXPR.GL:
            return self.diri.GL

        if subject is SRC_EXPR.INV_GL:
            return ~self.diri.GL

        if subject is SRC_EXPR.GGL:
            return self.diri.GGL

        if subject is SRC_EXPR.INV_GGL:
            return ~self.diri.GGL

        if subject is SRC_EXPR.RSP16:
            return self.diri.RSP16

        if subject is SRC_EXPR.INV_RSP16:
            return ~self.diri.RSP16

        if subject is RSP256_EXPR.RSP256:
            return self.diri.RSP256

        if subject is RSP2K_EXPR.RSP2K:
            return self.diri.RSP2K

        if subject is RSP32K_EXPR.RSP32K:
            return self.diri.RSP32K

        if subject is LGL_EXPR.LGL:
            return self.diri.LGL

        raise ValueError(
            f"Unsupported subject type ({subject.__class__.__name__}): {subject}")

    def format_subject_rows(
            self: "BLEIRInterpreter",
            glass_statement: GlassStatement) -> Sequence[str]:

        subject = self.subject_from_diri(glass_statement.subject)
        sections = list(glass_statement.sections)
        plats = list(glass_statement.plats)

        if np.ndim(subject) == 1:
            num_plats, num_sections = subject.shape[0], 1
        else:
            num_plats, num_sections = subject.shape

        if glass_statement.balloon:
            num_plat_repeats = NUM_PLATS_PER_APUC // num_plats
            num_sect_repeats = NSECTIONS // num_sections
            plats = [plat // num_plat_repeats for plat in plats]
            sections = [section // num_sect_repeats for section in sections]
            num_balloon_sections = NSECTIONS
        else:
            num_sect_repeats = 1
            num_balloon_sections = num_sections

        if np.ndim(subject) == 1:
            view = subject[plats][None][sections].astype(np.uint8)
        else:
            view = subject[plats].T[sections].astype(np.uint8)

        if glass_statement.order is GlassOrder.LEAST_SIGNIFICANT_BIT_FIRST:
            view = iter(view)
        elif glass_statement.order is GlassOrder.MOST_SIGNIFICANT_BIT_FIRST:
            view = reversed(view)
        else:
            raise ValueError(
                f"Unsupported glass order: {glass_statement.order}")

        hex_formatter = {'int_kind': lambda value: f"{value:X}"}

        rows = []

        if glass_statement.fmt is GlassFormat.HEX:
            num_nibbles = max(num_balloon_sections // 4, 1)
            num_secs_per_nibble = min(num_balloon_sections, 4)

            for nibble_index in range(num_nibbles):
                lower_nibble = nibble_index * num_secs_per_nibble
                upper_nibble = lower_nibble + num_secs_per_nibble

                if glass_statement.order is GlassOrder.LEAST_SIGNIFICANT_BIT_FIRST:
                    nibble_range = list(range(lower_nibble, upper_nibble))
                elif glass_statement.order is GlassOrder.MOST_SIGNIFICANT_BIT_FIRST:
                    nibble_range = list(range(-upper_nibble, -lower_nibble))
                else:
                    raise ValueError(
                        f"Unsupported glass order: {glass_statement.order}")

                row = np.zeros(len(plats), dtype=np.uint16)
                for shift, section in enumerate(nibble_range):
                    if section < 0:
                        section = num_balloon_sections + section
                        shift = num_secs_per_nibble - shift - 1
                    section //= num_sect_repeats
                    if section in sections:
                        nibble_section = next(view)
                        row |= (nibble_section << shift)

                row = np.array2string(row,
                                      formatter=hex_formatter,
                                      max_line_width=np.inf)

                rows.append(row)

        else:
            if glass_statement.order is GlassOrder.LEAST_SIGNIFICANT_BIT_FIRST:
                section_range = list(range(num_balloon_sections))
            elif glass_statement.order is GlassOrder.MOST_SIGNIFICANT_BIT_FIRST:
                section_range = list(range(-1, -num_balloon_sections - 1, -1))
            else:
                raise ValueError(
                    f"Unsupported glass order: {glass_statement.order}")

            zeros = f"[{' '.join(repeat('0', len(plats)))}]"
            for section in section_range:
                if section < 0:
                    section = num_balloon_sections + section
                section //= num_sect_repeats
                if section in sections:
                    row = np.array2string(next(view),
                                          formatter=hex_formatter,
                                          max_line_width=np.inf)
                else:
                    row = zeros
                rows.append(row)

        rewrite = glass_statement.rewrite
        if rewrite is not None:
            for index, row in enumerate(rows):
                for pattern, replacement in rewrite.items():
                    row = row.replace(pattern, replacement)
                rows[index] = row

        return rows

    def visit_glass_statement(self: "BLEIRInterpreter",
                              glass_statement: GlassStatement) -> None:

        log_message = "fragment: %s; instruction: %d; variable: %s (%s)"
        log_args = [self.frag_nym,
                    self.instruction_num,
                    (glass_statement.subject.identifier
                     if hasattr(glass_statement.subject, "identifier")
                     else glass_statement.subject),
                    glass_statement.order.name.lower().replace("_", " ")]

        if glass_statement.comment is not None:
            log_message = f"{log_message}; comment = %s"
            log_args.append(glass_statement.comment)

        print(log_message % tuple(log_args))
        LOGGER.warning(log_message, *log_args)

        rows = self.format_subject_rows(glass_statement)
        for row in rows:
            print(row)
            LOGGER.warning(row)

    def visit_masked(self: "BLEIRInterpreter",
                     masked: MASKED) -> Sequence[PatchPair]:

        self.mask = self.visit_mask(masked.mask)

        patch_pairs = []
        if masked.assignment is not None:
            assignment_patches = self.visit_assignment(masked.assignment)
            for patch_pair in assignment_patches:
                patch_pairs.append(patch_pair)
        else:
            # internal compiler error if None
            assert masked.read_write_inhibit is not None

        if masked.read_write_inhibit is ReadWriteInhibit.RWINH_SET:
            patch_pair = self.diri.rwinh_set(self.mask)
            patch_pairs.append(patch_pair)
        elif masked.read_write_inhibit is ReadWriteInhibit.RWINH_RST:
            patch_pair = self.diri.rwinh_rst(self.mask, self.has_read)
            patch_pairs.append(patch_pair)

        self.mask = None
        return patch_pairs

    def visit_mask(self: "BLEIRInterpreter", mask: MASK) -> Mask:
        if isinstance(mask.expression, SM_REG):
            sections = self.visit_sm_reg(mask.expression)
        elif isinstance(mask.expression, SHIFTED_SM_REG):
            sections = self.visit_shifted_sm_reg(mask.expression)
        else:
            raise NotImplementedError(
                f"Unsupported expression type: {type(mask.expression)}")

        if mask.operator is UNARY_OP.NEGATE:
            return ~sections

        return sections

    def visit_shifted_sm_reg(self: "BLEIRInterpreter",
                             shifted_sm_reg: SHIFTED_SM_REG) -> Mask:
        mask = self.visit_sm_reg(shifted_sm_reg.register)
        return mask << shifted_sm_reg.num_bits

    def visit_assignment(self: "BLEIRInterpreter",
                         assignment: ASSIGNMENT) -> Sequence[PatchPair]:
        if isinstance(assignment.operation, READ):
            return self.visit_read(assignment.operation)
        if isinstance(assignment.operation, WRITE):
            return self.visit_write(assignment.operation)
        if isinstance(assignment.operation, BROADCAST):
            return self.visit_broadcast(assignment.operation)
        if isinstance(assignment.operation, RSP16_ASSIGNMENT):
            return self.visit_rsp16_assignment(assignment.operation)
        if isinstance(assignment.operation, RSP256_ASSIGNMENT):
            return self.visit_rsp256_assignment(assignment.operation)
        if isinstance(assignment.operation, RSP2K_ASSIGNMENT):
            return self.visit_rsp2k_assignment(assignment.operation)
        if isinstance(assignment.operation, RSP32K_ASSIGNMENT):
            return self.visit_rsp32k_assignment(assignment.operation)
        raise NotImplementedError(f"Unsupported operation type: {type(assignment.operation)}")

    def visit_read(self: "BLEIRInterpreter",
                   read: READ) -> Sequence[PatchPair]:
        operator_patterns = READ_PATTERNS[read.operator]
        expression_patterns = operator_patterns[type(read.rvalue)]
        self.has_read = True

        if isinstance(read.rvalue, UNARY_EXPR):
            vrs_or_src_or_bit = self.visit_unary_expr(read.rvalue)
            rvalue_patterns = expression_patterns[type(read.rvalue.expression)]
            if callable(rvalue_patterns):
                dispatch = rvalue_patterns
            else:
                dispatch = rvalue_patterns[read.rvalue.expression.operator]
            patch_pair = dispatch(self.diri, self.mask, vrs_or_src_or_bit)
            return [patch_pair]

        if isinstance(read.rvalue, BINARY_EXPR):
            sb, src = self.visit_binary_expr(read.rvalue)
            if callable(expression_patterns):
                dispatch = expression_patterns
            else:
                rvalue_patterns = expression_patterns[read.rvalue.operator]
                if callable(rvalue_patterns):
                    dispatch = rvalue_patterns
                else:
                    rvalue = read.rvalue.left_operand
                    r2value = read.rvalue.right_operand
                    negation_key = (rvalue.operator, r2value.operator)
                    dispatch = rvalue_patterns[negation_key]
            patch_pair = dispatch(self.diri, self.mask, sb, src)
            return [patch_pair]

        raise NotImplementedError(
            f"Unsupported rvalue type: {type(read.rvalue)}")

    def visit_write(self: "BLEIRInterpreter",
                    write: WRITE) -> Sequence[PatchPair]:

        if write.operator is ASSIGN_OP.COND_EQ \
           and write.rvalue.expression in INVERT_SRC_EXPR:
            rvalue = write.rvalue
            expression = INVERT_SRC_EXPR[rvalue.expression]
            if rvalue.operator is UNARY_OP.NEGATE:
                operator = None
            else:
                operator = UNARY_OP.NEGATE
            rvalue = rvalue.having(expression=expression, operator=operator)
            write = write.having(rvalue=rvalue)

        vrs = self.visit_sb_expr(write.lvalue)
        src = self.visit_unary_src(write.rvalue)

        if write.operator is ASSIGN_OP.EQ:
            if write.rvalue.operator is UNARY_OP.NEGATE:
                patch_pair = self.diri.sb_from_inv_src(self.mask, vrs, src)
            else:
                patch_pair = self.diri.sb_from_src(self.mask, vrs, src)
        elif write.operator is ASSIGN_OP.COND_EQ:
            if write.rvalue.operator is UNARY_OP.NEGATE:
                patch_pair = self.diri.sb_cond_equals_inv_src(self.mask, vrs, src)
            else:
                patch_pair = self.diri.sb_cond_equals_src(self.mask, vrs, src)
        else:
            raise NotImplementedError(
                f"Unsupported WRITE operator: {write.operator}")

        return [patch_pair]

    def visit_broadcast(self: "BLEIRInterpreter",
                        broadcast: BROADCAST) -> PatchPair:

        if broadcast.lvalue == BROADCAST_EXPR.GL:
            patch_pair = self.diri.gl_from_rl(self.mask)
            return [patch_pair]

        if broadcast.lvalue == BROADCAST_EXPR.GGL:
            if isinstance(broadcast.rvalue, BINARY_EXPR):
                binary_expr = broadcast.rvalue
                if binary_expr.operator is not BINOP.AND:
                    raise ValueError(
                        f"Only conjunction is supported: {broadcast}")
                if binary_expr.left_operand is not RL_EXPR.RL:
                    raise ValueError(
                        f"Only RL is supported as a <SRC>: {broadcast}")
                lx_addr = binary_expr.right_operand
                if not isinstance(lx_addr, L1_REG) \
                   and not (isinstance(lx_addr, LXRegWithOffsets)
                            and isinstance(lx_addr.parameter, L1_REG)):
                    raise ValueError(
                        f"Only an L1 address is supported: {broadcast}")
                l1_addr = self.visit_lx_addr(lx_addr)
                patch_pair = self.diri.ggl_from_rl_and_l1(self.mask, l1_addr)
            else:
                patch_pair = self.diri.ggl_from_rl(self.mask)
            return [patch_pair]

        if broadcast.lvalue == BROADCAST_EXPR.RSP16:
            patch_pair = self.diri.rsp16_from_rl(self.mask)
            return [patch_pair]

        raise NotImplementedError(
            f"Unsupported lvalue type ({type(broadcast.lvalue).__name__}): "
            f"{broadcast.lvalue}")

    def visit_ggl_assignment(
                self: "BLEIRInterpreter",
                ggl_assignment: GGL_ASSIGNMENT) \
            -> Sequence[PatchPair]:
        lx_addr = self.visit_lx_addr(ggl_assignment.rvalue)
        patch_pair = self.diri.ggl_from_l1(lx_addr)
        return [patch_pair]

    def visit_lgl_assignment(
                self: "BLEIRInterpreter",
                lgl_assignment: LGL_ASSIGNMENT) \
            -> Sequence[PatchPair]:
        lx_reg_or_offset = lgl_assignment.rvalue
        lx_addr = self.visit_lx_addr(lx_reg_or_offset)

        if isinstance(lx_reg_or_offset, L1_REG) \
           or (isinstance(lx_reg_or_offset, LXRegWithOffsets)
               and isinstance(lx_reg_or_offset.parameter, L1_REG)):
            patch_pair = self.diri.lgl_from_l1(lx_addr)

        elif isinstance(lx_reg_or_offset, L2_REG) \
             or (isinstance(lx_reg_or_offset, LXRegWithOffsets)
                 and isinstance(lx_reg_or_offset.parameter, L2_REG)):
            patch_pair = self.diri.lgl_from_l2(lx_addr)

        else:
            raise ValueError(
                f"Unsupported type for rvalue "
                f"({lx_reg_or_offset.__class__.__name__}): "
                f"{lx_reg_or_offset}")

        return [patch_pair]

    def visit_lx_assignment(
                self: "BLEIRInterpreter",
                lx_assignment: LX_ASSIGNMENT) \
            -> Sequence[PatchPair]:

        lx_reg_or_offset = lx_assignment.lvalue
        lx_addr = self.visit_lx_addr(lx_reg_or_offset)
        rvalue = lx_assignment.rvalue

        if isinstance(lx_reg_or_offset, L1_REG) \
           or (isinstance(lx_reg_or_offset, LXRegWithOffsets)
               and isinstance(lx_reg_or_offset.parameter, L1_REG)):

            if isinstance(rvalue, GGL_EXPR):
                update_fn = self.diri.l1_from_ggl
            elif isinstance(rvalue, LGL_EXPR):
                update_fn = self.diri.l1_from_lgl
            else:
                raise ValueError(
                    f"Unsupported type for rvalue "
                    f"({rvalue.__class__.__name__}): {lx_assignment}")

        elif isinstance(lx_reg_or_offset, L2_REG) \
             or (isinstance(lx_reg_or_offset, LXRegWithOffsets)
                 and isinstance(lx_reg_or_offset.parameter, L2_REG)):

            if isinstance(rvalue, LGL_EXPR):
                update_fn = self.diri.l2_from_lgl
            else:
                raise ValueError(
                    f"Unsupported type for rvalue "
                    f"({rvalue.__class__.__name__}): {lx_assignment}")

        else:
            raise ValueError(
                f"Unsupported type for rvalue "
                f"({lx_reg_or_offset.__class__.__name__}): "
                f"{lx_reg_or_offset}")

        patch_pair = update_fn(lx_addr)
        return [patch_pair]

    def visit_rsp16_assignment(self: "BLEIRInterpreter",
                               rsp16_assignment: RSP16_ASSIGNMENT) \
            -> Sequence[PatchPair]:

        if rsp16_assignment.rvalue is RSP16_RVALUE.RSP256:
            patch_pair = self.diri.rsp16_from_rsp256()
        else:
            raise NotImplementedError(
                f"Unsupported RSP16_RVALUE: {rsp16_assignment.rvalue}")
        return [patch_pair]

    def visit_rsp256_assignment(self: "BLEIRInterpreter",
                                rsp256_assignment: RSP256_ASSIGNMENT) \
            -> Sequence[PatchPair]:

        if rsp256_assignment.rvalue is RSP256_RVALUE.RSP16:
            patch_pair = self.diri.rsp256_from_rsp16()
        elif rsp256_assignment.rvalue is RSP256_RVALUE.RSP2K:
            patch_pair = self.diri.rsp256_from_rsp2k()
        else:
            raise NotImplementedError(
                f"Unsupported RSP256_RVALUE: {rsp256_assignment.rvalue}")
        return [patch_pair]

    def visit_rsp2k_assignment(self: "BLEIRInterpreter",
                               rsp2k_assignment: RSP2K_ASSIGNMENT) \
            -> Sequence[PatchPair]:

        if rsp2k_assignment.rvalue is RSP2K_RVALUE.RSP256:
            patch_pair = self.diri.rsp2k_from_rsp256()
        elif rsp2k_assignment.rvalue is RSP2K_RVALUE.RSP32K:
            patch_pair = self.diri.rsp2k_from_rsp32k()
        else:
            raise NotImplementedError(
                f"Unsupported RSP2K_RVALUE: {rsp2k_assignment.rvalue}")
        return [patch_pair]

    def visit_rsp32k_assignment(
                self: "BLEIRInterpreter",
                rsp32k_assignment: RSP32K_ASSIGNMENT) \
            -> Sequence[PatchPair]:

        if rsp32k_assignment.rvalue is RSP32K_RVALUE.RSP2K:
            patch_pair = self.diri.rsp32k_from_rsp2k()
        else:
            raise NotImplementedError(
                f"Unsupported RSP32K_RVALUE: {rsp32k_assignment.rvalue}")
        return [patch_pair]

    def visit_binary_expr(self: "BLEIRInterpreter",
                          binary_expr: BINARY_EXPR) \
            -> Tuple[Union[np.ndarray, Sequence[np.ndarray]], np.ndarray]:

        vrs = self.visit_unary_sb(binary_expr.left_operand)
        src = self.visit_unary_src(binary_expr.right_operand)
        return vrs, src

    def visit_unary_expr(self: "BLEIRInterpreter",
                         unary_expr: UNARY_EXPR) \
            -> Union[int, np.ndarray, Sequence[int], Sequence[np.ndarray]]:

        if isinstance(unary_expr.expression, UNARY_SB):
            return self.visit_unary_sb(unary_expr.expression)
        elif isinstance(unary_expr.expression, UNARY_SRC):
            return self.visit_unary_src(unary_expr.expression)
        elif isinstance(unary_expr.expression, BIT_EXPR):
            return self.visit_bit_expr(unary_expr.expression)
        else:
            raise NotImplementedError(
                f"Unsupported expression type: {type(unary_expr.expression)}")

    def visit_unary_src(self: "BLEIRInterpreter",
                        unary_src: UNARY_SRC) -> np.ndarray:
        src = self.visit_src_expr(unary_src.expression)
        # Not negating the src, here, is intentional. Negation is handled by the DIRI callback
        # function (e.g. DIRI.rl_and_equals_inv_src)
        # ------------------------------------------------------------------------------------
        # if unary_src.operator is UNARY_OP.NEGATE:
        #     src = ~src
        return src

    def visit_unary_sb(self: "BLEIRInterpreter",
                       unary_sb: UNARY_SB) -> Union[np.ndarray, Sequence[np.ndarray]]:
        return self.visit_sb_expr(unary_sb.expression)

    def visit_sb_expr(self: "BLEIRInterpreter",
                      sb_expr: SB_EXPR) \
            -> Union[np.ndarray, Sequence[np.ndarray]]:
        vrs = []
        for parameter in sb_expr:
            if isinstance(parameter, RN_REG):
                rn_reg = parameter
                vrs.append(self.visit_rn_reg(rn_reg))
            elif isinstance(parameter, RE_REG):
                re_reg = parameter
                for vr in self.visit_re_reg(re_reg):
                    vrs.append(vr)
            elif isinstance(parameter, EWE_REG):
                ewe_reg = parameter
                for vr in self.visit_ewe_reg(ewe_reg):
                    vrs.append(vr)
            elif isinstance(parameter, ExtendedRegister):
                xe_reg = parameter
                for vr in self.visit_extended_register(xe_reg):
                    vrs.append(vr)
            elif parameter is not None:
                raise ValueError(
                    f"Unsupported parameter type "
                    f"({parameter.__class__.__name__}): {parameter}")
        return vrs

    def visit_register_parameter(
                self: "BLEIRInterpreter",
                register: RegisterParameter) \
            -> Union[int, np.ndarray, Sequence[int]]:

        if self.data_by_parameter_id is not None \
           and register.identifier in self.data_by_parameter_id:
            return self.data_by_parameter_id[register.identifier]

        if isinstance(register, RE_REG):
            return self.visit_re_reg(register)

        if isinstance(register, EWE_REG):
            return self.visit_ewe_reg(register)

        if register.identifier in self.values_by_parameter_id:
            return self.values_by_parameter_id[register.identifier]

        if isinstance(register, SM_REG) and register.constant_value is not None:
            return register.constant_value

        if isinstance(register, RN_REG) and register.row_number is not None:
            return register.row_number

        if isinstance(register, L1_REG) and register.bank_group_row is not None:
            return register.bank_group_row

        if isinstance(register, L2_REG) and register.value is not None:
            return register.value

        raise ValueError(
            f"Unsupported register type ({register.__class__.__name__}): "
            f"{register}")

    def visit_rn_reg(self: "BLEIRInterpreter",
                     rn_reg: RN_REG) -> Union[Integer, np.ndarray]:
        return self.visit_register_parameter(rn_reg)

    def visit_re_reg(self: "BLEIRInterpreter",
                     re_reg: RE_REG) -> Sequence[Union[Integer, np.ndarray]]:

        row_mask = None

        if re_reg.identifier in self.values_by_parameter_id:
            row_mask = self.values_by_parameter_id[re_reg.identifier]
        elif re_reg.row_mask is not None:
            row_mask = re_reg.row_mask
        elif re_reg.rows is not None:
            sbs_and_vrs = []
            for row in re_reg.rows:
                if isinstance(row, RE_REG):
                    sbs_and_vrs.extend(self.visit_re_reg(row))
                elif isinstance(row, RN_REG):
                    sbs_and_vrs.append(self.visit_rn_reg(row))
                else:
                    raise ValueError(
                        f"Unsupported RE_REG row type "
                        f"({row.__class__.__name__}): {row}")
        else:
            raise ValueError(
                f"Unable to determine row_mask for RE_REG: {re_reg}")

        if row_mask is not None:
            sbs_and_vrs = sorted(sb for sb in range(NSB)
                                 if row_mask & (0x000001 << sb) != 0)

        return sbs_and_vrs

    def lookup_wordline_mask(self: "BLEIRInterpreter",
                             ewe_reg: EWE_REG) -> int:
        if ewe_reg.identifier in self.values_by_parameter_id:
            wordline_mask = self.values_by_parameter_id[ewe_reg.identifier]
        elif ewe_reg.wordline_mask is not None:
            wordline_mask = ewe_reg.wordline_mask
        else:
            raise ValueError(
                f"Unable to determine wordline_mask for EWE_REG: "
                f"{ewe_reg}")
        return wordline_mask

    def lookup_ewe_group(self: "BLEIRInterpreter",
                         ewe_reg: EWE_REG) -> int:
        wordline_mask = self.lookup_wordline_mask(ewe_reg)
        group = wordline_mask >> 8

        if group > 2:
            raise ValueError(
                f"Group for EWE_REG must be 0, 1, or 2: "
                f"{wordline_mask:03X}")

        return group

    def visit_ewe_reg(self: "BLEIRInterpreter",
                      ewe_reg: EWE_REG) -> Sequence[Integer]:

        wordline_mask = self.lookup_wordline_mask(ewe_reg)
        group = wordline_mask >> 8
        offset = group * 8

        sbs = [sb + offset for sb in range(8)
               if wordline_mask & (0x01 << sb) != 0]

        return sbs

    def visit_extended_register(
            self: "BLEIRInterpreter",
            extended_register: ExtendedRegister) -> Sequence[Integer]:

        register_parameter = extended_register.register
        sbs = self.visit_register_parameter(register_parameter)

        num_shifted_bits = extended_register.num_shifted_bits
        if num_shifted_bits > 0:
            if isinstance(register_parameter, RE_REG):
                upper_bound = NSB
            elif isinstance(register_parameter, EWE_REG):
                group = self.lookup_ewe_group(register_parameter)
                upper_bound = (1 + group) * 8

            sbs = [sb + num_shifted_bits for sb in sbs
                   if sb + num_shifted_bits < upper_bound]

        if extended_register.operator is UNARY_OP.NEGATE:
            if isinstance(register_parameter, RE_REG):
                available_sbs = set(range(NSB))
            elif isinstance(register_parameter, EWE_REG):
                group = self.lookup_ewe_group(register_parameter)
                lower_bound = group * 8
                upper_bound = lower_bound + 8
                available_sbs = set(range(lower_bound, upper_bound))
            else:
                raise ValueError(
                    f"Unsupported register type "
                    f"({register_parameter.__class__.__name__}): "
                    f"{register_parameter}")

            sbs = sorted(available_sbs - set(sbs))

        return sbs

    def visit_sm_reg(self: "BLEIRInterpreter", sm_reg: SM_REG) -> Mask:
        sections = self.visit_register_parameter(sm_reg)
        if isinstance(sections, Integer.__args__):
            sections = f"0x{sections:04X}"
        return Mask(sections)

    def visit_lx_addr(self: "BLEIRInterpreter",
                      lx_addr: LX_ADDR) -> int:

        if isinstance(lx_addr, LXParameter.__args__):
            return self.visit_lx_parameter(lx_addr)

        if isinstance(lx_addr, LXRegWithOffsets):
            return self.visit_lx_reg_with_offsets(lx_addr)

        raise ValueError(
            f"Unsupported type for lx_addr "
            f"({lx_addr.__class__.__name__}): {lx_addr}")

    def visit_lx_reg_with_offsets(
            self: "BLEIRInterpreter",
            lx_reg_with_offsets: LXRegWithOffsets) -> int:

        offset = lx_reg_with_offsets.offset
        lx_reg = lx_reg_with_offsets.parameter
        lx_addr = self.visit_lx_parameter(lx_reg)
        return lx_addr + offset

    def visit_lx_parameter(self: "BLEIRInterpreter",
                           lx_parameter: LXParameter) -> int:

        if isinstance(lx_parameter, L1_REG):
            l1_reg = lx_parameter
            return self.visit_l1_reg(l1_reg)

        if isinstance(lx_parameter, L2_REG):
            l2_reg = lx_parameter
            return self.visit_l2_reg(l2_reg)

        raise ValueError(
            f"Unsupported type for lx_parameter "
            f"({lx_parameter.__class__.__name__}): {lx_parameter}")

    def visit_l1_reg(self: "BLEIRInterpreter", l1_reg: L1_REG) -> int:
        return self.visit_register_parameter(l1_reg)

    def visit_l2_reg(self: "BLEIRInterpreter", l2_reg: L2_REG) -> int:
        return self.visit_register_parameter(l2_reg)

    def visit_bit_expr(self: "BLEIRInterpreter", bit_expr: BIT_EXPR) -> int:
        return int(bit_expr.value)

    def visit_src_expr(self: "BLEIRInterpreter",
                       src_expr: SRC_EXPR) -> np.ndarray:
        if src_expr == SRC_EXPR.RL:
            return self.diri.RL()
        elif src_expr == SRC_EXPR.NRL:
            return self.diri.NRL()
        elif src_expr == SRC_EXPR.ERL:
            return self.diri.ERL()
        elif src_expr == SRC_EXPR.WRL:
            return self.diri.WRL()
        elif src_expr == SRC_EXPR.SRL:
            return self.diri.SRL()
        elif src_expr == SRC_EXPR.GL:
            return self.diri.GL
        elif src_expr == SRC_EXPR.GGL:
            return self.diri.GGL
        elif src_expr == SRC_EXPR.RSP16:
            return self.diri.RSP16
        elif src_expr in INVERT_SRC_EXPR:
            return np.logical_not(self.visit_src_expr(INVERT_SRC_EXPR[src_expr]))
        else:
            raise NotImplementedError(f"Unsupported SRC_EXPR: {src_expr}")

    def visit_special(
            self: "BLEIRInterpreter",
            special: SPECIAL) -> Sequence[PatchPair]:
        if special == SPECIAL.NOOP:
            patch_pair = self.diri.noop()
        elif special == SPECIAL.FSEL_NOOP:
            patch_pair = self.diri.fsel_noop()
        elif special == SPECIAL.RSP_END:
            patch_pair = self.diri.rsp_end()
        elif special == SPECIAL.RSP_START_RET:
            patch_pair = self.diri.rsp_start_ret()
        elif special == SPECIAL.L2_END:
            patch_pair = self.diri.l2_end()
        else:
            raise NotImplementedError(f"Unsupported special value: {special}")
        return [patch_pair]
