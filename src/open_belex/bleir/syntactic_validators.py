r"""
By Dylon Edwards
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Sequence, Type, Union

from open_belex.bleir.inspectors import is_bleir, is_type_hint
from open_belex.bleir.types import (ASSIGN_OP, ASSIGNMENT, BINARY_EXPR, BINOP,
                                    BIT_EXPR, BROADCAST, BROADCAST_EXPR, MASK,
                                    MASKED, READ, RL_EXPR, RN_REG,
                                    RSP2K_ASSIGNMENT, RSP2K_EXPR,
                                    RSP16_ASSIGNMENT, RSP16_EXPR,
                                    RSP32K_ASSIGNMENT, RSP32K_EXPR,
                                    RSP256_ASSIGNMENT, RSP256_EXPR, SB_EXPR,
                                    SHIFTED_SM_REG, SM_REG, SPECIAL, SRC_EXPR,
                                    STATEMENT, UNARY_EXPR, UNARY_OP, UNARY_SB,
                                    UNARY_SRC, WRITE, AllocatedRegister,
                                    Example, Fragment, FragmentCaller,
                                    FragmentCallerCall, InlineComment,
                                    MultiLineComment, MultiStatement,
                                    ReadWriteInhibit, SingleLineComment,
                                    Snippet, SyntacticError, TrailingComment,
                                    ValueParameter, instance_members_of)
from open_belex.bleir.walkables import BLEIRListener, BLEIRVisitor, BLEIRWalker

MAX_C_ID_LEN = 31

Fn_or_BLEIR = Any


def compile_id_pattern(suffix=""):
    max_id_len = MAX_C_ID_LEN - len(suffix)
    non_digit = "[a-zA-Z_]"
    digit = "[0-9]"
    # hex_digit = "[0-9a-fA-F]"
    # hex_quad = f"(?:{hex_digit}{{4,4}})"
    # TODO: Investigate which, if any, unicode escape sequences are supported by the APL compiler
    # As it stands, the compiler throws a lot of erors like the following:
    #     - error: universal character \ueb09 is not valid in an identifier
    #     - error: universal character \U00109d2A is not valid in an identifier
    # universal_char_name = f"(?:\\\\u{hex_quad}|\\\\U00(?:0[0-9]|10){hex_quad}|[\x80-\U0010FFFF])"
    # TODO: Even literal unicode chars cause errors:
    #     - fe1eadb07c8e0f1e82a1d4ede399ce2b-apl-funcs.apl:14: error: syntax error
    # universal_char_name = f"[\x80-\U0010FFFF]"
    # id_non_digit = f"(?:{non_digit}|{universal_char_name})"
    id_non_digit = non_digit
    id_pattern = f"{id_non_digit}(?:{id_non_digit}|{digit}){{0,{max_id_len - 1}}}"
    return re.compile(id_pattern, re.UNICODE)


VALUE_PARAM_ID_PATTERN = compile_id_pattern(suffix="")
FRAGMENT_ID_PATTERN = compile_id_pattern(suffix="_caller")
# REGISTER_PARAM_ID_PATTERN = compile_id_pattern(suffix="_rp")
REGISTER_PARAM_ID_PATTERN = compile_id_pattern()

# Exception to the name length rule: users are expected to keep the name of the
# fragment caller less than 31 chars without requiring the "_caller" suffix.
CUSTOM_FRAGMENT_ID_PATTERN = compile_id_pattern(suffix="")


class IdentifierValidator(BLEIRListener):

    def enter_value_parameter(self: "IdentifierValidator",
                              value_parameter: ValueParameter) -> None:
        if not VALUE_PARAM_ID_PATTERN.fullmatch(value_parameter.identifier):
            raise SyntacticError(
                f"Value id [{value_parameter.identifier}] failed to validation: "
                f"{VALUE_PARAM_ID_PATTERN}")

    def enter_fragment(self: "IdentifierValidator",
                       fragment: Fragment) -> None:
        if not FRAGMENT_ID_PATTERN.fullmatch(fragment.identifier):
            raise SyntacticError(
                f"Fragment id [{fragment.identifier}] "
                f"(length={len(fragment.identifier)}) failed to validation: "
                f"{FRAGMENT_ID_PATTERN}")

    def enter_rn_reg(self: "IdentifierValidator",
                     rn_reg: RN_REG) -> None:
        if not REGISTER_PARAM_ID_PATTERN.fullmatch(rn_reg.identifier):
            raise SyntacticError(
                f"Register id [{rn_reg.identifier}] "
                f"(length={len(rn_reg.identifier)}) failed to validation: "
                f"{REGISTER_PARAM_ID_PATTERN}")

    def enter_sm_reg(self: "IdentifierValidator",
                     sm_reg: SM_REG) -> None:
        if not REGISTER_PARAM_ID_PATTERN.fullmatch(sm_reg.identifier):
            raise SyntacticError(
                f"Register id [{sm_reg.identifier}] "
                f"(length={len(sm_reg.identifier)}) failed to validation: "
                f"{REGISTER_PARAM_ID_PATTERN}")


# See: APL Command Set (README.md)
ASSIGNMENT_PATTERNS = set([
    "SB = <SRC>",   # WRITE LOGIC

    # FIXME: Add these to the README.md
    "SB = ~<SRC>",

    # FIXME: Add these to the README.md
    "SB ?= <SRC>",

    # FIXME: Add these to the README.md
    "SB ?= ~<SRC>",

    "RL = <BIT>",          # READ LOGIG #1 and #2

    "RL = <SB>",           # READ LOGIC #3
    "RL = <SRC>",          # READ LOGIC #4
    "RL = <SB> & <SRC>",   # READ LOGIC #5

    # FIXME: Add these to the README.md
    "RL = ~<SB>",
    "RL = ~<SRC>",

    "RL |= <SB>",          # READ LOGIC #10
    "RL |= <SRC>",         # READ LOGIC #11
    "RL |= ~<SRC>",        # READ LOGIC #11
    "RL |= <SB> & <SRC>",  # READ LOGIC #12
    "RL |= <SB> & ~<SRC>", # READ LOGIC #12

    "RL &= <SB>",          # READ LOGIC #13
    "RL &= <SRC>",         # READ LOGIC #14
    "RL &= <SB> & <SRC>",  # READ LOGIC #15
    "RL &= <SB> & ~<SRC>", # READ LOGIC #15

    "RL ^= <SB>",          # READ LOGIC #18
    "RL ^= <SRC>",         # READ LOGIC #19
    "RL ^= ~<SRC>",        # READ LOGIC #19
    "RL ^= <SB> & <SRC>",  # READ LOGIC #20
    "RL ^= <SB> & ~<SRC>", # READ LOGIC #20

    "RL = <SB> | <SRC>",   # READ LOGIC #6
    "RL = <SB> | ~<SRC>",  # READ LOGIC #6
    "RL = <SB> ^ <SRC>",   # READ LOGIC #7

    "RL = ~<SB> & <SRC>",  # READ LOGIC #8
    "RL = <SB> & ~<SRC>",  # READ LOGIC #9
    "RL = <SB> ^ ~<SRC>",  # FIXME: UNDOCUMENTED VARIATION OF READ LOGIC #9

    "RL &= ~<SB>",         # READ LOGIC #16
    "RL &= ~<SRC>",        # READ LOGIC #17

    "RL = ~<SB> & ~<SRC>",

    "GL = RL",             # R-SEL LOGIC
    "GGL = RL",            # R-SEL LOGIC
    "RSP16 = RL",          # R-SEL LOGIC

    "RSP256 = RSP16",      # SPECIAL ASSIGNMENT
    "RSP2K = RSP256",      # SPECIAL ASSIGNMENT
    "RSP32K = RSP2K",      # SPECIAL ASSIGNMENT

    "RWINH_SET",
    "RWINH_RST",
])


SRC_PATTERNS = {
    SRC_EXPR.RL: "<SRC>",
    SRC_EXPR.NRL: "<SRC>",
    SRC_EXPR.ERL: "<SRC>",
    SRC_EXPR.WRL: "<SRC>",
    SRC_EXPR.SRL: "<SRC>",
    SRC_EXPR.GL: "<SRC>",
    SRC_EXPR.GGL: "<SRC>",
    SRC_EXPR.RSP16: "<SRC>",

    SRC_EXPR.INV_RL: "~<SRC>",
    SRC_EXPR.INV_NRL: "~<SRC>",
    SRC_EXPR.INV_ERL: "~<SRC>",
    SRC_EXPR.INV_WRL: "~<SRC>",
    SRC_EXPR.INV_SRL: "~<SRC>",
    SRC_EXPR.INV_GL: "~<SRC>",
    SRC_EXPR.INV_GGL: "~<SRC>",
    SRC_EXPR.INV_RSP16: "~<SRC>",
}


UNARY_SRC_PATTERNS = {
    (None, "<SRC>"): "<SRC>",
    (None, "~<SRC>"): "~<SRC>",
    (UNARY_OP.NEGATE, "<SRC>"): "~<SRC>",
    (UNARY_OP.NEGATE, "~<SRC>"): "<SRC>",
}


def has_context(context):

    def decorator(fn):

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            self.context = context
            retval = fn(self, *args, **kwargs)
            self.context = None
            return retval

        return wrapper

    return decorator


@dataclass
class AssignmentPatternVisitor(BLEIRVisitor):
    context: Optional[Type[Union[READ, WRITE, BROADCAST]]] = None

    @has_context(None)
    def visit_assignment(self: "AssignmentPatternVisitor", assignment: ASSIGNMENT) -> str:
        if isinstance(assignment.operation, READ):
            pattern = self.visit_read(assignment.operation)
        elif isinstance(assignment.operation, WRITE):
            pattern = self.visit_write(assignment.operation)
        elif isinstance(assignment.operation, BROADCAST):
            pattern = self.visit_broadcast(assignment.operation)
        elif isinstance(assignment.operation, RSP16_ASSIGNMENT):
            pattern = self.visit_rsp16_assignment(assignment.operation)
        elif isinstance(assignment.operation, RSP256_ASSIGNMENT):
            pattern = self.visit_rsp256_assignment(assignment.operation)
        elif isinstance(assignment.operation, RSP2K_ASSIGNMENT):
            pattern = self.visit_rsp2k_assignment(assignment.operation)
        elif isinstance(assignment.operation, RSP32K_ASSIGNMENT):
            pattern = self.visit_rsp32k_assignment(assignment.operation)
        elif isinstance(assignment.operation, ReadWriteInhibit):
            pattern = self.visit_read_write_inhibit(assignment.operation)
        else:
            raise NotImplementedError(f"Unsupported operation type ({assignment.operation.__class__.__name__}): {assignment.operation}")

        return pattern

    def visit_read_write_inhibit(self: "AssignmentPatternVisitor",
                                 read_write_inhibit: ReadWriteInhibit) -> str:
        return read_write_inhibit.value

    @has_context(READ)
    def visit_read(self: "AssignmentPatternVisitor", read: READ) -> str:
        if isinstance(read.rvalue, UNARY_EXPR):
            rvalue = self.visit_unary_expr(read.rvalue)
        elif isinstance(read.rvalue, BINARY_EXPR):
            rvalue = self.visit_binary_expr(read.rvalue)
        else:
            raise NotImplementedError(f"Unsupported rvalue type: {type(read.rvalue).__name__}: {read.rvalue}")
        operator = self.visit_assign_op(read.operator)
        return f"RL {operator} {rvalue}"

    @has_context(WRITE)
    def visit_write(self: "AssignmentPatternVisitor", write: WRITE) -> str:
        operator = self.visit_assign_op(write.operator)
        lvalue = self.visit_sb_expr(write.lvalue)
        rvalue = self.visit_unary_src(write.rvalue)
        return f"{lvalue} {operator} {rvalue}"

    @has_context(BROADCAST)
    def visit_broadcast(self: "AssignmentPatternVisitor", broadcast: BROADCAST) -> str:
        lvalue = self.visit_broadcast_expr(broadcast.lvalue)
        return f"{lvalue} = RL"

    def visit_rsp16_assignment(self: "AssignmentPatternVisitor", rsp16_assignment: RSP16_ASSIGNMENT) -> str:
        return f"RSP16 = {rsp16_assignment.rvalue}"

    def visit_rsp256_assignment(self: "AssignmentPatternVisitor", rsp256_assignment: RSP256_ASSIGNMENT) -> str:
        return f"RSP256 = {rsp256_assignment.rvalue}"

    def visit_rsp2k_assignment(self: "AssignmentPatternVisitor", rsp2k_assignment: RSP2K_ASSIGNMENT) -> str:
        return f"RSP2K = {rsp2k_assignment.rvalue}"

    def visit_rsp32k_assignment(self: "AssignmentPatternVisitor", rsp32k_assignment: RSP32K_ASSIGNMENT) -> str:
        return f"RSP32K = {rsp32k_assignment.rvalue}"

    def visit_binary_expr(self: "AssignmentPatternVisitor", binary_expr: BINARY_EXPR) -> str:
        operator = self.visit_binop(binary_expr.operator)
        left_operand = self.visit_unary_sb(binary_expr.left_operand)
        right_operand = self.visit_unary_src(binary_expr.right_operand)
        return f"{left_operand} {operator} {right_operand}"

    def visit_unary_expr(self: "AssignmentPatternVisitor", unary_expr: UNARY_EXPR) -> str:
        if isinstance(unary_expr.expression, UNARY_SB):
            return self.visit_unary_sb(unary_expr.expression)
        elif isinstance(unary_expr.expression, UNARY_SRC):
            return self.visit_unary_src(unary_expr.expression)
        elif isinstance(unary_expr.expression, BIT_EXPR):
            return self.visit_bit_expr(unary_expr.expression)
        else:
            raise NotImplementedError(f"Unsupported expression type: {type(unary_expr.expression).__name__}: {unary_expr.expression}")

    def visit_unary_src(self: "AssignmentPatternVisitor", unary_src: UNARY_SRC) -> str:
        expression = self.visit_src_expr(unary_src.expression)
        pattern = (unary_src.operator, expression)
        return UNARY_SRC_PATTERNS[pattern]

    def visit_unary_sb(self: "AssignmentPatternVisitor", unary_sb: UNARY_SB) -> str:
        expression = self.visit_sb_expr(unary_sb.expression)
        operator = self.visit_unary_op(unary_sb.operator)
        return f"{operator}{expression}"

    def visit_sb_expr(self: "AssignmentPatternVisitor", sb_expr: SB_EXPR) -> str:
        if self.context is READ:
            return "<SB>"
        return "SB"

    def visit_bit_expr(self: "AssignmentPatternVisitor", bit_expr: BIT_EXPR) -> str:
        return "<BIT>"

    def visit_src_expr(self: "AssignmentPatternVisitor", src_expr: SRC_EXPR) -> str:
        return SRC_PATTERNS[src_expr]

    def visit_broadcast_expr(self: "AssignmentPatternVisitor", broadcast_expr: BROADCAST_EXPR) -> str:
        return str(broadcast_expr)

    def visit_assign_op(self: "AssignmentPatternVisitor", assign_op: ASSIGN_OP) -> str:
        return str(assign_op)

    def visit_binop(self: "AssignmentPatternVisitor", binop: BINOP) -> str:
        return str(binop)

    def visit_unary_op(self: "AssignmentPatternVisitor", unary_op: Optional[UNARY_OP]) -> str:
        if unary_op is None:
            return ""
        return str(unary_op)


@dataclass
class AssignmentPatternValidator(BLEIRListener):
    visitor: AssignmentPatternVisitor = field(default_factory=AssignmentPatternVisitor)

    def enter_assignment(self: "AssignmentPatternValidator", assignment: ASSIGNMENT) -> None:
        pattern = self.visitor.visit_assignment(assignment)
        if pattern not in ASSIGNMENT_PATTERNS:
            raise SyntacticError(f"Unsupported assignment pattern: {pattern}")
        return assignment


def validate_assignment_pattern(fn_or_bleir: Fn_or_BLEIR) -> Optional[Callable]:
    """Validates the assignment patterns of the values returned from the given
    function. If a BLEIR instance is provided instead, it is validated and no
    value is returned."""

    walker = BLEIRWalker()
    validator = AssignmentPatternValidator()

    def validate(bleir: Any) -> None:
        walker.walk(validator, bleir)

    if is_bleir(fn_or_bleir):
        bleir = fn_or_bleir
        validate(bleir)
        return None

    # If it isn't a BLEIR or a function to decorate, raise an error
    if not callable(fn_or_bleir):
        raise NotImplementedError(f"Unsupported fn_or_bleir type ({fn_or_bleir.__class__.__name__}): {fn_or_bleir}")

    fn = fn_or_bleir

    @wraps(fn)
    def wrapper(*args, **kwargs):
        bleir = fn(*args, **kwargs)
        validate(bleir)
        return bleir

    return wrapper


class BLEIRTypeValidator(BLEIRVisitor):

    @staticmethod
    def assert_type_is(actual_value: Any, expected_type: Type) -> bool:
        actual_type = type(actual_value)
        if actual_type is not expected_type:
            raise SyntacticError(f"{actual_type.__name__} is not the expected type: {expected_type.__name__}")

    @staticmethod
    def assert_optional_type_is(actual_value: Any, expected_type: Type) -> bool:
        actual_type = type(actual_value)
        if actual_value is not None and actual_type is not expected_type:
            raise SyntacticError(f"{actual_type.__name__} is not the expected type: {expected_type.__name__}")

    @staticmethod
    def assert_type_in(actual_value: Any, expected_types: Sequence[Type]) -> bool:
        actual_type = type(actual_value)
        if actual_type not in expected_types:
            expected_type_names = ", ".join([expected_type.__name__ for expected_type in expected_types])
            raise SyntacticError(f"{actual_type.__name__} is not among the expected types: [{expected_type_names}]")

    @staticmethod
    def assert_optional_type_in(actual_value: Any, expected_types: Sequence[Type]) -> bool:
        actual_type = type(actual_value)
        if actual_value is not None and actual_type not in expected_types:
            expected_type_names = ", ".join([expected_type.__name__ for expected_type in expected_types])
            raise SyntacticError(f"{actual_type.__name__} is not among the expected types: [{expected_type_names}]")

    def assert_satisfies_dict(self: "BLEIRTypeValidator", context: Any, hint: Any) -> bool:
        if not isinstance(context, dict):
            return False

        key_hint, value_hint = hint.__args__
        for key, value in context.items():
            if key_hint is not Any \
               and not self.assert_satisfies_hint(key, key_hint):
                return False
            if value_hint is not Any \
               and not self.assert_satisfies_hint(value, value_hint):
                return False

        return True

    def assert_satisfies_tuple(self: "BLEIRTypeValidator", context: Any, hint: Any) -> bool:
        if len(context) != len(hint.__args__):
            return False

        for elem, elem_hint in zip(context, hint.__args__):
            if not self.assert_satisfies_hint(elem, elem_hint):
                return False

        return True

    def assert_satisfies_sequence(self: "BLEIRTypeValidator", context: Any, hint: Any) -> bool:
        if not isinstance(context, (list, tuple)):
            return False

        elem_hint = hint.__args__[0]
        for elem in context:
            if not self.assert_satisfies_hint(elem, elem_hint):
                return False

        return True

    def assert_satisfies_set(self: "BLEIRTypeValidator", context: Any, hint: Any) -> bool:
        if not isinstance(context, set):
            return False

        elem_hint = hint.__args__[0]
        for elem in context:
            if not self.assert_satisfies_hint(elem, elem_hint):
                return False

        return True

    def assert_satisfies_union(self: "BLEIRTypeValidator", context: Any, hint: Any) -> bool:
        if hint._name == "Any":
            return True
        for kind in hint.__args__:
            if self.assert_satisfies_hint(context, kind):
                return True
        return False

    def assert_satisfies_hint(self: "BLEIRTypeValidator", context: Any, hint: Any) -> bool:
        if not is_type_hint(hint):
            return isinstance(context, hint)

        if hint._name in ["Sequence", "List", "Iterable"]:
            return self.assert_satisfies_sequence(context, hint)

        if hint._name == "Set":
            return self.assert_satisfies_set(context, hint)

        if hint._name == "Dict":
            return self.assert_satisfies_dict(context, hint)

        if hint._name == "Tuple":
            return self.assert_satisfies_tuple(context, hint)

        return self.assert_satisfies_union(context, hint)

    def validate_attrs(self: "BLEIRTypeValidator", context: Any) -> None:
        if not hasattr(context.__class__, "__annotations__"):
            raise SyntacticError(f"Expected a BLEIR instance but received a type: {context}")
        for attr, hint in context.__class__.__annotations__.items():
            value = getattr(context, attr)
            if not self.assert_satisfies_hint(value, hint):
                raise SyntacticError(f"Type of {attr} ({value}) does not satisfy hint {hint}")

    def validate_tree(self: "BLEIRTypeValidator", context: Any) -> None:
        if is_bleir(context) and not isinstance(context, Enum):
            self.validate_attrs(context)
            for attr, value in instance_members_of(context):
                self.validate_tree(value)

    def validate_type_and_tree(self: "BLEIRTypeValidator", context: Any, kind: Type) -> None:
        self.assert_type_is(context, kind)
        self.validate_tree(context)

    def visit_snippet(self: "BLEIRTypeValidator", snippet: Snippet) -> None:
        self.validate_type_and_tree(snippet, Snippet)

    def visit_multi_line_comment(self: "BLEIRTypeValidator", multi_line_comment: MultiLineComment) -> None:
        self.validate_type_and_tree(multi_line_comment, MultiLineComment)

    def visit_single_line_comment(self: "BLEIRTypeValidator", single_line_comment: SingleLineComment) -> None:
        self.validate_type_and_tree(single_line_comment, SingleLineComment)

    def visit_trailing_comment(self: "BLEIRTypeValidator", trailing_comment: TrailingComment) -> None:
        self.validate_type_and_tree(trailing_comment, TrailingComment)

    def visit_inline_comment(self: "BLEIRTypeValidator", inline_comment: InlineComment) -> None:
        self.validate_type_and_tree(inline_comment, InlineComment)

    def visit_example(self: "BLEIRTypeValidator", example: Example) -> None:
        self.validate_type_and_tree(example, Example)

    def visit_value_parameter(self: "BLEIRTypeValidator", value_parameter: ValueParameter) -> None:
        self.validate_type_and_tree(value_parameter, ValueParameter)

    def visit_fragment_caller_call(self: "BLEIRTypeValidator", fragment_caller_call: FragmentCallerCall) -> None:
        self.validate_type_and_tree(fragment_caller_call, FragmentCallerCall)

    def visit_fragment_caller(self: "BLEIRTypeValidator", fragment_caller: FragmentCaller) -> None:
        self.validate_type_and_tree(fragment_caller, FragmentCaller)

    def visit_allocated_register(self: "BLEIRTypeValidator", allocated_register: AllocatedRegister) -> None:
        self.validate_type_and_tree(allocated_register, AllocatedRegister)

    def visit_fragment(self: "BLEIRTypeValidator", fragment: Fragment) -> None:
        self.validate_type_and_tree(fragment, Fragment)

    def visit_multi_statement(self: "BLEIRTypeValidator", multi_statement: MultiStatement) -> None:
        self.validate_type_and_tree(multi_statement, MultiStatement)

    def visit_statement(self: "BLEIRTypeValidator", statement: STATEMENT) -> None:
        self.validate_type_and_tree(statement, STATEMENT)

    def visit_masked(self: "BLEIRTypeValidator", masked: MASKED) -> None:
        self.validate_type_and_tree(masked, MASKED)

    def visit_mask(self: "BLEIRTypeValidator", mask: MASK) -> None:
        self.validate_type_and_tree(mask, MASK)

    def visit_shifted_sm_reg(self: "BLEIRTypeValidator", shifted_sm_reg: SHIFTED_SM_REG) -> None:
        self.validate_type_and_tree(shifted_sm_reg, SHIFTED_SM_REG)

    def visit_assignment(self: "BLEIRTypeValidator", assignment: ASSIGNMENT) -> None:
        self.validate_type_and_tree(assignment, ASSIGNMENT)

    def visit_read(self: "BLEIRTypeValidator", read: READ) -> None:
        self.validate_type_and_tree(read, READ)

    def visit_write(self: "BLEIRTypeValidator", write: WRITE) -> None:
        self.validate_type_and_tree(write, WRITE)

    def visit_broadcast(self: "BLEIRTypeValidator", broadcast: BROADCAST) -> None:
        self.validate_type_and_tree(broadcast, BROADCAST)

    def visit_rsp16_assignment(self: "BLEIRTypeValidator", rsp16_assignment: RSP16_ASSIGNMENT) -> None:
        self.validate_type_and_tree(rsp16_assignment, RSP16_ASSIGNMENT)

    def visit_rsp256_assignment(self: "BLEIRTypeValidator", rsp256_assignment: RSP256_ASSIGNMENT) -> None:
        self.validate_type_and_tree(rsp256_assignment, RSP256_ASSIGNMENT)

    def visit_rsp2k_assignment(self: "BLEIRTypeValidator", rsp2k_assignment: RSP2K_ASSIGNMENT) -> None:
        self.validate_type_and_tree(rsp2k_assignment, RSP2K_ASSIGNMENT)

    def visit_rsp32k_assignment(self: "BLEIRTypeValidator", rsp32k_assignment: RSP32K_ASSIGNMENT) -> None:
        self.validate_type_and_tree(rsp32k_assignment, RSP32K_ASSIGNMENT)

    def visit_binary_expr(self: "BLEIRTypeValidator", binary_expr: BINARY_EXPR) -> None:
        self.validate_type_and_tree(binary_expr, BINARY_EXPR)

    def visit_unary_expr(self: "BLEIRTypeValidator", unary_expr: UNARY_EXPR) -> None:
        self.validate_type_and_tree(unary_expr, UNARY_EXPR)

    def visit_unary_src(self: "BLEIRTypeValidator", unary_src: UNARY_SRC) -> None:
        self.validate_type_and_tree(unary_src, UNARY_SRC)

    def visit_unary_sb(self: "BLEIRTypeValidator", unary_sb: UNARY_SB) -> None:
        self.validate_type_and_tree(unary_sb, UNARY_SB)

    def visit_sb_expr(self: "BLEIRTypeValidator", sb_expr: SB_EXPR) -> None:
        self.validate_type_and_tree(sb_expr, SB_EXPR)

    def visit_rn_reg(self: "BLEIRTypeValidator", rn_reg: RN_REG) -> None:
        self.validate_type_and_tree(rn_reg, RN_REG)

    def visit_sm_reg(self: "BLEIRTypeValidator", sm_reg: SM_REG) -> None:
        self.validate_type_and_tree(sm_reg, SM_REG)

    def visit_rl_expr(self: "BLEIRTypeValidator", rl_expr: RL_EXPR) -> None:
        self.validate_type_and_tree(rl_expr, RL_EXPR)

    def visit_rsp16_expr(self: "BLEIRTypeValidator", rsp16_expr: RSP16_EXPR) -> None:
        self.validate_type_and_tree(rsp16_expr, RSP16_EXPR)

    def visit_rsp256_expr(self: "BLEIRTypeValidator", rsp256_expr: RSP256_EXPR) -> None:
        self.validate_type_and_tree(rsp256_expr, RSP256_EXPR)

    def visit_rsp2k_expr(self: "BLEIRTypeValidator", rsp2k_expr: RSP2K_EXPR) -> None:
        self.validate_type_and_tree(rsp2k_expr, RSP2K_EXPR)

    def visit_rsp32k_expr(self: "BLEIRTypeValidator", rsp32k_expr: RSP32K_EXPR) -> None:
        self.validate_type_and_tree(rsp32k_expr, RSP32K_EXPR)

    def visit_bit_expr(self: "BLEIRTypeValidator", bit_expr: BIT_EXPR) -> None:
        self.validate_type_and_tree(bit_expr, BIT_EXPR)

    def visit_src_expr(self: "BLEIRTypeValidator", src_expr: SRC_EXPR) -> None:
        self.validate_type_and_tree(src_expr, SRC_EXPR)

    def visit_broadcast_expr(self: "BLEIRTypeValidator", broadcast_expr: BROADCAST_EXPR) -> None:
        self.validate_type_and_tree(broadcast_expr, BROADCAST_EXPR)

    def visit_special(self: "BLEIRTypeValidator", special: SPECIAL) -> None:
        self.validate_type_and_tree(special, SPECIAL)

    def visit_assign_op(self: "BLEIRTypeValidator", assign_op: ASSIGN_OP) -> None:
        self.validate_type_and_tree(assign_op, ASSIGN_OP)

    def visit_binop(self: "BLEIRTypeValidator", binop: BINOP) -> None:
        self.validate_type_and_tree(binop, BINOP)

    def visit_unary_op(self: "BLEIRTypeValidator", unary_op: Optional[UNARY_OP]) -> None:
        self.validate_type_and_tree(unary_op, UNARY_OP)


def validate_types(fn_or_bleir: Fn_or_BLEIR) -> Optional[Callable]:
    """Validates the BLEIR tree of the value returned from the provided
    function. If a BLEIR instance is provided instead of a function, the
    BLEIR instance will be validated and no value will be returned."""

    validator = BLEIRTypeValidator()

    def validate(param):
        if is_bleir(param):
            validator.validate_tree(param)
        elif isinstance(param, (list, tuple, set)):
            for elem in param:
                validate(elem)
        elif isinstance(param, dict):
            for key, value in param.items():
                validate(key)  # In case the key is a BLEIR
                validate(value)

    if is_bleir(fn_or_bleir) or isinstance(fn_or_bleir, (list, tuple, set, dict)):
        param = fn_or_bleir
        validate(param)
        return None

    # If it isn't a BLEIR or a function to decorate, raise an error
    if not callable(fn_or_bleir):
        raise NotImplementedError(f"Unsupported fn_or_bleir type ({fn_or_bleir.__class__.__name__}): {fn_or_bleir}")

    fn = fn_or_bleir

    @wraps(fn)
    def wrapper(*args, **kwargs):
        bleir = fn(*args, **kwargs)
        validate(bleir)
        return bleir

    return wrapper
