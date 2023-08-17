r"""
By Dylon Edwards
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from typing import Deque, Dict, Optional, Sequence, Set, Tuple, Type, Union

from open_belex.bleir.types import (ASSIGN_OP, ASSIGNMENT, BINARY_EXPR, BINOP,
                                    BIT_EXPR, BROADCAST, BROADCAST_EXPR,
                                    EWE_REG, GGL_ASSIGNMENT, GGL_EXPR, L1_REG,
                                    L2_REG, LGL_ASSIGNMENT, LGL_EXPR, LX_ADDR,
                                    LX_ASSIGNMENT, MASK, MASKED, RE_REG, READ,
                                    RL_EXPR, RN_REG, RSP2K_ASSIGNMENT,
                                    RSP2K_RVALUE, RSP16_ASSIGNMENT,
                                    RSP16_RVALUE, RSP32K_ASSIGNMENT,
                                    RSP32K_RVALUE, RSP256_ASSIGNMENT,
                                    RSP256_RVALUE, SB_EXPR, SHIFTED_SM_REG,
                                    SM_REG, SPECIAL, SRC_EXPR, STATEMENT,
                                    UNARY_EXPR, UNARY_OP, UNARY_SB, UNARY_SRC,
                                    WRITE, ExtendedRegister, Fragment,
                                    GlassStatement, LineComment, LXParameter,
                                    LXRegWithOffsets, MultiStatement,
                                    Operation, Operation_or_LineComment,
                                    ReadWriteInhibit, Snippet, bleir_dataclass,
                                    collectible)
from open_belex.bleir.walkables import BLEIRVisitor

LOGGER = logging.getLogger()


@bleir_dataclass
class SetInPlace:
    in_place: bool


class Command:
    pass


class Mimic(Command):
    pass


@bleir_dataclass
class MimicRSP16FromRSP256(Mimic):
    pass


@bleir_dataclass
class MimicRSP256FromRSP16(Mimic):
    pass


@bleir_dataclass
class MimicRSP256FromRSP2K(Mimic):
    pass


@bleir_dataclass
class MimicRSP2KFromRSP256(Mimic):
    pass


@bleir_dataclass
class MimicRSP2KFromRSP32K(Mimic):
    pass


@bleir_dataclass
class MimicRSP32KFromRSP2K(Mimic):
    pass


@bleir_dataclass
class MimicNoOp(Mimic):
    pass


@bleir_dataclass
class MimicFSelNoOp(Mimic):
    pass


@bleir_dataclass
class MimicRSPEnd(Mimic):
    pass


@bleir_dataclass
class MimicRSPStartRet(Mimic):
    pass


@bleir_dataclass
class MimicL2End(Mimic):
    pass


class RegisterKind(Enum):
    SM_REG = auto()
    RN_REG = auto()
    RE_REG = auto()
    EWE_REG = auto()
    L1_REG = auto()
    L2_REG = auto()


@bleir_dataclass
class LoadRegister(Command):
    kind: RegisterKind
    symbol: str
    shift_width: int = 0  # SM_REG / RE_REG / EWE_REG
    invert: bool = False  # SM_REG / RE_REG / EWE_REG
    offset: int = 0       # L1_REG / L2_REG

@bleir_dataclass
@collectible("registers")
class LoadRegisters(Command):
    registers: Sequence[LoadRegister]

    @property
    def kind(self: "LoadRegister") -> RegisterKind:
        return self.registers[0].kind


@bleir_dataclass
@collectible("registers")
class UnifySMRegs(Command):
    registers: Sequence[LoadRegister]


@bleir_dataclass
class LoadSrc(Command):
    src: SRC_EXPR


@bleir_dataclass
class MimicSBFromSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicSBFromInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicSBCondEqSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicSBCondEqInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicSetRL(Mimic):
    mask: LoadRegister
    bit: bool


@bleir_dataclass
class MimicRLFromSrc(Mimic):
    mask: LoadRegister
    src: LoadSrc


@bleir_dataclass
class MimicRLFromInvSrc(Mimic):
    mask: LoadRegister
    src: LoadSrc


@bleir_dataclass
class MimicRLOrEqSrc(Mimic):
    mask: LoadRegister
    src: LoadSrc


@bleir_dataclass
class MimicRLOrEqInvSrc(Mimic):
    mask: LoadRegister
    src: LoadSrc


@bleir_dataclass
class MimicRLAndEqSrc(Mimic):
    mask: LoadRegister
    src: LoadSrc


@bleir_dataclass
class MimicRLXorEqSrc(Mimic):
    mask: LoadRegister
    src: LoadSrc


@bleir_dataclass
class MimicRLXorEqInvSrc(Mimic):
    mask: LoadRegister
    src: LoadSrc


@bleir_dataclass
class MimicRLFromSB(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters


@bleir_dataclass
class MimicRLFromInvSB(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters


@bleir_dataclass
class MimicRLOrEqSB(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters


@bleir_dataclass
class MimicRLAndEqSB(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters


@bleir_dataclass
class MimicRLAndEqInvSB(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters


@bleir_dataclass
class MimicRLXorEqSB(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters


@bleir_dataclass
class MimicRLFromSBAndSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLOrEqSBAndSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLOrEqSBAndInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLAndEqSBAndSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLAndEqSBAndInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLXorEqSBAndSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLXorEqSBAndInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLFromSBOrSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLFromSBOrInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLFromSBXorSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLFromSBXorInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLFromInvSBAndSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLFromInvSBAndInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRLFromSBAndInvSrc(Mimic):
    mask: LoadRegister
    vrs: LoadRegisters
    src: LoadSrc


@bleir_dataclass
class MimicRSP16FromRL(Mimic):
    mask: Union[LoadRegister, UnifySMRegs]


@bleir_dataclass
class MimicGLFromRL(Mimic):
    mask: Union[LoadRegister, UnifySMRegs]


@bleir_dataclass
class MimicGGLFromRL(Mimic):
    mask: Union[LoadRegister, UnifySMRegs]


@bleir_dataclass
class MimicL1FromGGL(Mimic):
    l1_addr: LoadRegister


@bleir_dataclass
class MimicLGLFromL1(Mimic):
    l1_addr: LoadRegister


@bleir_dataclass
class MimicL2FromLGL(Mimic):
    l2_addr: LoadRegister


@bleir_dataclass
class MimicLGLFromL2(Mimic):
    l2_addr: LoadRegister


@bleir_dataclass
class MimicL1FromLGL(Mimic):
    l1_addr: LoadRegister


@bleir_dataclass
class MimicGGLFromL1(Mimic):
    l1_addr: LoadRegister


@bleir_dataclass
class MimicGGLFromRLAndL1(Mimic):
    mask: Union[LoadRegister, UnifySMRegs]
    l1_addr: LoadRegister


@bleir_dataclass
class MimicRWInhSet(Mimic):
    mask: LoadRegister


@bleir_dataclass
class MimicRWInhRst(Mimic):
    mask: LoadRegister
    has_read: bool


@bleir_dataclass
class ApplyPatch(Command):
    mimic: Mimic


@bleir_dataclass
class IncrementInstructions(Command):
    num_instructions: int


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

RWINH_RST_STATEMENT_WEIGHTS: Dict[Union[Type, ReadWriteInhibit], int] = {
    statement_type: weight
    for weight, statement_type
    in enumerate([
        WRITE,
        LX_ASSIGNMENT,
        LGL_ASSIGNMENT,
        GGL_ASSIGNMENT,
        BROADCAST,
        READ,
        ReadWriteInhibit.RWINH_SET,
        ReadWriteInhibit.RWINH_RST,
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
        return operation.assignment.operation.__class__
    return operation.__class__


def statement_weight(resets_rwinh: bool, statement: STATEMENT) -> int:
    statement_type = statement_type_of(statement)
    if not resets_rwinh and statement_type in STATEMENT_WEIGHTS:
        return STATEMENT_WEIGHTS[statement_type]
    if resets_rwinh and statement_type in RWINH_RST_STATEMENT_WEIGHTS:
        return RWINH_RST_STATEMENT_WEIGHTS[statement_type]
    raise ValueError(
        f"Unsupported statement type ({statement.__class__.__name__}): "
        f"{statement}")


def has_rwinh_rst(multi_statement: MultiStatement) -> bool:
    for statement_or_comment in multi_statement:
        operation = statement_or_comment.operation
        if isinstance(operation, MASKED) \
           and operation.read_write_inhibit is not None:
            return True
    return False


def has_multiple_reads(multi_statement: MultiStatement) -> bool:
    has_read = False
    for statement_or_comment in filter(is_statement, multi_statement):
        operation = statement_or_comment.operation
        if not isinstance(operation, MASKED):
            continue
        masked = operation
        assignment = masked.assignment
        if assignment is None:
            continue
        if not isinstance(assignment.operation, READ):
            continue
        if has_read:
            return True
        has_read = True
    return False


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


@dataclass
class ReadScheduler(BLEIRVisitor):
    provides: Dict[int, STATEMENT] = field(default_factory=dict)
    consumes: Dict[STATEMENT, int] = field(default_factory=dict)

    statements: Optional[Sequence[STATEMENT]] = None
    codependencies: Optional[Set[STATEMENT]] = None

    statement: Optional[STATEMENT] = None
    mask: int = 0x0000

    def visit_multi_statement(
            self: "ReadScheduler",
            multi_statement: MultiStatement) -> None:

        statements = list(filter(is_statement, multi_statement))
        for statement in statements:
            self.visit_statement(statement)

        dependents = defaultdict(set)
        dependencies = defaultdict(set)

        for consumer, mask in self.consumes.items():
            for section in range(16):
                if mask & (1 << section) != 0x0000 \
                   and section in self.provides:
                    provider = self.provides[section]
                    dependents[provider].add(consumer)
                    dependencies[consumer].add(provider)

        root = []
        for provider in dependents.keys():
            if provider not in dependencies \
               or len(dependencies[provider]) == 0:
                root.append(provider)

        while len(root) > 0:
            provider = root[0]
            root.remove(provider)
            statements.remove(provider)
            statements.append(provider)
            consumers = dependents[provider]
            for consumer in consumers:
                if len(dependencies[consumer]) == 0:
                    root.append(consumer)

        self.statements = statements
        self.codependencies = set()
        for dependent in dependents:
            for dependency in dependencies[dependent]:
                if dependent in dependents[dependency]:
                    self.codependencies.add(dependent)
                    self.codependencies.add(dependency)

    def visit_statement(self: "ReadScheduler", statement: STATEMENT) -> None:
        if isinstance(statement.operation, MASKED):
            self.statement = statement
            self.visit_masked(statement.operation)

    def visit_masked(self: "ReadScheduler", masked: MASKED) -> None:
        self.mask = self.visit_mask(masked.mask)
        for section in range(16):
            if self.mask & (1 << section) != 0x0000:
                self.provides[section] = self.statement
        if masked.assignment is not None:
            self.visit_assignment(masked.assignment)

    def visit_assignment(self: "ReadScheduler", assignment: ASSIGNMENT) -> None:
        if isinstance(assignment.operation, READ):
            self.visit_read(assignment.operation)

    def visit_read(self: "ReadScheduler", read: READ) -> None:
        if isinstance(read.rvalue, UNARY_EXPR):
            self.visit_unary_expr(read.rvalue)
        elif isinstance(read.rvalue, BINARY_EXPR):
            self.visit_binary_expr(read.rvalue)
        else:
            raise NotImplementedError(
                f"Unsupported rvalue type: {read.rvalue.__class__.__name__}: "
                f"{read}")

    def visit_binary_expr(self: "ReadScheduler", binary_expr: BINARY_EXPR) -> None:
        if isinstance(binary_expr.right_operand, UNARY_SRC):
            self.visit_unary_src(binary_expr.right_operand)

    def visit_unary_expr(self: "ReadScheduler", unary_expr: UNARY_EXPR) -> None:
        if isinstance(unary_expr.expression, UNARY_SRC):
            self.visit_unary_src(unary_expr.expression)

    def visit_unary_src(self: "ReadScheduler", unary_src: UNARY_SRC) -> None:
        if unary_src.expression in RL_SRCS:
            if unary_src.expression in [SRC_EXPR.NRL, SRC_EXPR.INV_NRL]:
                self.consumes[self.statement] = self.mask >> 1
            elif unary_src.expression in [SRC_EXPR.SRL, SRC_EXPR.INV_SRL]:
                self.consumes[self.statement] = self.mask << 1
            else:
                self.consumes[self.statement] = self.mask

    def visit_mask(self: "ReadScheduler", mask: MASK) -> int:
        if isinstance(mask.expression, SM_REG):
            is_param, value = self.visit_sm_reg(mask.expression)
        elif isinstance(mask.expression, SHIFTED_SM_REG):
            is_param, value = self.visit_shifted_sm_reg(mask.expression)
        else:
            raise NotImplementedError(
                f"Unsupported expression type: "
                f"{mask.expression.__class__.__name__}")

        if is_param:
            return value

        if mask.operator is UNARY_OP.NEGATE:
            value = (0xFFFF - value)

        return value

    def visit_shifted_sm_reg(self: "ReadScheduler",
                             shifted_sm_reg: SHIFTED_SM_REG) -> Tuple[bool, int]:
        is_param, value = self.visit_sm_reg(shifted_sm_reg.register)
        if is_param:
            return is_param, value
        value = (value << shifted_sm_reg.num_bits) & 0xFFFF
        return is_param, value

    def visit_sm_reg(self: "ReadScheduler", sm_reg: SM_REG) -> Tuple[bool, int]:
        value = sm_reg.constant_value
        if value is None:
            return True, 0xFFFF
        if sm_reg.is_section:
            value = 1 << value;
        return False, value

    def weight_of(self: "ReadScheduler", statement: STATEMENT) -> int:
        return 0


@dataclass
class CommandScheduler(BLEIRVisitor):
    commands_by_frag_nym: Dict[str, Sequence[Command]] = \
        field(default_factory=dict)
    patches_by_command_by_frag_nym: Dict[str, Dict[Command, ApplyPatch]] = \
        field(default_factory=dict)
    frag_nym: Optional[str] = None
    commands: Optional[Deque[Command]] = None
    patches_by_command: Optional[Dict[Command, ApplyPatch]] = None
    mask: Optional[LoadRegister] = None
    in_multi_statement: bool = False
    num_instructions: int = 0
    has_read: bool = False

    def visit_snippet(self: "CommandScheduler", snippet: Snippet) -> None:
        for fragment in snippet.fragments:
            self.visit_fragment(fragment)

    def visit_fragment(self: "CommandScheduler", fragment: Fragment) -> None:
        if fragment.children is not None:
            for child in fragment.children:
                self.visit_fragment(child)
        else:
            self.num_instructions = 0
            self.frag_nym = fragment.identifier
            self.commands = deque()
            self.patches_by_command = {}
            for operation in fragment.operations:
                self.visit_operation(operation)
            self.commands.append(IncrementInstructions(self.num_instructions))
            self.commands_by_frag_nym[self.frag_nym] = self.commands
            self.patches_by_command_by_frag_nym[self.frag_nym] = \
                self.patches_by_command
            self.frag_nym = None
            self.commands = None

    def visit_operation(self: "CommandScheduler",
                        operation: Operation) -> None:
        if isinstance(operation, MultiStatement):
            self.visit_multi_statement(operation)
            self.num_instructions += 1
        elif isinstance(operation, STATEMENT):
            commands = self.visit_statement(operation)
            if commands is not None:
                self.num_instructions += 1
        elif not isinstance(operation, LineComment.__args__):
            raise NotImplementedError(
                f"Unsupported operation type: {operation.__class__.__name__}")

    def set_in_place(self: "CommandScheduler", in_place: bool) -> None:
        if len(self.commands) > 0 \
           and isinstance(self.commands[-1], SetInPlace) \
           and self.commands[-1].in_place != in_place:
            self.commands.pop()
        else:
            self.commands.append(SetInPlace(in_place))

    def apply_patch(self: "CommandScheduler", command: Command) -> None:
        commands = deque()
        while isinstance(self.commands[-1], ApplyPatch):
            commands.append(self.commands.pop())
        if self.commands[-1] is not command:
            patch = ApplyPatch(command)
            self.commands.append(patch)
            self.patches_by_command[command] = patch
        else:
            self.commands.pop()
            self.set_in_place(True)
            self.commands.append(command)
            self.set_in_place(False)
        self.commands.extend(commands)

    def visit_multi_statement(self: "CommandScheduler",
                              multi_statement: MultiStatement) -> None:
        self.in_multi_statement = True

        # Group multiple GL broadcasts within a multi-statement into a single
        # broadcast unified over their section masks
        sm_regs_by_broadcast = defaultdict(list)
        statements_by_type = defaultdict(list)
        for statement in multi_statement:
            if isinstance(statement, (STATEMENT, MultiStatement)):
                statement_type = statement_type_of(statement)
                statements_by_type[statement_type].append(statement)
                if statement_type is BROADCAST:
                    broadcast = statement.operation.assignment.operation
                    mask = statement.operation.mask
                    load_register = self.visit_mask(mask)
                    sm_regs_by_broadcast[broadcast].append(load_register)
        unified_broadcasts = set()

        # Re-order in half-clock friendly manner
        may_parallelize = has_multiple_reads(multi_statement)
        if may_parallelize:
            read_scheduler = ReadScheduler()
            read_scheduler.visit_multi_statement(multi_statement)
            codependencies = read_scheduler.codependencies
            statements = read_scheduler.statements
        else:
            statements = list(filter(is_statement, multi_statement))
        resets_rwinh = has_rwinh_rst(statements)
        half_clock_friendly_statements = \
            sorted(statements, key=partial(statement_weight, resets_rwinh))

        commands = []
        has_pending_read = False
        for statement in half_clock_friendly_statements:
            # simulate half-clock operations
            statement_type = statement_type_of(statement)

            if statement_type is READ:
                if may_parallelize and statement in codependencies:
                    self.set_in_place(False)
                has_pending_read = True

            elif statement_type is BROADCAST:
                broadcast = statement.operation.assignment.operation
                if broadcast in unified_broadcasts:
                    continue

                if has_pending_read:
                    # flush the buffer to simulate half-clock operations
                    # read-before-broadcast is a valid half-clock operation
                    if may_parallelize:
                        for command in commands:
                            self.apply_patch(command)
                        commands.clear()
                    has_pending_read = False

            statement_commands = self.visit_statement(statement)

            if statement_type is BROADCAST:
                sm_regs = sm_regs_by_broadcast[broadcast]
                if len(sm_regs) > 1:
                    unify_sm_regs = UnifySMRegs(sm_regs)
                    self.commands.append((statement_type, unify_sm_regs))
                    command = statement_commands[0].having(mask=unify_sm_regs)
                    statement_commands = [unify_sm_regs, command]
                unified_broadcasts.add(broadcast)

            if statement_commands is not None:
                self.commands.extend(statement_commands)
                if statement_type is READ \
                   and may_parallelize \
                   and statement in codependencies:
                    self.set_in_place(True)
                    for command in statement_commands:
                        commands.append(command)

        if may_parallelize:
            for command in commands:
                self.apply_patch(command)

        self.in_multi_statement = False
        self.has_read = False

    def visit_statement(self: "CommandScheduler",
                        statement: STATEMENT) -> Optional[Sequence[Command]]:

        commands = None

        if isinstance(statement.operation, MASKED):
            commands = self.visit_masked(statement.operation)
        elif isinstance(statement.operation, GGL_ASSIGNMENT):
            commands = self.visit_ggl_assignment(statement.operation)
        elif isinstance(statement.operation, LX_ASSIGNMENT):
            commands = self.visit_lx_assignment(statement.operation)
        elif isinstance(statement.operation, LGL_ASSIGNMENT):
            commands = self.visit_lgl_assignment(statement.operation)
        elif isinstance(statement.operation, SPECIAL):
            commands = self.visit_special(statement.operation)
        elif isinstance(statement.operation, RSP16_ASSIGNMENT):
            commands = self.visit_rsp16_assignment(statement.operation)
        elif isinstance(statement.operation, RSP256_ASSIGNMENT):
            commands = self.visit_rsp256_assignment(statement.operation)
        elif isinstance(statement.operation, RSP2K_ASSIGNMENT):
            commands = self.visit_rsp2k_assignment(statement.operation)
        elif isinstance(statement.operation, RSP32K_ASSIGNMENT):
            commands = self.visit_rsp32k_assignment(statement.operation)
        elif isinstance(statement.operation, GlassStatement):
            pass
        else:
            raise NotImplementedError(
                f"Unsupported operation type: {statement.operation.__class__.__name__}: "
                f"{statement}")

        if not self.in_multi_statement and commands is not None:
            self.commands.extend(commands)
            self.has_read = False

        return commands

    def visit_masked(self: "CommandScheduler",
                     masked: MASKED) -> Sequence[Command]:

        self.mask = self.visit_mask(masked.mask)

        commands = []

        if masked.assignment is not None:
            for command in self.visit_assignment(masked.assignment):
                commands.append(command)

        if masked.read_write_inhibit is ReadWriteInhibit.RWINH_SET:
            command = MimicRWInhSet(self.mask)
            commands.append(command)

        elif masked.read_write_inhibit is ReadWriteInhibit.RWINH_RST:
            command = MimicRWInhRst(self.mask, self.has_read)
            commands.append(command)

        self.mask = None
        return commands

    def visit_mask(self: "CommandScheduler", mask: MASK) -> LoadRegister:
        if isinstance(mask.expression, SM_REG):
            command = self.visit_sm_reg(mask.expression)
        elif isinstance(mask.expression, SHIFTED_SM_REG):
            command = self.visit_shifted_sm_reg(mask.expression)
        else:
            raise NotImplementedError(
                f"Unsupported expression type: {mask.expression.__class__.__name__}")
        command = command.having(invert=mask.operator is UNARY_OP.NEGATE)
        return command

    def visit_shifted_sm_reg(self: "CommandScheduler",
                             shifted_sm_reg: SHIFTED_SM_REG) -> LoadRegister:
        command = self.visit_sm_reg(shifted_sm_reg.register)
        command = command.having(shift_width=shifted_sm_reg.num_bits)
        return command

    def visit_assignment(self: "CommandScheduler",
                         assignment: ASSIGNMENT) -> Sequence[Command]:
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
        raise NotImplementedError(
            f"Unsupported operation type: {assignment.operation.__class__.__name__}: "
            f"{assignment}")

    def visit_read(self: "CommandScheduler",
                   read: READ) -> Sequence[Command]:

        self.has_read = True

        if isinstance(read.rvalue, UNARY_EXPR):
            vrs_or_src_or_bit = self.visit_unary_expr(read.rvalue)
            if isinstance(vrs_or_src_or_bit, LoadRegisters):
                sb = vrs_or_src_or_bit
                if read.operator is ASSIGN_OP.EQ:
                    if read.rvalue.expression.operator is UNARY_OP.NEGATE:
                        return [MimicRLFromInvSB(self.mask, sb)]
                    return [MimicRLFromSB(self.mask, sb)]
                if read.operator is ASSIGN_OP.AND_EQ:
                    if read.rvalue.expression.operator is UNARY_OP.NEGATE:
                        return [MimicRLAndEqInvSB(self.mask, sb)]
                    return [MimicRLAndEqSB(self.mask, sb)]
                if read.operator is ASSIGN_OP.OR_EQ:
                    return [MimicRLOrEqSB(self.mask, sb)]
                return [MimicRLXorEqSB(self.mask, sb)]
            if isinstance(vrs_or_src_or_bit, LoadSrc):
                src = vrs_or_src_or_bit
                if read.operator is ASSIGN_OP.EQ:
                    if read.rvalue.expression.operator is UNARY_OP.NEGATE:
                        return [MimicRLFromInvSrc(self.mask, src)]
                    return [MimicRLFromSrc(self.mask, src)]
                if read.operator is ASSIGN_OP.AND_EQ:
                    return [MimicRLAndEqSrc(self.mask, src)]
                if read.operator is ASSIGN_OP.OR_EQ:
                    if read.rvalue.expression.operator is UNARY_OP.NEGATE:
                        return [MimicRLOrEqInvSrc(self.mask, src)]
                    return [MimicRLOrEqSrc(self.mask, src)]
                if read.operator is ASSIGN_OP.XOR_EQ:
                    if read.rvalue.expression.operator is UNARY_OP.NEGATE:
                        return [MimicRLXorEqInvSrc(self.mask, src)]
                    return [MimicRLXorEqSrc(self.mask, src)]
            if isinstance(vrs_or_src_or_bit, bool):
                bit = vrs_or_src_or_bit
                return [MimicSetRL(self.mask, bit)]

        if isinstance(read.rvalue, BINARY_EXPR):
            sb, src = self.visit_binary_expr(read.rvalue)

            if read.operator is ASSIGN_OP.EQ:
                if read.rvalue.operator is BINOP.AND:
                    if read.rvalue.left_operand.operator is UNARY_OP.NEGATE:
                        if read.rvalue.right_operand.operator is UNARY_OP.NEGATE:
                            return [MimicRLFromInvSBAndInvSrc(self.mask, sb, src)]
                        return [MimicRLFromInvSBAndSrc(self.mask, sb, src)]
                    else:
                        if read.rvalue.right_operand.operator is UNARY_OP.NEGATE:
                            return [MimicRLFromSBAndInvSrc(self.mask, sb, src)]
                        return [MimicRLFromSBAndSrc(self.mask, sb, src)]
                if read.rvalue.operator is BINOP.OR:
                    if read.rvalue.right_operand.operator is UNARY_OP.NEGATE:
                        return [MimicRLFromSBOrInvSrc(self.mask, sb, src)]
                    return [MimicRLFromSBOrSrc(self.mask, sb, src)]
                if read.rvalue.right_operand.operator is UNARY_OP.NEGATE:
                    return [MimicRLFromSBXorInvSrc(self.mask, sb, src)]
                return [MimicRLFromSBXorSrc(self.mask, sb, src)]

            if read.operator is ASSIGN_OP.AND_EQ:
                if read.rvalue.right_operand.operator is UNARY_OP.NEGATE:
                    return [MimicRLAndEqSBAndInvSrc(self.mask, sb, src)]
                return [MimicRLAndEqSBAndSrc(self.mask, sb, src)]

            if read.operator is ASSIGN_OP.OR_EQ:
                if read.rvalue.right_operand.operator is UNARY_OP.NEGATE:
                    return [MimicRLOrEqSBAndInvSrc(self.mask, sb, src)]
                return [MimicRLOrEqSBAndSrc(self.mask, sb, src)]

            if read.rvalue.right_operand.operator is UNARY_OP.NEGATE:
                return [MimicRLXorEqSBAndInvSrc(self.mask, sb, src)]
            return [MimicRLXorEqSBAndSrc(self.mask, sb, src)]

        raise NotImplementedError(
            f"Unsupported rvalue type: {read.rvalue.__class__.__name__}: {read}")

    def visit_write(self: "CommandScheduler",
                    write: WRITE) -> Sequence[Command]:

        vrs = self.visit_sb_expr(write.lvalue)
        src = self.visit_unary_src(write.rvalue)

        if write.operator is ASSIGN_OP.EQ:
            if write.rvalue.operator is UNARY_OP.NEGATE:
                command = MimicSBFromInvSrc(self.mask, vrs, src)
            else:
                command = MimicSBFromSrc(self.mask, vrs, src)
        elif write.operator is ASSIGN_OP.COND_EQ:
            if write.rvalue.operator is UNARY_OP.NEGATE:
                command = MimicSBCondEqInvSrc(self.mask, vrs, src)
            else:
                command = MimicSBCondEqSrc(self.mask, vrs, src)
        else:
            raise NotImplementedError(
                f"Unsupported WRITE operator: {write.operator}")

        return [command]

    def visit_broadcast(self: "CommandScheduler",
                        broadcast: BROADCAST) -> Sequence[Command]:

        if broadcast.lvalue == BROADCAST_EXPR.GL:
            command = MimicGLFromRL(self.mask)
            return [command]

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
                command = MimicGGLFromRLAndL1(self.mask, l1_addr)
            else:
                command = MimicGGLFromRL(self.mask)
            return [command]

        if broadcast.lvalue == BROADCAST_EXPR.RSP16:
            command = MimicRSP16FromRL(self.mask)
            return [command]

        raise NotImplementedError(
            f"Unsupported lvalue type ({broadcast.lvalue.__class__.__name__.__name__}): "
            f"{broadcast.lvalue}")

    def visit_ggl_assignment(
                self: "CommandScheduler",
                ggl_assignment: GGL_ASSIGNMENT) \
            -> Sequence[Command]:
        l1_addr = self.visit_lx_addr(ggl_assignment.rvalue)
        command = MimicGGLFromL1(l1_addr)
        return [command]

    def visit_lgl_assignment(
                self: "CommandScheduler",
                lgl_assignment: LGL_ASSIGNMENT) \
            -> Sequence[Command]:
        lx_reg_or_offset = lgl_assignment.rvalue
        lx_addr = self.visit_lx_addr(lx_reg_or_offset)

        if isinstance(lx_reg_or_offset, L1_REG) \
           or (isinstance(lx_reg_or_offset, LXRegWithOffsets)
               and isinstance(lx_reg_or_offset.parameter, L1_REG)):
            command = MimicLGLFromL1(lx_addr)

        elif isinstance(lx_reg_or_offset, L2_REG) \
             or (isinstance(lx_reg_or_offset, LXRegWithOffsets)
                 and isinstance(lx_reg_or_offset.parameter, L2_REG)):
            command = MimicLGLFromL2(lx_addr)

        else:
            raise ValueError(
                f"Unsupported type for rvalue "
                f"({lx_reg_or_offset.__class__.__name__}): "
                f"{lx_reg_or_offset}")

        return [command]

    def visit_lx_assignment(
                self: "CommandScheduler",
                lx_assignment: LX_ASSIGNMENT) \
            -> Sequence[Command]:

        lx_reg_or_offset = lx_assignment.lvalue
        lx_addr = self.visit_lx_addr(lx_reg_or_offset)
        rvalue = lx_assignment.rvalue

        if isinstance(lx_reg_or_offset, L1_REG) \
           or (isinstance(lx_reg_or_offset, LXRegWithOffsets)
               and isinstance(lx_reg_or_offset.parameter, L1_REG)):

            if isinstance(rvalue, GGL_EXPR):
                command = MimicL1FromGGL(lx_addr)
            elif isinstance(rvalue, LGL_EXPR):
                command = MimicL1FromLGL(lx_addr)
            else:
                raise ValueError(
                    f"Unsupported type for rvalue "
                    f"({rvalue.__class__.__name__}): {lx_assignment}")

        elif isinstance(lx_reg_or_offset, L2_REG) \
             or (isinstance(lx_reg_or_offset, LXRegWithOffsets)
                 and isinstance(lx_reg_or_offset.parameter, L2_REG)):

            if isinstance(rvalue, LGL_EXPR):
                command = MimicL2FromLGL(lx_addr)
            else:
                raise ValueError(
                    f"Unsupported type for rvalue "
                    f"({rvalue.__class__.__name__}): {lx_assignment}")

        else:
            raise ValueError(
                f"Unsupported type for rvalue "
                f"({lx_reg_or_offset.__class__.__name__}): "
                f"{lx_reg_or_offset}")

        return [command]

    def visit_rsp16_assignment(self: "CommandScheduler",
                               rsp16_assignment: RSP16_ASSIGNMENT) \
            -> Sequence[Command]:

        if rsp16_assignment.rvalue is RSP16_RVALUE.RSP256:
            command = MimicRSP16FromRSP256()
            return [command]
        raise NotImplementedError(
            f"Unsupported RSP16_RVALUE: {rsp16_assignment.rvalue}")

    def visit_rsp256_assignment(self: "CommandScheduler",
                                rsp256_assignment: RSP256_ASSIGNMENT) \
            -> Sequence[Command]:

        if rsp256_assignment.rvalue is RSP256_RVALUE.RSP16:
            command = MimicRSP256FromRSP16()
        elif rsp256_assignment.rvalue is RSP256_RVALUE.RSP2K:
            command = MimicRSP256FromRSP2K()
        else:
            raise NotImplementedError(
                f"Unsupported RSP256_RVALUE: {rsp256_assignment.rvalue}")
        return [command]

    def visit_rsp2k_assignment(self: "CommandScheduler",
                               rsp2k_assignment: RSP2K_ASSIGNMENT) \
            -> Sequence[Command]:

        if rsp2k_assignment.rvalue is RSP2K_RVALUE.RSP256:
            command = MimicRSP2KFromRSP256()
        elif rsp2k_assignment.rvalue is RSP2K_RVALUE.RSP32K:
            command = MimicRSP2KFromRSP32K()
        else:
            raise NotImplementedError(
                f"Unsupported RSP2K_RVALUE: {rsp2k_assignment.rvalue}")
        return [command]

    def visit_rsp32k_assignment(
                self: "CommandScheduler",
                rsp32k_assignment: RSP32K_ASSIGNMENT) -> Sequence[Command]:
        if rsp32k_assignment.rvalue is RSP32K_RVALUE.RSP2K:
            command = MimicRSP32KFromRSP2K()
            return [command]
        raise NotImplementedError(
            f"Unsupported RSP32K_RVALUE: {rsp32k_assignment.rvalue}")

    def visit_binary_expr(self: "CommandScheduler",
                          binary_expr: BINARY_EXPR) \
            -> Tuple[Command, Command]:
        vrs = self.visit_unary_sb(binary_expr.left_operand)
        src = self.visit_unary_src(binary_expr.right_operand)
        return vrs, src

    def visit_unary_expr(self: "CommandScheduler",
                         unary_expr: UNARY_EXPR) -> Union[Command, bool]:
        if isinstance(unary_expr.expression, UNARY_SB):
            return self.visit_unary_sb(unary_expr.expression)
        if isinstance(unary_expr.expression, UNARY_SRC):
            return self.visit_unary_src(unary_expr.expression)
        if isinstance(unary_expr.expression, BIT_EXPR):
            return self.visit_bit_expr(unary_expr.expression)
        raise NotImplementedError(
            f"Unsupported expression type: {unary_expr.expression.__class__.__name__}")

    def visit_unary_src(self: "CommandScheduler",
                        unary_src: UNARY_SRC) -> Command:
        command = self.visit_src_expr(unary_src.expression)
        return command

    def visit_unary_sb(self: "CommandScheduler", unary_sb: UNARY_SB) -> Command:
        return self.visit_sb_expr(unary_sb.expression)

    def visit_sb_expr(self: "CommandScheduler", sb_expr: SB_EXPR) -> Command:
        registers = []
        for parameter in sb_expr:
            if isinstance(parameter, RN_REG):
                rn_reg = parameter
                registers.append(self.visit_rn_reg(rn_reg))
            elif isinstance(parameter, RE_REG):
                re_reg = parameter
                registers.append(self.visit_re_reg(re_reg))
            elif isinstance(parameter, EWE_REG):
                ewe_reg = parameter
                registers.append(self.visit_ewe_reg(ewe_reg))
            elif isinstance(parameter, ExtendedRegister):
                xe_reg = parameter
                registers.append(self.visit_extended_register(xe_reg))
            elif parameter is not None:
                raise ValueError(
                    f"Unsupported parameter type "
                    f"({parameter.__class__.__name__}): {parameter}")
        return LoadRegisters(registers)

    def visit_rn_reg(self: "CommandScheduler", rn_reg: RN_REG) -> Command:
        return LoadRegister(kind=RegisterKind.RN_REG,
                            symbol=rn_reg.identifier)

    def visit_re_reg(self: "CommandScheduler", re_reg: RE_REG) -> Command:
        return LoadRegister(kind=RegisterKind.RE_REG,
                            symbol=re_reg.identifier)

    def visit_ewe_reg(self: "CommandScheduler", ewe_reg: EWE_REG) -> Command:
        return LoadRegister(kind=RegisterKind.EWE_REG,
                            symbol=ewe_reg.identifier)

    def visit_extended_register(
            self: "CommandScheduler",
            extended_register: ExtendedRegister) -> LoadRegister:
        register = extended_register.register
        if isinstance(register, RE_REG):
            command = self.visit_re_reg(register)
        elif isinstance(register, EWE_REG):
            command = self.visit_ewe_reg(register)
        else:
            raise ValueError(
                f"Unsupported register type ({register.__class__.__name__}): "
                f"{extended_register}")
        command = command.having(
            shift_width=extended_register.num_shifted_bits,
            invert=(extended_register.operator is UNARY_OP.NEGATE))
        return command

    def visit_sm_reg(self: "CommandScheduler", sm_reg: SM_REG) -> LoadRegister:
        return LoadRegister(kind=RegisterKind.SM_REG,
                            symbol=sm_reg.identifier)

    def visit_lx_addr(self: "CommandScheduler",
                      lx_addr: LX_ADDR) -> LoadRegister:

        if isinstance(lx_addr, LXParameter.__args__):
            return self.visit_lx_parameter(lx_addr)

        if isinstance(lx_addr, LXRegWithOffsets):
            return self.visit_lx_reg_with_offsets(lx_addr)

        raise ValueError(
            f"Unsupported type for lx_addr "
            f"({lx_addr.__class__.__name__}): {lx_addr}")

    def visit_lx_reg_with_offsets(
            self: "CommandScheduler",
            lx_reg_with_offsets: LXRegWithOffsets) -> LoadRegister:

        offset = lx_reg_with_offsets.offset
        lx_reg = lx_reg_with_offsets.parameter
        command = self.visit_lx_parameter(lx_reg)
        command = command.having(offset=offset)
        return command

    def visit_lx_parameter(self: "CommandScheduler",
                           lx_parameter: LXParameter) -> LoadRegister:

        if isinstance(lx_parameter, L1_REG):
            l1_reg = lx_parameter
            return self.visit_l1_reg(l1_reg)

        if isinstance(lx_parameter, L2_REG):
            l2_reg = lx_parameter
            return self.visit_l2_reg(l2_reg)

        raise ValueError(
            f"Unsupported type for lx_parameter "
            f"({lx_parameter.__class__.__name__}): {lx_parameter}")

    def visit_l1_reg(self: "CommandScheduler", l1_reg: L1_REG) -> LoadRegister:
        return LoadRegister(kind=RegisterKind.L1_REG,
                            symbol=l1_reg.identifier)

    def visit_l2_reg(self: "CommandScheduler", l2_reg: L2_REG) -> LoadRegister:
        return LoadRegister(kind=RegisterKind.L2_REG,
                            symbol=l2_reg.identifier)

    def visit_bit_expr(self: "CommandScheduler", bit_expr: BIT_EXPR) -> bool:
        return bool(int(bit_expr.value))

    def visit_src_expr(self: "CommandScheduler",
                       src_expr: SRC_EXPR) -> Command:
        return LoadSrc(src_expr)

    def visit_special(
            self: "CommandScheduler",
            special: SPECIAL) -> Sequence[Command]:
        if special == SPECIAL.NOOP:
            command = MimicNoOp()
        elif special == SPECIAL.FSEL_NOOP:
            command = MimicFSelNoOp()
        elif special == SPECIAL.RSP_END:
            command = MimicRSPEnd()
        elif special == SPECIAL.RSP_START_RET:
            command = MimicRSPStartRet()
        elif special == SPECIAL.L2_END:
            command = MimicL2End()
        else:
            raise NotImplementedError(f"Unsupported special value: {special}")
        return [command]
