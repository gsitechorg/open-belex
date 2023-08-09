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
import re
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from inspect import getfullargspec, getsourcelines
from itertools import chain, islice, product
from typing import (Any, Callable, Deque, Dict, Iterator, List,
                    MutableSequence, Optional, Sequence, Set, Tuple, Type,
                    TypeVar, Union)
from warnings import warn

import numpy as np

import open_belex.bleir.types as BLEIR
from open_belex.bleir.analyzers import RegisterParameterFinder
from open_belex.bleir.types import (ASSIGN_OP, BINOP, BleirEnum,
                                    BleirSerializable, bleir_dataclass,
                                    collectible, immutable, is_sequence)
from open_belex.bleir.walkables import BLEIRWalker
from open_belex.common.constants import (NSB, NSECTIONS, NUM_PLATS_PER_APUC,
                                         NUM_PLATS_PER_HALF_BANK)
from open_belex.common.seu_layer import SEULayer
from open_belex.common.stack_manager import StackManager, contextual
from open_belex.common.types import Indices, Integer
from open_belex.diri.half_bank import DIRI, make_vector_register
from open_belex.pyble.blecci import BLECCI
from open_belex.utils.example_utils import u16_to_bool
from open_belex.utils.index_utils import parse_indices
from open_belex.utils.section_utils import parse_sections

LOGGER = logging.getLogger()

MaskLiteral = Indices

# Reserves variable identifiers for the compiler
RESERVED_IDENTIFIERS: Set[str] = set()

SNIPPET_BUILDER: str = "SnippetBuilder"


def in_belex_temporary(fn: Callable) -> Callable:

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not Belex.has_context():
            return fn(*args, **kwargs)

        belex = Belex.context()
        _in_temporary = belex.in_temporary
        belex.in_temporary = True
        retval = fn(*args, **kwargs)
        belex.in_temporary = _in_temporary
        return retval

    return wrapper


class Instruction(ABC):
    pass


@bleir_dataclass
class NoopInstruction(Instruction, BleirSerializable):
    __eq_excludes__ = ["file_path", "line_number"]

    file_path: Optional[str] = None
    line_number: int = -1

    def as_bleir(self: "NoopInstruction") -> BLEIR.SPECIAL:
        metadata = {
            BLEIR.StatementMetadata.FILE_PATH: self.file_path,
            BLEIR.StatementMetadata.LINE_NUMBER: self.line_number,
        }
        return BLEIR.STATEMENT(BLEIR.SPECIAL.NOOP) \
                    .having(metadata=metadata)


@bleir_dataclass
class FselNoopInstruction(Instruction, BleirSerializable):
    __eq_excludes__ = ["file_path", "line_number"]

    file_path: Optional[str] = None
    line_number: int = -1

    def as_bleir(self: "FselNoopInstruction") -> BLEIR.SPECIAL:
        metadata = {
            BLEIR.StatementMetadata.FILE_PATH: self.file_path,
            BLEIR.StatementMetadata.LINE_NUMBER: self.line_number,
        }
        return BLEIR.STATEMENT(BLEIR.SPECIAL.FSEL_NOOP) \
                    .having(metadata=metadata)


@bleir_dataclass
class RspEndInstruction(Instruction, BleirSerializable):
    __eq_excludes__ = ["file_path", "line_number"]

    file_path: Optional[str] = None
    line_number: int = -1

    def as_bleir(self: "RspEndInstruction") -> BLEIR.SPECIAL:
        metadata = {
            BLEIR.StatementMetadata.FILE_PATH: self.file_path,
            BLEIR.StatementMetadata.LINE_NUMBER: self.line_number,
        }
        return BLEIR.STATEMENT(BLEIR.SPECIAL.RSP_END) \
                    .having(metadata=metadata)


@bleir_dataclass
class RspStartRetInstruction(Instruction, BleirSerializable):
    __eq_excludes__ = ["file_path", "line_number"]

    file_path: Optional[str] = None
    line_number: int = -1

    def as_bleir(self: "RspStartRetInstruction") -> BLEIR.SPECIAL:
        metadata = {
            BLEIR.StatementMetadata.FILE_PATH: self.file_path,
            BLEIR.StatementMetadata.LINE_NUMBER: self.line_number,
        }
        return BLEIR.STATEMENT(BLEIR.SPECIAL.RSP_START_RET) \
                    .having(metadata=metadata)


@bleir_dataclass
class L2EndInstruction(Instruction, BleirSerializable):
    __eq_excludes__ = ["file_path", "line_number"]

    file_path: Optional[str] = None
    line_number: int = -1

    def as_bleir(self: "L2EndInstruction") -> BLEIR.SPECIAL:
        metadata = {
            BLEIR.StatementMetadata.FILE_PATH: self.file_path,
            BLEIR.StatementMetadata.LINE_NUMBER: self.line_number,
        }
        return BLEIR.STATEMENT(BLEIR.SPECIAL.L2_END) \
                    .having(metadata=metadata)


SPECIAL_INSTRUCTIONS = {
    "NOOP": NoopInstruction(),
    "FSEL_NOOP": FselNoopInstruction(),
    "RSP_END": RspEndInstruction(),
    "RSP_START_RET": RspStartRetInstruction(),
    "L2_END": L2EndInstruction(),
}


@bleir_dataclass
class BelexSymbol(BleirSerializable):
    symbol: str
    is_internal: bool = False
    is_external: bool = False


def file_line() -> Tuple[str, int]:
    for frame_info in inspect.stack():
        if frame_info.filename != __file__:
            return frame_info.filename, frame_info.lineno


@bleir_dataclass
class BelexSpecial(BelexSymbol):

    def as_bleir(self: "BelexSymbol") -> BLEIR.STATEMENT:
        operation = BLEIR.SPECIAL.find_by_value(self.symbol)
        return BLEIR.STATEMENT(operation=operation)

    def __call__(self: "BelexAccess") -> None:
        file_path, line_number = file_line()
        instruction = SPECIAL_INSTRUCTIONS[self.symbol].having(
            file_path=file_path,
            line_number=line_number)
        Belex.add_instruction(instruction)


@bleir_dataclass
class ReadWriteInhibitFactory:
    read_write_inhibit: bool

    @in_belex_temporary
    def __getitem__(self: "ReadWriteInhibitFactory",
                    expr: Optional[Any]) -> None:

        if expr is None:
            # msk: RL eqop SB RWINH_SET
            # msk: RL eqop SB RWINH_RST

            belex = Belex.context()
            instr = belex.instructions[-1]

            lvalue = instr.lvalue
            if not isinstance(lvalue, BelexAccess):
                raise ValueError(
                    f"Unsupported assignment type ({lvalue.__class__.__name__}): {lvalue}")

            mask = lvalue.mask
            if mask is None:
                raise AssertionError(f"Assignment must be masked: {instr}")

            mask = mask.having(read_write_inhibit=self.read_write_inhibit)
            lvalue = lvalue.having(mask=mask)
            instr = instr.having(lvalue=lvalue)
            belex.instructions[-1] = instr
            # This will result in a double READ during interactive debugging mode
            # (only), but it is necessary for the correct behavior of RWINH since
            # it behaves differently within the same instruction as the READ than
            # without.
            Belex.interpret(instr)

        elif isinstance(expr, Mask):
            # msk: RWINH_SET
            # msk: RWINH_RST

            mask = expr.having(read_write_inhibit=self.read_write_inhibit)
            Belex.add_instruction(mask)

        else:
            mask_literal = expr
            mask = Mask.parse_literal(mask_literal)
            self.__getitem__(mask)


RWINH_SET = ReadWriteInhibitFactory(True)
RWINH_RST = ReadWriteInhibitFactory(False)


class Operable(ABC):
    """Vestigial docstring"""
    @abstractmethod
    def operands(self: "Operable") -> List[Any]:
        raise NotImplementedError


class Masqued(ABC):

    @property
    def mask(self: "Masqued") -> Optional["Mask"]:
        return None

    def masquerade(self: "Masqued") -> List["Masqued"]:
        return [self]


@dataclass
class Expressible:

    @in_belex_temporary
    def __invert__(self: "Expressible") -> "Expressible":
        is_negated = (not self.is_negated)
        return self.having(is_negated=is_negated)

    def __or__(self: "Expressible", other: "Conjunctible") -> "BinaryOperation":
        return BinaryOperation(operator=BINOP.OR, lhs=self, rhs=other)

    def __and__(self: "Expressible", other: "Conjunctible") -> "BinaryOperation":
        return BinaryOperation(operator=BINOP.AND, lhs=self, rhs=other)

    def __xor__(self: "Expressible", other: "Conjunctible") -> "BinaryOperation":
        return BinaryOperation(operator=BINOP.XOR, lhs=self, rhs=other)

    def __le__(self: "Expressible", other: "Operand") -> None:
        file_path, line_number = file_line()
        assignment = AssignOperation(operator=ASSIGN_OP.EQ,
                                     lvalue=self,
                                     rvalue=other,
                                     file_path=file_path,
                                     line_number=line_number)
        Belex.add_instruction(assignment)

    def __ior__(self: "Expressible", other: "Operand") -> None:
        file_path, line_number = file_line()
        assignment = AssignOperation(operator=ASSIGN_OP.OR_EQ,
                                     lvalue=self,
                                     rvalue=other,
                                     file_path=file_path,
                                     line_number=line_number)
        Belex.add_instruction(assignment)

    def __iand__(self: "Expressible", other: "Operand") -> None:
        file_path, line_number = file_line()
        assignment = AssignOperation(operator=ASSIGN_OP.AND_EQ,
                                     lvalue=self,
                                     rvalue=other,
                                     file_path=file_path,
                                     line_number=line_number)
        Belex.add_instruction(assignment)

    def __ixor__(self: "Expressible", other: "Operand") -> None:
        file_path, line_number = file_line()
        assignment = AssignOperation(operator=ASSIGN_OP.XOR_EQ,
                                     lvalue=self,
                                     rvalue=other,
                                     file_path=file_path,
                                     line_number=line_number)
        Belex.add_instruction(assignment)


@bleir_dataclass
class SymbolTable:
    lookup: Dict[str, BelexSymbol] = field(default_factory=dict)
    serial_vr_index: int = 0
    serial_re_index: int = 0
    serial_ewe_index: int = 0
    serial_l1_index: int = 0
    serial_l2_index: int = 0
    serial_sm_index: int = 0

    def gen_vr_nym(self: "SymbolTable") -> str:
        vr_nym = f"_INTERNAL_VR_{self.serial_vr_index:03d}"
        self.serial_vr_index += 1
        return vr_nym

    def gen_re_nym(self: "SymbolTable") -> str:
        re_nym = f"_INTERNAL_RE_{self.serial_re_index:03d}"
        self.serial_re_index += 1
        return re_nym

    def gen_ewe_nym(self: "SymbolTable") -> str:
        ewe_nym = f"_INTERNAL_EWE_{self.serial_ewe_index:03d}"
        self.serial_ewe_index += 1
        return ewe_nym

    def gen_l1_nym(self: "SymbolTable") -> str:
        l1_nym = f"_INTERNAL_L1_{self.serial_l1_index:03d}"
        self.serial_l1_index += 1
        return l1_nym

    def gen_l2_nym(self: "SymbolTable") -> str:
        l2_nym = f"_INTERNAL_L2_{self.serial_l2_index:03d}"
        self.serial_l2_index += 1
        return l2_nym

    def gen_sm_nym(self: "SymbolTable") -> str:
        sm_nym = f"_INTERNAL_SM_{self.serial_sm_index:03d}"
        self.serial_sm_index += 1
        return sm_nym

    def __contains__(self: "SymbolTable", symbol: str) -> bool:
        return symbol in self.lookup

    def __getitem__(self: "SymbolTable", symbol: str) -> BelexSymbol:
        if symbol in self.lookup:
            return self.lookup[symbol]
        raise ValueError(f"Cannot find symbol: {symbol}")

    def __setitem__(self: "SymbolTable",
                    symbol: str,
                    value: BelexSymbol) -> None:
        self.assert_is_available(symbol, value)
        self.lookup[symbol] = value

    def assert_is_available(self: "SymbolTable",
                            symbol: str,
                            value: BelexSymbol) -> None:

        if symbol in RESERVED_IDENTIFIERS:
            raise ValueError(f"Symbol is a reserved term: {symbol}")

        if symbol.startswith("_INTERNAL"):
            pass
        elif symbol in self and self[symbol] != value:
            raise ValueError(
                f"Already defined symbol ({symbol}): {self[symbol]}")

        if not (value.is_internal or value.is_external) \
           and symbol.startswith("_INTERNAL"):
            raise ValueError(
                f"User-defined symbols may not begin with \"_INTERNAL\": "
                f"{symbol}")

    def make_vector_register(
            self: "SymbolTable",
            symbol: Optional[str] = None,
            initial_value: Optional[int] = None,
            register: Optional[int] = None,
            row_number: Optional[int] = None) -> "VR":

        is_internal = False
        if symbol is None:
            symbol = self.gen_vr_nym()
            is_internal = True

        vr = VR(symbol=symbol,
                initial_value=initial_value,
                register=register,
                row_number=row_number,
                is_temporary=True,
                is_internal=is_internal)

        self.assert_is_available(symbol, vr)
        self.lookup[symbol] = vr
        return vr

    def make_re_register(
            self: "SymbolTable",
            symbol: Optional[str] = None,
            register: Optional[int] = None,
            row_mask: Optional[int] = None,
            rows: Optional[Sequence[Union["VR", "RE"]]] = None) -> "RE":

        is_internal = False
        if symbol is None:
            symbol = self.gen_re_nym()
            is_internal = True

        re = RE(symbol=symbol,
                register=register,
                row_mask=row_mask,
                rows=rows,
                is_internal=is_internal)

        self.assert_is_available(symbol, re)
        self.lookup[symbol] = re
        return re

    def make_ewe_register(
            self: "SymbolTable",
            symbol: Optional[str] = None,
            register: Optional[int] = None,
            wordline_mask: Optional[int] = None) -> "EWE":

        is_internal = False
        if symbol is None:
            symbol = self.gen_ewe_nym()
            is_internal = True

        ewe = EWE(symbol=symbol,
                  register=register,
                  wordline_mask=wordline_mask,
                  is_internal=is_internal)

        self.assert_is_available(symbol, ewe)
        self.lookup[symbol] = ewe
        return ewe

    def make_l1_register(
            self: "SymbolTable",
            symbol: Optional[str] = None,
            register: Optional[int] = None,
            bank_group_row: Optional[int] = None) -> "L1":

        is_internal = False
        if symbol is None:
            symbol = self.gen_l1_nym()
            is_internal = True

        l1 = L1(symbol=symbol,
                register=register,
                bank_group_row=bank_group_row,
                is_internal=is_internal)

        self.assert_is_available(symbol, l1)
        self.lookup[symbol] = l1
        return l1

    def make_l2_register(
            self: "SymbolTable",
            symbol: Optional[str] = None,
            register: Optional[int] = None,
            value: Optional[int] = None) -> "L2":

        is_internal = False
        if symbol is None:
            symbol = self.gen_l2_nym()
            is_internal = True

        l2 = L2(symbol=symbol,
                register=register,
                value=value,
                is_internal=is_internal)

        self.assert_is_available(symbol, l2)
        self.lookup[symbol] = l2
        return l2

    def make_mask(
            self: "SymbolTable",
            symbol: Optional[str] = None,
            constant_value: Optional[int] = None,
            check_availability: bool = True,
            register: Optional[int] = None) -> "Mask":

        is_internal = False
        if symbol is None:
            if constant_value is not None:
                symbol = f"_INTERNAL_SM_0X{constant_value:04X}"
            else:
                symbol = self.gen_sm_nym()
            is_internal = True

        sm = Mask(symbol=symbol,
                  constant_value=constant_value,
                  register=register,
                  is_internal=is_internal)

        if check_availability:
            self.assert_is_available(symbol, sm)

        self.lookup[symbol] = sm
        return sm

    def make_section(
            self: "SymbolTable",
            symbol: Optional[str] = None,
            constant_value: Optional[int] = None,
            register: Optional[int] = None) -> "Section":

        is_internal = False
        if symbol is None:
            if constant_value is not None:
                symbol = f"_INTERNAL_SM_0X{1<<constant_value:04X}"
            else:
                symbol = self.gen_sm_nym()
            is_internal = True

        section = Section(symbol=symbol,
                          constant_value=constant_value,
                          register=register,
                          is_internal=is_internal)

        self.assert_is_available(symbol, section)
        self.lookup[symbol] = section
        return section


T = TypeVar("T")
XS = List[T]
XSS = Union[XS, List["XSS"], Iterator["XSS"]]


def flatten(xss: XSS) -> List[T]:
    flattened = []

    if not hasattr(xss, "__iter__"):
        x = xss
        flattened.append(x)
        return flattened

    for x_or_xs in xss:
        if isinstance(x_or_xs, list):
            xs = x_or_xs
            flattened += flatten(xs)
        else:
            x = x_or_xs
            flattened.append(x)

    return flattened


@bleir_dataclass
@collectible("instructions")
class Lane(Instruction, BleirSerializable):
    __eq_excludes__ = ["file_path", "line_number"]

    instructions: Deque[Instruction] = field(default_factory=deque)
    comment: Optional[str] = None

    file_path: Optional[str] = None
    line_number: Optional[int] = None

    def as_bleir(self: "Lane") -> BLEIR.Operation:
        statements = flatten(instruction.as_bleir()
                             for instruction in self.instructions)

        comment = None
        if self.comment is not None:
            comment = BLEIR.SingleLineComment(line=self.comment)

        metadata = {
            BLEIR.StatementMetadata.FILE_PATH: self.file_path,
            BLEIR.StatementMetadata.LINE_NUMBER: self.line_number,
        }

        return BLEIR.MultiStatement(statements=statements,
                                    comment=comment,
                                    metadata=metadata)


@bleir_dataclass
class SingleLineComment(Instruction, BleirSerializable):
    line: str

    def as_bleir(self: "SingleLineComment") -> BLEIR.SingleLineComment:
        return BLEIR.SingleLineComment(line=self.line)


@bleir_dataclass
class MultiLineComment(Instruction, BleirSerializable):
    lines: Sequence[str]

    def as_bleir(self: "MultiLineComment") -> BLEIR.MultiLineComment:
        return BLEIR.MultiLineComment(lines=self.lines)


def comment(line_or_lines: Union[str, Sequence[str]]) -> None:
    belex = Belex.context()
    belex.comment(line_or_lines)


class GlassFormat(BleirEnum):
    BIN = "bin"
    HEX = "hex"

    def as_bleir(self: "GlassFormat") -> BLEIR.GlassFormat:
        return BLEIR.GlassFormat.find_by_value(self.value)


class GlassOrder(BleirEnum):
    LEAST_SIGNIFICANT_BIT_FIRST = "lsb"
    MOST_SIGNIFICANT_BIT_FIRST = "msb"

    def as_bleir(self: "GlassOrder") -> BLEIR.GlassFormat:
        return BLEIR.GlassOrder.find_by_value(self.value)


def value_by_parameter_id(fn: Callable) -> Callable:

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        register = fn(self, *args, **kwargs)
        if self.debug:
            value = None
            data = None
            if isinstance(register, VR):
                value = register.row_number
                if register.initial_value is not None:
                    data = u16_to_bool(register.initial_value)
                else:
                    data = make_vector_register()
            elif isinstance(register, RE):
                value = register.row_mask
            elif isinstance(register, EWE):
                value = register.wordline_mask
            elif isinstance(register, L1):
                value = register.bank_group_row
            elif isinstance(register, L2):
                value = register.value
            elif isinstance(register, Section):
                shift_width = register.constant_value
                value = (0x0001 << shift_width)
                # value = register.constant_value
            elif isinstance(register, Mask):
                value = register.constant_value
            else:
                raise ValueError(
                    f"Unsupported register type ({register.__class__.__name__}): "
                    f"{register}")
            parameter_id = register.symbol
            if value is not None:
                self.values_by_parameter_id[parameter_id] = value
            if data is not None:
                self.data_by_parameter_id[parameter_id] = data
        return register

    return wrapper


@contextual
@bleir_dataclass
@collectible("instructions")
class Belex(BleirSerializable):
    instructions: Deque[Instruction] = field(default_factory=deque)
    symbol_table: SymbolTable = field(default_factory=SymbolTable)
    parent: Optional["Belex"] = None
    is_bleir: bool = False
    is_appendable: bool = False

    frag_nym: Optional[str] = None
    debug_parameters: Optional[Sequence["FormalParameter"]] = None
    values_by_parameter_id: Optional[Dict[str, Union[int, str]]] = None
    data_by_parameter_id: Optional[Dict[str, Union[int, str, np.ndarray]]] = None
    debug: bool = False
    captured_glass: Optional[MutableSequence[str]] = None

    interpreter: Optional["BLEIRInterpreter"] = None

    in_fragment: bool = False
    in_temporary: bool = False

    @staticmethod
    def assert_constructible() -> None:
        if not Belex.has_context():
            return

        belex = Belex.context()
        if belex.in_fragment and not belex.in_temporary:
            raise AssertionError(
                f"You must use the Belex version of the constructor within a "
                f"fragment, e.g. Belex.VR(...) instead of calling VR(...) "
                f"directly.")

    @property
    def diri(self: "Belex") -> Optional[DIRI]:
        if self.interpreter is not None:
            return self.interpreter.diri
        return None

    def __post_init__(self: "Belex") -> None:
        if self.debug and self.parent is None:
            from belex.bleir.interpreters import BLEIRInterpreter
            if BLEIRInterpreter.has_context():
                self.interpreter = BLEIRInterpreter.context()
            else:
                diri = DIRI.context()
                self.interpreter = BLEIRInterpreter(diri=diri)
            self.interpreter.values_by_parameter_id = self.values_by_parameter_id
            self.interpreter.data_by_parameter_id = self.data_by_parameter_id
            self.interpreter.frag_nym = self.frag_nym

            if StackManager.has_elem("used_sbs"):
                used_sbs = StackManager.peek("used_sbs")

                _visit_rn_reg = self.interpreter.visit_rn_reg
                def visit_rn_reg(rn_reg):
                    # We only want to capture user parameters (non-temporaries)
                    # and globals (lowered parameters)
                    sb = _visit_rn_reg(rn_reg)
                    if (not rn_reg.is_temporary
                        and not rn_reg.identifier.startswith("_INTERNAL")) \
                       or rn_reg.is_lowered:
                        used_sbs.add(sb)
                    return sb
                self.interpreter.visit_rn_reg = visit_rn_reg

                in_extended_register = False

                _visit_re_reg = self.interpreter.visit_re_reg
                def visit_re_reg(re_reg):
                    sbs_and_vrs = _visit_re_reg(re_reg)
                    if not in_extended_register:
                        for sb_or_vr in sbs_and_vrs:
                            if not isinstance(sb_or_vr, np.ndarray):
                                sb = sb_or_vr
                                used_sbs.add(sb)
                    return sbs_and_vrs
                self.interpreter.visit_re_reg = visit_re_reg

                _visit_ewe_reg = self.interpreter.visit_ewe_reg
                def visit_ewe_reg(ewe_reg):
                    sbs_and_vrs = _visit_ewe_reg(ewe_reg)
                    if not in_extended_register:
                        for sb_or_vr in sbs_and_vrs:
                            if not isinstance(sb_or_vr, np.ndarray):
                                sb = sb_or_vr
                                used_sbs.add(sb)
                    return sbs_and_vrs
                self.interpreter.visit_ewe_reg = visit_ewe_reg

                _visit_extended_register = self.interpreter.visit_extended_register
                def visit_extended_register(extended_register):
                    nonlocal in_extended_register
                    in_extended_register = True
                    sbs_and_vrs = _visit_extended_register(extended_register)
                    for sb_or_vr in sbs_and_vrs:
                        if not isinstance(sb_or_vr, np.ndarray):
                            sb = sb_or_vr
                            used_sbs.add(sb)
                    in_extended_register = False
                    return sbs_and_vrs
                self.interpreter.visit_extended_register = visit_extended_register

    def comment(self: "Belex",
                line_or_lines: Union[str, Sequence[str]]) -> None:

        if isinstance(line_or_lines, str):
            line = line_or_lines
            comment = SingleLineComment(line=line)

        elif is_sequence(line_or_lines):
            lines = tuple(line_or_lines)
            comment = MultiLineComment(lines=lines)

        else:
            raise ValueError(
                f"Unsupported line type ({line_or_lines.__class__.__name__}): "
                f"{line_or_lines}")

        self.add_instruction(comment)

    def assert_true(self: "Belex",
                    predicate: Union[bool, Callable[[], bool]],
                    message: Optional[str] = None) -> None:
        if not self.debug:
            return
        if callable(predicate):
            predicate = predicate()
        if not predicate:
            raise AssertionError(message)

    def assert_eq(self: "Belex",
                  left_operand: Union[Any, Callable[[], bool]],
                  right_operand: Union[Any, Callable[[], bool]],
                  message: Optional[str] = None) -> None:
        if not self.debug:
            return
        if callable(left_operand):
            left_operand = left_operand()
        if callable(right_operand):
            right_operand = right_operand()
        if left_operand != right_operand:
            if message is not None:
                raise AssertionError(message)
            raise AssertionError("\n".join([
                "left_operand != right_operand",
                "-------------",
                "left_operand:",
                "-------------",
                str(left_operand),
                "--------------",
                "right_operand:",
                "--------------",
                str(right_operand),
            ]))

    def glass(self: "Belex", subject: "Glassible",
              comment: Optional[str] = None,
              plats: Optional[Indices] = None,
              sections: Optional[Indices] = None,
              fmt: Union[str, GlassFormat] = "hex",
              order: Optional[Union[str, GlassOrder]] = None,
              balloon: bool = True,
              rewrite: Optional[Dict[str, str]] = None) -> Optional[str]:

        if isinstance(fmt, str):
            fmt = GlassFormat.find_by_value(fmt.lower())

        if not isinstance(fmt, GlassFormat):
            raise ValueError(
                f"Unsupported fmt type ({fmt.__class__.__name__}): {fmt}")

        if order is None:
            if fmt is GlassFormat.HEX:
                order = "msb"
            elif fmt is GlassFormat.BIN:
                order = "lsb"
            else:
                raise ValueError(
                    f"Unsupported fmt type ({fmt.__class__.__name__}): {fmt}")

        if isinstance(order, str):
            order = GlassOrder.find_by_value(order.lower())

        if not isinstance(order, GlassOrder):
            raise ValueError(
                f"Unsupported order type ({order.__class__.__name__}): {order}")

        # Ballooning doesn't make sense in the context of LX or LGL
        if balloon and \
           (subject is LGL or isinstance(subject, (L1, L2, OffsetLX))):
            balloon = False

        nplats = NUM_PLATS_PER_APUC if balloon else glassible_plats(subject)
        plats = parse_indices(plats, upper_bound=nplats)

        nsections = NSECTIONS if balloon else glassible_sections(subject)
        sections = parse_indices(sections, upper_bound=nsections)

        file_path, line_number = file_line()

        instruction = GlassStatement(
            subject=subject,
            comment=comment,
            sections=sections,
            plats=plats,
            fmt=fmt,
            order=order,
            balloon=balloon,
            rewrite=rewrite,
            file_path=file_path,
            line_number=line_number)

        # self.instructions.append(instruction)
        self.add_instruction(instruction)

        if self.debug and self.parent is None:
            glass_statement = instruction.as_bleir()[0]
            snapshot = "\n".join(
                self.interpreter.format_subject_rows(glass_statement))
            if self.captured_glass is not None:
                self.captured_glass.append(snapshot)
            return snapshot

    @in_belex_temporary
    @value_by_parameter_id
    def VR(self: "Belex", initial_value: Optional[Integer] = None) -> "VR":
        if initial_value is not None and not 0x0000 <= initial_value <= 0xFFFF:
            raise AssertionError(
                f"initial_value must be between 0x0000 and 0xFFFF: "
                f"0x{initial_value:04X}")

        return self.symbol_table.make_vector_register(
            initial_value=initial_value)

    @in_belex_temporary
    @value_by_parameter_id
    def Mask(self: "Belex",
             constant_value: Indices,
             check_availability: bool = True) -> "Mask":
        """Vestigial docstring"""

        if not isinstance(constant_value, Integer.__args__):
            mask_literal = constant_value
            return Mask.parse_literal(mask_literal)

        Mask.validate(constant_value)

        return self.symbol_table.make_mask(
            constant_value=constant_value,
            check_availability=check_availability)

    @in_belex_temporary
    @value_by_parameter_id
    def Section(self: "Belex",
                constant_value: Optional[Integer] = None) -> "Section":
        if constant_value is not None:
            Section.validate(constant_value)
        return self.symbol_table.make_section(
            constant_value=constant_value)

    @classmethod
    def coerce_statement(
            cls: Type["Belex"],
            operation: Union[BLEIR.Operation, BLEIR.MASK]) -> BLEIR.Operation:

        if isinstance(operation, BLEIR.MASK):
            mask = operation
            if mask.read_write_inhibit is None:
                raise RuntimeError(
                    f"If an operation is a BLEIR.MASK it is expected to "
                    f"have a ReadWriteInhibit: {mask}")
            masked = BLEIR.MASKED(mask=mask)
            statement = BLEIR.STATEMENT(operation=masked)
            return statement

        elif isinstance(operation, BLEIR.GlassStatement):
            metadata = {
                BLEIR.StatementMetadata.FILE_PATH: operation.file_path,
                BLEIR.StatementMetadata.LINE_NUMBER: operation.line_number,
            }
            statement = BLEIR.STATEMENT(
                operation=operation,
                metadata=metadata)
            return statement

        elif isinstance(operation, BLEIR.STATEMENT):
            return operation

        elif isinstance(operation, BLEIR.MultiStatement):
            return operation.having(
                statements=list(map(cls.coerce_statement,
                                    operation.statements)))

        elif isinstance(operation, BLEIR.LineComment.__args__):
            return operation

        else:
            raise ValueError(
                f"Unsupported type ({operation.__class__.__name__}): {operation}")

    def as_bleir(self: "Belex") -> List[BLEIR.Operation]:
        operations = flatten(instruction.as_bleir()
                             for instruction in self.instructions)
        operations = list(map(self.coerce_statement, operations))
        return operations

    def add_symbol(self: "Belex", symbol: BelexSymbol) -> None:
        if symbol.symbol.startswith("_INTERNAL"):
            pass
        elif symbol.symbol in self.symbol_table:
            if self.symbol_table[symbol.symbol] != symbol:
                raise ValueError(
                    f"Already defined symbol: "
                    f"{self.symbol_table[symbol.symbol]}")
        else:
            self.symbol_table[symbol.symbol] = symbol

    def as_lane(self: "Belex",
                comment: Optional[str] = None,
                file_path: Optional[str] = None,
                line_number: Optional[int] = None) -> Lane:
        return Lane(instructions=self.instructions,
                    comment=comment,
                    file_path=file_path,
                    line_number=line_number)

    @staticmethod
    def push_context(*args, **kwargs) -> "Belex":
        if StackManager.has_elem(Belex.__STACK_NYM__):
            parent = StackManager.peek(Belex.__STACK_NYM__)
            symbol_table = parent.symbol_table
            belex = Belex(*args,
                          symbol_table=symbol_table,
                          parent=parent,
                          debug=parent.debug,
                          captured_glass=parent.captured_glass,
                          values_by_parameter_id=parent.values_by_parameter_id,
                          data_by_parameter_id=parent.data_by_parameter_id,
                          in_fragment=parent.in_fragment,
                          in_temporary=parent.in_temporary,
                          is_appendable=parent.is_appendable,
                          **kwargs)

        else:
            belex = Belex(*args, **kwargs)

        StackManager.push(Belex.__STACK_NYM__, belex)
        return belex

    @classmethod
    def interpret(cls: Type["Belex"], instruction: Instruction) -> None:
        belex = cls.context()
        if belex.debug and belex.parent is None:
            statements_and_multi_statements = instruction.as_bleir()
            if not isinstance(statements_and_multi_statements, list):
                statements_and_multi_statements = [statements_and_multi_statements]
            statements_and_multi_statements = \
                tuple(map(cls.coerce_statement,
                          flatten(statements_and_multi_statements)))
            for statement_or_multi_statement in statements_and_multi_statements:
                belex.interpreter.visit_operation(statement_or_multi_statement)

    @classmethod
    def add_instruction(cls: Type["Belex"], instruction: Instruction) -> None:
        belex = cls.context()
        belex.instructions.append(instruction)
        cls.interpret(instruction)


# The `int` Operand type is only for assigning to RL
Operand = Union["BelexAccess", "BinaryOperation", int]


class CallAccessible(ABC):

    def __call__(self: "CallAccessible") -> "BelexAccess":
        return BelexAccess(var=self)


class Accessible(CallAccessible):

    @in_belex_temporary
    def __getitem__(
            self: "Accessible",
            mask_or_literal: Optional[Union["Mask", MaskLiteral]] = None) \
            -> "BelexAccess":
        return BelexAccess(var=self, mask=mask_or_literal)

    def __setitem__(
            self: "Accessible",
            mask_or_literal: Optional[Union["Mask", MaskLiteral]],
            rvalue: Optional[Operand]) -> None:
        if rvalue is not None:
            self[mask_or_literal] <= rvalue


class Validatable(ABC):

    @staticmethod
    @abstractmethod
    def is_valid(value: Any) -> bool:
        raise NotImplementedError

    @classmethod
    def validate(cls: Type["Validatable"], value: Any) -> None:
        # A "str" value implies pass-by-reference, which is not something we can
        # validate here.
        if not isinstance(value, str) and not cls.is_valid(value):
            raise ValueError(f"Invalid value for type {cls.__name__}: {value}")


@bleir_dataclass
class VR(BelexSymbol, Accessible, Validatable, BleirSerializable, Operable):
    initial_value: Optional[int] = None  # Initial value for each plat in the VR
    register: Optional[int] = None       # Register id ϵ [0, 16)
    row_number: Optional[int] = None     # Row number
    is_lowered: bool = False
    is_literal: bool = False
    is_temporary: bool = False
    rn_reg: Optional[BLEIR.RN_REG] = None

    def __post_init__(self: "VR") -> None:
        Belex.assert_constructible()

    def as_bleir(self: "VR") -> BLEIR.RN_REG:
        if self.rn_reg is not None:
            return self.rn_reg
        return BLEIR.RN_REG(
            identifier=self.symbol,
            initial_value=self.initial_value,
            register=self.register,
            row_number=self.row_number,
            is_lowered=self.is_lowered,
            is_literal=self.is_literal,
            is_temporary=self.is_temporary)

    @property
    def reg_nym(self: "VR") -> Optional[str]:
        if self.register is not None:
            return f"RN_REG_{self.register}"
        return None

    def operands(self: "VR") -> List["VR"]:
        return [self]

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0 <= value < NSB

    @classmethod
    def parse_literal(cls: Type["VR"],
                      vr_literal: int) -> "VR":
        belex = Belex.context()
        return belex.VR(row_number=vr_literal)


@bleir_dataclass
class RE(BelexSymbol, Accessible, Validatable, BleirSerializable, Operable):
    row_mask: Optional[int] = None
    rows: Optional[Sequence[Union[VR, "RE"]]] = None
    register: Optional[int] = None       # Register id ϵ [0, 16)
    is_lowered: bool = False
    is_literal: bool = False
    re_reg: Optional[BLEIR.RE_REG] = None

    def __post_init__(self: "RE") -> None:
        Belex.assert_constructible()

    def as_bleir(self: "RE") -> BLEIR.RE_REG:
        if self.re_reg is not None:
            return self.re_reg
        rows = None
        if self.rows is not None:
            rows = tuple(row.as_bleir() for row in self.rows)
        return BLEIR.RE_REG(
            identifier=self.symbol,
            row_mask=self.row_mask,
            rows=rows,
            register=self.register,
            is_lowered=self.is_lowered,
            is_literal=self.is_literal)

    @property
    def reg_nym(self: "RE") -> Optional[str]:
        if self.register is not None:
            return f"RE_REG_{self.register}"
        return None

    def operands(self: "RE") -> List[Union[VR, "RE"]]:
        if self.rows is None:
            return [self]
        return [operand
                for row in self.rows
                for operand in row.operands()]

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0 <= value <= 0xFFFFFF

    @classmethod
    def parse_literal(cls: Type["RE"],
                      re_literal: int) -> "RE":
        belex = Belex.context()
        return belex.RE(row_mask=re_literal)


@bleir_dataclass
class EWE(BelexSymbol, Accessible, Validatable, BleirSerializable, Operable):
    wordline_mask: Optional[int] = None
    register: Optional[int] = None       # Register id ϵ [0, 16)
    is_lowered: bool = False
    is_literal: bool = False
    ewe_reg: Optional[BLEIR.EWE_REG] = None

    def __post_init__(self: "EWE") -> None:
        Belex.assert_constructible()

    def as_bleir(self: "EWE") -> BLEIR.EWE_REG:
        if self.ewe_reg is not None:
            return self.ewe_reg
        return BLEIR.EWE_REG(
            identifier=self.symbol,
            wordline_mask=self.wordline_mask,
            register=self.register,
            is_lowered=self.is_lowered,
            is_literal=self.is_literal)

    @property
    def reg_nym(self: "EWE") -> Optional[str]:
        if self.register is not None:
            return f"EWE_REG_{self.register}"
        return None

    def operands(self: "EWE") -> List["EWE"]:
        return [self]

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0 <= value < 0x300

    @classmethod
    def parse_literal(cls: Type["EWE"],
                      ewe_literal: int) -> "EWE":
        belex = Belex.context()
        return belex.EWE(wordline_mask=ewe_literal)


@bleir_dataclass
class XE(CallAccessible, Validatable, BleirSerializable, Operable, Expressible, Masqued):
    accessor: "BelexAccess"
    is_negated: bool = False
    num_shifted_bits: int = 0

    def __post_init__(self: "XE") -> None:
        Belex.assert_constructible()

    @property
    def var(self: "XE") -> "XE":
        return self

    @property
    def mask(self: "XE") -> Optional["Mask"]:
        return self.accessor.mask

    def as_bleir(self: "XE") -> BLEIR.ExtendedRegister:
        operator = None if not self.is_negated else BLEIR.UNARY_OP.NEGATE
        return BLEIR.ExtendedRegister(
            register=self.accessor.var.as_bleir(),
            operator=operator,
            num_shifted_bits=self.num_shifted_bits)

    def operands(self: "XE") -> List[BelexSymbol]:
        return self.accessor.operands()

    @in_belex_temporary
    def __invert__(self: "BelexAccess") -> "XE":
        is_negated = (not self.is_negated)
        return self.having(is_negated=is_negated)

    @in_belex_temporary
    def __lshift__(self: "BelexAccess", num_shifted_bits: int) -> "XE":
        num_shifted_bits += self.num_shifted_bits
        return self.having(num_shifted_bits=num_shifted_bits)

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0 <= value < NSB


@bleir_dataclass
class L1(BelexSymbol, CallAccessible, Validatable, BleirSerializable, Operable, Expressible, Masqued):
    register: Optional[int] = None  # Register id ϵ [0, 16)
    bank_group_row: Optional[int] = None     # Encoded bank/group/row IDs
    is_lowered: bool = False
    is_negated: bool = False
    is_literal: bool = False
    l1_reg: Optional[BLEIR.L1_REG] = None

    def __post_init__(self: "L1") -> None:
        Belex.assert_constructible()

    def as_bleir(self: "L1") -> BLEIR.L1_REG:
        if self.l1_reg is not None:
            return self.l1_reg
        return BLEIR.L1_REG(
            identifier=self.symbol,
            register=self.register,
            bank_group_row=self.bank_group_row,
            is_lowered=self.is_lowered,
            is_literal=self.is_literal)

    @property
    def reg_nym(self: "L1") -> Optional[str]:
        if self.register is not None:
            return f"L1_ADDR_REG_{self.register}"
        return None

    def operands(self: "L1") -> List["L1"]:
        return [self]

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0 <= value < (1 << 13)



@bleir_dataclass
class L2(BelexSymbol,
         CallAccessible,
         Validatable,
         BleirSerializable,
         Operable,
         Expressible,
         Masqued):

    register: Optional[int] = None  # Register id ϵ [0, 16)
    value: Optional[int] = None     # Encoded bank/group/row IDs
    is_lowered: bool = False
    is_negated: bool = False
    is_literal: bool = False
    l2_reg: Optional[BLEIR.L2_REG] = None

    def __post_init__(self: "L2") -> None:
        Belex.assert_constructible()

    def as_bleir(self: "L2") -> BLEIR.L1_REG:
        if self.l2_reg is not None:
            return self.l2_reg
        return BLEIR.L2_REG(
            identifier=self.symbol,
            register=self.register,
            value=self.value,
            is_lowered=self.is_lowered,
            is_literal=self.is_literal)

    @property
    def reg_nym(self: "L2") -> Optional[str]:
        if self.register is not None:
            return f"L2_ADDR_REG_{self.register}"
        return None

    def operands(self: "L2") -> List["L2"]:
        return [self]

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0 <= value < (1 << 7)


LX = Union[L1, L2]


@bleir_dataclass
class Mask(BelexSymbol, Validatable, BleirSerializable, Instruction):
    is_negated: bool = False
    shift_width: int = 0
    constant_value: Optional[int] = None
    register: Optional[int] = None
    is_section: bool = False
    is_lowered: bool = False
    read_write_inhibit: Optional[bool] = None
    is_literal: bool = False
    sm_reg: Optional[BLEIR.SM_REG] = None

    def __post_init__(self: "Mask") -> None:
        Belex.assert_constructible()

    def as_bleir(self: "Mask") -> Union[BLEIR.MASK, BLEIR.STATEMENT]:
        if self.sm_reg is not None:
            return self.sm_reg

        expression = BLEIR.SM_REG(
            identifier=self.symbol,
            constant_value=self.constant_value,
            register=self.register,
            is_section=self.is_section,
            is_lowered=self.is_lowered,
            is_literal=self.is_literal)

        if self.shift_width > 0:
            expression = BLEIR.SHIFTED_SM_REG(
                register=expression,
                num_bits=self.shift_width)

        mask = BLEIR.MASK(expression=expression)

        if self.read_write_inhibit is not None:
            read_write_inhibit = \
                BLEIR.ReadWriteInhibit.find_by_value(self.read_write_inhibit)
            mask = mask.having(read_write_inhibit=read_write_inhibit)

        if self.is_negated:
            return ~mask

        return mask

    @property
    def reg_nym(self: "Mask") -> Optional[str]:
        if self.register is not None:
            return f"SM_REG_{self.register}"
        return None

    @in_belex_temporary
    def __invert__(self: "Invertible") -> "Invertible":
        is_negated = (not self.is_negated)
        return self.having(is_negated=is_negated)

    @in_belex_temporary
    def __lshift__(self: "Mask", num_bits: int) -> "Mask":
        if num_bits == 0:
            return self

        if num_bits < 0:
            raise ValueError(f"num_bits must be >= 0: {num_bits}")

        shift_width = self.shift_width + num_bits

        if shift_width >= NSECTIONS:
            raise ValueError(
                f"Shift width would be out of bounds: "
                f"{self.shift_width} + {num_bits} = {shift_width}")

        return self.having(shift_width=shift_width)

    def __getitem__(self: "Mask", vrs: Sequence[VR]) -> "MultiAccess":
        if isinstance(vrs, VR):
            vrs = [vrs]
        return MultiAccess(vrs, self)

    def __setitem__(
            self: "Mask",
            vrs: Sequence[VR],
            _: Optional[Any]) -> None:
        # shim to make Python happy with statements like, "msk[a,b] |= RL()"
        pass

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0x0000 <= value <= 0xFFFF

    @staticmethod
    def get_shift(mask: int) -> int:
        for i in range(NSECTIONS):
            if mask & 0x0001 == 1:
                return i
            mask = mask >> 1
        return 0

    @classmethod
    def normalize(cls: Type["Mask"], mask: int) -> "Mask":
        shift_width = cls.get_shift(mask)
        belex = Belex.context()
        if (mask >> shift_width) == (0xFFFF >> shift_width):
            normalized_mask = belex.Mask(0xFFFF, check_availability=False)
        else:
            normalized_mask = belex.Mask(mask >> shift_width,
                                         check_availability=False)
        return normalized_mask << shift_width

    @classmethod
    def parse_literal(cls: Type["Mask"],
                      mask_literal: MaskLiteral) -> "Mask":
        mask = 0x0000
        for index in parse_sections(mask_literal):
            mask |= (0x0001 << index)
        return cls.normalize(mask)


u16 = Mask


@immutable
@dataclass
class Section(Mask):

    def __post_init__(self: "Section") -> None:
        super().__post_init__()
        self.is_section = True

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0 <= value < NSECTIONS


BleirRP = Union[BLEIR.RN_REG,
                BLEIR.RE_REG,
                BLEIR.EWE_REG,
                BLEIR.L1_REG,
                BLEIR.L2_REG,
                BLEIR.SM_REG]


BelexRP = Union[VR, RE, EWE, L1, L2, Mask]


RP_MAP: Dict[BleirRP, Callable[[BleirRP], BelexRP]] = {
    BLEIR.RN_REG:
        lambda rn_reg: VR(symbol=rn_reg.identifier,
                          is_external=True,
                          rn_reg=rn_reg),
    BLEIR.RE_REG:
        lambda re_reg: RE(symbol=re_reg.identifier,
                          is_external=True,
                          re_reg=re_reg),
    BLEIR.EWE_REG:
        lambda ewe_reg: EWE(symbol=ewe_reg.identifier,
                            is_external=True,
                            ewe_reg=ewe_reg),
    BLEIR.L1_REG:
        lambda l1_reg: L1(symbol=l1_reg.identifier,
                          is_external=True,
                          l1_reg=l1_reg),
    BLEIR.L2_REG:
        lambda l2_reg: L2(symbol=l2_reg.identifier,
                          is_external=True,
                          l2_reg=l2_reg),
    BLEIR.SM_REG:
        lambda sm_reg: Mask(symbol=sm_reg.identifier,
                            is_external=True,
                            sm_reg=sm_reg),
}


# ================ #
# RN_REG Literals: #
# ================ #

RN_REG_0 = VR(symbol="RN_REG_0", register=0, is_lowered=True, is_literal=True)
RN_REG_1 = VR(symbol="RN_REG_1", register=1, is_lowered=True, is_literal=True)
RN_REG_2 = VR(symbol="RN_REG_2", register=2, is_lowered=True, is_literal=True)
RN_REG_3 = VR(symbol="RN_REG_3", register=3, is_lowered=True, is_literal=True)
RN_REG_4 = VR(symbol="RN_REG_4", register=4, is_lowered=True, is_literal=True)
RN_REG_5 = VR(symbol="RN_REG_5", register=5, is_lowered=True, is_literal=True)
RN_REG_6 = VR(symbol="RN_REG_6", register=6, is_lowered=True, is_literal=True)
RN_REG_7 = VR(symbol="RN_REG_7", register=7, is_lowered=True, is_literal=True)
RN_REG_8 = VR(symbol="RN_REG_8", register=8, is_lowered=True, is_literal=True)
RN_REG_9 = VR(symbol="RN_REG_9", register=9, is_lowered=True, is_literal=True)
RN_REG_10 = VR(symbol="RN_REG_10", register=10, is_lowered=True, is_literal=True)
RN_REG_11 = VR(symbol="RN_REG_11", register=11, is_lowered=True, is_literal=True)
RN_REG_12 = VR(symbol="RN_REG_12", register=12, is_lowered=True, is_literal=True)
RN_REG_13 = VR(symbol="RN_REG_13", register=13, is_lowered=True, is_literal=True)
RN_REG_14 = VR(symbol="RN_REG_14", register=14, is_lowered=True, is_literal=True)
RN_REG_15 = VR(symbol="RN_REG_15", register=15, is_lowered=True, is_literal=True)

# ===================== #
# L1_ADDR_REG Literals: #
# ===================== #

L1_ADDR_REG_0 = L1(symbol="L1_ADDR_REG_0", register=0, is_lowered=True, is_literal=True)
L1_ADDR_REG_1 = L1(symbol="L1_ADDR_REG_1", register=1, is_lowered=True, is_literal=True)
L1_ADDR_REG_2 = L1(symbol="L1_ADDR_REG_2", register=2, is_lowered=True, is_literal=True)
L1_ADDR_REG_3 = L1(symbol="L1_ADDR_REG_3", register=3, is_lowered=True, is_literal=True)

# ===================== #
# L2_ADDR_REG Literals: #
# ===================== #

L2_ADDR_REG_0 = L2(symbol="L2_ADDR_REG_0", register=0, is_lowered=True, is_literal=True)

# ================ #
# SM_REG Literals: #
# ================ #

SM_REG_0 = Mask(symbol="SM_REG_0", register=0, is_lowered=True, is_literal=True)
SM_REG_1 = Mask(symbol="SM_REG_1", register=1, is_lowered=True, is_literal=True)
SM_REG_2 = Mask(symbol="SM_REG_2", register=2, is_lowered=True, is_literal=True)
SM_REG_3 = Mask(symbol="SM_REG_3", register=3, is_lowered=True, is_literal=True)
SM_REG_4 = Mask(symbol="SM_REG_4", register=4, is_lowered=True, is_literal=True)
SM_REG_5 = Mask(symbol="SM_REG_5", register=5, is_lowered=True, is_literal=True)
SM_REG_6 = Mask(symbol="SM_REG_6", register=6, is_lowered=True, is_literal=True)
SM_REG_7 = Mask(symbol="SM_REG_7", register=7, is_lowered=True, is_literal=True)
SM_REG_8 = Mask(symbol="SM_REG_8", register=8, is_lowered=True, is_literal=True)
SM_REG_9 = Mask(symbol="SM_REG_9", register=9, is_lowered=True, is_literal=True)
SM_REG_10 = Mask(symbol="SM_REG_10", register=10, is_lowered=True, is_literal=True)
SM_REG_11 = Mask(symbol="SM_REG_11", register=11, is_lowered=True, is_literal=True)
SM_REG_12 = Mask(symbol="SM_REG_12", register=12, is_lowered=True, is_literal=True)
SM_REG_13 = Mask(symbol="SM_REG_13", register=13, is_lowered=True, is_literal=True)
SM_REG_14 = Mask(symbol="SM_REG_14", register=14, is_lowered=True, is_literal=True)
SM_REG_15 = Mask(symbol="SM_REG_15", register=15, is_lowered=True, is_literal=True)

# ================= #
# EWE_REG Literals: #
# ================= #

EWE_REG_0 = EWE(symbol="EWE_REG_0", register=0, is_lowered=True, is_literal=True)
EWE_REG_1 = EWE(symbol="EWE_REG_1", register=1, is_lowered=True, is_literal=True)
EWE_REG_2 = EWE(symbol="EWE_REG_2", register=2, is_lowered=True, is_literal=True)
EWE_REG_3 = EWE(symbol="EWE_REG_3", register=3, is_lowered=True, is_literal=True)

# ================= #
# RE_REG Literals: #
# ================= #

RE_REG_0 = RE(symbol="RE_REG_0", register=0, is_lowered=True, is_literal=True)
RE_REG_1 = RE(symbol="RE_REG_1", register=1, is_lowered=True, is_literal=True)
RE_REG_2 = RE(symbol="RE_REG_2", register=2, is_lowered=True, is_literal=True)
RE_REG_3 = RE(symbol="RE_REG_3", register=3, is_lowered=True, is_literal=True)

# ============================= #
# GVML predefined RN registers: #
# ============================= #

RN_REG_G0 = VR(symbol="RN_REG_G0", register=0, row_number=0, is_lowered=True)
RN_REG_G1 = VR(symbol="RN_REG_G1", register=1, row_number=1, is_lowered=True)
RN_REG_G2 = VR(symbol="RN_REG_G2", register=2, row_number=2, is_lowered=True)
RN_REG_G3 = VR(symbol="RN_REG_G3", register=3, row_number=3, is_lowered=True)
RN_REG_G4 = VR(symbol="RN_REG_G4", register=4, row_number=4, is_lowered=True)
RN_REG_G5 = VR(symbol="RN_REG_G5", register=5, row_number=5, is_lowered=True)
RN_REG_G6 = VR(symbol="RN_REG_G6", register=6, row_number=6, is_lowered=True)
RN_REG_G7 = VR(symbol="RN_REG_G7", register=7, row_number=7, is_lowered=True)

# The symbols in the following VRs follow the convention above but they have
# not yet been defined by GVML:
# --------------------------------------------------------------------------
# RN_REG_G8 = VR(symbol="RN_REG_G8", row_number=8, is_lowered=True)
# RN_REG_G9 = VR(symbol="RN_REG_G9", row_number=9, is_lowered=True)
# RN_REG_G10 = VR(symbol="RN_REG_G10", row_number=10, is_lowered=True)
# RN_REG_G11 = VR(symbol="RN_REG_G11", row_number=11, is_lowered=True)
# RN_REG_G12 = VR(symbol="RN_REG_G12", row_number=12, is_lowered=True)
# RN_REG_G13 = VR(symbol="RN_REG_G13", row_number=13, is_lowered=True)
# RN_REG_G14 = VR(symbol="RN_REG_G14", row_number=14, is_lowered=True)

# Row number 15 is treated specially by GVML but there does not seem to be an
# RN_REG associated with it:
# ---------------------------------------------------------------------------
# RN_REG_M4_IDX = VR(symbol="VR16_M4_IDX", row_number=15, is_lowered=True)

RN_REG_T0 = VR(symbol="RN_REG_T0", register=8, row_number=16, is_lowered=True)
RN_REG_T1 = VR(symbol="RN_REG_T1", register=9, row_number=17, is_lowered=True)
RN_REG_T2 = VR(symbol="RN_REG_T2", register=10, row_number=18, is_lowered=True)
RN_REG_T3 = VR(symbol="RN_REG_T3", register=11, row_number=19, is_lowered=True)
RN_REG_T4 = VR(symbol="RN_REG_T4", register=12, row_number=20, is_lowered=True)
RN_REG_T5 = VR(symbol="RN_REG_T5", register=13, row_number=21, is_lowered=True)
RN_REG_T6 = VR(symbol="RN_REG_T6", register=14, row_number=22, is_lowered=True)

RN_REG_FLAGS = VR(symbol="RN_REG_FLAGS", register=15, row_number=23, is_lowered=True)


# ============================= #
# GVML predefined SM registers: #
# ============================= #

SM_0XFFFF = Mask(symbol="SM_0XFFFF", register=4, constant_value=0xFFFF, is_lowered=True)
SM_0X0001 = Mask(symbol="SM_0X0001", register=5, constant_value=0x0001, is_lowered=True)
SM_0X1111 = Mask(symbol="SM_0X1111", register=6, constant_value=0x1111, is_lowered=True)
SM_0X0101 = Mask(symbol="SM_0X0101", register=7, constant_value=0x0101, is_lowered=True)
SM_0X000F = Mask(symbol="SM_0X000F", register=8, constant_value=0x000F, is_lowered=True)
SM_0X0F0F = Mask(symbol="SM_0X0F0F", register=9, constant_value=0x0F0F, is_lowered=True)
SM_0X0707 = Mask(symbol="SM_0X0707", register=10, constant_value=0x0707, is_lowered=True)
SM_0X5555 = Mask(symbol="SM_0X5555", register=11, constant_value=0x5555, is_lowered=True)
SM_0X3333 = Mask(symbol="SM_0X3333", register=12, constant_value=0x3333, is_lowered=True)
SM_0X00FF = Mask(symbol="SM_0X00FF", register=13, constant_value=0x00FF, is_lowered=True)
SM_0X001F = Mask(symbol="SM_0X001F", register=14, constant_value=0x001F, is_lowered=True)
SM_0X003F = Mask(symbol="SM_0X003F", register=15, constant_value=0x003F, is_lowered=True)

# ============================================== #
# l1_addr registers used for function parameters #
# ============================================== #
L1_ADDR_REG0 = L1_ADDR_REG_0
L1_ADDR_REG1 = L1_ADDR_REG_1

# =========================================== #
# l1_addr registers holding predefined values #
# =========================================== #
L1_ADDR_REG_RESERVED = L1_ADDR_REG_2
L1_ADDR_REG_INDEX = L1_ADDR_REG_3

L2_ADDR_REG0 = L2_ADDR_REG_0

RE_REG_G0 = RE_REG_0
RE_REG_G1 = RE_REG_1
RE_REG_T0 = RE_REG_2
RE_REG_NO_RE = RE_REG_3

EWE_REG_G0 = EWE_REG_0
EWE_REG_G1 = EWE_REG_1
EWE_REG_T0 = EWE_REG_2
EWE_REG_NO_EWE = EWE_REG_3

# general purpose smaps registers #
SM_REG0 = SM_REG_0
SM_REG1 = SM_REG_1
SM_REG2 = SM_REG_2
SM_REG3 = SM_REG_3

VR16_T0 = 16
VR16_T1 = 17
VR16_T2 = 18
VR16_T3 = 19
VR16_T4 = 20
VR16_T5 = 21
VR16_T6 = 22
VR16_FLAGS = 23

# pre-defined flags for the flags vector-register
C_FLAG = 0
B_FLAG = 1
OF_FLAG = 2
PE_FLAG = 3

# Markers for general purpose usage Callee must preserve
GP0_MRK = 4
GP1_MRK = 5
GP2_MRK = 6
GP3_MRK = 7
GP4_MRK = 8
GP5_MRK = 9
GP6_MRK = 10
GP7_MRK = 11

# Markers for general purpose usage Callee may overwrite preserve
TMP0_MRK = 12
TMP1_MRK = 13
TMP2_MRK = 14
TMP3_MRK = 15

# General purpose vector-markers
FIRST_GP0_MRK = GP0_MRK
LAST_GP7_MRK = GP7_MRK

# Temporary vector-markers
FIRST_TMP0_MRK = TMP0_MRK
LAST_TMP3_MRK = TMP3_MRK

GVML_SM_REGS_BY_VAL: Dict[int, Mask] = {
    0xFFFF: SM_0XFFFF,
    0x0001: SM_0X0001,
    0x1111: SM_0X1111,
    0x0101: SM_0X0101,
    0x000F: SM_0X000F,
    0x0F0F: SM_0X0F0F,
    0x0707: SM_0X0707,
    0x5555: SM_0X5555,
    0x3333: SM_0X3333,
    0x00FF: SM_0X00FF,
    0x001F: SM_0X001F,
    0x003F: SM_0X003F,
}


@bleir_dataclass
class BelexLiteral(BelexSymbol, Accessible, Operable):
    bleir: Any = None  # None for parameter compatibility with BelexSymbol

    def as_bleir(self: "BelexLiteral") -> BLEIR.SRC_EXPR:
        return self.bleir

    def operands(self: "BelexLiteral") -> List["BelexLiteral"]:
        return [self]


VarType = Union[VR, BelexLiteral, L1, L2, RE, EWE]


Mask_or_Literal = Union[Mask, MaskLiteral]


@bleir_dataclass
class BelexAccess(CallAccessible, Operable, Expressible, Masqued):
    var: VarType
    mask: Optional[Mask_or_Literal] = None
    is_negated: bool = False

    def __post_init__(self: "BelexAccess") -> None:
        if self.mask is not None and not isinstance(self.mask, Mask):
            mask_literal = self.mask
            self.mask = Mask.parse_literal(mask_literal)

    def operands(self: "BelexAccess") -> List[BelexSymbol]:
        operands = self.var.operands()
        return operands

    def __call__(self: "BelexAccess") -> "BelexAccess":
        return self

    def __ior__(self: "BelexAccess", other: Operand) -> None:
        if isinstance(self.var, BelexLiteral):
            return super().__ior__(other)

        if isinstance(self.var, VR):
            assignment = AssignOperation(operator=ASSIGN_OP.COND_EQ,
                                         lvalue=self,
                                         rvalue=other)

        else:
            raise ValueError(
                f"Unsupported var type ({self.var.__class__.__name__}): {var}")

        Belex.add_instruction(assignment)

    def __add__(self: "BelexAccess",
                offsets: Union[BLEIR.Offsets, Integer]) -> Union["OffsetLX", "BelexAccess"]:

        if not isinstance(self.var, LX.__args__):
            raise AssertionError(
                f"Adding offsets is only supported for L1 and L2 register "
                f"types, not {self.var.__class__.__name__}: {self.var}")

        if isinstance(offsets, Integer.__args__):
            row_id = offsets
            offsets = (row_id,)

        OffsetLX.validate(offsets)
        return OffsetLX(self, *reversed(offsets))

    @in_belex_temporary
    def __invert__(self: "BelexAccess") -> Expressible:
        if isinstance(self.var, (RE, EWE)):  # special case
            return XE(accessor=self, is_negated=True)
        return super().__invert__()

    @in_belex_temporary
    def __lshift__(self: "BelexAccess", num_shifted_bits: int) -> XE:
        if isinstance(self.var, (RE, EWE)):
            return XE(accessor=self,
                      num_shifted_bits=num_shifted_bits)
        raise RuntimeError(
            f"Left-shift operations on variable accesses are only supported "
            f"for vars of type RE or EWE, not {self.var.__class__.__name__}: "
            f"{self.var}")


@bleir_dataclass
class MultiAccess(CallAccessible, Operable, Expressible, Masqued):
    vrs: Sequence[VR]
    mask: Optional[Mask_or_Literal] = None
    is_negated: bool = False

    def __post_init__(self: "MultiAccess") -> None:
        BelexAccess.__post_init__(self)

    def operands(self: "MultiAccess") -> List[BelexSymbol]:
        operands = chain.from_iterable(vr.operands() for vr in self.vrs)
        return list(operands)

    def __ior__(self: "BelexAccess", other: Operand) -> None:
        assignment = AssignOperation(operator=ASSIGN_OP.COND_EQ,
                                     lvalue=self,
                                     rvalue=other)

        Belex.add_instruction(assignment)


@bleir_dataclass
class OffsetLX(CallAccessible, Validatable, BleirSerializable, Operable, Expressible, Masqued):
    lhs: BelexAccess
    row_id: int
    group_id: int = 0
    bank_id: int = 0
    is_negated: bool = False

    def __call__(self: "OffsetLX") -> "OffsetLX":
        return self

    @property
    def var(self: "OffsetLX") -> "OffsetLX":
        return self

    @property
    def mask(self: "OffsetLX") -> Optional[Mask]:
        return self.lhs.mask

    def as_bleir(self: "OffsetLX") -> BLEIR.LXRegWithOffsets:
        return BLEIR.LXRegWithOffsets(
            parameter=self.lhs.var.as_bleir(),
            row_id=self.row_id,
            group_id=self.group_id,
            bank_id=self.bank_id)

    def operands(self: "OffsetLX") -> List[LX]:
        return self.lhs.operands()

    @staticmethod
    def is_valid(value: Any) -> bool:
        return isinstance(value, Integer.__args__) and 0 <= value < (1 << 9) \
            or BLEIR.is_sequence(value) and (
                len(value) == 1
                and isinstance(value[0], Integer.__args__) and 0 <= value[0] < (1 << 9)  # row_id
                or
                len(value) == 2
                and isinstance(value[1], Integer.__args__) and 0 <= value[1] < (1 << 9)  # row_id
                and isinstance(value[0], Integer.__args__) and 0 <= value[0] < (1 << 2)  # group_id
                or
                len(value) == 3
                and isinstance(value[2], Integer.__args__) and 0 <= value[2] < (1 << 9)  # row_id
                and isinstance(value[1], Integer.__args__) and 0 <= value[1] < (1 << 2)  # group_id
                and isinstance(value[0], Integer.__args__) and 0 <= value[0] < (1 << 2)  # bank_id
            )

    def __add__(self: "OffsetLX",
                offsets: Union[BLEIR.Offsets, Integer]) -> "OffsetLX":

        if self.is_negated:
            raise RuntimeError(
                f"Cannot add offsets {offsets} to negated LX parameter: {self}")

        OffsetLX.validate(offsets)

        if isinstance(offsets, Integer.__args__):
            row_id = offsets
            offsets = (row_id,)

        row_id = self.row_id
        group_id = self.group_id
        bank_id = self.bank_id

        row_id += offsets[-1]
        if len(offsets) > 1:
            group_id += offsets[-2]
        if len(offsets) > 2:
            bank_id += offsets[-3]

        offsets = (bank_id, group_id, row_id)
        OffsetLX.validate(offsets)

        return OffsetLX(lhs=self.lhs,
                        row_id=row_id,
                        group_id=group_id,
                        bank_id=bank_id)


@bleir_dataclass
class BinaryOperation(Operable):
    operator: BINOP
    lhs: Union[BelexAccess, LX, OffsetLX, "BinaryOperation"]
    rhs: Union[BelexAccess, LX, OffsetLX, "BinaryOperation"]

    def masquerade(self: "BinaryOperation") -> List[BelexAccess]:
        masquerade = []

        for operand in [self.lhs, self.rhs]:
            if hasattr(operand, "mask"):
                masquerade.append(operand)
            else:
                masquerade += operand.masquerade()

        return masquerade

    def operands(self: "BinaryOperation") -> List[Union[VR, BelexLiteral]]:
        return self.lhs.operands() + self.rhs.operands()

    def __and__(self: "BinaryOperation", other: BelexAccess) -> "BinaryOperation":
        return BinaryOperation(
            operator=BINOP.AND,
            lhs=self,
            rhs=other)

    def __or__(self: "BinaryOperation", other: BelexAccess) -> "BinaryOperation":
        return BinaryOperation(
            operator=BINOP.OR,
            lhs=self,
            rhs=other)

    def __xor__(self: "BinaryOperation", other: BelexAccess) -> "BinaryOperation":
        return BinaryOperation(
            operator=BINOP.XOR,
            lhs=self,
            rhs=other)

    def __invert__(self: "BinaryOperation") -> "BinaryOperation":
        if self.operator is BINOP.AND:
            # If we have A & B, by DeMorgan's Law,
            #    ~(A & B) == ~A | ~B

            return BinaryOperation(
                operator=BINOP.OR,
                lhs=~self.lhs,
                rhs=~self.rhs)

        if self.operator is BINOP.OR:
            # If we have A | B, by DeMorgan's Law,
            #    ~(A | B) == ~A & ~B

            return BinaryOperation(
                operator=BINOP.AND,
                lhs=~self.lhs,
                rhs=~self.rhs)

        if self.operator is BINOP.XOR:
            # If we have A ^ B, it is equivalent to:
            #     A ^ B == (A & ~B) | (~A & B)
            # Therefore, by DeMorgan's Law,
            #     ~(A ^ B) == ~((A & ~B) | (~A & B))
            #              == ~(A & ~B) & ~(~A & B)
            #              == (~A | B) & (A | ~B)

            inv_a_or_b = BinaryOperation(
                operator=BINOP.OR,
                lhs=~self.lhs,
                rhs=self.rhs)

            a_or_inv_b = BinaryOperation(
                operator=BINOP.OR,
                lhs=self.lhs,
                rhs=~self.rhs)

            return BinaryOperation(
                operator=BINOP.AND,
                lhs=inv_a_or_b,
                rhs=a_or_inv_b)

        raise RuntimeError(
            f"Unable to determine how to negate operator: {self.operator}")


GRAMMAR: Dict[str, Callable] = {

    # WRITE LOGIC
    "SB = <SRC>": lambda msk, lhss, rhss: BLECCI.sb_from_src(msk, lhss, *rhss),

    # FIXME: Add these to the README.md
    "SB = ~<SRC>": lambda msk, lhss, rhss: BLECCI.sb_from_inv_src(msk, lhss, *rhss),

    # FIXME: Add these to the README.md
    "SB ?= <SRC>": lambda msk, lhss, rhss: BLECCI.sb_cond_equals_src(msk, lhss, *rhss),

    # FIXME: Add these to the README.md
    "SB ?= ~<SRC>": lambda msk, lhss, rhss: BLECCI.sb_cond_equals_inv_src(msk, lhss, *rhss),

    # READ LOGIG #1 and #2
    "RL = <BIT>": lambda msk, lhs, rhss: BLECCI.set_rl(msk, *rhss),

    # READ LOGIC #3, #4, #5
    "RL = <SB>": lambda msk, lhs, rhss: BLECCI.rl_from_sb(msk, rhss),

    "RL = <SRC>": lambda msk, lhs, rhss: BLECCI.rl_from_src(msk, *rhss),

    "RL = (<SB> & <SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_from_sb_and_src(msk, rhss[:-1], rhss[-1]),

    # BUGFIX: Issue #68: The former yield, "error: Unsupported logic"
    # "RL = ~<SB>": lambda msk, lhs, rhss: BLECCI.rl_from_inv_sb(msk, rhss),
    "RL = ~<SB>": lambda msk, lhs, rhss: [
        BLECCI.set_rl(msk, 1),
        BLECCI.rl_and_equals_inv_sb(msk, rhss),
    ],

    # FIXME: Add these to the README.md
    "RL = ~<SRC>": lambda msk, lhs, rhss: BLECCI.rl_from_inv_src(msk, *rhss),

    # READ LOGIC #10, #11, #12
    "RL |= <SB>": lambda msk, lhs, rhss: BLECCI.rl_or_equals_sb(msk, rhss),

    "RL |= <SRC>": lambda msk, lhs, rhss: BLECCI.rl_or_equals_src(msk, *rhss),
    "RL |= ~<SRC>": lambda msk, lhs, rhss: BLECCI.rl_or_equals_inv_src(msk, *rhss),

    "RL |= (<SB> & <SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_or_equals_sb_and_src(msk, rhss[:-1], rhss[-1]),
    "RL |= (<SB> & ~<SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_or_equals_sb_and_inv_src(msk, rhss[:-1], rhss[-1]),

    # READ LOGIC #13, #14, #15
    "RL &= <SB>": lambda msk, lhs, rhss: BLECCI.rl_and_equals_sb(msk, rhss),

    "RL &= <SRC>": lambda msk, lhs, rhss: BLECCI.rl_and_equals_src(msk, *rhss),

    "RL &= (<SB> & <SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_and_equals_sb_and_src(msk, rhss[:-1], rhss[-1]),
    "RL &= (<SB> & ~<SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_and_equals_sb_and_inv_src(msk, rhss[:-1], rhss[-1]),

    # READ LOGIC #18, #19, #20
    "RL ^= <SB>": lambda msk, lhs, rhss: BLECCI.rl_xor_equals_sb(msk, rhss),

    "RL ^= <SRC>": lambda msk, lhs, rhss: BLECCI.rl_xor_equals_src(msk, *rhss),
    "RL ^= ~<SRC>": lambda msk, lhs, rhss: BLECCI.rl_xor_equals_inv_src(msk, *rhss),

    "RL ^= (<SB> & <SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_xor_equals_sb_and_src(msk, rhss[:-1], rhss[-1]),
    "RL ^= (<SB> & ~<SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_xor_equals_sb_and_inv_src(msk, rhss[:-1], rhss[-1]),

    # READ LOGIC #6, #7
    "RL = (<SB> | <SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_from_sb_or_src(msk, rhss[:-1], rhss[-1]),
    "RL = (<SB> | ~<SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_from_sb_or_inv_src(msk, rhss[:-1], rhss[-1]),

    "RL = (<SB> ^ <SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_from_sb_xor_src(msk, rhss[:-1], rhss[-1]),

    # READ LOGIC #8, #9
    "RL = (~<SB> & <SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_from_inv_sb_and_src(msk, rhss[:-1], rhss[-1]),

    "RL = (<SB> & ~<SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_from_sb_and_inv_src(msk, rhss[:-1], rhss[-1]),

    "RL = (<SB> ^ ~<SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_from_sb_xor_inv_src(msk, rhss[:-1], rhss[-1]),

    # READ LOGIC #16, #17
    "RL &= ~<SB>": \
        lambda msk, lhs, rhss: BLECCI.rl_and_equals_inv_sb(msk, rhss),

    "RL &= ~<SRC>": lambda msk, lhs, rhss: BLECCI.rl_and_equals_inv_src(msk, *rhss),

    "RL = (~<SB> & ~<SRC>)": \
        lambda msk, lhs, rhss: BLECCI.rl_from_inv_sb_and_inv_src(msk, rhss[:-1], rhss[-1]),

    # R-SEL LOGIC
    "GL = RL": lambda msk, lhs, rhss: BLECCI.gl_from_rl(msk),
    "RSP16 = RL": lambda msk, lhs, rhss: BLECCI.rsp16_from_rl(msk),

    "GGL = RL": lambda msk, lhs, rhss: BLECCI.ggl_from_rl(msk),
    "GGL = (RL & <LX>)": lambda msk, lhs, rhss: BLECCI.ggl_from_rl_and_lx(msk, rhss[-1]),
    "GGL = <LX>": lambda msk, lhs, rhss: BLECCI.ggl_from_lx(rhss[0]),

    # SPECIAL ASSIGNMENT
    "RSP16 = RSP256": lambda msk, lhs, rhss: BLECCI.rsp16_from_rsp256(),

    # SPECIAL ASSIGNMENT
    "RSP256 = RSP16": lambda msk, lhs, rhss:  BLECCI.rsp256_from_rsp16(),
    "RSP256 = RSP2K": lambda msk, lhs, rhss: BLECCI.rsp256_from_rsp2k(),

    # SPECIAL ASSIGNMENT
    "RSP2K = RSP256": lambda msk, lhs, rhss: BLECCI.rsp2k_from_rsp256(),
    "RSP2K = RSP32K": lambda msk, lhs, rhss: BLECCI.rsp2k_from_rsp32k(),

    # SPECIAL ASSIGNMENT
    "RSP32K = RSP2K": lambda msk, lhs, rhss: BLECCI.rsp32k_from_rsp2k(),

    "LGL = <LX>": lambda msk, lhs, rhss: BLECCI.lgl_from_lx(rhss[0]),

    "LX = GGL": lambda msk, lhs, rhss: BLECCI.lx_from_ggl(lhs),

    "LX = LGL": lambda msk, lhs, rhss: BLECCI.lx_from_lgl(lhs),
}


LEFT_MORPHOLOGY = {
    VR: lambda vr: "SB",
    RE: lambda re: "SB",
    EWE: lambda re: "SB",
    L1: lambda l1: "LX",
    L2: lambda l1: "LX",
    OffsetLX: lambda l1: "LX",
    BelexLiteral: lambda literal: literal.symbol,
}


RIGHT_MORPHOLOGY: Dict[Union[Type, str], Union[Callable, str]] = {
    BinaryOperation: lambda binop: \
        [
            f"({left_morphology} {binop.operator} {right_morphology})"
            for left_morphology, right_morphology in
            product(morphology(binop.lhs, RIGHT_MORPHOLOGY),
                    morphology(binop.rhs, RIGHT_MORPHOLOGY))
        ],

    BelexAccess: lambda access:
        morphology(access.var, RIGHT_MORPHOLOGY)
        if not access.is_negated
        else [f"~{morph}"
              for morph in morphology(access.var, RIGHT_MORPHOLOGY)],

    BelexLiteral: lambda literal: morphology(literal.symbol, RIGHT_MORPHOLOGY),

    VR: "<SB>",
    RE: "<SB>",
    EWE: "<SB>",
    XE: "<SB>",
    L1: "<LX>",
    L2: "<LX>",
    OffsetLX: "<LX>",

    int: "<BIT>",
    1: "<BIT>",
    0: "<BIT>",

    "RL": ["<SRC>", "RL"],
    "NRL": ["<SRC>", "NRL"],
    "ERL": ["<SRC>", "ERL"],
    "WRL": ["<SRC>", "WRL"],
    "SRL": ["<SRC>", "SRL"],

    "GL": ["<SRC>", "GL"],
    "GGL": ["<SRC>", "GGL"],
    "RSP16": ["<SRC>", "RSP16"],

    "LGL": "LGL",

    "RSP256": "RSP256",
    "RSP2K": "RSP2K",
    "RSP32K": "RSP32K",

    "INV_RL": ["<SRC>", "INV_RL"],
    "INV_NRL": ["<SRC>", "INV_NRL"],
    "INV_ERL": ["<SRC>", "INV_ERL"],
    "INV_WRL": ["<SRC>", "INV_WRL"],
    "INV_SRL": ["<SRC>", "INV_SRL"],

    "INV_GL": ["<SRC>", "INV_GL"],
    "INV_GGL": ["<SRC>", "INV_GGL"],
    "INV_RSP16": ["<SRC>", "INV_RSP16"],
}


AS_BLEIR = {
    # BelexAccess: lambda access: AS_BLEIR[access.var.__class__](access.operands()[0]),
    BelexAccess: lambda access: AS_BLEIR[access.var.__class__](access.var),
    VR: lambda vr: [vr.as_bleir()],
    RE: lambda re: [re.as_bleir()],
    EWE: lambda ewe: [ewe.as_bleir()],
    XE: lambda xe: [xe.as_bleir()],
    L1: lambda l1: [l1.as_bleir()],
    L2: lambda l2: [l2.as_bleir()],
    OffsetLX: lambda lx: [lx.as_bleir()],
    BelexLiteral: lambda literal: [literal.as_bleir()],
    BinaryOperation: lambda binop: \
        [operand.as_bleir() for operand in binop.operands()],
    int: lambda bit: [BLEIR.BIT_EXPR.find_by_value(bit)],
}


def morphology(value, mapping):
    if isinstance(value, (str, *Integer.__args__)):
        morph = mapping[value]
    else:
        morph = mapping[value.__class__]

    if callable(morph):
        morph = morph(value)

    if isinstance(morph, str):
        morph = [morph]

    return morph


@in_belex_temporary
def simplify(expression):
    if not isinstance(expression, BinaryOperation):
        return expression

    op = expression.operator
    lhs = simplify(expression.lhs)
    rhs = simplify(expression.rhs)

    if not isinstance(lhs, (BelexAccess, XE)) \
       or not isinstance(rhs, (BelexAccess, XE)) \
       or isinstance(lhs, BelexAccess) and not isinstance(lhs.var, (VR, RE)) \
       or isinstance(rhs, BelexAccess) and not isinstance(rhs.var, (VR, RE)):
        return expression

    belex = Belex.context()
    symbol_table = belex.symbol_table

    if isinstance(lhs, XE) and lhs.num_shifted_bits > 0 \
       and (not isinstance(rhs, XE)
            or lhs.num_shifted_bits != rhs.num_shifted_bits):
        return expression

    if op is BINOP.AND and not lhs.is_negated and not rhs.is_negated:
        expression = symbol_table.make_re_register(
            rows=(lhs.operands() + rhs.operands()))
        expression = BelexAccess(var=expression)

        if isinstance(lhs, XE) and lhs.num_shifted_bits > 0:
            expression = XE(accessor=expression,
                            num_shifted_bits=lhs.num_shifted_bits)

    elif op is BINOP.OR and lhs.is_negated and rhs.is_negated:
        expression = symbol_table.make_re_register(
            rows=(lhs.operands() + rhs.operands()))
        expression = BelexAccess(var=expression)
        expression = XE(accessor=expression,
                        is_negated=True)

        if isinstance(lhs, XE) and lhs.num_shifted_bits > 0:
            expression = expression.having(
                num_shifted_bits=lhs.num_shifted_bits)

    return expression


def is_rn_reg(value: Any) -> bool:
    return isinstance(value, BLEIR.RN_REG)


@bleir_dataclass
class AssignOperation(Instruction, BleirSerializable):
    __eq_excludes__ = ["file_path", "line_number"]

    operator: ASSIGN_OP
    lvalue: Union[BelexAccess, OffsetLX]
    rvalue: Operand

    file_path: Optional[str] = None
    line_number: int = -1

    def morph_left(self: "AssignOperation") -> Sequence[str]:
        operands = self.lvalue.operands()
        if len(operands) == 0:
            raise RuntimeError(f"At least one lvalue is required")

        morphologies = list(chain.from_iterable(
            morphology(operand, LEFT_MORPHOLOGY)
            for operand in operands))

        if not all(morphologies[i] == morphologies[0]
                   for i in range(1, len(morphologies))):
            raise RuntimeError(
                f"All morphologies must be the same: {morphologies}")

        return [morphologies[0]]

    def morph_right(self: "AssignOperation") -> Sequence[str]:
        rvalue = self.rvalue
        if isinstance(rvalue, BinaryOperation):
            lhs = rvalue.lhs
            # FIXME: Make this more general purpose (its purpose is to collapse
            # multiple <SB>s)
            while isinstance(lhs, BinaryOperation) \
                  and isinstance(lhs.rhs, BelexAccess) \
                  and isinstance(lhs.rhs.var, VR):
                lhs = lhs.rhs
            rvalue = rvalue.having(lhs=lhs)
        morphologies = morphology(rvalue, RIGHT_MORPHOLOGY)
        if isinstance(morphologies, str):
            morphologies = [morphologies]
        return morphologies

    def as_patterns(self: "AssignOperation") -> Sequence[str]:
        left_patterns = self.morph_left()
        right_patterns = self.morph_right()
        return [
            f"{left_pattern} {self.operator} {right_pattern}"
            for left_pattern, right_pattern in
            product(left_patterns, right_patterns)
        ]

    def assert_mask_consistency(self: "AssignOperation") -> None:
        if isinstance(self.rvalue, Integer.__args__):
            return

        mask = self.lvalue.mask
        for masqued in self.rvalue.masquerade():
            if masqued.mask is None or masqued.mask == mask:
                continue
            raise AssertionError(
                f"Expected assignment operand to be the same as the "
                f"recipient: {masqued.mask} != {mask}")

    @in_belex_temporary
    def as_bleir(self: "AssignOperation") -> List[BLEIR.STATEMENT]:
        simplification = self.having(rvalue=simplify(self.rvalue))
        patterns = simplification.as_patterns()

        as_bleir = None
        for pattern in patterns:
            if pattern in GRAMMAR:
                if as_bleir is not None:
                    raise AssertionError(
                        f"Grammar redundancy detected: {patterns}")
                as_bleir = GRAMMAR[pattern]

        if as_bleir is None:
            raise AssertionError(f"Unsupported assignment patterns: {patterns}")

        msk = None
        if simplification.lvalue.mask is not None:
            msk = simplification.lvalue.mask
            if not isinstance(simplification.lvalue, XE) and simplification.lvalue.is_negated:
                msk = ~msk
            msk = msk.as_bleir()

        if isinstance(simplification.lvalue, (BelexAccess, OffsetLX, XE)):
            lhs = simplification.lvalue.var.as_bleir()
        elif isinstance(simplification.lvalue, MultiAccess):
            lhs = [vr.as_bleir() for vr in simplification.lvalue.vrs]
        else:
            raise RuntimeError(
                f"Unsupported lvalue type ({simplification.lvalue.__class__.__name__}): "
                f"{simplification.lvalue}")

        simplification.assert_mask_consistency()
        if simplification.rvalue.__class__ not in AS_BLEIR:
            raise AssertionError(
                f"Unsupported rvalue type ({simplification.rvalue.__class__.__name__}): "
                f"{simplification.rvalue}")

        rhss = AS_BLEIR[simplification.rvalue.__class__](simplification.rvalue)
        if len(rhss) == 1 and isinstance(rhss[0], BLEIR.RE_REG) \
           and rhss[0].rows is not None \
           and len(rhss[0].rows) <= 3 and all(map(is_rn_reg, rhss[0].rows)):
            rhss = rhss[0].rows

        bleir = as_bleir(msk, lhs, rhss)

        # Return a list so we can expand some patterns to multiple statements,
        # e.g. "GL = <SB>" would become ["RL = <SB>", "GL = RL"],
        if not isinstance(bleir, list):
            bleir = [bleir]

        metadata = {
            BLEIR.StatementMetadata.FILE_PATH: self.file_path,
            BLEIR.StatementMetadata.LINE_NUMBER: self.line_number,
        }

        return [stmt.having(metadata=metadata) for stmt in bleir]


def reserve_special(symbol: str) -> BelexSymbol:
    if symbol in RESERVED_IDENTIFIERS:
        raise ValueError(f"Already declared BelexSymbol: {symbol}")

    RESERVED_IDENTIFIERS.add(symbol)
    return BelexSpecial(symbol=symbol)


def reserve_literal(symbol: str, bleir: Any) -> BelexLiteral:
    if symbol in RESERVED_IDENTIFIERS:
        raise ValueError(f"Already declared BelexSymbol: {symbol}")

    RESERVED_IDENTIFIERS.add(symbol)
    return BelexLiteral(symbol=symbol, bleir=bleir)


RL: BelexLiteral = reserve_literal("RL", BLEIR.SRC_EXPR.RL)
NRL: BelexLiteral = reserve_literal("NRL", BLEIR.SRC_EXPR.NRL)
ERL: BelexLiteral = reserve_literal("ERL", BLEIR.SRC_EXPR.ERL)
WRL: BelexLiteral = reserve_literal("WRL", BLEIR.SRC_EXPR.WRL)
SRL: BelexLiteral = reserve_literal("SRL", BLEIR.SRC_EXPR.SRL)

GL: BelexLiteral = reserve_literal("GL", BLEIR.SRC_EXPR.GL)
GGL: BelexLiteral = reserve_literal("GGL", BLEIR.SRC_EXPR.GGL)
RSP16: BelexLiteral = reserve_literal("RSP16", BLEIR.SRC_EXPR.RSP16)
RSP256: BelexLiteral = reserve_literal("RSP256", BLEIR.RSP256_EXPR.RSP256)
RSP2K: BelexLiteral = reserve_literal("RSP2K", BLEIR.RSP2K_EXPR.RSP2K)
RSP32K: BelexLiteral = reserve_literal("RSP32K", BLEIR.RSP32K_EXPR.RSP32K)

LGL: BelexLiteral = reserve_literal("LGL", BLEIR.LGL_EXPR.LGL)

INV_RL: BelexLiteral = reserve_literal("INV_RL", BLEIR.SRC_EXPR.INV_RL)
INV_NRL: BelexLiteral = reserve_literal("INV_NRL", BLEIR.SRC_EXPR.INV_NRL)
INV_ERL: BelexLiteral = reserve_literal("INV_ERL", BLEIR.SRC_EXPR.INV_ERL)
INV_WRL: BelexLiteral = reserve_literal("INV_WRL", BLEIR.SRC_EXPR.INV_WRL)
INV_SRL: BelexLiteral = reserve_literal("INV_SRL", BLEIR.SRC_EXPR.INV_SRL)

INV_GL: BelexLiteral = reserve_literal("INV_GL", BLEIR.SRC_EXPR.INV_GL)
INV_GGL: BelexLiteral = reserve_literal("INV_GGL", BLEIR.SRC_EXPR.INV_GGL)
INV_RSP16: BelexLiteral = reserve_literal("INV_RSP16", BLEIR.SRC_EXPR.INV_RSP16)

NOOP: BelexSpecial = reserve_special("NOOP")
FSEL_NOOP: BelexSpecial = reserve_special("FSEL_NOOP")
RSP_END: BelexSpecial = reserve_special("RSP_END")
RSP_START_RET: BelexSpecial = reserve_special("RSP_START_RET")
L2_END: BelexSpecial = reserve_special("L2_END")


APPENDABLE_COMMANDS: str = "__APPENDABLE_COMMANDS__"

CONTEXTLIB_FILE_PATH = inspect.getfile(contextmanager)


@contextmanager
def apl_commands(comment: Optional[str] = None) -> Belex:
    if not Belex.has_context():
        raise RuntimeError(
            f"This must be executed within a @belex_apl context")

    for frame_info in inspect.stack():
        if frame_info.filename not in [__file__, CONTEXTLIB_FILE_PATH]:
            file_path, line_number = frame_info.filename, frame_info.lineno
            break

    belex = Belex.push_context()

    try:
        yield belex
    finally:
        popped = Belex.pop_context()
        assert belex is popped
        lane = belex.as_lane(comment=comment,
                             file_path=file_path,
                             line_number=line_number)

        if StackManager.has_elem(APPENDABLE_COMMANDS):
            prev = StackManager.pop(APPENDABLE_COMMANDS)
            prev.instructions.extend(lane.instructions)
            lane = prev

        if len(lane.instructions) > 0:
            Belex.add_instruction(lane)


@contextmanager
def append_commands() -> Belex:
    belex = Belex.context()
    belex.is_appendable = True

    try:
        yield belex
    finally:
        belex.is_appendable = False
        if StackManager.has_elem(APPENDABLE_COMMANDS):
            lane = StackManager.pop(APPENDABLE_COMMANDS)
            if len(lane.instructions) > 0:
                belex.add_instruction(lane)


@contextmanager
def appendable_commands(comment: Optional[str] = None) -> Belex:
    if not Belex.has_context():
        raise RuntimeError(
            f"This must be executed within a @belex_apl context")

    belex = Belex.push_context()

    try:
        yield belex
    finally:
        popped = Belex.pop_context()
        assert belex is popped
        if len(belex.instructions) > 0:
            lane = belex.as_lane(comment=comment)
            if belex.is_appendable:
                StackManager.push(APPENDABLE_COMMANDS, lane)
            else:
                Belex.add_instruction(lane)


@contextmanager
def inline_bleir() -> Sequence[BLEIR.Operation]:
    belex = Belex.push_context(is_bleir=True)
    try:
        yield belex
    finally:
        popped = Belex.pop_context()
        assert belex is popped


FormalParameter = Union[VR, RE, EWE, L1, L2, Mask, Section]
ActualParameter = BLEIR.ActualParameter


def build_or_call_fragment_caller(
        fn: Callable,
        formal_parameters: Sequence[FormalParameter],
        debug_parameters: Optional[Sequence[FormalParameter]] = None,
        is_inline: bool = False,
        values_by_parameter_id: Optional[Dict[str, Union[int, str]]] = None,
        data_by_parameter_id: Optional[Dict[str, Union[int, str, np.ndarray]]] = None,
        debug: bool = False,
        captured_glass: Optional[MutableSequence[str]] = None,
        fragment_file_path: Optional[str] = None,
        fragment_line_number: Optional[int] = None,
        fragment_num_lines: Optional[int] = None) \
        -> Optional[BLEIR.FragmentCaller]:

    if debug_parameters is not None:
        belex_parameters = debug_parameters
    else:
        belex_parameters = formal_parameters
        debug = False

    if is_inline:
        belex = Belex.context()
    else:
        belex = Belex.push_context(
            frag_nym=fn.__name__,
            debug_parameters=belex_parameters,
            values_by_parameter_id=values_by_parameter_id,
            data_by_parameter_id=data_by_parameter_id,
            debug=debug, captured_glass=captured_glass)

    try:
        for belex_parameter in belex_parameters:
            if hasattr(belex_parameter, "is_lowered") \
               and not belex_parameter.is_lowered:
                belex.add_symbol(belex_parameter)
            elif isinstance(belex_parameter, OffsetLX) \
                 and not belex_parameter.lhs.var.is_lowered:
                belex.add_symbol(belex_parameter.lhs.var)

        _in_fragment = belex.in_fragment
        belex.in_fragment = True
        # NOTE: Break on the following line to step into the fragment
        fn(belex, *belex_parameters)

        if StackManager.has_elem(APPENDABLE_COMMANDS):
            lane = StackManager.pop(APPENDABLE_COMMANDS)
            if len(lane.instructions) > 0:
                belex.add_instruction(lane)

        belex.in_fragment = _in_fragment

        if is_inline:
            return None

        bleir_operations = belex.as_bleir()

        register_parameter_finder = RegisterParameterFinder()
        walker = BLEIRWalker()

        for bleir_operation in bleir_operations:
            walker.walk(register_parameter_finder, bleir_operation)

        register_parameters = register_parameter_finder.register_parameters
        parameters_by_id = {
            parameter.identifier: parameter
            for parameter in register_parameters
        }

        bleir_parameters = []

        # Maintain order for caller parameters
        for formal_parameter in formal_parameters:
            register_parameter = formal_parameter.as_bleir()
            if isinstance(register_parameter, BLEIR.MASK):
                register_parameter = register_parameter.sm_reg
            if register_parameter.identifier in parameters_by_id:
                del parameters_by_id[register_parameter.identifier]
            else:
                warn(f"unused register parameter in fragment {fn.__name__}: {register_parameter}")
            bleir_parameters.append(register_parameter)

        intern_parameters = sorted(parameters_by_id.values(),
                                   key=lambda param: param.identifier)
        bleir_parameters.extend(intern_parameters)

    except Exception as exception:
        LOGGER.exception(f"Failed to generate {fn.__name__}")
        raise exception

    finally:
        if not is_inline:
            popped = Belex.pop_context()
            assert belex is popped

    fragment_metadata = {
        BLEIR.FragmentMetadata.IS_LOW_LEVEL: True,
        BLEIR.FragmentMetadata.FILE_PATH: fragment_file_path,
        BLEIR.FragmentMetadata.LINE_NUMBER: fragment_line_number,
        BLEIR.FragmentMetadata.NUM_LINES: fragment_num_lines,
    }

    fragment = BLEIR.Fragment(
        identifier=fn.__name__,
        parameters=bleir_parameters,
        operations=bleir_operations,
        metadata=fragment_metadata)

    caller_metadata = {
        BLEIR.CallerMetadata.IS_LOW_LEVEL: True,
    }

    fragment_caller = BLEIR.FragmentCaller(
        fragment=fragment,
        metadata=caller_metadata)

    return fragment_caller


BelexOutput = Union[
    Optional[BLEIR.FragmentCallerCall],
    Sequence[BLEIR.Operation],
]


Glassible = Union[
    BelexLiteral,
    EWE,
    L1,
    L2,
    OffsetLX,
    RE,
    VR,
]


RE_VR_PLATS = re.compile(r"(?:INV_)?(?:[NEWS]?RL|GL|GGL)")
RE_RSP16_PLATS = re.compile(r"(?:INV_)?RSP16")
RE_RSP256_PLATS = re.compile(r"RSP256")
RE_RSP2K_PLATS = re.compile(r"RSP2K")
RE_RSP32K_PLATS = re.compile(r"RSP32K")


def glassible_plats(subject: Glassible) -> int:
    if subject is LGL \
       or isinstance(subject, L2) \
       or isinstance(subject, OffsetLX) and isinstance(subject.lhs.var, L2):
        return NUM_PLATS_PER_HALF_BANK

    if isinstance(subject, (EWE, L1, OffsetLX, RE, VR)):
        return NUM_PLATS_PER_APUC

    if RE_VR_PLATS.fullmatch(subject.symbol):
        return NUM_PLATS_PER_APUC

    if RE_RSP16_PLATS.fullmatch(subject.symbol):
        return NUM_PLATS_PER_APUC // 16

    if RE_RSP256_PLATS.fullmatch(subject.symbol):
        return NUM_PLATS_PER_APUC // 256

    if RE_RSP2K_PLATS.fullmatch(subject.symbol):
        return NUM_PLATS_PER_APUC // 2048

    if RE_RSP32K_PLATS.fullmatch(subject.symbol):
        return 1

    raise ValueError(
        f"Unsupported glassible type ({subject.__class__.__name__}): {subject}")


RE_16_SECTIONS = re.compile(r"(?:(?:INV_)?(?:[NEWS]?RL|RSP16)|RSP256|RSP2K|RSP32K)")
RE_4_SECTIONS = re.compile(r"(?:INV_)?GGL")
RE_1_SECTION = re.compile(r"(?:(?:INV_)?GL|LGL)")


def glassible_sections(subject: Glassible) -> int:
    if subject is LGL \
       or isinstance(subject, L2) \
       or isinstance(subject, OffsetLX) and isinstance(subject.lhs.var, L2):
        return 1

    if isinstance(subject, (EWE, RE, VR)):
        return 16

    if isinstance(subject, (L1, OffsetLX)):
        return 4

    if RE_16_SECTIONS.fullmatch(subject.symbol):
        return 16

    if RE_4_SECTIONS.fullmatch(subject.symbol):
        return 4

    if RE_1_SECTION.fullmatch(subject.symbol):
        return 1

    raise ValueError(
        f"Unsupported glassible type ({subject.__class__.__name__}): {subject}")


@bleir_dataclass
class GlassStatement(Instruction, BleirSerializable):
    __eq_excludes__ = ["file_path", "line_number"]

    subject: Glassible
    comment: Optional[str]
    sections: Sequence[int]
    plats: Sequence[int]
    fmt: GlassFormat
    order: GlassOrder
    balloon: bool
    rewrite: Optional[Dict[str, str]]
    file_path: Optional[str] = None
    line_number: int = -1

    def as_bleir(self: "GlassStatement") -> List[BLEIR.GlassStatement]:
        return [
            BLEIR.GlassStatement(
                subject=self.subject.as_bleir(),
                comment=self.comment,
                sections=self.sections,
                plats=self.plats,
                fmt=self.fmt.as_bleir(),
                order=self.order.as_bleir(),
                balloon=self.balloon,
                rewrite=self.rewrite,
                file_path=self.file_path,
                line_number=self.line_number),
        ]


def belex_apl(fn: Optional[Callable] = None, **manual_allocs) -> Callable:
    # Sometimes it is necessary to explicitly specify register allocations to
    # collaborate between manual and automatic allocations. NOTE: This method
    # does not reserve the allocations, it just ensures the caller uses them if
    # they are not overridden by calls to apl_set_*_reg.

    def decorator(fn: Callable) -> Callable:
        fragment_file_path = inspect.getfile(fn)
        fragment_source_lines, fragment_line_number = getsourcelines(fn)
        fragment_num_lines = len(fragment_source_lines)

        spec = getfullargspec(fn)

        param_specs = {}
        for name, kind in spec.annotations.items():
            if name != "return":
                param_specs[name] = kind

        belex_arg = spec.args[0]
        if belex_arg in param_specs:
            if param_specs[belex_arg] is not Belex:
                raise RuntimeError(
                    f"Expected the first parameter to be of type Belex")

            if len(param_specs) != len(spec.args):
                raise NotImplementedError(
                    f"Inferred parameter types are not supported yet")

            del param_specs[belex_arg]

        elif len(param_specs) != len(spec.args) - 1:
            raise NotImplementedError(
                f"Inferred parameter types are not supported yet")

        formal_parameters = []
        for name, kind in param_specs.items():
            if kind not in FormalParameter.__args__:
                if hasattr(kind, "__name__"):
                    raise NotImplementedError(
                        f"Unsupported parameter type ({kind.__name__}) for {name}")
                else:
                    raise NotImplementedError(
                        f"Unsupported parameter type ({kind}) for {name}")

            # if name.endswith("_rp"):
            #     name = name[:-len("_rp")]

            if name in manual_allocs:
                manual_alloc = manual_allocs[name]
                if kind is not manual_alloc.__class__:
                    raise RuntimeError(
                        f"Manual allocation for parameter {name} does not "
                        f"match its type ({kind.__name__}: {manual_alloc}")

            formal_parameters.append(kind(name))

        def build_fragment_caller() -> BLEIR.FragmentCaller:
            fragment_caller = build_or_call_fragment_caller(
                fn, formal_parameters,
                fragment_file_path=fragment_file_path,
                fragment_line_number=fragment_line_number,
                fragment_num_lines=fragment_num_lines)

            fragment = fragment_caller.fragment

            allocated_registers = []
            for register_parameter in fragment.parameters:
                if register_parameter.identifier in manual_allocs:
                    manual_alloc = \
                        manual_allocs[register_parameter.identifier]
                    local_register = manual_alloc.reg_nym
                    allocated_register = BLEIR.AllocatedRegister(
                        parameter=register_parameter,
                        register=local_register)
                    allocated_registers.append(allocated_register)
                else:
                    allocated_registers.append(None)

            fragment_caller = fragment_caller.having(
                registers=allocated_registers)

            return fragment_caller

        def build_fragment() -> BLEIR.Fragment:
            fragment_caller = build_fragment_caller()
            fragment = fragment_caller.fragment
            return fragment

        Actual_or_FormalParameters = Union[
            Sequence[int],
            Sequence[FormalParameter]]

        @wraps(fn)
        def wrapper(*actual_or_formal_parameters: Actual_or_FormalParameters,
                    is_initializer: bool = False,
                    debug: bool = True,
                    captured_glass: Optional[MutableSequence[str]] = None,
                    **kwargs: Dict[str, Any]) \
                -> BelexOutput:

            nonlocal fn, formal_parameters, param_specs

            actual_or_formal_parameters = list(actual_or_formal_parameters)
            kwargs = dict(kwargs)

            # Parse the args and kwargs
            # -------------------------

            pending_params = deque(param_specs.keys())
            for _ in actual_or_formal_parameters:
                pending_params.popleft()

            for param_id in pending_params:
                if param_id in kwargs:
                    actual_or_formal_parameter = kwargs[param_id]
                    actual_or_formal_parameters \
                        .append(actual_or_formal_parameter)
                    del kwargs[param_id]
                else:
                    raise RuntimeError(
                        f"Did not receive a value for parameter {param_id}")

            if len(kwargs) > 0:
                raise RuntimeError(
                    f"Found extraneous kwargs: {kwargs}")

            # Case 1: Inline the fragment call (part of fragment definition)
            # --------------------------------------------------------------

            if Belex.has_context():
                local_formals = actual_or_formal_parameters
                if len(local_formals) != len(formal_parameters):
                    raise AssertionError(
                        f"Expected length of parameters to be "
                        f"{len(formal_parameters)} but was "
                        f"{len(local_formals)}: {local_formals}")

                zipped_formals = list(zip(local_formals, formal_parameters))
                belex = Belex.context()

                # Special case for inlining BLEIR operations
                if belex.is_bleir:
                    belex_formals = []
                    for index, (local_formal, formal_parameter) \
                            in enumerate(zipped_formals):
                        if isinstance(local_formal, BleirRP.__args__):
                            as_belex_formal = RP_MAP[local_formal.__class__]
                            belex_formal = as_belex_formal(local_formal)
                        else:
                            belex_formal = local_formal
                            if isinstance(formal_parameter, Mask) \
                               and isinstance(belex_formal, Integer.__args__):
                                belex_formal = f"0x{belex_formal:04X}"
                        belex_formals.append(belex_formal)
                    belex.is_bleir = False
                    wrapper(*belex_formals)
                    belex.is_bleir = True
                    return flatten([instr.as_bleir()
                                    for instr in belex.instructions])

                for index, (local_formal, formal_parameter) \
                        in enumerate(zipped_formals):
                    formal_type = formal_parameter.__class__
                    if not isinstance(local_formal, formal_type) \
                       and (formal_type not in [L1, L2]
                            or not isinstance(local_formal, OffsetLX)
                            and isinstance(local_formal.lhs, BelexAccess)
                            and isinstance(local_formal.lhs.var, formal_type)):
                        local_formal = formal_type.parse_literal(local_formal)
                        local_formals[index] = local_formal

                build_or_call_fragment_caller(
                    fn, local_formals,
                    is_inline=True,
                    fragment_file_path=fragment_file_path,
                    fragment_line_number=fragment_line_number,
                    fragment_num_lines=fragment_num_lines)

                return None

            # Case 2: Call the fragment
            # -------------------------

            local_formals = formal_parameters
            local_actuals = []
            local_registers = []

            seu = SEULayer.context()

            for local_formal, local_actual in \
                    zip(local_formals, actual_or_formal_parameters):

                if isinstance(local_actual, ActualParameter.__args__):
                    local_registers.append(None)

                elif isinstance(local_actual, local_formal.__class__):
                    local_register = local_actual.register

                    if local_register is None:
                        raise ValueError(
                            f"Register not specified for "
                            f"{local_formal.__class__.__name__}: "
                            f"{local_actual}")

                    if local_formal.__class__ is Mask:
                        local_actual = seu.sm_regs[local_register]
                        register_prefix = seu.sm_regs.prefix
                    elif local_formal.__class__ is VR:
                        local_actual = seu.rn_regs[local_register]
                        register_prefix = seu.rn_regs.prefix
                    elif local_formal.__class__ is RE:
                        local_actual = seu.re_regs[local_register]
                        register_prefix = seu.re_regs.prefix
                    elif local_formal.__class__ is EWE:
                        local_actual = seu.ewe_regs[local_register]
                        register_prefix = seu.ewe_regs.prefix
                    elif local_formal.__class__ is L1:
                        local_actual = seu.l1_regs[local_register]
                        register_prefix = seu.l1_regs.prefix
                    elif local_formal.__class__ is L2:
                        local_actual = seu.l2_regs[local_register]
                        register_prefix = seu.l2_regs.prefix
                    else:
                        raise ValueError(
                            f"Unsupported register type "
                            f"({local_formal.__class__.__name__}): {local_actual}")

                    local_registers.append(f"{register_prefix}{local_register}")

                else:
                    raise ValueError(
                        f"Expected an ActualParameter or "
                        f"{local_formal.__class__.__name__} but received a "
                        f"{local_actual.__class__.__name__}: "
                        f"{local_actual}")

                local_actuals.append(local_actual)

            if len(local_actuals) != len(local_formals):
                raise AssertionError(
                    f"Expected {len(local_formals)} parameters but received "
                    f"{len(local_actuals)}: {local_actuals}")

            values_by_parameter_id = {}
            data_by_parameter_id = {}
            local_pairs = zip(local_formals, local_actuals)
            for index, (local_formal, local_actual) in enumerate(local_pairs):
                local_formal.validate(local_actual)
                if isinstance(local_formal, Section):
                    values_by_parameter_id[local_formal.symbol] = (0x0001 << local_actual)
                else:
                    values_by_parameter_id[local_formal.symbol] = local_actual

            call_file_path, call_line_number = file_line()

            call_metadata = {
                BLEIR.CallMetadata.IS_INITIALIZER: is_initializer,
                BLEIR.CallMetadata.IS_LOW_LEVEL: True,
                BLEIR.CallMetadata.FILE_PATH: call_file_path,
                BLEIR.CallMetadata.LINE_NUMBER: call_line_number,
            }

            # Rebuild the fragment while manipulating an inlined DIRI
            # instance for debugging.
            debug_parameters = []
            parameter_pairs = zip(local_formals, local_actuals)
            for formal_parameter, actual_parameter in parameter_pairs:
                if isinstance(formal_parameter, VR):
                    debug_parameter = \
                        formal_parameter.having(row_number=actual_parameter)
                elif isinstance(formal_parameter, RE):
                    debug_parameter = \
                        formal_parameter.having(row_mask=actual_parameter)
                elif isinstance(formal_parameter, EWE):
                    debug_parameter = \
                        formal_parameter.having(wordline_mask=actual_parameter)
                elif isinstance(formal_parameter, L1):
                    debug_parameter = \
                        formal_parameter.having(bank_group_row=actual_parameter)
                elif isinstance(formal_parameter, L2):
                    debug_parameter = \
                        formal_parameter.having(value=actual_parameter)
                elif isinstance(formal_parameter, (Mask, Section)):
                    debug_parameter = \
                        formal_parameter.having(constant_value=actual_parameter)
                else:
                    raise ValueError(
                        f"Unsupported parameter type "
                        f"({formal_parameter.__class__.__name__}): "
                        f"{formal_parameter}={actual_parameter}")
                debug_parameters.append(debug_parameter)

            # NOTIFY HERE
            from belex.bleir.interpreters import BLEIRInterpreter
            if BLEIRInterpreter.has_context():
                interpreter = BLEIRInterpreter.context()
                if len(interpreter.subject.observers) > 0:
                    parameters = []
                    for param_nym, formal_param, actual_param \
                            in zip(islice(spec.args, 1, None),
                                   formal_parameters,
                                   local_actuals):
                        parameters.append((param_nym,
                                           formal_param.__class__.__name__,
                                           int(actual_param)))
                    interpreter.subject.on_next(
                        ("fragment::enter",
                         [fragment_file_path,
                          fragment_line_number,
                          parameters]))

            fragment_caller = build_or_call_fragment_caller(
                fn, formal_parameters, debug_parameters,
                values_by_parameter_id=values_by_parameter_id,
                data_by_parameter_id=data_by_parameter_id,
                debug=debug, captured_glass=captured_glass,
                fragment_file_path=fragment_file_path,
                fragment_line_number=fragment_line_number,
                fragment_num_lines=fragment_num_lines)

            fragment = fragment_caller.fragment

            allocated_registers = []
            for register_parameter, local_register in \
                    zip(fragment.parameters, local_registers):
                if local_register is None \
                   and register_parameter.identifier in manual_allocs:
                    manual_alloc = \
                        manual_allocs[register_parameter.identifier]
                    local_register = manual_alloc.reg_nym
                if local_register is None:
                    allocated_registers.append(None)
                else:
                    allocated_register = BLEIR.AllocatedRegister(
                        parameter=register_parameter,
                        register=local_register)
                    allocated_registers.append(allocated_register)

            fragment_caller = fragment_caller.having(
                registers=allocated_registers)

            fragment_caller_call = BLEIR.FragmentCallerCall(
                caller=fragment_caller,
                parameters=local_actuals,
                metadata=call_metadata)

            # TODO: Determine if appending the call to the builder is still
            # necessary or whether I may utilize the side-effects of calling
            # build_or_call_fragment_caller in debug mode.
            if StackManager.has_elem(SNIPPET_BUILDER):
                snippet_builder = StackManager.peek(SNIPPET_BUILDER)
                fragment_caller_call = \
                    snippet_builder.compile_and_append(
                        fragment_caller_call)

            return fragment_caller_call

        wrapper.__low_level_block__ = True
        wrapper.__fragment__ = build_fragment
        wrapper.__caller__ = build_fragment_caller
        return wrapper

    if callable(fn):
        return decorator(fn)

    return decorator


def apl_init() -> int:
    apl_set_sm_reg(SM_REG_4, 0xFFFF)
    apl_set_sm_reg(SM_REG_5, 0x0001)
    apl_set_sm_reg(SM_REG_6, 0x1111)
    apl_set_sm_reg(SM_REG_7, 0x0101)
    apl_set_sm_reg(SM_REG_8, 0x000F)
    apl_set_sm_reg(SM_REG_9, 0x0F0F)
    apl_set_sm_reg(SM_REG_10, 0x0707)
    apl_set_sm_reg(SM_REG_11, 0x5555)
    apl_set_sm_reg(SM_REG_12, 0x3333)
    apl_set_sm_reg(SM_REG_13, 0x00FF)
    apl_set_sm_reg(SM_REG_14, 0x001F)
    apl_set_sm_reg(SM_REG_15, 0x003F)

    apl_set_rn_reg(RN_REG_8, VR16_T0)
    apl_set_rn_reg(RN_REG_9, VR16_T1)
    apl_set_rn_reg(RN_REG_10, VR16_T2)
    apl_set_rn_reg(RN_REG_11, VR16_T3)
    apl_set_rn_reg(RN_REG_12, VR16_T4)
    apl_set_rn_reg(RN_REG_13, VR16_T5)
    apl_set_rn_reg(RN_REG_14, VR16_T6)
    apl_set_rn_reg(RN_REG_15, VR16_FLAGS)

    # return apl_init_vals()
    return 0


def apl_set_sm_reg(literal: Mask, value: int) -> None:
    seu = SEULayer.context()
    seu.sm_regs[literal.register] = value


def apl_set_rn_reg(literal: VR, value: int) -> None:
    seu = SEULayer.context()
    seu.rn_regs[literal.register] = value


def apl_set_re_reg(literal: RE, value: int) -> None:
    seu = SEULayer.context()
    seu.re_regs[literal.register] = value


def apl_set_ewe_reg(literal: EWE, value: int) -> None:
    seu = SEULayer.context()
    seu.ewe_regs[literal.register] = value


def apl_set_l1_reg(literal: L1, value: int) -> None:
    seu = SEULayer.context()
    seu.l1_regs[literal.register] = value


def apl_set_l1_reg_ext(literal: L1,
                       bank_id: int,
                       grp_id: int,
                       grp_row: int) -> None:
    l1_addr = (bank_id << 11) | (grp_id << 9) | grp_row
    apl_set_l1_reg(literal, l1_addr)


def apl_set_l2_reg(literal: L2, value: int) -> None:
    seu = SEULayer.context()
    seu.l2_regs[literal.register] = value
