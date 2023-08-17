r"""
By Dylon Edwards
"""

import logging
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from dataclasses import field
from typing import Any, Dict, List, Optional, Sequence, Type
from warnings import warn

from open_belex.bleir.types import bleir_dataclass
from open_belex.common.types import Indices
from open_belex.literal import RESERVED_IDENTIFIERS
from open_belex.renderable import Renderable, Renderer, Rendition
from open_belex.utils.section_utils import parse_sections

LOGGER = logging.getLogger()


class Operable(ABC):

    def __or__(self: "Operable", other: "Operable") -> "Operable":
        return OrOperator(self, other)

    def __and__(self: "Operable", other: "Operable") -> "Operable":
        return AndOperator(self, other)

    def __xor__(self: "Operable", other: "Operable") -> "Operable":
        return XorOperator(self, other)

    def __le__(self: "Operable", other: "Operable") -> "Operable":
        return self.assign(other)

    def __ior__(self: "Operable", other: "Operable") -> "Operable":
        return self.assign(self.__or__(other))

    def __iand__(self: "Operable", other: "Operable") -> "Operable":
        return self.assign(self.__and__(other))

    def __ixor__(self: "Operable", other: "Operable") -> "Operable":
        return self.assign(self.__xor__(other))

    def assign(self: "Operable", other: "Operable") -> "Operable":
        raise NotImplementedError


def reduce_ggl(a : "Operable") -> "Operable":
    return ReduceGGLOperator(a)


def nibble_and_fold(a : "Operable") -> "Operable":
    return NibbleAndFoldOperator(a)


# HIR - handles assignment statements directly
@bleir_dataclass
class ReduceGGLOperator(Operable, Renderable):
    a: "VariableAccess"

    def __str__(self: "ReduceGGLOperator") -> str:
        return f"reduce_ggl({self.a})"

    def operands(self: "AndOperator") -> List["VariableAccess"]:
        return self.a.operands()

    def morphology(self: "AndOperator") -> str:
        return f"reduce_ggl({self.a.morphology()})"

    def clone(self: "AndOperator") -> "AndOperator":
        return ReduceGGLOperator(self.a.clone())

    def render(self: "AndOperator", ir: Renderer) -> "VariableAccess":
        a = self.a.render(ir)
        return ir.REDUCE_GGL(a)

    def belex_representation(self: "AndOperator") -> "BelexRepresentation":
        return self.a.belex_representation()


@bleir_dataclass
class NibbleAndFoldOperator(Operable, Renderable):
    a: "VariableAccess"

    def __str__(self: "NibbleAndFoldOperator") -> str:
        return f"nibble_and_fold({self.a})"

    def operands(self: "NibbleAndFoldOperator") -> List["VariableAccess"]:
        return self.a.operands()

    def morphology(self: "NibbleAndFoldOperator") -> str:
        return f"nibble_and_fold({self.a.morphology()})"

    def clone(self: "NibbleAndFoldOperator") -> "NibbleAndFoldOperator":
        return NibbleAndFoldOperator(self.a.clone())

    def render(self: "NibbleAndFoldOperator", ir: Renderer) -> "VariableAccess":
        a = self.a.render(ir)
        return ir.NIBBLE_AND_FOLD(a)

    def belex_representation(self: "NibbleAndFoldOperator") -> "BelexRepresentation":
        return self.a.belex_representation()


@bleir_dataclass
class AndOperator(Operable, Renderable):
    a: "VariableAccess"
    b: "VariableAccess"

    def __str__(self: "AndOperator") -> str:
        return f"{self.a}&{self.b}"

    def operands(self: "AndOperator") -> List["VariableAccess"]:
        return self.a.operands() + self.b.operands()

    def morphology(self: "AndOperator") -> str:
        return f"{self.a.morphology()}&{self.b.morphology()}"

    def clone(self: "AndOperator") -> "AndOperator":
        return AndOperator(self.a.clone(), self.b.clone())

    def render(self: "AndOperator", ir: Renderer) -> "VariableAccess":
        a = self.a.render(ir)
        b = self.b.render(ir)
        return ir.AND(a, b)

    def belex_representation(self: "AndOperator") -> "BelexRepresentation":
        return self.a.belex_representation()


@bleir_dataclass
class OrOperator(Operable, Renderable):
    a: "VariableAccess"
    b: "VariableAccess"

    def __str__(self: "OrOperator") -> str:
        return f"{self.a}|{self.b}"

    def operands(self: "OrOperator") -> List["VariableAccess"]:
        return self.a.operands() + self.b.operands()

    def morphology(self: "OrOperator") -> str:
        return f"{self.a.morphology()}|{self.b.morphology()}"

    def clone(self: "OrOperator") -> "OrOperator":
        return OrOperator(self.a.clone(), self.b.clone())

    def render(self: "OrOperator", ir: Renderer) -> "VariableAccess":
        a = self.a.render(ir)
        b = self.b.render(ir)
        return ir.OR(a, b)

    def belex_representation(self: "OrOperator") -> "BelexRepresentation":
        return self.a.belex_representation()


@bleir_dataclass
class XorOperator(Operable, Renderable):
    a: "VariableAccess"
    b: "VariableAccess"

    def __str__(self: "XorOperator") -> str:
        return str(self.a)+"^"+str(self.b)

    def operands(self: "XorOperator") -> List["VariableAccess"]:
        return self.a.operands() + self.b.operands()

    def morphology(self: "XorOperator") -> str:
        return f"{self.a.morphology()}^{self.b.morphology()}"

    def clone(self: "XorOperator") -> "XorOperator":
        return XorOperator(self.a.clone(), self.b.clone())

    def render(self: "XorOperator", ir: Renderer) -> "VariableAccess":
        a = self.a.render(ir)
        b = self.b.render(ir)
        return ir.XOR(a, b)

    def belex_representation(self: "XorOperator") -> "BelexRepresentation":
        return self.a.belex_representation()


@bleir_dataclass
class AssignOperator(Renderable):
    a: "VariableAccess"
    b: Operable

    def __str__(self: "AssignOperator") -> str:
        return f"{self.a}:={self.b}"

    def lhs(self: "AssignOperator") -> List[Operable]:
        return self.a.operands()

    def rhs(self: "AssignOperator") -> List[Operable]:
        return self.b.operands()

    def morphology(self: "AssignOperator") -> str:
        return f"{self.a.morphology()}:={self.b.morphology()}"

    def operands(self: "AssignOperator") -> List["VariableAccess"]:
        return self.a.operands() + self.b.operands()

    def clone(self: "AssignOperator") -> "AssignOperator":
        return assign(self.a.clone(), self.b.clone())

    def render(self: "AssignOperator", ir: Renderer) -> Rendition:
        lhs = self.a.render(ir)
        rhs = self.b.render(ir)
        return ir.ASSIGN(lhs, rhs)

    def is_dep(self: "AssignOperator",
               other: "AssignOperator") -> bool:
        # LOGGER.debug('-------- IS_DEP --------')
        data_dep = self.data_dep(self, other)
        anti_dep = self.anti_dep(self, other)
        output_dep = self.output_dep(self, other)
        return data_dep | anti_dep | output_dep

    @staticmethod
    def vars_and_sections_overlap(accesses1, accesses2) -> bool:
        lh_var_section_map = defaultdict(set)
        rh_var_section_map = defaultdict(set)

        for v in accesses1:
            # Concatenate in case a register appears multiple times
            lh_var_section_map[v.var.symbol] |= set(v.indices)

        for v in accesses2:
            # Concatenate in case a register appears multiple times
            rh_var_section_map[v.var.symbol] |= set(v.indices)

        lhss = set(map(lambda v: v.var.symbol, accesses1))
        rhss = set(map(lambda v: v.var.symbol, accesses2))

        result = False
        for common_v in lhss & rhss:
            lh_section_map = lh_var_section_map[common_v]
            rh_section_map = rh_var_section_map[common_v]
            result = bool(lh_section_map & rh_section_map)
            if result:
                break

        # LOGGER.debug(f'lh_var_section_map: {lh_var_section_map}')
        # LOGGER.debug(f'rh_var_section_map: {rh_var_section_map}')
        # LOGGER.debug(f'lhss: {lhss}, rhss: {rhss}')
        # LOGGER.debug(f'result: {result}')

        return result

    @classmethod
    def data_dep(cls: Type["AssignOperator"],
                 inst1: "AssignOperator",
                 inst2: "AssignOperator") -> bool:

        # LOGGER.debug('-------- INSIDE DATA_DEP --------')
        lhs1 = inst1.lhs()
        rhs2 = inst2.rhs()
        result = cls.vars_and_sections_overlap(lhs1, rhs2)
        return result

    @classmethod
    def anti_dep(cls: Type["AssignOperator"],
                 inst1: "AssignOperator",
                 inst2: "AssignOperator") -> bool:
        # Useful for loop analysis (e.g. inner loop analysis).
        # This will be important for Tartan.
        return cls.data_dep(inst2, inst1)

    @classmethod
    def output_dep(cls: Type["AssignOperator"],
                   inst1: "AssignOperator",
                   inst2: "AssignOperator") -> bool:

        # LOGGER.debug('-------- INSIDE OUTPUT_DEP --------')
        result = cls.vars_and_sections_overlap(inst1.lhs(), inst2.lhs())
        return result

    def belex_representation(self: "AssignOperator") -> "BelexRepresentation":
        return self.a.belex_representation()


def assign(lhs: Operable, rhs: Operable) -> AssignOperator:
    assign_operator = AssignOperator(lhs, rhs)
    lhs.belex_representation().ir.append(assign_operator)
    return assign_operator


@bleir_dataclass
class VariableAccess(Operable, Renderable, Rendition):
    var: "Variable"
    indices: Sequence[int]

    def __init__(self: "VariableAccess",
                 var: "Variable",
                 indices: Indices = [],
                 is_clone: bool = False) -> None:

        self.var = var

        if not is_clone:
            # CAREFUL: here an empty indices array means all indices
            self.indices = parse_sections(indices)
        else:
            self.indices = deepcopy(indices)

    def __str__(self: "VariableAccess") -> str:
        if len(self.indices) == 0:
            return str(self.var)
        elif len(self.indices) == 16:
            return f"{self.var}[:]"

        return f"{self.var}{self.indices}"

    def operands(self: "VariableAccess") -> List["VariableAccess"]:
        return [self]

    def morphology(self: "VariableAccess") -> str:
        return f"{self.var}[]"

    def assign(self: "VariableAccess", other: Operable) -> AssignOperator:
        if self.var.is_tmp():
            raise RuntimeError("explicit assignment to temporaries is invalid")

        if isinstance(other, Variable):
            other = other(self.indices)

        # TODO: Recursively transform any VarTypes on the RHS to be instances
        # of VariableAccess. This includes recursively transforming
        # AndOperator, OrOperator, and XorOperator.

        return assign(self, other)

    def clone(self: "VariableAccess") -> "VariableAccess":
        return VariableAccess(self.var, indices=self.indices, is_clone=True)

    def render(self: "VariableAccess", ir: Renderer) -> Rendition:
        return self.clone()

    def belex_representation(self: "VariableAccess") -> "BelexRepresentation":
        return self.var.belex_representation()


@bleir_dataclass
class Variable(Operable):
    # Hints to instance_members_of to only consider the following fields, which
    # avoid recursing infinitely into belex_repr since it maintains a list to
    # this Variable.
    __walk__ = ["symbol", "is_temp"]

    symbol: str
    is_temp: bool = False
    belex_repr: Optional["BelexRepresentation"] = None

    def __call__(self: "Variable", indices: Indices = []) -> VariableAccess:
        return VariableAccess(self, indices)

    def __getitem__(self: "Variable", indices: Indices) -> VariableAccess:
        return VariableAccess(self, indices)

    def __setitem__(self: "Variable",
                    indices: Indices,
                    value: Operable) -> AssignOperator:
        return self(indices) <= value

    def __str__(self: "Variable") -> str:
        return self.symbol

    def operands(self: "Variable") -> List[VariableAccess]:
        return [self()]

    def morphology(self: "Variable") -> str:
        return str(self)

    def is_tmp(self: "Variable") -> bool:
        return self.is_temp

    def belex_representation(self: "Variable") -> "BelexRepresentation":
        return self.belex_repr

    def assign(self: "Variable", other: Operable) -> AssignOperator:
        if self.is_tmp():
            raise RuntimeError("explicit assignment to temporaries is invalid")

        if isinstance(other, Variable):
            return self() <= other()

        while not isinstance(other, VariableAccess):
            other = other.a

        return self(other.indices) <= other


@bleir_dataclass
class SymbolTable:
    symbols: Dict[str, Variable] = field(default_factory=dict)
    temp_count: int = 0

    def get_temp(self: "SymbolTable") -> Variable:
        symstr = "t_{}".format(self.temp_count)
        self.temp_count += 1
        return self.variable_for_symstr(symstr, is_temp=True)

    def variable_for_symstr(self: "SymbolTable",
                            symstr: str,
                            is_temp: bool = False) -> Variable:

        if symstr in RESERVED_IDENTIFIERS:
            raise ValueError(f"{symstr} is reserved, please choose another name.")

        variable = Variable(symstr, is_temp=is_temp)
        self.symbols[symstr] = variable
        return variable

    def exist(self: "SymbolTable", symbol: str) -> bool:
        return symbol in self.symbols

    def __getitem__(self: "SymbolTable", symbol: str) -> Variable:
        return self.symbols[symbol]

    def __contains__(self: "SymbolTable", symbol: str) -> bool:
        return symbol in self.symbols


@bleir_dataclass
class BelexRepresentation:
    symbol_table: SymbolTable = field(default_factory=SymbolTable)
    ir: Sequence[AssignOperator] = field(default_factory=list)
    internal_count: int = 0

    def apl(self: "BelexRepresentation", *statements: Sequence[Any]) -> None:
        pass

    def install_symbol(self: "BelexRepresentation", symstr: str) -> Variable:
        r"""Install a string that specifies a symbol into the symbol table
        of this BelexRepresentation."""
        t = self.symbol_table.variable_for_symstr(symstr)
        t.belex_repr = self
        return t

    def var(self: "BelexRepresentation",
            value: Optional[int] = None) -> Variable:

        # FIXME: Add support for initial values
        if value is not None:
            warn(f"initial values are not yet supported")

        symbol = f"_INTERNAL{self.internal_count}"
        self.internal_count += 1
        return self.install_symbol(symbol)
