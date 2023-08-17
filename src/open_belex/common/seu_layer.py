r"""
By Dylon Edwards
"""

from dataclasses import dataclass, field
from typing import (Any, Callable, ClassVar, Iterator, Sequence, Tuple, Type,
                    Union)

import numpy as np

from reactivex.subject import Subject

from open_belex.common.constants import (MAX_EWE_VALUE, MAX_L1_VALUE,
                                         MAX_L2_VALUE, MAX_RE_VALUE,
                                         MAX_RN_VALUE, MAX_SM_VALUE,
                                         NUM_EWE_REGS, NUM_L1_REGS,
                                         NUM_L2_REGS, NUM_RE_REGS, NUM_RN_REGS,
                                         NUM_SM_REGS)
from open_belex.common.stack_manager import contextual
from open_belex.common.types import Integer

Subscriber = Callable[[Any], None]


@dataclass
class RegisterTable:
    prefix: str
    num_regs: Integer
    max_value: Integer
    _regs: Sequence[Integer] = None
    subject: Subject = field(default_factory=Subject)

    def __post_init__(self: "RegisterTable") -> None:
        if len(self.prefix.strip()) == 0:
            raise ValueError(
                f"Expected non-empty prefix, but received \"{self.prefix}\"")

        if self.num_regs < 0:
            raise ValueError(
                f"Expected num_regs to be non-negative, but was "
                f"{self.num_regs}")

        if self.max_value < 0:
            raise ValueError(
                f"Expected max_value to be non-negative, but was "
                f"{self.max_value}")

        self._regs = np.repeat(-1, self.num_regs)

    def __eq__(self: "RegisterTable", other: Any) -> bool:
        return isinstance(other, self.__class__) \
            and self.prefix == other.prefix \
            and self.num_regs == other.num_regs \
            and self.max_value == other.max_value \
            and np.array_equal(self._regs, other._regs)

    def __ne__(self: "RegisterTable", other: Any) -> bool:
        return not self.__eq__(other)

    def __iter__(self: "RegisterTable") -> Iterator[Tuple[Integer, Integer]]:
        for index in range(self.num_regs):
            reg_val = self._regs[index]
            if reg_val >= 0:
                yield index, reg_val

    def __len__(self: "RegisterTable") -> Integer:
        return self.num_regs

    def __getitem__(self: "RegisterTable",
                    index_or_nym: Union[Integer, str],
                    is_property: bool = False) -> Integer:

        if isinstance(index_or_nym, str):
            nym = index_or_nym
            # NOTE: Avoid calling "hasattr(self, nym)" because hasattr will
            # invoke the property ...
            if not nym.startswith(self.prefix) or not hasattr(self.__class__, nym):
                raise ValueError(
                    f"{self.__class__.__name__} has no register named, {nym}")
            return getattr(self, nym)

        index = index_or_nym

        if not 0 <= index < self.num_regs:
            raise ValueError(
                f"Out of bounds: Expected index = {index} to be in the range "
                f"[0, {self.num_regs})")

        reg_val = self._regs[index]

        if reg_val < 0:
            raise KeyError(
                f"Register value for {self.prefix}{index} has not been "
                f"initialized.")

        return reg_val

    def __setitem__(self: "RegisterTable",
                    index_or_nym: Union[Integer, str],
                    reg_val: Integer,
                    is_property: bool = False) -> None:

        if isinstance(index_or_nym, str):
            nym = index_or_nym
            # NOTE: Avoid calling "hasattr(self, nym)" because hasattr will
            # invoke the property ...
            if not nym.startswith(self.prefix) or not hasattr(self.__class__, nym):
                raise ValueError(
                    f"{self.__class__.__name__} has no register named, {nym}")
            return setattr(self, nym, reg_val)

        index = index_or_nym

        if not 0 <= index < self.num_regs:
            raise ValueError(
                f"Out of bounds: Expected index = {index} to be in the range "
                f"[0, {self.num_regs})")

        if not 0 <= reg_val <= self.max_value:
            raise ValueError(
                f"Out of bounds: expected reg_val = {reg_val} to be in the "
                f"range [0, {self.max_value}]")

        self._regs[index] = reg_val

        if len(self.subject.observers) > 0:
            event_class = self.prefix[:-1].lower()
            event_class = f"seu::{event_class}"
            self.subject.on_next((event_class, index, reg_val))

    def __repr__(self: "RegisterTable") -> str:
        reg_vals = []
        for index in range(self.num_regs):
            reg_val = self._regs[index]
            reg_vals.append(f"{self.prefix}{index}: {reg_val}")
        return "{" + ", ".join(reg_vals) + "}"

    def subscribe(self: "RegisterTable", subscriber: Subscriber) -> None:
        self.subject.subscribe(subscriber)


# Delegate the property initializer to capture the value of index
def def_reg(cls: Type["RegisterTable"], prefix: str, index: Integer) -> None:
    setattr(cls, f"{prefix}{index}",
            property(fget=lambda self: cls.__getitem__(self, index,
                                                       is_property=True),
                     fset=lambda self, value: cls.__setitem__(self, index,
                                                              value, is_property=True)))


class SMRegTable(RegisterTable):
    PREFIX: ClassVar[str] = "SM_REG_"

    def __init__(self: "SMRegTable") -> None:
        super().__init__(self.PREFIX, NUM_SM_REGS, MAX_SM_VALUE)

    def __repr__(self: "SMRegTable") -> str:
        reg_vals = []
        for index in range(self.num_regs):
            reg_val = self._regs[index]
            if reg_val >= 0:
                reg_vals.append(f"{self.prefix}{index}: 0x{reg_val:04X}")
            else:
                reg_vals.append(f"{self.prefix}{index}: {reg_val}")
        return "{" + ", ".join(reg_vals) + "}"


for index in range(NUM_SM_REGS):
    def_reg(SMRegTable, SMRegTable.PREFIX, index)


class RNRegTable(RegisterTable):
    PREFIX: ClassVar[str] = "RN_REG_"

    def __init__(self: "RNRegTable") -> None:
        super().__init__(self.PREFIX, NUM_RN_REGS, MAX_RN_VALUE)


for index in range(NUM_RN_REGS):
    def_reg(RNRegTable, RNRegTable.PREFIX, index)


class RERegTable(RegisterTable):
    PREFIX: ClassVar[str] = "RE_REG_"

    def __init__(self: "RERegTable") -> None:
        super().__init__(self.PREFIX, NUM_RE_REGS, MAX_RE_VALUE)

    def __repr__(self: "RERegTable") -> str:
        reg_vals = []
        for index in range(self.num_regs):
            reg_val = self._regs[index]
            if reg_val >= 0:
                reg_vals.append(f"{self.prefix}{index}: 0x{reg_val:06X}")
            else:
                reg_vals.append(f"{self.prefix}{index}: {reg_val}")
        return "{" + ", ".join(reg_vals) + "}"


for index in range(NUM_RE_REGS):
    def_reg(RERegTable, RERegTable.PREFIX, index)


class EWERegTable(RegisterTable):
    PREFIX: ClassVar[str] = "EWE_REG_"

    def __init__(self: "EWERegTable") -> None:
        super().__init__(self.PREFIX, NUM_EWE_REGS, MAX_EWE_VALUE)

    def __repr__(self: "EWERegTable") -> str:
        reg_vals = []
        for index in range(self.num_regs):
            reg_val = self._regs[index]
            if reg_val >= 0:
                group = (reg_val >> 8)
                mask = (reg_val & 0xFF)
                reg_vals.append(f"{self.prefix}{index}: ({group}, 0x{mask:02X})")
            else:
                reg_vals.append(f"{self.prefix}{index}: {reg_val}")
        return "{" + ", ".join(reg_vals) + "}"

    def __getitem__(self: "EWERegTable",
                    index_or_nym: Union[Integer, str],
                    is_property: bool = False) \
            -> Union[Integer, Tuple[Integer, Integer]]:
        reg_val = super().__getitem__(index_or_nym,
                                      is_property=is_property)
        if not is_property:
            return reg_val
        group = (reg_val >> 8)
        mask = (reg_val & 0xFF)
        return (group, mask)

    def __setitem__(self: "EWERegTable",
                    index_or_nym: Union[Integer, str],
                    reg_val: Union[Integer, Tuple[Integer, Integer]],
                    is_property: bool = False) -> None:
        if not isinstance(reg_val, Integer.__args__):
            group, mask = reg_val
            reg_val = (group << 8) | mask
        super().__setitem__(index_or_nym, reg_val,
                            is_property=is_property)


for index in range(NUM_EWE_REGS):
    def_reg(EWERegTable, EWERegTable.PREFIX, index)


class L1RegTable(RegisterTable):
    PREFIX: ClassVar[str] = "L1_ADDR_REG_"

    def __init__(self: "L1RegTable") -> None:
        super().__init__(self.PREFIX, NUM_L1_REGS, MAX_L1_VALUE)

    def __repr__(self: "L1RegTable") -> str:
        reg_vals = []
        for index in range(self.num_regs):
            reg_val = self._regs[index]
            if reg_val >= 0:
                bank = (reg_val >> 11)
                group = ((reg_val >> 9) & 0b11)
                row = (reg_val & 0xFF)
                reg_vals.append(f"{self.prefix}{index}: ({bank}, {group}, {row})")
            else:
                reg_vals.append(f"{self.prefix}{index}: {reg_val}")
        return "{" + ", ".join(reg_vals) + "}"

    def __getitem__(self: "L1RegTable",
                    index_or_nym: Union[Integer, str],
                    is_property: bool = False) \
            -> Union[Integer, Tuple[Integer, Integer, Integer]]:
        reg_val = super().__getitem__(index_or_nym,
                                      is_property=is_property)
        if not is_property:
            return reg_val
        bank = (reg_val >> 11)
        group = ((reg_val >> 9) & 0b11)
        row = (reg_val & 0xFF)
        return (bank, group, row)

    def __setitem__(self: "L1RegTable",
                    index_or_nym: Union[Integer, str],
                    reg_val: Union[Integer,
                                   Tuple[Integer],
                                   Tuple[Integer, Integer],
                                   Tuple[Integer, Integer, Integer]],
                    is_property: bool = False) -> None:
        if not isinstance(reg_val, Integer.__args__):
            if len(reg_val) == 3:
                bank, group, row = reg_val
            elif len(reg_val) == 2:
                bank, group, row = 0, *reg_val
            elif len(reg_val) == 1:
                bank, group, row = 0, 0, *reg_val
            else:
                raise ValueError(
                    f"Expected reg_val to be either an Integer literal or a tuple "
                    f"of 1, 2, or 3 elements, but was: {reg_val}")
            reg_val = ((bank << 11) | (group << 9) | row)
        super().__setitem__(index_or_nym, reg_val,
                            is_property=is_property)


for index in range(NUM_L1_REGS):
    def_reg(L1RegTable, L1RegTable.PREFIX, index)


class L2RegTable(RegisterTable):
    PREFIX: ClassVar[str] = "L2_ADDR_REG_"

    def __init__(self: "L2RegTable") -> None:
        super().__init__(self.PREFIX, NUM_L2_REGS, MAX_L2_VALUE)


for index in range(NUM_L2_REGS):
    def_reg(L2RegTable, L2RegTable.PREFIX, index)


@contextual(lazy_init=True)
@dataclass
class SEULayer:
    sm_regs: Sequence[Integer] = field(default_factory=SMRegTable)
    rn_regs: Sequence[Integer] = field(default_factory=RNRegTable)
    re_regs: Sequence[Integer] = field(default_factory=RERegTable)
    ewe_regs: Sequence[Integer] = field(default_factory=EWERegTable)
    l1_regs: Sequence[Integer] = field(default_factory=L1RegTable)
    l2_regs: Sequence[Integer] = field(default_factory=L2RegTable)

    def __iter__(self: "SEULayer") -> Iterator[Tuple[str, RegisterTable]]:
        yield "sm_regs", self.sm_regs
        yield "rn_regs", self.rn_regs
        yield "re_regs", self.re_regs
        yield "ewe_regs", self.ewe_regs
        yield "l1_regs", self.l1_regs
        yield "l2_regs", self.l2_regs

    def __eq__(self: "SEULayer", other: Any) -> bool:
        return isinstance(other, SEULayer) \
            and self.sm_regs == other.sm_regs \
            and self.rn_regs == other.rn_regs \
            and self.re_regs == other.re_regs \
            and self.ewe_regs == other.ewe_regs \
            and self.l1_regs == other.l1_regs \
            and self.l2_regs == other.l2_regs

    def subscribe_to_sm_regs(self: "SEULayer", subscriber: Subscriber) -> None:
        self.sm_regs.subscribe(subscriber)

    def subscribe_to_rn_regs(self: "SEULayer", subscriber: Subscriber) -> None:
        self.rn_regs.subscribe(subscriber)

    def subscribe_to_re_regs(self: "SEULayer", subscriber: Subscriber) -> None:
        self.re_regs.subscribe(subscriber)

    def subscribe_to_ewe_regs(self: "SEULayer", subscriber: Subscriber) -> None:
        self.ewe_regs.subscribe(subscriber)

    def subscribe_to_l1_regs(self: "SEULayer", subscriber: Subscriber) -> None:
        self.l1_regs.subscribe(subscriber)

    def subscribe_to_l2_regs(self: "SEULayer", subscriber: Subscriber) -> None:
        self.l2_regs.subscribe(subscriber)

    def subscribe(self: "SEULayer", subscriber: Subscriber) -> None:
        self.subscribe_to_sm_regs(subscriber)
        self.subscribe_to_rn_regs(subscriber)
        self.subscribe_to_re_regs(subscriber)
        self.subscribe_to_ewe_regs(subscriber)
        self.subscribe_to_l1_regs(subscriber)
        self.subscribe_to_l2_regs(subscriber)
