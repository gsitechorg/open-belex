r"""
By Dylon Edwards and Brian Beckman
"""


# BLEIR := Bit-Level Engine Intermediate Representation


import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (Any, Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Sequence, Set, Tuple, Type, Union)
from warnings import warn

import numpy as np

from open_belex.common.mask import Mask
from open_belex.common.types import Integer

RESERVED_TERMS: Set[str] = set()


def reserve_term(term: str) -> str:
    global RESERVED_TERMS
    if term in RESERVED_TERMS:
        raise ValueError(f"Term is already reserved: {term}")
    RESERVED_TERMS.add(term)
    return term


RL: str = reserve_term("RL")
NRL: str = reserve_term("NRL")
ERL: str = reserve_term("ERL")
WRL: str = reserve_term("WRL")
SRL: str = reserve_term("SRL")
GL: str = reserve_term("GL")
GGL: str = reserve_term("GGL")
RSP16: str = reserve_term("RSP16")

INV_RL: str = reserve_term("INV_RL")
INV_NRL: str = reserve_term("INV_NRL")
INV_ERL: str = reserve_term("INV_ERL")
INV_WRL: str = reserve_term("INV_WRL")
INV_SRL: str = reserve_term("INV_SRL")
INV_GL: str = reserve_term("INV_GL")
INV_GGL: str = reserve_term("INV_GGL")
INV_RSP16: str = reserve_term("INV_RSP16")

LGL: str = reserve_term("LGL")

RSP256: str = reserve_term("RSP256")
RSP2K: str = reserve_term("RSP2K")
RSP32K: str = reserve_term("RSP32K")

NOOP: str = reserve_term("NOOP")
FSEL_NOOP: str = reserve_term("FSEL_NOOP")
RSP_END: str = reserve_term("RSP_END")
RSP_START_RET: str = reserve_term("RSP_START_RET")
L2_END: str = reserve_term("L2_END")

RWINH_SET: str = reserve_term("RWINH_SET")
RWINH_RST: str = reserve_term("RWINH_RST")


class BleirSerializable(ABC):
    """Any type that satisfies this interface may be passed to the BLEIR
    factory functions (e.g. `statement`). The corresponding `as_bleir` method
    will be invoked and the return value treated as the parameter to the
    respective constructor function. For examples, please reference the various
    types defined by the `open_belex.literal` package (e.g.
    `open_belex.literal.AssignOperation`)."""

    @abstractmethod
    def as_bleir(self: "BleirSerializable") -> Any:
        """Helper method used by the BLEIR factory functions that returns a
        BLEIR representation of the current instance of the implementing type."""
        raise NotImplementedError


class BLEIRError(RuntimeError):
    """Top-level error class for BLEIR-related classes."""


class SyntacticError(BLEIRError):
    """Specifies there was such an error in the BLEIR structure as the wrong
    type for a field, the wrong number of parameters to a Fragment call,
    etc."""


class SemanticError(BLEIRError):
    """Specifies there is an error in the BLEIR application that is not
    syntactic. It may include such things as duplicated parameter IDs or
    assigning the same register to multiple register params for the same
    Fragment."""


class BleirEnum(Enum):

    @classmethod
    def values(self: Type["Token"]) -> Sequence[str]:
        return [enumerated.value for enumerated in self]

    @classmethod
    def value_map(self: Type["Token"]) -> Dict[str, "Token"]:
        """Returns a mapping of token names to values."""
        return OrderedDict((enumerated.value, enumerated)
                           for enumerated in self)

    @classmethod
    def find_by_value(self: Type["Token"], value: Any) -> "Token":
        """Returns the token associated with the given value, or raises an
        error if none exists."""
        for enumerated in list(self):
            if enumerated.value == value:
                return enumerated
        raise ValueError(
            f"No {self.__class__.__name__} exists for value: {value}")

    def __str__(self: "Symbol") -> str:
        """Returns a string representation of this token for debuggin."""
        return self.value


class Token(BleirEnum):
    """Top-level enum class for BLEIR types that represents such things as
    symbols and operators. A few helper methods are provided for common
    enum-related tasks."""

    @property
    def operands(self: "Token") -> List[str]:
        """Returns the operands associated with the current instance of this
        type that are used within a higher-level BLEIR expression. In the case
        of a Token type, the operands include only itself."""
        return [self.value]


class Symbol(Token):
    """A string literal that has been enumerated."""


class ReadWriteInhibit(Token):
    """A special token type used to represent read-write inhibited sections.
    When masked sections are read-write inhibited they are disabled to reduce
    power consumption. You should never enable read-write inhibit over sections
    you want to use."""

    RWINH_SET: str = RWINH_SET  # enable read/write inhibit
    RWINH_RST: str = RWINH_RST  # disable read/write inhibit

    @classmethod
    def find_by_value(cls: Type["ReadWriteInhibit"], value: Union[str, bool]):
        """Returns the ReadWriteInhibit Token associated with either the string
        name or boolean value.

        Parameters:
            value: Either the string name or boolean value representing the
                   desired Token.

        Returns:
            The ReadWriteInhibit Token associated with the value."""

        if isinstance(value, str):
            return super().find_by_value(value)

        if isinstance(value, bool):
            if value:
                return ReadWriteInhibit.RWINH_SET
            return ReadWriteInhibit.RWINH_RST

        raise KeyError(f"No {cls.__name__}")


class RL_EXPR(Symbol):
    """Possible lvalues for READ and rvalues for BROADCAST."""
    RL: str = RL


class GGL_EXPR(Symbol):
    """Special Token type for expressions that accept only GGL."""
    GGL: str = GGL


class LGL_EXPR(Symbol):
    """Special Token type for expressions that accept only LGL."""
    LGL: str = LGL


class RSP16_EXPR(Symbol):
    """Represents the lvalue of RSP16_ASSIGNMENT."""
    RSP16: str = RSP16


class RSP256_EXPR(Symbol):
    """Possible lvalues for RSP256_ASSIGNMENT."""
    RSP256: str = RSP256


class RSP2K_EXPR(Symbol):
    """Possible lvalues for RSP2K_ASSIGNMENT."""
    RSP2K: str = RSP2K


class RSP32K_EXPR(Symbol):
    """Possible lvalues for RSP32K_ASSIGNMENT."""
    RSP32K: str = RSP32K


class RSP16_RVALUE(Symbol):
    """Possible rvalues for RSP16_ASSIGNMENT."""
    RSP256: str = RSP256

    # NOTE: We do not include RL because it implies a BROADCAST
    # RL: str = RL


class RSP256_RVALUE(Symbol):
    """Possible rvalues for RSP256_ASSIGNMENT."""
    RSP16: str = RSP16
    RSP2K: str = RSP2K


class RSP2K_RVALUE(Symbol):
    """Possible rvalues for RSP2K_ASSIGNMENT."""
    RSP256: str = RSP256
    RSP32K: str = RSP32K


class RSP32K_RVALUE(Symbol):
    """Possible rvalues for RSP32K_ASSIGNMENT."""
    RSP2K: str = RSP2K


class BIT_EXPR(Symbol):
    """Possible bit values for the rvalue of READ."""

    ZERO: str = "0"
    ONE: str = "1"

    @classmethod
    def find_by_value(self: Type["BIT_EXPR"], value: Union[str, int]) -> "BIT_EXPR":
        """Finds the BIT_EXPR Token associated with the string name or int value requested.

        Parameters:
            value: Either the string name or int value for the needed BIT_EXPR Token.

        Returns:
            The BIT_EXPR Token associated with the given value."""

        value = str(value)
        for enum in self:
            if enum.value == value:
                return enum
        raise ValueError(f"No enum exists for value: {value}")


class SRC_EXPR(Symbol):
    """Available <SRC> values for READ and WRITE."""

    RL: str = RL
    NRL: str = NRL
    ERL: str = ERL
    WRL: str = WRL
    SRL: str = SRL
    GL: str = GL
    GGL: str = GGL
    RSP16: str = RSP16

    INV_RL: str = INV_RL
    INV_NRL: str = INV_NRL
    INV_ERL: str = INV_ERL
    INV_WRL: str = INV_WRL
    INV_SRL: str = INV_SRL
    INV_GL: str = INV_GL
    INV_GGL: str = INV_GGL
    INV_RSP16: str = INV_RSP16

    def __invert__(self: "SRC_EXPR") -> "UNARY_SRC":
        """Returns a negated (inverted) representation of the current SRC_EXPR.
        For example, `~RL` with return a representation equivalent to `INV_RL`.
        Likewise, `~INV_RL` will return a representation equivalent to `RL`."""
        return UNARY_SRC(expression=self, operator=UNARY_OP.NEGATE)


class BROADCAST_EXPR(Symbol):
    """Possible lvalues for BROADCAST."""
    GL: str = GL
    GGL: str = GGL
    RSP16: str = RSP16


class SPECIAL(Symbol):
    """Statements that don't belong anywhere else."""
    NOOP: str = NOOP
    FSEL_NOOP: str = FSEL_NOOP
    RSP_END: str = RSP_END
    RSP_START_RET: str = RSP_START_RET
    L2_END: str = L2_END


class Operator(Token):
    """Represents an n-ary operation, such as negation, assignment, or
    conjunction."""


class ASSIGN_OP(Operator):
    """Represents the various assignment operations.

        1. EQ      := direct assignment from the rvalue to the lvalue
                      (e.g. `x = y`).
        2. AND_EQ  := assigns the lvalue the conjunction of the lvalue and the
                      rvalue (e.g. `x &= y`).
        3. OR_EQ   := assigns the lvalue the disjunction of the lvalue and the
                      rvalue (e.g. `x |= y`).
        4. XOR_EQ  := assigns the lvalue the exclusive disjunction of the
                      lvalue and the rvalue (1s everywhere the lvalue is 1 or 0
                      and the rvalue is its complement, and 0s everywhere the
                      lvalue is equal to the rvalue) (e.g. `x ^= y`).
        5. COND_EQ := equivalent to OR_EQ at the logical level but potentially
                      implemented differently at the hardware level
                      (e.g. `x ?= y`)."""

    EQ: str = "="
    AND_EQ: str = "&="
    OR_EQ: str = "|="
    XOR_EQ: str = "^="
    COND_EQ: str = "?="


class BINOP(Operator):
    """Represents the various binary operations that may be performed on rvalue
    operands.

        1. AND := performs a conjunction (e.g. `x & y`).
        2. OR  := performs a disjunction (e.g. `x | y`).
        3. XOR := performs an exclusive disjunction (e.g. `x ^ y`)."""
    AND: str = "&"
    OR: str = "|"
    XOR: str = "^"


class UNARY_OP(Operator):
    """Negates the bits of the operand such that its 1s become 0s and its 0s
    become 1s, (e.g. `~0b10 == 0b01`)."""
    NEGATE: str = "~"


class FrozenValue(NamedTuple):
    """Used to serialize non-serializable types (e.g. dicts). `value` is the
    serialized value of the respective type and `kind` tells the deserializer
    how to return it to its original form."""
    value: Any
    kind: str


def instance_members_of(obj: Any) -> Iterator[Tuple[str, Any]]:
    """Yields (attr, value) pairs of all instance attributes of BLEIR types.

    You may define a __walk__ attribute containing a list of attribute names
    to walk in the specified order (e.g. Fragment or FragmentCaller).
    """

    # [!! NOTE !!] :: The first time I implemented this, I used the `inspect`
    # module, but it was very slow. Not only that but it returned all methods,
    # properties, etc. of `obj`. Studying how to make it faster, I discovered
    # the `__annotations__` attribute which contains exactly the information
    # needed to implement this function.  Not only is it fast but it returns
    # only the field annotations.

    if isinstance(obj, Enum):
        attrs = ["value"]
    elif hasattr(obj, "__walk__"):
        # TODO: Refer to __walk__ as a partial ordering over attributes rather
        # than an explicit ordering. This is to avoid the common regression of
        # adding new fields and forgetting to update the __walk__ attribute.
        attrs = obj.__walk__
    elif hasattr(obj, "__annotations__"):
        attrs = obj.__annotations__.keys()
    else:
        attrs = []

    for attr in attrs:
        yield attr, getattr(obj, attr)


def metadatable(meta_kind: Enum) -> Callable:
    """Specifies that a Type supports metadata. The type is required to define
    its own `metadata` dict because I have not been able to determine who to
    inject a field into @dataclass types such that they will be instantiate
    upon construction."""

    # TODO: Determine how (if possible) to dynamically add an attribute to the
    # class with a type hint. I would like to dynamically assign the `metadata`
    # attribute as a type Dict[meta_kind, Any].

    def has_metadata(self: Any,
                     key: Optional[meta_kind] = None,
                     value: Optional[Any] = None) -> bool:
        """Determines whether this instance has the specified key and value in
        its metadata (if provided). If no value is provided then the method
        looks for the existence of the metadata key. If no key is provided then
        the method checks whether the instance has metadata at all."""

        if not hasattr(self, "metadata"):
            return False

        has_metadata = self.metadata is not None
        if has_metadata and key is not None:
            has_metadata = key in self.metadata
            if has_metadata and value is not None:
                has_metadata = (self.metadata[key] == value)

        return has_metadata

    def get_metadata(self: Any, key: meta_kind, **kwargs) -> Any:
        """Returns the metadata value mapped-to by the key."""

        if self.has_metadata(key):
            return self.metadata[key]

        if "default_value" in kwargs:
            return kwargs["default_value"]

        raise KeyError(f"metadata has no entry for {key}")

    def wrapper(cls: Type) -> Type:
        cls.__metadatable__ = True
        cls.has_metadata = has_metadata
        cls.get_metadata = get_metadata
        return cls

    return wrapper


def immutable(cls: Type) -> Type:
    """Specifies the cls is immutable and provides helper methods to examine
    and update its attributes. Since the cls is immutable its attributes may
    not be updated in-place. Rather, a new instance having the updated
    attributes will be returned."""

    def attr_map(self: cls, kind: Optional[Type] = None) -> Dict[str, Any]:
        """Returns a mapping of field names to their corresponding values."""

        attr_values = OrderedDict()

        this = self if kind is None else kind

        if kind is None:
            kind = self.__class__

        for base in reversed(kind.__bases__):
            if hasattr(base, "__annotations__"):
                base_attr_values = attr_map(self, base)
                attr_values.update(base_attr_values)

        for attr, type_hint in this.__annotations__.items():
            value = getattr(self, attr)
            attr_values[attr] = value

        return attr_values

    def having(self: cls, **kwargs) -> cls:
        """Returns a new instance of cls having the original's attributes
        updated with the new values."""

        attr_map = self.attr_map()

        for attr, value in kwargs.items():
            if attr not in attr_map:
                raise AssertionError(f"Expected \"{attr}\" to be among {attr_map.keys()}")
            attr_map[attr] = value

        return cls(**attr_map)

    cls.__immutable__ = True
    cls.attr_map = attr_map
    cls.having = having
    return cls


def stateful(cls):
    """Specifies that cls can be serialized and deserialized in a Pythonic
    manner. Helper methods, __getstate__ and __setstate__, are defined for the
    process."""

    def _encode_dict(d):
        """Helper method to encode Python dicts in a hash-friendly manner."""
        attrs = {}
        for attr, value in d.items():
            if isinstance(value, list):
                attrs[attr] = FrozenValue(tuple(value), 'list')
            elif isinstance(value, dict):
                attrs[attr] = _encode_dict(value)
            elif isinstance(value, np.ndarray):
                attrs[attr] = FrozenValue(value.data.tobytes(), 'numpy.uint16')
            else:
                attrs[attr] = value
        return FrozenValue(tuple(attrs.items()), 'dict')

    def _decode_dict(d):
        """Helper method to return an encoded Python dict to its original
        form."""
        attrs = {}
        for attr, value in d.items():
            if isinstance(value, FrozenValue):
                if value.kind == 'list':
                    attrs[attr] = list(value.value)
                elif value.kind == 'dict':
                    attrs[attr] = _decode_dict(value.value)
                elif value.kind == 'numpy.uint16':
                    attrs[attr] = np.frombuffer(value.value, dtype=np.uint16)
                else:
                    raise NotImplementedError(
                        f"Unsupported value type "
                        f"({value.__class__.__name__}): {value}")
            else:
                attrs[attr] = value
        return attrs

    def __getstate__(self):
        """Returns a tuple consisting of a mapping from attributes to values of
        the respective object.

        See: https://docs.python.org/3/library/pickle.html#object.__getstate__"""

        attrs = {}
        for attr, value in instance_members_of(self):
            if isinstance(value, list):
                attrs[attr] = FrozenValue(tuple(value), 'list')
            elif isinstance(value, dict):
                attrs[attr] = _encode_dict(value)
            elif isinstance(value, np.ndarray):
                attrs[attr] = FrozenValue(value.data.tobytes(), 'numpy.uint16')
            else:
                attrs[attr] = value
        return tuple(attrs.items())

    def __setstate__(self, state):
        """Restores the state of this object from a tuple consisting of a
        mapping from attributes to values.

        See: https://docs.python.org/3/library/pickle.html#object.__setstate__"""

        attrs = dict(state)
        for attr, value in attrs.items():
            if isinstance(value, FrozenValue):
                if value.kind == 'list':
                    setattr(self, attr, list(value.value))
                elif value.kind == 'dict':
                    setattr(self, attr, _decode_dict(value.value))
                elif value.kind == 'numpy.uint16':
                    setattr(self, attr, np.frombuffer(value.value, dtype=np.uint16))
                else:
                    raise NotImplementedError(f"Unsupported value type ({value.__class__.__name__}): {value}")
            else:
                setattr(self, attr, value)

    cls.__stateful__ = True
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
    return cls


def hashable(cls):
    """Defines common methods for BLEIR types that have dictionary or list
    attributes, which are not hashable without coercion.  Please note that
    __hash__ depends on the value of __getstate__ being hashable."""

    def __eq__(this: cls, that: Any) -> bool:
        """Returns whether this cls instance is equivalent to the other in
        terms of fields and values."""
        if this is that:
            return True
        if not isinstance(that, type(this)):
            return False
        if hasattr(this, "__eq_excludes__"):
            exclusions = this.__eq_excludes__
        else:
            exclusions = []
        for attr, this_value in instance_members_of(this):
            if attr == "metadata" or attr in exclusions:
                continue
            that_value = getattr(that, attr)
            if this_value is None and that_value is None:
                continue
            elif this_value is None and that_value is not None:
                return False
            elif this_value is not None and that_value is None:
                return False
            elif isinstance(this_value, np.ndarray):
                if not np.array_equal(this_value, that_value):
                    return False
            elif not isinstance(this_value, dict) \
                    and hasattr(this_value, "__getitem__") \
                    and hasattr(this_value, "__len__") \
                    and not isinstance(that_value, dict) \
                    and hasattr(that_value, "__getitem__") \
                    and hasattr(that_value, "__len__"):
                # this is important since lists and tuples will never equal each other even if they
                # contain the same elements
                if len(this_value) != len(that_value):
                    return False
                for this_elem, that_elem in zip(this_value, that_value):
                    if this_elem != that_elem:
                        return False
            elif this_value != that_value:
                return False
        return True

    def __ne__(this: cls, that: Any) -> bool:
        """Returns whether this cls instance is inequivalent to the other in
        terms of fields and values."""
        return not this.__eq__(that)

    def __hash__(self: cls) -> int:
        """Returns in integer representing a hash code over this cls' fields
        and values. The hash code is drawn over the state object returned from
        __getstate__(), so __getstate__() must return a hashable structure."""
        try:
            return hash(self.__getstate__())
        except Exception as e:
            raise e

    cls.__hashable__ = True
    cls.__eq__ = __eq__
    cls.__ne__ = __ne__
    cls.__hash__ = __hash__
    return cls


def bleir_dataclass(cls):
    """Shared set of decorators for most BLEIR @dataclass classes."""
    cls = dataclass(cls)
    cls = stateful(cls)
    cls = hashable(cls)
    cls = immutable(cls)
    return cls


def seqable(attr):
    """Defines common methods for BLEIR types that serve as wrappers over a
    Sequence."""

    def __len__(self):
        value = getattr(self, attr)
        return len(value)

    def __getitem__(self, index):
        value = getattr(self, attr)
        return value[index]

    def decorator(cls):
        cls.__len__ = __len__
        cls.__getitem__ = __getitem__
        return cls

    return decorator


def iterable(attr):
    """Defines common methods for BLEIR types that serve as wrappers over an
    Iterable."""

    def __iter__(self):
        value = getattr(self, attr)
        return iter(value)

    def decorator(cls):
        cls.__iter__ = __iter__
        return cls

    return decorator


def collectible(attr):
    """Specifies that cls represents an iterable and seqable collection over
    its field identified by `attr`."""

    def decorator(cls):
        nonlocal attr
        cls = iterable(attr)(cls)
        cls = seqable(attr)(cls)
        return cls

    return decorator


@immutable
class SingleLineComment(NamedTuple):
    """Comments that appear by themselves on a single-line, like so:

        /* this is a single-line comment */

    Parameters:
        line: string content of the comment."""

    line: str

    def __str__(self: "SingleLineComment") -> str:
        return f"/* {self.line} */"


@bleir_dataclass
@collectible("lines")
class MultiLineComment:
    """Comments that appear by themselves over multiple lines, like so:

        /**
         * this is
         * a
         * multi-line
         * comment
         */

    Parameters:
        lines: sequence of strings to print, one per line."""

    lines: Sequence[str]

    def __str__(self: "MultiLineComment") -> str:
        lines = "\n".join(f" * {line}" for line in self.lines)
        return f"/**\n{lines}\n */"


@immutable
class TrailingComment(NamedTuple):
    """Comments that appear after statements, like so:

        msk_rp: RL = 1;  /* assign 1 to RL guided by msk_rp */

    Parameters:
        value: string content of the comment."""

    value: str

    def __str__(self: "TrailingComment") -> str:
        return f"/* {self.value} */"


@immutable
class InlineComment(NamedTuple):
    """Comments that appear, intermingled, within expressions, parameters,
    etc., like so:

        APL_FRAG foo(RN_REG lvr_rp /* lvalue */,
                     RN_REG rvr_rp /* rvalue */,
                     SM_REG msk_rp /* section mask */)

    Parameters:
        value: string content of the comment."""

    value: str

    def __str__(self: "InlineComment") -> str:
        return f"/* {self.value} */"


LineComment = Union[MultiLineComment, SingleLineComment]


# InlineComment is intentionally left out of the Union because it won't be used
# where this type variable is referenced (propose a better name?)
Comment = Union[LineComment, TrailingComment]


def register_parameter(cls):
    """Annotates register parameter types with common properties and
    methods."""

    def register_type(self) -> str:
        return cls.__name__

    def value_param(self) -> str:
        # return f"{self.identifier}_vp"
        return f"{self.identifier}"

    def register_param(self) -> str:
        # return f"{self.identifier}_rp"
        return f"{self.identifier}"

    def __str__(self) -> str:
        if self.is_lowered:
            return self.identifier
        return self.register_param

    cls.register_type = property(register_type)
    cls.value_param = property(value_param)
    cls.register_param = property(register_param)
    cls.__str__ = __str__

    return cls


@immutable
@register_parameter
class RN_REG(NamedTuple):
    """Identifies a vector register (VR).

    Parameters:
        identifier: name representing the VR, e.g. `lvr`.
        comment: optional comment to associate with this RN_REG.
        initial_value: (optional) initial value to write across all plats of
                       the VR associated with this register (identified by its
                       row number). RN_REGs with initial values do not have
                       corresponding parameters. Instead, they will be
                       initialized as local constant variables within the
                       fragment caller body.
        register: pre-defined register id to allocate for this RN_REG.
        row_number: constant row number to assign the corresponding RN_REG (not
                    its corresponding VR value).
        is_lowered: whether this register is lowered, or referenced from a
                    global context instead of from the parameter list or local
                    variables. Lowering is the opposite of lambda lifting (see:
                    https://en.wikipedia.org/wiki/Lambda_lifting).
        is_literal: whether this instance represents an RN_REG literal and not
                    a variable (e.g. RN_REG_0).
        is_temporary: Whether this register is part of the fragment but not its
                      caller."""

    identifier: str
    comment: Optional[InlineComment] = None
    initial_value: Optional[int] = None
    register: Optional[int] = None
    row_number: Optional[int] = None
    is_lowered: bool = False
    is_literal: bool = False
    is_temporary: bool = False

    @property
    def operands(self: "RN_REG") -> List[str]:
        return [self.identifier]


@immutable
@register_parameter
class RE_REG(NamedTuple):
    """Identifies a special type of register that represents combinations over
    all 24 VRs.

    Parameters:
        identifier: name representing the register parameter.
        comment: optional comment to associate with this RE_REG.
        row_mask: 24-bit mask representing the combination of VRs associated
                  with this register.
        rows: a sequence of RE_REGs and RN_REGs to combine into a single mask.
        register: pre-defined register id to allocate for this RE_REG.
        is_lowered: whether this register is lowered, or referenced from a
                    global context instead of from the parameter list or local
                    variables. Lowering is the opposite of lambda lifting (see:
                    https://en.wikipedia.org/wiki/Lambda_lifting).
        is_literal: whether this instance represents an RE_REG literal and not
                    a variable (e.g. RE_REG_0)."""

    identifier: str
    comment: Optional[InlineComment] = None
    row_mask: Optional[int] = None  # 24-bit integer
    rows: Optional[Sequence[Union[RN_REG, "RE_REG"]]] = None
    register: Optional[int] = None
    is_lowered: bool = False
    is_literal: bool = False

    @property
    def operands(self: "RE_REG") -> List[str]:
        if self.rows is None:
            return [self.identifier]
        return [operand
                for row in self.rows
                for operand in row.operands]


@immutable
@register_parameter
class EWE_REG(NamedTuple):
    """Identifies a special type of register that represents combinations over
    half word-lines (1024 of the 2048 plats).

    Parameters:
        identifier: name representing the register parameter.
        comment: optional comment to associate with this EWE_REG.
        wordline_mask: 10-bit mask representing the combination of plats
                       associated with this register.
        register: pre-defined register id to allocate for this EWE_REG.
        is_lowered: whether this register is lowered, or referenced from a
                    global context instead of from the parameter list or local
                    variables. Lowering is the opposite of lambda lifting (see:
                    https://en.wikipedia.org/wiki/Lambda_lifting).
        is_literal: whether this instance represents an EWE_REG literal and not
                    a variable (e.g. EWE_REG_0)."""

    identifier: str
    comment: Optional[InlineComment] = None
    wordline_mask: Optional[int] = None  # 10-bit integer
    register: Optional[int] = None
    is_lowered: bool = False
    is_literal: bool = False

    @property
    def operands(self: "EWE_REG") -> List[str]:
        return [self.identifier]


@bleir_dataclass
class ExtendedRegister:
    register: Union[RE_REG, EWE_REG, "ExtendedRegister"]
    operator: Optional[UNARY_OP] = None
    num_shifted_bits: int = 0

    @property
    def operands(self: "ExtendedRegister") -> List[str]:
        if isinstance(self.register, RE_REG) and self.register.rows is not None:
            return self.register.operands
        return [self.register.identifier]

    @property
    def identifier(self: "ExtendedRegister") -> str:
        return self.register.identifier

    @property
    def register_type(self: "ExtendedRegister") -> str:
        return self.register.register_type

    @property
    def value_param(self: "ExtendedRegister") -> str:
        return self.register.value_param

    @property
    def register_param(self: "ExtendedRegister") -> str:
        return self.register.register_param

    def __post_init__(self: "ExtendedRegister") -> None:
        if isinstance(self.register, ExtendedRegister):
            xe_reg = self.register
            self.register = xe_reg.register
            self.operator ^= xe_reg.operator
            self.num_shifted_bits += xe_reg.num_shifted_bits

    def __str__(self: "ExtendedRegister") -> str:
        representation = str(self.register)
        if self.num_shifted_bits > 0:
            representation = f"{representation}<<{self.num_shifted_bits}"
        if self.operator is UNARY_OP.NEGATE:
            representation = f"~({representation})"
        return representation


Offsets = Union[Tuple[int],
                Tuple[int, int],
                Tuple[int, int, int]]


@bleir_dataclass
@register_parameter
class L1_REG:
    """Identifies a register associated with the level of memory just above
    MMB. This level is used for I/O and spilling and restoring MMB registers.
    It is slower to access this level than MMB but not as slow as accessing L2.

    Parameters:
        identifier: name representing the register parameter.
        comment: optional comment to associate with this register parameter.
        register: pre-defined register id to allocate for this register
                  parameter.
        bank_group_row: 13-bit scalar representing the encoded bank_id,
                        group_id, and row_id used to identify the memory for
                        this register parameter. A bank_id is comprised of 2
                        bits, a group_id is comprised of 2 bits, and a row_id
                        is comprised of 9 bits. The various ids are combined as
                        follows:

                            bank_group_row = (bank_id << 11) \\
                                           | (group_id << 9) \\
                                           | row_id.

                        It is important to understand that although the maximum
                        value of the bank_group_row is (1 << 13) = 8192, the
                        maximum value supported by the hardware is 6144.
        is_lowered: whether this register is lowered, or referenced from a
                    global context instead of from the parameter list or local
                    variables. Lowering is the opposite of lambda lifting (see:
                    https://en.wikipedia.org/wiki/Lambda_lifting).
        is_literal: whether this instance represents a register literal and not
                    a variable (e.g. L1_ADDR_REG_0)."""

    identifier: str
    comment: Optional[InlineComment] = None
    register: Optional[int] = None
    bank_group_row: Optional[Union[int, Tuple[int, int, int]]] = None  # 13-bit integer
    is_lowered: bool = False
    is_literal: bool = False

    def __post_init__(self: "L1_REG") -> None:
        if isinstance(self.bank_group_row, tuple):
            bank, group, row = self.bank_group_row
            self.bank_group_row = ((bank << 11) | (group << 9) | row)

    @property
    def operands(self: "L1_REG") -> List[str]:
        return [self.identifier]

    @property
    def bank_id(self: "L1_REG") -> Optional[int]:
        if self.bank_group_row is None:
            return None
        return self.bank_group_row >> 11

    @property
    def group_id(self: "L1_REG") -> Optional[int]:
        if self.bank_group_row is None:
            return None
        return (self.bank_group_row >> 9) & ((1 << 2) - 1)

    @property
    def row_id(self: "L1_REG") -> Optional[int]:
        if self.bank_group_row is None:
            return None
        return self.bank_group_row & ((1 << 9) - 1)

    def __add__(self: "L1_REG", offsets: Union[Offsets, int]) -> "LXRegWithOffsets":
        if isinstance(offsets, int):
            row_id = offsets
            offsets = (row_id,)
        return LXRegWithOffsets(self, *reversed(offsets))


@bleir_dataclass
@register_parameter
class L2_REG:
    """Identifies a register associated with the level of memory just above L1.
    This level is used for I/O as data must be written to L2 before L1 and
    vice-versa.

    Parameters:
        identifier: name representing the register parameter.
        comment: optional comment to associate with this register parameter.
        register: pre-defined register id to allocate for this register
                  parameter.
        value: register value to assign this parameter.
        is_lowered: whether this register is lowered, or referenced from a
                    global context instead of from the parameter list or local
                    variables. Lowering is the opposite of lambda lifting (see:
                    https://en.wikipedia.org/wiki/Lambda_lifting).
        is_literal: whether this instance represents a register literal and not
                    a variable (e.g. L2_ADDR_REG_0)."""

    identifier: str
    comment: Optional[InlineComment] = None
    register: Optional[int] = None
    value: Optional[int] = None  # 7-bit integer
    is_lowered: bool = False
    is_literal: bool = False

    def __add__(self: "L2_REG", offsets: Union[Offsets, int]) -> "LXRegWithOffsets":
        if isinstance(offsets, int):
            row_id = offsets
            offsets = (row_id,)
        return LXRegWithOffsets(self, *reversed(offsets))

    @property
    def operands(self: "L2_REG") -> List[str]:
        return [self.identifier]


LXParameter = Union[L1_REG, L2_REG]

# TODO: Merge offset logic into L1/L2_REG types to reduce code (give them optional offsets)

@bleir_dataclass
class LXRegWithOffsets:
    """Represents an offset L1_REG or L2_REG. The offset may contain a bank,
    group, and row only for L1_REG. If it is for an L2_REG it may contain only
    a row offset. Each L1_REG and L2_REG value represents a coordinate to which
    the offset values are added. Offsets avoid unnecessary parameters.

    Parameters:
        parameter: the L1_REG or L2_REG to which the offset is applicable.
        row_id: an offset added to the row coordinate of the LX_REG.
        group_id: an offset added to the group coordinate of the LX_REG.
        bank_id: an offset added to the bank coordinate of the LX_REG.
        comment: an optional comment describing this offset."""

    parameter: Union[LXParameter, "LXRegWithOffsets"]
    row_id: int
    group_id: int = 0
    bank_id: int = 0
    comment: Optional[Comment] = None

    def __post_init__(self: "LXRegWithOffsets") -> None:
        while isinstance(self.parameter, LXRegWithOffsets):
            self.row_id += self.parameter.row_id
            self.group_id += self.parameter.group_id
            self.bank_id += self.parameter.bank_id
            self.parameter = self.parameter.parameter

    @property
    def operands(self: "LXRegWithOffsets") -> List[str]:
        if self.row_id == 0 and self.group_id == 0 and self.bank_id == 0:
            return self.parameter.operands
        return [f"{self.parameter.identifier} + ({self.bank_id}, {self.group_id}, {self.row_id})"]

    @property
    def identifier(self: "LXRegWithOffsets") -> str:
        return self.parameter.identifier

    @property
    def register_type(self: "LXRegWithOffsets") -> str:
        return self.parametere.register_type

    @property
    def value_param(self: "LXRegWithOffsets") -> str:
        return self.parameter.value_param

    @property
    def register_param(self: "LXRegWithOffsets") -> str:
        return self.parameter.register_param

    @property
    def offset(self: "LXRegWithOffsets") -> int:
        return self.row_id \
            | (self.group_id << 9) \
            | (self.bank_id << 11)

    def __str__(self: "LXRegWithOffsets") -> str:
        if self.bank_id > 0:
            offsets = f"{self.bank_id},{self.group_id},{self.row_id}"
        elif self.group_id > 0:
            offsets = f"{self.group_id},{self.row_id}"
        else:
            offsets = f"{self.row_id}"

        return f"{self.identifier} + {offsets}"


LX_ADDR = Union[LXParameter, LXRegWithOffsets]


@immutable
class GGL_ASSIGNMENT(NamedTuple):
    """Represents an assignment to GGL from an LX register. It is important to
    note that only L1_REG is supported. Assigning to GGL from L2 is untested
    and may not work. The LX_ADDR type was extracted from the APL grammar and
    may not semantically apply to both L1 and L2.

    Parameters:
        rvalue: The L1_REG or L2_REG (or an offset) to assign GGL."""

    rvalue: LX_ADDR

    @property
    def lvalue(self: "GGL_ASSIGNMENT") -> GGL_EXPR:
        return GGL_EXPR.GGL

    @property
    def operator(self: "GGL_ASSIGNMENT") -> ASSIGN_OP:
        return ASSIGN_OP.EQ

    @property
    def left_operands(self: "GGL_ASSIGNMENT") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "GGL_ASSIGNMENT") -> List[str]:
        return self.rvalue.operands

    def __str__(self: "GGL_ASSIGNMENT") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
class LGL_ASSIGNMENT(NamedTuple):
    """Assigns to LGL the memory associated with the corresponding L1_REG,
    L2_REG, or an offset of either. LGL is used to shuttle data between L1 and
    L2 registers.

    Parameters:
        rvalue: Either an L1_REG, L2_REG, or offset to assign LGL."""

    rvalue: Union[LXParameter, LXRegWithOffsets]

    @property
    def lvalue(self: "LGL_ASSIGNMENT") -> LGL_EXPR:
        return LGL_EXPR.LGL

    @property
    def operator(self: "LGL_ASSIGNMENT") -> ASSIGN_OP:
        return ASSIGN_OP.EQ

    @property
    def left_operands(self: "LX_ASSIGNMENT") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "LX_ASSIGNMENT") -> List[str]:
        return self.rvalue.operands

    def __str__(self: "LGL_ASSIGNMENT") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
class LX_ASSIGNMENT(NamedTuple):
    """Assigns to either an L1_REG or L2_REG the value contained within GGL or
    LGL. It is important to note that a GGL rvalue is only applicable to L1,
    but LGL may apply to either L1 or L2.

    Parameters:
        lvalue: the L1 or L2 register recipient.
        rvalue: the storage device of the data, either GGL or LGL."""

    lvalue: LX_ADDR
    rvalue: Union[GGL_EXPR, LGL_EXPR]

    @property
    def operator(self: "LX_ASSIGNMENT") -> ASSIGN_OP:
        return ASSIGN_OP.EQ

    @property
    def left_operands(self: "LX_ASSIGNMENT") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "LX_ASSIGNMENT") -> List[str]:
        return self.rvalue.operands

    def __str__(self: "LX_ASSIGNMENT") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
@register_parameter
class SM_REG(NamedTuple):
    """Identifies a section mask (SM).

    Parameters:
        identifier: name representing the SM, e.g. `fs`.
        comment: optional comment to associate with this SM_REG.
        negated: helper attribute for parsing APL, specifies whether this
                 SM_REG should be negated (have its 1s flipped to 0s, and
                 vice-versa).
        constant_value: optional constant value to assign this register.
                        SM_REGs with constant values do not have corresponding
                        parameters generated for them. Instead, they will be
                        allocated within the fragment caller body.
        register: pre-defined register to allocate for this SM_REG.
        is_section: whether this SM_REG should be treated as a section value
                    instead of a mask value. SM_REGs marked as sections will
                    have their corresponding actual (section) values
                    transformed into mask values as follows:
                    sm_vp = (0x0001 << sm_vp).
        is_lowered: whether this register is lowered, or referenced from a
                    global context instead of from the parameter list or local
                    variables. Lowering is the opposite of lambda lifting (see:
                    https://en.wikipedia.org/wiki/Lambda_lifting).
        is_literal: whether this instance represents a register literal and not
                    a variable (e.g. SM_REG_0)."""

    identifier: str
    comment: Optional[InlineComment] = None
    negated: Optional[bool] = False
    constant_value: Optional[int] = None
    register: Optional[int] = None
    is_section: bool = False
    is_lowered: bool = False
    is_literal: bool = False

    def __invert__(self: "SM_REG") -> "MASK":
        return MASK(expression=self,
                    operator=UNARY_OP.NEGATE)

    def __lshift__(self: "SM_REG", num_bits: int) -> "SHIFTED_SM_REG":
        return SHIFTED_SM_REG(self, num_bits)

    @property
    def is_constant(self: "SM_REG") -> bool:
        return self.constant_value is not None

    # helper method
    def resolve(self: "SM_REG", mask: int) -> int:
        return mask


RE_REG_ID = re.compile(r".*?(\d+)")
RegisterParameter = Union[RN_REG,
                          RE_REG,
                          EWE_REG,
                          L1_REG,
                          L2_REG,
                          SM_REG]
FormalParameter = RegisterParameter


Glassible = Union[
    EWE_REG,
    L1_REG,
    L2_REG,
    LGL_EXPR,
    LXRegWithOffsets,
    RE_REG,
    RN_REG,
    RSP256_EXPR,
    RSP2K_EXPR,
    RSP32K_EXPR,
    SRC_EXPR,
]


class GlassFormat(BleirEnum):
    BIN = "bin"
    HEX = "hex"


class GlassOrder(BleirEnum):
    LEAST_SIGNIFICANT_BIT_FIRST = "lsb"
    MOST_SIGNIFICANT_BIT_FIRST = "msb"


@immutable
class GlassStatement(NamedTuple):
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

    @property
    def right_operands(self: "GlassStatement") -> List[str]:
        return self.subject.operands

    def __str__(self: "GlassStatement") -> str:
        return f"Glass[{self.subject.name}]"


@immutable
class AllocatedRegister(NamedTuple):
    """Represents a register parameter that has been assigned a specific
    register, e.g. RN_REG_0 or SM_REG_11.

    Parameters:
        parameter: RN_REG or SM_REG being assigned a register.
        register: name of the specific register being assigned.
        comment: optional comment to associate with this AllocatedRegister."""

    parameter: RegisterParameter
    register: str
    comment: Optional[Comment] = None

    @property
    def identifier(self: "AllocatedRegister") -> str:
        return self.parameter.identifier

    @property
    def register_type(self: "AllocatedRegister") -> str:
        return self.parameter.register_type

    @property
    def value_param(self: "AllocatedRegister") -> str:
        return self.parameter.value_param

    @property
    def register_param(self: "AllocatedRegister") -> str:
        return self.parameter.register_param

    @property
    def reg_id(self: "AllocatedRegister") -> int:
        global RE_REG_ID

        if self.parameter.register is not None:
            return self.parameter.register

        match = RE_REG_ID.fullmatch(self.register)
        if match is None:
            raise RuntimeError(f"Cannot determine reg_id for register: {self.register}")

        reg_id = match.group(1)
        return int(reg_id)

    def isa(self: "AllocatedRegister", kind: Type[FormalParameter]) -> bool:
        return isinstance(self.parameter, kind)

    def __str__(self: "AllocatedRegister") -> str:
        return f"{self.parameter}={self.register}"


@bleir_dataclass
class ValueParameter:
    """Represents the value of an `Example` parameter and references the vector
    register that should receive it.

    Parameters:
        identifier: name of this parameter.
        row_number: identifies the vector register that should hold the value.
        value: numpy array of 2048, 16-bit unsigned integers."""

    identifier: str
    row_number: int
    value: np.ndarray  # shape=(2048,), dtype=np.uint16


@bleir_dataclass
class Example(Sequence):
    """Contains the initial and expected final states for a single execution of
    the application.

    Parameters:
        expected_value: expected output of a single VR.
        parameters: inital state of the application."""

    expected_value: ValueParameter
    parameters: Sequence[ValueParameter] = field(default_factory=tuple)

    def __len__(self: "Example") -> int:
        if self.parameters is not None:
            return 1 + len(self.parameters)
        return 1

    def __getitem__(self: "Example", index: int) -> ValueParameter:
        if index == 0:
            return self.expected_value
        return self.parameters[index - 1]

    def __iter__(self: "Example") -> Iterator[ValueParameter]:
        for index in range(len(self)):
            yield self[index]


ValueParameters = Tuple[str, int, np.ndarray]
ExampleParameters = Tuple[ValueParameters, Optional[Sequence[ValueParameters]]]


def build_examples(examples_parameters: Sequence[ExampleParameters]) -> Sequence[Example]:
    """Helper function used to construct a sequence of Examples for a Snippet,
    with the given parameters."""

    examples = []

    for example_parameters in examples_parameters:
        if not 1 <= len(example_parameters) <= 2:
            raise ValueError(
                f"Expected example_parameters to have a single expected value "
                f"and an arbitrary number of parameters: {example_parameter}")

        value_parameters = example_parameters[0]
        expected_value = ValueParameter(*value_parameters)

        parameters = []
        if len(example_parameters) == 2:
            for value_parameters in example_parameters[1]:
                value_parameter = ValueParameter(*value_parameters)
                parameters.append(value_parameter)

        example = Example(expected_value=expected_value, parameters=parameters)
        examples.append(example)

    return examples


@immutable
class UNARY_SRC(NamedTuple):
    """Optionally transforms a <SRC> with some unary operator.

    A UNARY_SRC may take any of the following forms:
        1. <SRC>
        2. ~<SRC>

    Parameters:
        expression: recipient (operand) of the provided operator.
        operator: performs an optional transformation on the expression."""

    expression: SRC_EXPR
    operator: Optional[UNARY_OP] = None

    @property
    def operands(self: "UNARY_SRC") -> List[str]:
        return self.expression.operands

    def __str__(self: "UNARY_SRC") -> str:
        if self.operator is not None:
            return f"{self.operator}{self.expression}"
        return f"{self.expression}"

    def __invert__(self: "UNARY_SRC") -> "UNARY_SRC":
        if self.operator is UNARY_OP.NEGATE:
            return self.having(operator=None)
        return self.having(operator=UNARY_OP.NEGATE)


SBParameter = Union[RN_REG, RE_REG, EWE_REG, ExtendedRegister]


@bleir_dataclass
@collectible("parameters")
class SB_EXPR:
    """Collection of 1-to-3 RN_REGs. When used as the lvalue of a WRITE, each
    register mapped to by the RN_REGs receives the same value. When used as the
    rvalue of a READ, the registers are conjoined togeter to form a single
    rvalue operand.

    An SB_EXPR takes any of the following forms:
        1. SB[x]
        2. SB[x,y]
        3. SB[x,y,z]
        4. SB:varnym
        5. SB:varnym<<num_bits
        6. SB:~(varnym)
        7. SB:~(varnym<<num_bits)

    - The form SB[x] is equivalent to SB[x,x,x].
    - The form SB[x,y] is equivalent to SB[x,y,y].
    - The form SB[x,y,z] is equivalent to itself.

    Parameters:
       parameters: A collection of a single RE_REG or 1-to-3 RN_REGs."""

    parameters: Union[Sequence[RN_REG],
                      Tuple[RE_REG],
                      Tuple[EWE_REG],
                      Tuple[ExtendedRegister]]

    def isa(self: "SB_EXPR",
            kind_or_kinds: Union[Type, Sequence[Type]]) -> bool:
        return isinstance(self.parameters[0], kind_or_kinds)

    @property
    def operands(self: "SB_EXPR") -> List[str]:
        return [operand
                for parameter in self.parameters
                for operand in parameter.operands]

    def __invert__(self: "SB_EXPR") -> "UNARY_SB":
        return UNARY_SB(expression=self,
                        operator=UNARY_OP.NEGATE)

    def __and__(this: "SB_EXPR", that: "SB_EXPR") -> Set[RN_REG]:
        return set(this.parameters) & set(that.parameters)

    def __or__(this: "SB_EXPR", that: "SB_EXPR") -> Set[RN_REG]:
        return set(this.parameters) | set(that.parameters)

    def __str__(self: "SB_EXPR") -> str:
        registers = ",".join(map(str, self.parameters))
        parameter = self.parameters[0]
        if isinstance(parameter, RN_REG):
            return f"SB[{registers}]"
        elif isinstance(parameter, (RE_REG, EWE_REG, ExtendedRegister)):
            register = registers
            return f"SB:{register}"
        else:
            raise AssertionError(
                f"Unsupported register type ({parameter.__class__.__name__})")


@immutable
class UNARY_SB(NamedTuple):
    """Optionally transforms an SB_EXPR operand with some unary operator.

    A UNARY_SB takes either of the following forms:
        1. <SB>
        2. ~<SB>

    Parameters:
        expression: transformable SB_EXPR operand.
        operator: optional unary operator to transform the expression."""

    expression: SB_EXPR
    operator: Optional[UNARY_OP] = None

    @property
    def operands(self: "UNARY_SB") -> List[str]:
        return self.expression.operands

    def __invert__(self: "UNARY_SB") -> "UNARY_SB":
        if self.operator is UNARY_OP.NEGATE:
            return UNARY_SB(expression=self.expression,
                            operator=None)
        return UNARY_SB(expression=self.expression,
                        operator=UNARY_OP.NEGATE)

    def __str__(self: "UNARY_SB") -> str:
        if self.operator is not None:
            return f"{self.operator}{self.expression}"
        return f"{self.expression}"


@immutable
class UNARY_EXPR(NamedTuple):
    """Type alias surrounding each of the transformable unary operands.

    A UNARY_EXPR takes any of the following forms:
        1. <SB>
        2. ~<SB>
        3. <SRC>
        4. ~<SRC>
        5. 1
        6. 0

    Parameters:
        expression: transformable unary expression."""

    expression: Union[UNARY_SB, UNARY_SRC, BIT_EXPR]

    @property
    def operands(self: "UNARY_EXPR") -> List[str]:
        return self.expression.operands

    def __str__(self: "UNARY_EXPR") -> str:
        return f"{self.expression}"


@immutable
class BINARY_EXPR(NamedTuple):
    """Represents a binary operation performed over two unary operands. The
    left_operand is always an <SB> or ~<SB> and the right_operand is always a
    <SRC> or ~<SRC>.

    A BINARY_EXPR takes any of the following forms:
        1. <SB> <op> <SRC>
        2. ~<SB> <op> <SRC>
        3. <SB> <op> ~<SRC>
        4. ~<SB> <op> ~<SRC>

    Parameters:
        operator: binary operator to apply to the operands.
        left_operand: appears on the left side of the operation.
        right_operand: appears on the right side of the operation."""

    operator: BINOP
    left_operand: Union[UNARY_SB, RL_EXPR]
    right_operand: Union[UNARY_SRC, L1_REG, L2_REG, LXRegWithOffsets]

    @property
    def operands(self: "BINARY_EXPR") -> List[str]:
        left_operands = self.left_operand.operands
        right_operands = self.right_operand.operands
        return left_operands + right_operands

    def __str__(self: "BINARY_EXPR") -> str:
        return f"{self.left_operand} {self.operator} {self.right_operand}"


@immutable
class READ(NamedTuple):
    """Assigns the rvalue to RL.

    Per the READ LOGIC table of the README.md, a READ may take any of the
    following forms:

        +----------------------------------|-------------------------+
        | APL                              | BEL                     |
        +----------------------------------|-------------------------+
        |      immediate APL commands      | op  arg1                |
        +----------------------------------|-------------------------+
        |  1.  msk: RL  = 0                | :=   0                  |
        |  2.  msk: RL  = 1                | :=   1                  |
        +----------------------------------|-------------------------+
        |      combining APL commands      | op  arg1   comb  arg2   |
        +----------------------------------|-------------------------+
        |  3.  msk: RL  =  <SB>            | :=  <SB>                |
        |  4.  msk: RL  =  <SRC>           | :=               <SRC>  |
        |  5.  msk: RL  =  <SB> &  <SRC>   | :=  <SB>    &    <SRC>  |
        |                                  |                         |
        | 10.  msk: RL |=  <SB>            | |=  <SB>                |
        | 11.  msk: RL |=  <SRC>           | |=               <SRC>  |
        | 12.  msk: RL |=  <SB> &  <SRC>   | |=  <SB>    &    <SRC>  |
        |                                  |                         |
        | 13.  msk: RL &=  <SB>            | &=  <SB>                |
        | 14.  msk: RL &=  <SRC>           | &=               <SRC>  |
        | 15.  msk: RL &=  <SB> &  <SRC>   | &=  <SB>    &    <SRC>  |
        |                                  |                         |
        | 18.  msk: RL ^=  <SB>            | ^=  <SB>                |
        | 19.  msk: RL ^=  <SRC>           | ^=               <SRC>  |
        | 20.  msk: RL ^=  <SB> &  <SRC>   | ^=  <SB>    &    <SRC>  |
        +----------------------------------|-------------------------+
        |      special cases               | op  arg1   comb  arg2   |
        +----------------------------------|-------------------------+
        |  6.  msk: RL  =  <SB> |  <SRC>   | :=  <SB>    |    <SRC>  |
        |  7.  msk: RL  =  <SB> ^  <SRC>   | :=  <SB>    ^    <SRC>  |
        |                                  |                         |
        |  8.  msk: RL  = ~<SB> &  <SRC>   | := ~<SB>    &    <SRC>  |
        |  9.  msk: RL  =  <SB> & ~<SRC>   | :=  <SB>    &   ~<SRC>  |
        |                                  |                         |
        | 16.  msk: RL &= ~<SB>            | &= ~<SB>                |
        | 17.  msk: RL &= ~<SRC>           | &= ~<SRC>               |
        +----------------------------------|-------------------------+

    In addition, the following APL commands may be supported by HW but not
    supported by APL concrete syntax because they have no dedicated
    read-control register:

        21.  msk: RL = ~RL & <SRC>
        22.  msk: RL = ~RL & <SB>
        23.  msk: RL = ~RL & (<SB> & <SRC>)
        24.  msk: RL &= ~<SB> | ~<SRC>

    Parameters:
        operator: any ASSIGN_OP except COND_EQ.
        rvalue: either a unary or binary expression whose evaluation should be
                assigned to RL in the method specified by the assignment
                operator."""

    operator: ASSIGN_OP
    rvalue: Union[UNARY_EXPR, BINARY_EXPR]

    @property
    def lvalue(self: "READ") -> RL_EXPR:
        return RL_EXPR.RL

    @property
    def left_operands(self: "READ") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "READ") -> List[str]:
        if self.operator is ASSIGN_OP.EQ:
            return self.rvalue.operands
        left_operands = self.lvalue.operands
        right_operands = self.rvalue.operands
        return left_operands + right_operands

    def __str__(self: "READ") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
class WRITE(NamedTuple):
    """Assigns a <SRC> to the given collection of vector registers.

    A WRITE may take any of the following assignment forms:
        1. msk: SB[x] = <SRC>;
        2. msk: SB[x,y] = <SRC>;
        3. msk: SB[x,y,z] = <SRC>;

    A WRITE may take any of the following conditional assignment forms:
        1. msk: SB[x] ?= <SRC>;
        2. msk: SB[x,y] ?= <SRC>;
        3. msk: SB[x,y,z] ?= <SRC>;

    Parameters:
        operator: specifies the method of assignment (either EQ or COND_EQ).
        lvalue: recipient collection of vector registers.
        rvalue: expression to assign the vector registers."""

    operator: ASSIGN_OP
    lvalue: SB_EXPR
    rvalue: UNARY_SRC

    @property
    def left_operands(self: "WRITE") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "WRITE") -> List[str]:
        if self.operator is ASSIGN_OP.EQ:
            return self.rvalue.operands
        left_operands = self.lvalue.operands
        right_operands = self.rvalue.operands
        return left_operands + right_operands

    def __str__(self: "WRITE") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
class BROADCAST(NamedTuple):
    """Performs a contraction over either the sections or plats of RL and
    assigns it to the lvalue.

    A BROADCAST may take any of the following forms:
        1. msk: GL = RL;
        2. msk: GGL = RL;
        3. msk: RSP16 = RL;

    Parameters:
        lvalue: recipient of the contracted value of RL.
        rvalue: what to broadcast to the lvalue; note that LX regs are only
                applicable for GGL."""

    lvalue: BROADCAST_EXPR
    rvalue: Union[RL_EXPR,
                  L1_REG,
                  L2_REG,
                  LXRegWithOffsets,
                  BINARY_EXPR] = RL_EXPR.RL

    @property
    def operator(self: "BROADCAST") -> ASSIGN_OP:
        return ASSIGN_OP.EQ

    @property
    def left_operands(self: "BROADCAST") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "BROADCAST") -> List[str]:
        return self.rvalue.operands

    def __str__(self: "BROADCAST") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
class RSP16_ASSIGNMENT(NamedTuple):
    """Assigns some value to RSP16.

    An RSP16_ASSIGNMENT may take the following form:
        1. RSP16 = RSP256;

    RSP16 may take the value of either RSP256 or RL, but an assignment of RL is
    considered a broadcast and requires a section mask. Assigning to RSP16 from
    RSP256 does not require a section mask.

    Parameters:
        rvalue: an allowed rvalue to assign to RSP16 (e.g. RSP256)"""

    rvalue: RSP16_RVALUE

    @property
    def lvalue(self: "RSP16_ASSIGNMENT") -> RSP16_EXPR:
        return RSP16_EXPR.RSP16

    @property
    def operator(self: "RSP16_ASSIGNMENT") -> ASSIGN_OP:
        return ASSIGN_OP.EQ

    @property
    def left_operands(self: "RSP16_ASSIGNMENT") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "RSP16_ASSIGNMENT") -> List[str]:
        return self.rvalue.operands

    def __str__(self: "RSP16_ASSIGNMENT") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
class RSP256_ASSIGNMENT(NamedTuple):
    """Assigns some value to RSP256.

    A RSP256_ASSIGNMENT may take either of the following forms:
        1. RSP256 = RSP16;
        2. RSP256 = RSP2K;

    Parameters:
        rvalue: an allowed rvalue to assign to RSP256 (either RSP16 or
                RSP2K)."""

    rvalue: RSP256_RVALUE

    @property
    def lvalue(self: "RSP256_ASSIGNMENT") -> RSP256_EXPR:
        return RSP256_EXPR.RSP256

    @property
    def operator(self: "RSP256_ASSIGNMENT") -> ASSIGN_OP:
        return ASSIGN_OP.EQ

    @property
    def left_operands(self: "RSP256_ASSIGNMENT") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "RSP256_ASSIGNMENT") -> List[str]:
        return self.rvalue.operands

    def __str__(self: "RSP256_ASSIGNMENT") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
class RSP2K_ASSIGNMENT(NamedTuple):
    """Assigns some value to RSP2K.

    An RSP2K_ASSIGNMENT may take either of the following forms:
        1. RSP2K = RSP256;
        2. RSP2K = RSP32K;

    Parameters:
        rvalue: an allowed rvalue to assign to RSP2K (either RSP256 or
                RSP32K)."""

    rvalue: RSP2K_RVALUE

    @property
    def lvalue(self: "RSP2K_ASSIGNMENT") -> RSP2K_EXPR:
        return RSP2K_EXPR.RSP2K

    @property
    def operator(self: "RSP2K_ASSIGNMENT") -> ASSIGN_OP:
        return ASSIGN_OP.EQ

    @property
    def left_operands(self: "RSP2K_ASSIGNMENT") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "RSP2K_ASSIGNMENT") -> List[str]:
        return self.rvalue.operands

    def __str__(self: "RSP2K_ASSIGNMENT") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


@immutable
class RSP32K_ASSIGNMENT(NamedTuple):
    """Assigns some value to RSP32K.

    An RSP32K_ASSIGNMENT may take the following form:
        1. RSP32K = RSP2K;

    Parameters:
        rvalue: an allowed rvalue to assign to RSP32K (e.g. RSP2K)."""

    rvalue: RSP32K_RVALUE

    @property
    def lvalue(self: "RSP32K_ASSIGNMENT") -> RSP32K_EXPR:
        return RSP32K_EXPR.RSP32K

    @property
    def operator(self: "RSP32K_ASSIGNMENT") -> ASSIGN_OP:
        return ASSIGN_OP.EQ

    @property
    def left_operands(self: "RSP32K_ASSIGNMENT") -> List[str]:
        return self.lvalue.operands

    @property
    def right_operands(self: "RSP32K_ASSIGNMENT") -> List[str]:
        return self.rvalue.operands

    def __str__(self: "RSP32K_ASSIGNMENT") -> str:
        return f"{self.lvalue} {self.operator} {self.rvalue}"


AssignmentType = Union[READ, WRITE, BROADCAST,
                       RSP16_ASSIGNMENT,
                       RSP256_ASSIGNMENT,
                       RSP2K_ASSIGNMENT,
                       RSP32K_ASSIGNMENT]


@immutable
class ASSIGNMENT(NamedTuple):
    """Performs an assignment to either RL, an SB set, or one of the RSPs.

    Parameters:
        operation: assigns to either RL, an SB set, or one of the RSPs."""

    operation: AssignmentType

    @property
    def left_operands(self: "ASSIGNMENT") -> Sequence[Any]:
        return self.operation.left_operands

    @property
    def right_operands(self: "ASSIGNMENT") -> List[str]:
        return self.operation.right_operands

    def __str__(self: "ASSIGNMENT") -> str:
        return f"{self.operation}"


@bleir_dataclass
class SHIFTED_SM_REG:
    """Shifts a section mask to the left by the specified number of bits.

    A SHIFTED_SM_REG takes the following form:
        1. msk<<num_bits

    where `num_bits` is between 0 and 16.

    Parameters:
        register: section mask whose value should be shifted.
        num_bits: number of bits to shift the section mask to the left.
        negated: helper attribute that specifies whether the section mask bits
                 should be negated after they have been shifted."""

    register: Union[SM_REG, "SHIFTED_SM_REG"]
    num_bits: int
    negated: Optional[bool] = False

    def __post_init__(self: "SHIFTED_SM_REG") -> None:
        if isinstance(self.register, SHIFTED_SM_REG):
            shifted_sm_reg = self.register
            self.register = shifted_sm_reg.register
            self.num_bits += shifted_sm_reg.num_bits
            self.negated ^= shifted_sm_reg.negated

    @property
    def constant_value(self: "SHIFTED_SM_REG") -> Optional[int]:
        mask = self.register.constant_value
        if mask is None:
            return None
        if self.register.is_section:
            mask = (0x0001 << mask)
        return (mask << self.num_bits) & 0xFFFF

    def __invert__(self: "SHIFTED_SM_REG") -> "MASK":
        return MASK(expression=self,
                    operator=UNARY_OP.NEGATE)

    def __str__(self: "SHIFTED_SM_REG") -> str:
        return f"{self.register}<<{self.num_bits}"

    # helper method
    def resolve(self: "SHIFTED_SM_REG", mask: int) -> int:
        mask = self.register.resolve(mask)
        return (mask << self.num_bits) & 0xFFFF


@immutable
class MASK(NamedTuple):
    """Optionally transforms a section mask operand with some unary operator.

    A MASK takes either of the following forms:
        1. msk
        2. ~msk
        3. msk<<num_bits
        4. ~(msk<<num_bits)

    Parameters:
        expression: section mask operand to transform.
        operator: optional unary operator to transform the expression.
        read_write_inhibit: whether this mask should be read/write inhibited,
                            that is, whether its sections should be disabled to
                            conserve energy."""

    expression: Union[SM_REG, SHIFTED_SM_REG]
    operator: Optional[UNARY_OP] = None
    read_write_inhibit: Optional[ReadWriteInhibit] = None

    def __invert__(self: "MASK") -> "MASK":
        if self.operator is UNARY_OP.NEGATE:
            return self.having(operator=None)
        return self.having(operator=UNARY_OP.NEGATE)

    def __str__(self: "MASK") -> str:
        if self.operator is not None:
            if isinstance(self.expression, SHIFTED_SM_REG):
                return f"{self.operator}({self.expression})"
            return f"{self.operator}{self.expression}"
        return f"{self.expression}"

    @property
    def sm_reg(self: "MASK") -> SM_REG:
        if isinstance(self.expression, SM_REG):
            return self.expression
        return self.expression.register

    @property
    def is_constant(self: "MASK") -> bool:
        return self.sm_reg.is_constant

    @property
    def constant_value(self: "MASK") -> Optional[int]:
        mask = self.expression.constant_value
        if mask is None:
            return None
        if self.operator is UNARY_OP.NEGATE:
            mask = 0xFFFF - mask
        return mask

    # helper method
    def resolve(self: "MASK", mask_value: Optional[int] = None) -> int:
        if self.is_constant:
            if mask_value is not None \
               and mask_value != self.sm_reg.constant_value:
                warn(f"Ignoring mask_value ({mask_value}) for constant SM_REG "
                     f"expression: {self}")
            mask_value = self.sm_reg.constant_value
            if self.sm_reg.is_section:
                mask_value = (0x0001 << mask_value)

        if mask_value is None:
            raise ValueError(
                f"Must specify mask_value for non-constant SM_REG "
                f"expressions: {self}")

        resolution = self.expression.resolve(mask_value)
        if self.operator is UNARY_OP.NEGATE:
            resolution = 0xFFFF - resolution
        return resolution


@immutable
class MASKED(NamedTuple):
    """Represents an operation that requires a section mask.

    Parameters:
        mask: section mask to apply to the assignment operation.
        assignment: assigns some rvalue to an lvalue under the given mask."""

    mask: MASK
    assignment: Optional[ASSIGNMENT] = None

    @property
    def read_write_inhibit(self: "MASKED") -> Optional[ReadWriteInhibit]:
        return self.mask.read_write_inhibit

    @property
    def left_operands(self: "MASKED") -> List[str]:
        return self.assignment.left_operands

    @property
    def right_operands(self: "MASKED") -> List[str]:
        return self.assignment.right_operands

    def __str__(self: "MASKED") -> str:
        message = f"{self.mask}:"
        if self.assignment is not None:
            message = f"{message} {self.assignment}"
        if self.read_write_inhibit is not None:
            # msk: RWINH_(SET|RST)
            # msk: RL <eqop> <SB> RWINH_(SET|RST)
            message = f"{message} {self.read_write_inhibit}"
        return message


STATEMENT_operation = Union[MASKED, SPECIAL,
                            RSP16_ASSIGNMENT,
                            RSP256_ASSIGNMENT,
                            RSP2K_ASSIGNMENT,
                            RSP32K_ASSIGNMENT,
                            GGL_ASSIGNMENT,
                            LGL_ASSIGNMENT,
                            LX_ASSIGNMENT,
                            GlassStatement]


class StatementMetadata(Enum):
    """Metadata associated explicitly with STATEMENT instances.
        1. FILE_PATH := relative path to the file containing this statement.
        2. LINE_NUMBER := line number of the file containing this statement."""
    FILE_PATH: str = "FILE_PATH"
    LINE_NUMBER: str = "LINE_NUMBER"


@metadatable(StatementMetadata)
@bleir_dataclass
class STATEMENT:
    """A semicolon-terminated operation that operates on the half-bank. A
    STATEMENT might be a MASKED operation, SPECIAL operation, or any of the RSP
    assignments.

    Parameters:
        operation: manipulates the state of the half-bank in some fashion.
        comment: an optional comment to associate with this STATEMENT."""

    operation: STATEMENT_operation
    comment: Optional[Comment] = None
    metadata: Optional[Dict[StatementMetadata, Any]] = None

    @property
    def left_operands(self: "STATEMENT") -> Sequence[Any]:
        return self.operation.left_operands

    @property
    def right_operands(self: "STATEMENT") -> List[str]:
        return self.operation.right_operands

    def __str__(self: "STATEMENT") -> str:
        return f"{self.operation};"


STATEMENT_or_LineComment = Union[STATEMENT, LineComment]


@metadatable(StatementMetadata)
@bleir_dataclass
@collectible("statements")
class MultiStatement:
    """A collection of 1-to-4 STATEMENTs that are executed in parallel as a
    single STATEMENT. With two exceptions, the STATEMENTs should not depend on
    each and should yield the same half-bank state after executing in any
    order. The two exceptions are WRITE-before-READ and READ-before-BROADCAST,
    which operate in terms of half clocks and can be combined with dependencies
    in the same MultiStatement.

    A MultiStatement takes any of the following forms:
        1. { STATEMENT; }
        2. { STATEMENT; STATEMENT; }
        3. { STATEMENT; STATEMENT; STATEMENT; }
        4. { STATEMENT; STATEMENT; STATEMENT; STATEMENT; }

    Parameters:
        statements: unordered collection of STATEMENTs to execute in parallel.
        comment: an optional comment to associate with this MultiStatement."""

    statements: Sequence[STATEMENT_or_LineComment]
    comment: Optional[Comment] = None
    metadata: Optional[Dict[StatementMetadata, Any]] = None

    def __str__(self: "MultiStatement") -> str:
        statements = " ".join(str(statement) for statement in self.statements)
        return f"{{{statements}}}"

    @property
    def masks(self: "MultiStatement") -> Set[MASK]:
        return set([statement.operation.mask
                    for statement in self.statements
                    if isinstance(statement.operation, MASKED)])

    @property
    def sm_regs(self: "MultiStatement") -> Set[SM_REG]:
        sm_regs = set()
        for statement in self.statements:
            if not isinstance(statement.operation, MASKED):
                continue
            mask = statement.operation.mask
            if isinstance(mask.expression, SHIFTED_SM_REG):
                sm_regs.add(mask.expression.register)
            else:
                sm_regs.add(mask.expression)
        return sm_regs


class FragmentMetadata(Enum):
    """Metadata associated explicitly with Fragment instances.
        1. ORIGINAL_IDENTIFIER := the name of the original fragment, such as that
                                  before it is partitioned or has its name
                                  obfuscated.
        2. IS_LOW_LEVEL := whether the Fragment is defined by Low-Level Belex
                           (LLB) instead of High-Level Belex (HLB).
        3. FILE_PATH := relative path to the file containing this fragment.
        4. LINE_NUMBER := line number of the file containing this fragment.
        5. NUM_LINES := number of source lines in this fragment."""
    ORIGINAL_IDENTIFIER: str = "ORIGINAL_IDENTIFIER"
    IS_LOW_LEVEL: str = "IS_LOW_LEVEL"
    FILE_PATH: str = "FILE_PATH"
    LINE_NUMBER: str = "LINE_NUMBER"
    NUM_LINES: str = "NUM_LINES"


Operation = Union[MultiStatement, STATEMENT]
Operation_or_LineComment = Union[Operation, LineComment]


ActualParameter = Union[Integer, str]


@metadatable(FragmentMetadata)
@bleir_dataclass
class Fragment:
    """Represents a block of STATEMENTs and MultiStatements are executed with a
    given list of parameters.

    An example Fragment might look like the following:

        APL_FRAG set_rl(
                RN_REG rvr_rp,
                SM_REG msk_rp)
        {   {
            msk_rp: RL = SB[rvr_rp];
            }   };

    Parameters:
        identifier: name to assign the Fragment.
        parameters: sequence of RN_REGs and SM_REGs covering those used by the
                    operations.
        operations: sequence of instructions to execute in the provided order.
        doc_comment: an optional comment describing the purpose of the
                     Fragment.
        metadata: a mapping over FragmentMetadata -> Value pairs containing
                  meta information about this Fragment."""

    __walk__ = ["identifier", "parameters", "operations", "doc_comment",
                "metadata", "children"]

    identifier: str
    parameters: Sequence[FormalParameter]
    operations: Sequence[Operation_or_LineComment]
    doc_comment: Optional[MultiLineComment] = None
    metadata: Optional[Dict[FragmentMetadata, Any]] = None
    children: Optional[Sequence["Fragment"]] = None

    @property
    def original_identifier(self: "Fragment") -> str:
        return self.get_metadata(FragmentMetadata.ORIGINAL_IDENTIFIER,
                                 default_value=self.identifier)

    def __call__(self: "Fragment",
                 *parameters: Sequence[ActualParameter],
                 is_initializer: bool = False,
                 comment: Optional[Comment] = None) -> "FragmentCallerCall":
        fragment_caller = FragmentCaller(fragment=self)
        metadata = {CallMetadata.IS_INITIALIZER: is_initializer}
        return FragmentCallerCall(caller=fragment_caller,
                                  parameters=parameters,
                                  metadata=metadata,
                                  comment=comment)

    def __str__(self: "Fragment") -> str:
        parameter_list = ", ".join(f"{parameter.register_type} {parameter.register_param}"
                                   for parameter in self.parameters)
        return f"APL_FRAG {self.identifier}({parameter_list})"

    @property
    def sm_regs(self: "Fragment") -> List[SM_REG]:
        return list(reg for reg in self.parameters if isinstance(reg, SM_REG))

    @property
    def rn_regs(self: "Fragment") -> List[SM_REG]:
        return list(reg for reg in self.parameters if isinstance(reg, RN_REG))

    @property
    def multi_statements(self: "Fragment") -> List[MultiStatement]:
        return [operation for operation in self.operations
                if isinstance(operation, MultiStatement)]

    @property
    def temporaries(self: "Fragment") -> List[FormalParameter]:
        return [parameter for parameter in self.parameters
                if parameter.identifier.startswith("_INTERNAL")]


class CallerMetadata(Enum):
    """Metadata attributes associated with fragment callers.
        1. BUILD_EXAMPLES := a function that returns a sequence of Examples
        2. REGISTER_MAP := a mapping of parameter ids to register indices
        3. ARGS_BY_REG_NYM := mapping over sequences of params having the same
                              registers; if there is more than one param
                              associated with a register an a copy operation is
                              implied (high-level BELEX)
        4. OUT_PARAM := identifier of the return parameter (high-level BELEX)
        5. IS_HIGH_LEVEL := specifies that the fragment was built with
                                  BELEX
        6. IS_LOW_LEVEL := specifies that the fragment was built with BLECCI
        7. SHOULD_FAIL := indicates that the generated unit test should be a
                          negative test (i.e. code failures imply test
                          successes).
        8. INITIAL_ACTIVE_REGISTERS := list of registers that begin life active
                                       in a spill/restore environment."""

    BUILD_EXAMPLES: str = "BUILD_EXAMPLES"
    REGISTER_MAP: str = "REGISTER_MAP"
    ARGS_BY_REG_NYM: str = "ARGS_BY_REG_NYM"
    OUT_PARAM: str = "OUT_PARAM"
    IS_HIGH_LEVEL: str = "IS_HIGH_LEVEL"
    IS_LOW_LEVEL: str = "IS_LOW_LEVEL"
    SHOULD_FAIL: str = "SHOULD_FAIL"
    INITIAL_ACTIVE_REGISTERS: str = "INITIAL_ACTIVE_REGISTERS"


AllocatedRegister_or_Comment = Union[AllocatedRegister, Comment]


@metadatable(CallerMetadata)
@bleir_dataclass
class FragmentCaller:
    """Represents a C function that calls a fragment. It is within the
    FragmentCaller that registers are allocated for parameters and it is the
    FragmentCaller that is called from the C main function.

    An example FragmentCaller might look like the following:

        void set_rl_caller(
                u16 lvr_vp,
                u16 msk_vp)
        {   apl_set_rn_reg(RN_REG_0, lvr_vp);
            apl_set_sm_reg(SM_REG_0, msk_vp);
            RUN_FRAG_ASYNC(
                set_rl(
                    lvr_rp=RN_REG_0,
                    msk_rp=SM_REG_0));   }

    Parameters:
        fragment: the Fragment to call.
        registers: an optional sequence of AllocatedRegisters that specifies
                   which registers should receive the actual parameters.
        initializers: fragment caller calls that are required to set up this call
                      for execution.
        finalizers: fragment caller calls that are required to clean up this call
                    after execution.
        metadata: mapping over meta-information about this fragment caller,
                  such as whether it was built by BELEX or BLECCI."""

    __walk__ = ["registers", "fragment", "initializers", "finalizers",
                "metadata"]

    fragment: Fragment
    registers: Optional[Sequence[Optional[AllocatedRegister_or_Comment]]] = None
    initializers: Optional[Sequence["FragmentCallerCall"]] = None
    finalizers: Optional[Sequence["FragmentCallerCall"]] = None
    metadata: Optional[Dict[CallerMetadata, Any]] = None

    @property
    def doc_comment(self: "FragmentCaller") -> Optional[MultiLineComment]:
        return self.fragment.doc_comment

    def __call__(self: "FragmentCaller",
                 *parameters: Sequence[ActualParameter],
                 is_initializer: bool = False,
                 comment: Optional[Comment] = None) -> "FragmentCallerCall":
        metadata = {CallMetadata.IS_INITIALIZER: is_initializer}
        return FragmentCallerCall(caller=self,
                                  parameters=parameters,
                                  metadata=metadata,
                                  comment=comment)

    def __str__(self: "FragmentCaller") -> str:
        parameter_list = ", ".join(f"u16 {parameter.value_param}"
                                   for parameter in self.parameters)
        return f"void {self.identifier}({parameter_list});"

    @property
    def parameters(self: "FragmentCaller") -> Sequence[Union[FormalParameter]]:
        return self.fragment.parameters

    @property
    def formal_parameters(self: "FragmentCaller") -> Sequence[Union[FormalParameter]]:
        return self.fragment.parameters

    @property
    def identifier(self: "FragmentCaller") -> str:
        # return f"{self.fragment.identifier}_caller"
        return f"belex_{self.fragment.identifier}"

    @property
    def belex_parameters(self: "FragmentCaller") -> Sequence[RN_REG]:
        if not self.has_metadata(CallerMetadata.IS_HIGH_LEVEL):
            return []

        args_by_reg_nym = self.metadata[CallerMetadata.ARGS_BY_REG_NYM]

        belex_parameters = []
        for reg_nym, args in args_by_reg_nym.items():
            if reg_nym == "IR":
                continue

            if len(args) > 2:
                raise RuntimeError(
                    f"Expected length of args to be <= 2, but was {len(args)}: {args}")

            register = None
            for allocated_register in self.registers:
                if allocated_register.parameter.identifier == reg_nym:
                    register = allocated_register.register
                    break

            if register is None:
                warn(
                    f"Register is not allocated for [{reg_nym}], it is likely "
                    f"an unused parameter: {self.fragment.identifier}")
                continue

            for param_nym in args:
                belex_parameter = RN_REG(identifier=param_nym)
                belex_parameter = AllocatedRegister(
                    parameter=belex_parameter,
                    register=register)
                belex_parameters.append(belex_parameter)

        return belex_parameters

    @property
    def register_map(self: "FragmentCaller") \
            -> Optional[Dict[FormalParameter, AllocatedRegister]]:
        if self.registers is None:
            return None
        return OrderedDict(zip(self.formal_parameters, self.registers))


class CallMetadata(Enum):
    """Metadata attributes to attach to function calls.

        1. IS_INITIALIZER := specifies whether the function call should be
                          executed before anything else, even before the
                          parameters are written to their corresponding vector
                          registers.
        2. IS_HIGH_LEVEL := specifies the fragment was defined by high-level BELEX
        3. IS_LOW_LEVEL := specifies the fragment was defined by low-level BELEX
                        or BLECCI
        4. FILE_PATH := relative path to the file containing this fragment call.
        5. LINE_NUMBER := line number of the file containing this fragment call."""

    IS_INITIALIZER: str = "IS_INITIALIZER"
    IS_HIGH_LEVEL: str = "IS_HIGH_LEVEL"
    IS_LOW_LEVEL: str = "IS_LOW_LEVEL"
    FILE_PATH: str = "FILE_PATH"
    LINE_NUMBER: str = "LINE_NUMBER"


ActualParameter_or_Any = Union[ActualParameter, Any]


@immutable
@hashable
@stateful
@metadatable(CallMetadata)
class FragmentCallerCall:
    """Represents a concrete call to a fragment through its fragment caller.

    For example, if you want to call the Fragment, `set_rl`, you would call it
    through `set_rl_caller` with actual parameters:

        set_rl_caller(4, 0xBEEF);

    Parameters:
        caller: FragmentCaller that ferries calls to the desired Fragment.
        parameters: actual parameters used by the Fragment's operations.
        metadata: bits of information about this FragmentCallerCall that might
                  affect such things as its code is generated.
        comment: an optional comment associated with this FragmentCallerCall
                 that might include such things as debugger information."""

    caller: FragmentCaller
    parameters: Sequence[ActualParameter]
    metadata: Optional[Dict[CallMetadata, Any]]
    comment: Optional[Comment]

    def __init__(self: "FragmentCallerCall",
                 caller: FragmentCaller,
                 parameters: Sequence[ActualParameter_or_Any],
                 metadata: Optional[Dict[CallMetadata, Any]] = None,
                 comment: Optional[Comment] = None) -> None:

        # Workaround for when unused parameters are removed from the fragment before transforming
        # the fragment caller call (see open_belex.bleir.optimizers.UnusedParameterRemover):
        if self.should_transform_parameters(parameters):
            parameters = self.transform_parameters(caller, parameters)

        self.caller = caller
        self.parameters = tuple(parameters)
        self.metadata = metadata
        self.comment = comment

    @staticmethod
    def transform_parameters(caller: FragmentCaller,
                             actual_parameters: Sequence[ActualParameter_or_Any]) -> Sequence[ActualParameter]:
        transformed_parameters = []
        formal_parameters = caller.fragment.parameters
        for formal_parameter, actual_parameter in zip(formal_parameters, actual_parameters):
            if isinstance(actual_parameter, ActualParameter.__args__):
                transformed_parameter = actual_parameter
            elif type(formal_parameter) is RN_REG and type(actual_parameter) is str:
                transformed_parameter = int(actual_parameter, 10)
            elif type(formal_parameter) is SM_REG and type(actual_parameter) is not int:
                transformed_parameter = Mask(actual_parameter).full_integer
            else:
                raise NotImplementedError(f"Unsupported actual_parameter type ({actual_parameter.__class__.__name__}): {actual_parameter}")
            transformed_parameters.append(transformed_parameter)
        return transformed_parameters

    @staticmethod
    def should_transform_parameters(actual_parameters: Sequence[ActualParameter_or_Any]) -> bool:
        for actual_parameter in actual_parameters:
            if not isinstance(actual_parameter, ActualParameter.__args__):
                return True
        return False

    def __str__(self: "FragmentCallerCall") -> str:
        parameters = []
        for formal_parameter, actual_parameter in self.parameter_map.items():
            if isinstance(formal_parameter, (RN_REG, RE_REG, EWE_REG, L1_REG, L2_REG)):
                parameters.append(
                    f"{formal_parameter.identifier}={actual_parameter}")
            elif isinstance(formal_parameter, SM_REG):
                parameters.append(
                    f"{formal_parameter.identifier}=0x{actual_parameter:04X}")
            else:
                raise RuntimeError(
                    f"Unsupported formal_parameter type "
                    f"({type(formal_parameter).__name__}): {formal_parameter}")
        parameters = ", ".join(parameters)
        return f"{self.identifier}({parameters});"

    @property
    def fragment(self: "FragmentCallerCall") -> Fragment:
        return self.caller.fragment

    @property
    def identifier(self: "FragmentCallerCall") -> str:
        return self.caller.identifier

    @property
    def formal_parameters(self: "FragmentCallerCall") -> Sequence[FormalParameter]:
        return self.caller.fragment.parameters

    @property
    def actual_parameters(self: "FragmentCallerCall") -> Sequence[ActualParameter]:
        return self.parameters

    @property
    def parameter_map(self: "FragmentCallerCall") -> Dict[FormalParameter, ActualParameter]:
        return OrderedDict(zip(self.formal_parameters, self.actual_parameters))


@bleir_dataclass
class CParameter:
    name: str
    kind: Type


class CFunctionMetadata(Enum):
    PYTHON_FUNCTION: str = "PYTHON_FUNCTION"


@metadatable(CFunctionMetadata)
@bleir_dataclass
class CFunction:
    identifier: str
    formal_parameters: Sequence[CParameter]
    metadata: Optional[Dict[CFunctionMetadata, Any]] = None

    def __call__(self: "CFunction",
                 *actual_parameters: Sequence[Any],
                 is_initializer: bool = False) -> "CFunctionCall":

        if len(actual_parameters) != len(self.formal_parameters):
            raise ValueError(
                f"Expected {len(self.formal_parameters)} parameters for a call "
                f"to {self.identifier} but received {len(actual_parameters)}: "
                f"{actual_parameters}")

        for formal_parameter, actual_parameter \
                in zip(self.formal_parameters, actual_parameters):
            if not isinstance(actual_parameter, formal_parameter.kind):
                raise ValueError(
                    f"Expected value of {formal_parameter.name} to be an "
                    f"instance of {formal_parameter.kind.__name__} but was "
                    f"{actual_parameter.__class__.__name__}: {actual_parameter}")

        metadata = {
            CallMetadata.IS_INITIALIZER: is_initializer,
        }

        c_function_call = CFunctionCall(c_function=self,
                                        actual_parameters=actual_parameters,
                                        metadata=metadata)

        return c_function_call


@metadatable(CallMetadata)
@bleir_dataclass
class CFunctionCall:
    c_function: CFunction
    actual_parameters: Sequence[Any]
    metadata: Optional[Dict[CallMetadata, Any]] = None

Call = Union[FragmentCallerCall, CFunctionCall]
Call_or_LineComment = Union[Call, LineComment]


FragmentCallerCall_or_LineComment = Union[FragmentCallerCall, LineComment]


def is_fragment_caller_call(call: Call_or_LineComment) -> bool:
    """Determines whether the parameter is a fragment caller call instead of a
    comment."""
    return isinstance(call, FragmentCallerCall)


def is_fragment_caller(caller: Any) -> bool:
    """Determines whether the parameter is a fragment caller."""
    return isinstance(caller, FragmentCaller)


class SnippetMetadata(Enum):
    """Meta-information about Snippet instances, including:
        1. HEADER_FILE := path to the generated header file
        2. SOURCE_FILE := path to the generated source file
        3. TARGET := target for generated sources, e.g. 'apl' or 'baryon'"""
    HEADER_FILE: str = "HEADER_FILE"
    SOURCE_FILE: str = "SOURCE_FILE"
    TARGET: str = "TARGET"


@metadatable(SnippetMetadata)
@bleir_dataclass
class Snippet:
    """Top-level element of a BLEIR application that describes the application
    and its operations.

    Parameters:
        name: identifies the application.
        examples: initial states and expected final states of executions of the
                  application.
        calls: FragmentCallerCalls that define the application.
        doc_comment: optional description of the application's purpose.
        metadata: meta-information about this Snippet such as the name of its
                  APL header.
        library_callers: sequence of fragment callers to include in the
                         generated APL that are not called explicitly."""

    name: str
    examples: Sequence[Example]
    calls: Sequence[Call_or_LineComment]
    doc_comment: Optional[MultiLineComment] = None
    metadata: Optional[Dict[SnippetMetadata, Any]] = None
    library_callers: Optional[Sequence[FragmentCaller]] = None

    @property
    def source_file(self: "Snippet") -> Union[str, Path]:
        if self.target == "baryon":
            source_ext = ".c"
        else:
            raise RuntimeError(
                f"Unsupported target: {self.target}")
        return self.get_metadata(
            SnippetMetadata.SOURCE_FILE,
            default_value=f"{self.name}-funcs{source_ext}")

    @property
    def header_file(self: "Snippet") -> Union[str, Path]:
        if self.target == "baryon":
            header_ext = ".h"
        else:
            raise RuntimeError(
                f"Unsupported target: {self.target}")
        return self.get_metadata(
            SnippetMetadata.HEADER_FILE,
            default_value=f"{self.name}-funcs{header_ext}")

    @property
    def target(self: "Snippet") -> str:
        return self.get_metadata(
            SnippetMetadata.TARGET,
            default_value="baryon")

    @property
    def callers(self: "Snippet") -> Sequence[FragmentCaller]:
        visited = set()

        if self.library_callers is not None:
            for library_caller in self.library_callers:
                visited.add(library_caller)

        for call in self.calls:
            if isinstance(call, FragmentCallerCall):
                visited.add(call.caller)

        return sorted(visited, key=lambda caller: caller.identifier)

    @property
    def fragment_caller_calls(self: "Snippet") -> Sequence[FragmentCallerCall]:
        return tuple(filter(is_fragment_caller_call, self.calls))

    @property
    def fragments(self: "Snippet") -> Sequence[Fragment]:
        fragments = []
        for caller in self.callers:
            if is_fragment_caller(caller):
                fragments.append(caller.fragment)
        return fragments

    @property
    def initializers(self: "Snippet") -> List[Call]:
        return [call for call in self.calls
                if call.has_metadata(CallMetadata.IS_INITIALIZER, True)]

    @property
    def body(self: "Snippet") -> List[Call]:
        return [call for call in self.calls
                if not call.has_metadata(CallMetadata.IS_INITIALIZER, True)]


Invertible = Union[UNARY_SRC, UNARY_SB, MASK]


SBParameter_or_str = Union[SBParameter, str]


class SBFactory:
    """Provides an SB_EXPR factory that accepts parameters in the forms:
    SB = SBFactory()
        1. SB[x]       == SB[(x,)]      == SB_EXPR(tuple(x))
        2. SB[x, y]    == SB[(x, y)]    == SB_EXPR(tuple(x, y))
        3. SB[x, y, z] == SB[(x, y, z)] == SB_EXPR(tuple(x, y, z))
    """

    def __getitem__(self: "SBFactory",
                    x_or_regs: Union[SBParameter_or_str,
                                     Sequence[SBParameter_or_str]]) -> SB_EXPR:
        if isinstance(x_or_regs, BleirSerializable):
            x_or_regs = x_or_regs.as_bleir()
        if isinstance(x_or_regs, SBParameter_or_str.__args__):
            return SB_EXPR((sb_parameter(x_or_regs),))
        return SB_EXPR(tuple(map(sb_parameter, x_or_regs)))


# Usage:
# 1. SB[x]       == SB[(x,)]      == SB_EXPR(tuple(x))
# 2. SB[x, y]    == SB[(x, y)]    == SB_EXPR(tuple(x, y))
# 3. SB[x, y, z] == SB[(x, y, z)] == SB_EXPR(tuple(x, y, z))
SB = SBFactory()


def strip_rp_suffix(reg_id: str) -> str:
    if reg_id.endswith("_rp"):
        return reg_id[:-3]
    return reg_id


def sm_reg(reg: Union[SM_REG, str]) -> SM_REG:
    if isinstance(reg, BleirSerializable):
        reg = reg.as_bleir()
    if isinstance(reg, str):
        assert not reg in RESERVED_TERMS
        reg = strip_rp_suffix(reg.strip())
        return SM_REG(reg)
    return reg


def rn_reg(reg: Union[RN_REG, str]) -> RN_REG:
    if isinstance(reg, BleirSerializable):
        reg = reg.as_bleir()
    if isinstance(reg, str):
        assert not reg in RESERVED_TERMS
        reg = strip_rp_suffix(reg.strip())
        return RN_REG(reg)
    return reg


def l1_reg(reg: Union[L1_REG, str, BleirSerializable]) -> L1_REG:
    if isinstance(reg, BleirSerializable):
        reg = reg.as_bleir()
    if isinstance(reg, str):
        assert not reg in RESERVED_TERMS
        reg = strip_rp_suffix(reg.strip())
        return L1_REG(reg)
    return reg


def l2_reg(reg: Union[L2_REG, str]) -> L2_REG:
    if isinstance(reg, BleirSerializable):
        reg = reg.as_bleir()
    if isinstance(reg, str):
        assert not reg in RESERVED_TERMS
        reg = strip_rp_suffix(reg.strip())
        return L2_REG(reg)
    return reg


def re_reg(reg: Union[RE_REG, str]) -> RE_REG:
    if isinstance(reg, BleirSerializable):
        reg = reg.as_bleir()
    if isinstance(reg, str):
        assert not reg in RESERVED_TERMS
        reg = strip_rp_suffix(reg.strip())
        return RE_REG(reg)
    return reg


def ewe_reg(reg: Union[EWE_REG, str]) -> EWE_REG:
    if isinstance(reg, BleirSerializable):
        reg = reg.as_bleir()
    if isinstance(reg, str):
        assert not reg in RESERVED_TERMS
        reg = strip_rp_suffix(reg.strip())
        return EWE_REG(reg)
    return reg


def sb_parameter(reg: SBParameter_or_str) -> SBParameter:
    if isinstance(reg, BleirSerializable):
        return reg.as_bleir()
    if isinstance(reg, SBParameter.__args__):
        return reg
    warn(f"Assuming {reg} is of type RN_REG")
    return RN_REG(reg)


def offset(
        lx_reg: LXParameter,
        *offsets: Sequence[Union[int, Offsets]]) \
        -> Union[LXParameter, LXRegWithOffsets]:

    if isinstance(lx_reg, BleirSerializable):
        lx_reg = lx_reg.as_bleir()

    if len(offsets) == 0:
        return lx_reg

    if isinstance(offsets[0], int):
        if len(offsets) > 3:
            raise ValueError(f"Offsets may consist of 1 to 3 numbers, not {len(offsets)}: {offsets}")
        if len(offsets) > 2 and not isinstance(offsets[2], int) \
           or len(offsets) > 1 and not isinstance(offsets[1], int):
            raise ValueError(
                f"Offsets must consist exclusively of ints, not "
                f"{offsets[2].__class__.__name__}: {offsets}")
        return LXRegWithOffsets(lx_reg, *reversed(offsets))

    if len(offsets) == 1 \
       and is_sequence(offsets[0]) \
       and (len(offsets[0]) == 0 or isinstance(offsets[0][0], int)):
        return offset(lx_reg, *(offsets[0]))

    raise ValueError(
        f"Offsets must consist exclusively of ints, not "
        f"{offsets[2].__class__.__name__}: {offsets}")


Parameterizable = Union[int, str, Mask]


def is_sequence(value: Any) -> bool:
    return hasattr(value, "__len__") \
        and callable(value.__len__) \
        and hasattr(value, "__getitem__") \
        and callable(value.__getitem__)


def is_iterable(value: Any) -> bool:
    return hasattr(value, "__iter__") \
        and callable(value.__iter__)


InvertibleExpr = Union[Invertible,
                       SRC_EXPR,
                       SB_EXPR,
                       SHIFTED_SM_REG,
                       SM_REG,
                       str]


def invert(expr: InvertibleExpr) -> Invertible:
    if isinstance(expr, str):
        expr = SRC_EXPR.find_by_value(expr)
    if isinstance(expr, SRC_EXPR):
        expr = UNARY_SRC(expr)
    if isinstance(expr, SB_EXPR):
        expr = UNARY_SB(expr)
    if isinstance(expr, SM_REG) or isinstance(expr, SHIFTED_SM_REG):
        expr = MASK(expr)
    Expr = type(expr) # Must be a type of Invertible
    if expr.operator is UNARY_OP.NEGATE:
        return Expr(expr.expression)
    return Expr(expr.expression, UNARY_OP.NEGATE)


ReadableExpr = Union[UNARY_EXPR,
                     BINARY_EXPR,
                     UNARY_SB,
                     UNARY_SRC,
                     SB_EXPR,
                     SRC_EXPR,
                     BIT_EXPR,
                     str, int]


def read_expr_factory(operator: ASSIGN_OP,
                      lvalue: Union[RL_EXPR, str],
                      rvalue: ReadableExpr) -> READ:
    if isinstance(lvalue, str):
        lvalue = RL_EXPR.find_by_value(lvalue)
    if lvalue is SRC_EXPR.RL:  # Workaround
        lvalue = RL_EXPR.RL
    if lvalue is not RL_EXPR.RL:
        raise SemanticError(f"lvalue may only be RL, but was: {lvalue}")
    if isinstance(rvalue, str):
        if rvalue == "0" or rvalue == "1":
            # special case
            rvalue = BIT_EXPR.find_by_value(rvalue)
        else:
            rvalue = SRC_EXPR.find_by_value(rvalue)
    if isinstance(rvalue, SRC_EXPR):
        rvalue = UNARY_SRC(rvalue)
    if isinstance(rvalue, UNARY_SRC):
        rvalue = UNARY_EXPR(rvalue)
    if isinstance(rvalue, int):
        rvalue = BIT_EXPR.find_by_value(rvalue)
    if isinstance(rvalue, BIT_EXPR):
        rvalue = UNARY_EXPR(rvalue)
    if isinstance(rvalue, SB_EXPR):
        rvalue = UNARY_SB(rvalue)
    if isinstance(rvalue, UNARY_SB):
        rvalue = UNARY_EXPR(rvalue)
    return ASSIGNMENT(operation=READ(operator, rvalue))


WritableExpr = Union[UNARY_SRC, SRC_EXPR, str]


def write_expr_factory(operator: ASSIGN_OP,
                       lvalue: SB_EXPR,
                       rvalue: WritableExpr) -> WRITE:
    if operator not in [ASSIGN_OP.EQ, ASSIGN_OP.COND_EQ]:
        raise SemanticError(f"operator must be ASSIGN_OP.EQ or ASSIGN_OP.COND_EQ")
    if isinstance(rvalue, str):
        rvalue = SRC_EXPR.find_by_value(rvalue)
    if isinstance(rvalue, SRC_EXPR):
        rvalue = UNARY_SRC(rvalue)
    return ASSIGNMENT(operation=WRITE(operator, lvalue, rvalue))


BroadcastableExpr = Union[RL_EXPR, str]


def broadcast_expr_factory(operator: ASSIGN_OP,
                           lvalue: Union[BROADCAST_EXPR, str],
                           rvalue: BroadcastableExpr) -> BROADCAST:
    if operator is not ASSIGN_OP.EQ:
        raise SemanticError(f"operator may only be ASSIGN_OP.EQ")
    if isinstance(rvalue, str):
        if rvalue == "RL":
            rvalue = RL_EXPR.find_by_value(rvalue)
        else:
            rvalue = l1_reg(rvalue)
    if rvalue is SRC_EXPR.RL:  # Workaround
        rvalue = RL_EXPR.RL
    if not isinstance(rvalue, (RL_EXPR, L1_REG, L2_REG, LXRegWithOffsets, BINARY_EXPR)):
        raise SemanticError(f"Unsupported rvalue type ({rvalue.__class__.__name__}): {rvalue}")
    if isinstance(lvalue, SRC_EXPR):  # Workaround
        lvalue = str(lvalue)
    if isinstance(lvalue, str):
        lvalue = BROADCAST_EXPR.find_by_value(lvalue)
    return ASSIGNMENT(operation=BROADCAST(lvalue, rvalue))


def ggl_assignment_factory(operator: ASSIGN_OP,
                           lvalue: Union[GGL_EXPR, str],
                           rvalue: Union[LX_ADDR, str]) -> GGL_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise SemanticError(f"operator may only be ASSIGN_OP.EQ")
    if isinstance(lvalue, str):
        lvalue = GGL_EXPR.find_by_value(lvalue)
    if lvalue is SRC_EXPR.GGL:
        lvalue = GGL_EXPR.GGL
    if not isinstance(lvalue, GGL_EXPR):
        raise SemanticError(f"Unsupported lvalue type ({lvalue.__class__.__name__}): {lvalue}")
    if isinstance(rvalue, str):
        rvalue = l1_reg(rvalue)
    if not isinstance(rvalue, LX_ADDR.__args__):
        raise SemanticError(f"Unsupported rvalue type ({rvalue.__class__.__name__}): {rvalue}")
    return GGL_ASSIGNMENT(rvalue=rvalue)


def lgl_assignment_factory(operator: ASSIGN_OP,
                           lvalue: Union[LGL_EXPR, str],
                           rvalue: Union[LX_ADDR, str]) -> LGL_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise SemanticError(f"operator may only be ASSIGN_OP.EQ")
    if isinstance(lvalue, str):
        lvalue = LGL_EXPR.find_by_value(lvalue)
    if not isinstance(lvalue, LGL_EXPR):
        raise SemanticError(f"Unsupported lvalue type ({lvalue.__class__.__name__}): {lvalue}")
    if isinstance(rvalue, str):
        rvalue = l1_reg(rvalue)
    if not isinstance(rvalue, LX_ADDR.__args__):
        raise SemanticError(f"Unsupported rvalue type ({rvalue.__class__.__name__}): {rvalue}")
    return LGL_ASSIGNMENT(rvalue=rvalue)


def lx_assignment_factory(operator: ASSIGN_OP,
                          lvalue: Union[LX_ADDR, str],
                          rvalue: Union[SRC_EXPR, GGL_EXPR, LGL_EXPR, str]) -> LX_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise SemanticError(f"operator may only be ASSIGN_OP.EQ")
    if rvalue == "LGL":
        rvalue = LGL_EXPR.find_by_value(rvalue)
    elif rvalue == "GGL":
        rvalue = GGL_EXPR.find_by_value(rvalue)
    elif rvalue is SRC_EXPR.GGL:
        rvalue = GGL_EXPR.GGL
    if not isinstance(rvalue, (GGL_EXPR, LGL_EXPR)):
        raise SemanticError(f"Unsupported rvalue type ({rvalue.__class__.__name__}): {rvalue}")
    if isinstance(lvalue, str):
        lvalue = l1_reg(lvalue)
    if not isinstance(lvalue, LX_ADDR.__args__):
        raise SemanticError(f"Unsupported lvalue type ({lvalue.__class__.__name__}): {lvalue}")
    return LX_ASSIGNMENT(lvalue=lvalue, rvalue=rvalue)


def rsp16_from_rsp256(operator: ASSIGN_OP,
                      lvalue: Union[RSP16_EXPR, str],
                      rvalue: Union[RSP256_EXPR, str]) -> RSP16_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise ValueError(f"Expected operator to be \"=\": {operator}")
    if isinstance(lvalue, str) and lvalue != RSP16:
        raise ValueError(f"Expected lvalue to be RSP16: {lvalue}")
    if isinstance(rvalue, str) and rvalue != RSP256:
        raise ValueError(f"Expected rvalue to be RSP256: {rvalue}")
    return RSP16_ASSIGNMENT(rvalue=RSP16_RVALUE.RSP256)


def rsp256_from_rsp16(operator: ASSIGN_OP,
                      lvalue: Union[RSP256_EXPR, str],
                      rvalue: Union[RSP16_EXPR, str]) -> RSP256_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise ValueError(f"Expected operator to be \"=\": {operator}")
    if isinstance(lvalue, str) and lvalue != RSP256:
        raise ValueError(f"Expected lvalue to be RSP256: {lvalue}")
    if isinstance(rvalue, str) and rvalue != RSP16:
        raise ValueError(f"Expected rvalue to be RSP16: {rvalue}")
    return RSP256_ASSIGNMENT(rvalue=RSP256_RVALUE.RSP16)


def rsp256_from_rsp2k(operator: ASSIGN_OP,
                      lvalue: Union[RSP256_EXPR, str],
                      rvalue: Union[RSP2K_EXPR, str]) -> RSP256_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise ValueError(f"Expected operator to be \"=\": {operator}")
    if isinstance(lvalue, str) and lvalue != RSP256:
        raise ValueError(f"Expected lvalue to be RSP256: {lvalue}")
    if isinstance(rvalue, str) and rvalue != RSP2K:
        raise ValueError(f"Expected rvalue to be RSP2K: {rvalue}")
    return RSP256_ASSIGNMENT(rvalue=RSP256_RVALUE.RSP2K)


def rsp2k_from_rsp256(operator: ASSIGN_OP,
                      lvalue: Union[RSP2K_EXPR, str],
                      rvalue: Union[RSP256_EXPR, str]) -> RSP2K_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise ValueError(f"Expected operator to be \"=\": {operator}")
    if isinstance(lvalue, str) and lvalue != RSP2K:
        raise ValueError(f"Expected lvalue to be RSP2K: {lvalue}")
    if isinstance(rvalue, str) and rvalue != RSP256:
        raise ValueError(f"Expected rvalue to be RSP256: {rvalue}")
    return RSP2K_ASSIGNMENT(rvalue=RSP2K_RVALUE.RSP256)


def rsp2k_from_rsp32k(operator: ASSIGN_OP,
                      lvalue: Union[RSP2K_EXPR, str],
                      rvalue: Union[RSP32K_EXPR, str]) -> RSP2K_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise ValueError(f"Expected operator to be \"=\": {operator}")
    if isinstance(lvalue, str) and lvalue != RSP2K:
        raise ValueError(f"Expected lvalue to be RSP2K: {lvalue}")
    if isinstance(rvalue, str) and rvalue != RSP32K:
        raise ValueError(f"Expected rvalue to be RSP32K: {rvalue}")
    return RSP2K_ASSIGNMENT(rvalue=RSP2K_RVALUE.RSP32K)


def rsp32k_from_rsp2k(operator: ASSIGN_OP,
                      lvalue: Union[RSP32K_EXPR, str],
                      rvalue: Union[RSP2K_EXPR, str]) -> RSP32K_ASSIGNMENT:
    if operator is not ASSIGN_OP.EQ:
        raise ValueError(f"Expected operator to be \"=\": {operator}")
    if isinstance(lvalue, str) and lvalue != RSP32K:
        raise ValueError(f"Expected lvalue to be RSP32K: {lvalue}")
    if isinstance(rvalue, str) and rvalue != RSP2K:
        raise ValueError(f"Expected rvalue to be RSP2K: {rvalue}")
    return RSP32K_ASSIGNMENT(rvalue=RSP32K_RVALUE.RSP2K)


AssignmentFactory = Callable[[ASSIGN_OP, Any, Any], AssignmentType]


ASSIGNMENT_FACTORIES: Dict[str, AssignmentFactory] = {

    # WRITE LOGIC
    "SB = <SRC>": write_expr_factory,

    # FIXME: Add these to the README.md
    "SB = ~<SRC>": write_expr_factory,

    # FIXME: Add these to the README.md
    "SB ?= <SRC>": write_expr_factory,

    # FIXME: Add these to the README.md
    "SB ?= ~<SRC>": write_expr_factory,

    # READ LOGIG #1 and #2
    "RL = <BIT>": read_expr_factory,

    # READ LOGIC #3, #4, #5
    "RL = <SB>": read_expr_factory,
    "RL = <SRC>": read_expr_factory,
    "RL = <SB> & <SRC>": read_expr_factory,

    # FIXME: Add these to the README.md
    "RL = ~<SB>": read_expr_factory,
    "RL = ~<SRC>": read_expr_factory,

    # READ LOGIC #10, #11, #12
    "RL |= <SB>": read_expr_factory,
    "RL |= <SRC>": read_expr_factory,
    "RL |= ~<SRC>": read_expr_factory,
    "RL |= <SB> & <SRC>": read_expr_factory,
    "RL |= <SB> & ~<SRC>": read_expr_factory,

    # READ LOGIC #13, #14, #15
    "RL &= <SB>": read_expr_factory,
    "RL &= <SRC>": read_expr_factory,
    "RL &= <SB> & <SRC>": read_expr_factory,
    "RL &= <SB> & ~<SRC>": read_expr_factory,

    # READ LOGIC #18, #19, #20
    "RL ^= <SB>": read_expr_factory,
    "RL ^= <SRC>": read_expr_factory,
    "RL ^= ~<SRC>": read_expr_factory,
    "RL ^= <SB> & <SRC>": read_expr_factory,
    "RL ^= <SB> & ~<SRC>": read_expr_factory,

    # READ LOGIC #6, #7
    "RL = <SB> | <SRC>": read_expr_factory,
    "RL = <SB> | ~<SRC>": read_expr_factory,
    "RL = <SB> ^ <SRC>": read_expr_factory,

    # READ LOGIC #8, #9
    "RL = ~<SB> & <SRC>": read_expr_factory,
    "RL = <SB> & ~<SRC>": read_expr_factory,

    # FIXME: UDOCUMENTED VARIATION OF READ LOGIC #9
    "RL = <SB> ^ ~<SRC>": read_expr_factory,

    # READ LOGIC #16, #17
    "RL &= ~<SB>": read_expr_factory,
    "RL &= ~<SRC>": read_expr_factory,

    # TODO: Determine whether this is a valid instruction and then add it to
    # the README.md:
    # - test_BLECCI_1.py::test_blecci_ex_5_drop_by_drop
    # - test_BLECCI_1.py::test_blecci_belop_1_test_5_bellib
    # - test_BLECCI_1.py::test_blecci_belop_1_test_5_better_parameters
    # - test_BLECCI_1.py::test_blecci_alices_lunch
    "RL = ~<SB> & ~<SRC>": read_expr_factory,

    # R-SEL LOGIC
    "GL = RL": broadcast_expr_factory,
    "RSP16 = RL": broadcast_expr_factory,

    "GGL = RL": broadcast_expr_factory,
    "GGL = RL & <LX>": broadcast_expr_factory,
    "GGL = <LX>": ggl_assignment_factory,

    # Workaround for pattern-matching caveate
    "GL = <SRC>": broadcast_expr_factory,
    "GGL = <SRC>": broadcast_expr_factory,
    "RSP16 = <SRC>": broadcast_expr_factory,

    # SPECIAL ASSIGNMENT
    "RSP16 = RSP256": rsp16_from_rsp256,

    # SPECIAL ASSIGNMENT
    "RSP256 = RSP16": rsp256_from_rsp16,
    "RSP256 = RSP2K": rsp256_from_rsp2k,

    # SPECIAL ASSIGNMENT
    "RSP2K = RSP256": rsp2k_from_rsp256,
    "RSP2K = RSP32K": rsp2k_from_rsp32k,

    # SPECIAL ASSIGNMENT
    "RSP32K = RSP2K": rsp32k_from_rsp2k,

    # Workaround for pattern-matching caveate
    "RSP256 = <SRC>": rsp256_from_rsp16,

    "LGL = <LX>": lgl_assignment_factory,

    "LX = GGL": lx_assignment_factory,
    "LX = <SRC>": lx_assignment_factory,

    "LX = LGL": lx_assignment_factory,
}


LVALUE_TO_STR = {
    str: lambda lvalue: lvalue,
    SB_EXPR: lambda lvalue: "SB",
    RL_EXPR: lambda lvalue: "RL",
    RSP16_EXPR: lambda lvalue: "RSP16",
    RSP256_EXPR: lambda lvalue: "RSP256",
    RSP2K_EXPR: lambda lvalue: "RSP2K",
    RSP32K_EXPR: lambda lvalue: "RSP32K",
    BROADCAST_EXPR: lambda lvalue: str(lvalue),
    # Workaround
    SRC_EXPR: lambda lvalue: str(lvalue),
    LGL_EXPR: lambda lvalue: str(lvalue),
    GGL_EXPR: lambda lvalue: str(lvalue),
    # lx_regs
    L1_REG: lambda lvalue: "LX",
    L2_REG: lambda lvalue: "LX",
    LXRegWithOffsets: lambda lvalue: "LX"
}


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


UNARY_SB_PATTERNS = {
    (None, "<SB>"): "<SB>",
    (None, "~<SB>"): "~<SB>",
    (UNARY_OP.NEGATE, "<SB>"): "~<SB>",
    (UNARY_OP.NEGATE, "~<SB>"): "<SB>",
}


UNARY_SRC_PATTERNS = {
    (None, "<SRC>"): "<SRC>",
    (None, "~<SRC>"): "~<SRC>",
    (UNARY_OP.NEGATE, "<SRC>"): "~<SRC>",
    (UNARY_OP.NEGATE, "~<SRC>"): "<SRC>",
}


BIT_PATTERNS = {
    0: "<BIT>",
    1: "<BIT>",
}


RVALUE_STR_PATTERNS = {
    "0": "<BIT>",
    "1": "<BIT>",

    RL: "<SRC>",
    NRL: "<SRC>",
    ERL: "<SRC>",
    WRL: "<SRC>",
    SRL: "<SRC>",
    GL: "<SRC>",
    GGL: "<SRC>",
    RSP16: "<SRC>",

    LGL: "LGL",

    INV_RL: "~<SRC>",
    INV_NRL: "~<SRC>",
    INV_ERL: "~<SRC>",
    INV_WRL: "~<SRC>",
    INV_SRL: "~<SRC>",
    INV_GL: "~<SRC>",
    INV_GGL: "~<SRC>",
    INV_RSP16: "~<SRC>",

    RSP256: "RSP256",
    RSP2K: "RSP2K",
    RSP32K: "RSP32K",
}


RVALUE_TO_STR = {
    str: lambda rvalue: RVALUE_STR_PATTERNS[rvalue],
    int: lambda rvalue: BIT_PATTERNS[rvalue],
    SB_EXPR: lambda rvalue: "<SB>",
    SRC_EXPR: lambda rvalue: SRC_PATTERNS[rvalue],
    RL_EXPR: lambda rvalue: "RL",
    GGL_EXPR: lambda rvalue: "GGL",
    LGL_EXPR: lambda rvalue: "LGL",
    RSP16_EXPR: lambda rvalue: "RSP16",
    RSP256_EXPR: lambda rvalue: "RSP256",
    RSP2K_EXPR: lambda rvalue: "RSP2K",
    RSP32K_EXPR: lambda rvalue: "RSP32K",
    BIT_EXPR: lambda rvalue: "<BIT>",
    UNARY_SB: lambda rvalue: UNARY_SB_PATTERNS[(rvalue.operator, rvalue_to_str(rvalue.expression))],
    UNARY_SRC: lambda rvalue: UNARY_SRC_PATTERNS[(rvalue.operator, rvalue_to_str(rvalue.expression))],
    UNARY_EXPR: lambda rvalue: rvalue_to_str(rvalue.expression),
    BINARY_EXPR: lambda rvalue: f"{rvalue_to_str(rvalue.left_operand)} {rvalue.operator} {rvalue_to_str(rvalue.right_operand)}",
    L1_REG: lambda rvalue: "<LX>",
    L2_REG: lambda rvalue: "<LX>",
    LXRegWithOffsets: lambda rvalue: "<LX>",
}


def rvalue_to_str(rvalue: Any) -> str:
    global RVALUE_TO_STR
    if isinstance(rvalue, BleirSerializable):
        rvalue = rvalue.as_bleir()
    return RVALUE_TO_STR[type(rvalue)](rvalue)


def lvalue_to_str(lvalue: Any) -> str:
    global LVALUE_TO_STR
    if isinstance(lvalue, BleirSerializable):
        lvalue = lvalue.as_bleir()
    return LVALUE_TO_STR[type(lvalue)](lvalue)


def build_assignment(operator: Union[ASSIGN_OP, str], lvalue: Any, rvalue: Any) -> ASSIGNMENT:
    """Builds an ASSIGNMENT operation from the given parameters by iterating over possible
    assignment pattern matches until it finds one that fits. Exceptions are raised if either no
    pattern matches the parameters or more than one pattern matches (redundancy)."""

    if isinstance(lvalue, BleirSerializable):
        lvalue = lvalue.as_bleir()

    if isinstance(operator, BleirSerializable):
        operator = operator.as_bleir()

    if isinstance(rvalue, BleirSerializable):
        rvalue = rvalue.as_bleir()

    if isinstance(operator, str):
        operator = ASSIGN_OP.find_by_value(operator)

    pattern = f"{lvalue_to_str(lvalue)} {operator} {rvalue_to_str(rvalue)}"
    if pattern not in ASSIGNMENT_FACTORIES:
        raise SemanticError(f"No pattern matches for: {pattern}")

    assignment_fn = ASSIGNMENT_FACTORIES[pattern]
    return assignment_fn(operator, lvalue, rvalue)


def assign(lvalue: Any, rvalue: Any, operator: ASSIGN_OP = ASSIGN_OP.EQ) -> ASSIGNMENT:
    return build_assignment(operator, lvalue, rvalue)


def cond_eq(lvalue: Any, rvalue: Any) -> ASSIGNMENT:
    return build_assignment(ASSIGN_OP.COND_EQ, lvalue, rvalue)


def and_eq(lvalue: Any, rvalue: Any) -> ASSIGNMENT:
    return build_assignment(ASSIGN_OP.AND_EQ, lvalue, rvalue)


def or_eq(lvalue: Any, rvalue: Any) -> ASSIGNMENT:
    return build_assignment(ASSIGN_OP.OR_EQ, lvalue, rvalue)


def xor_eq(lvalue: Any, rvalue: Any) -> ASSIGNMENT:
    return build_assignment(ASSIGN_OP.XOR_EQ, lvalue, rvalue)


def build_binary_expr(operator: BINOP,
                      left_operand: Union[UNARY_SB,
                                          SB_EXPR,
                                          RL_EXPR,
                                          str],
                      right_operand: Union[UNARY_SRC,
                                           SRC_EXPR,
                                           L1_REG,
                                           L2_REG,
                                           LXRegWithOffsets,
                                           str]) -> BINARY_EXPR:
    if isinstance(operator, BleirSerializable):
        operator = operator.as_bleir()
    if isinstance(left_operand, BleirSerializable):
        left_operand = left_operand.as_bleir()
    if isinstance(right_operand, BleirSerializable):
        right_operand = right_operand.as_bleir()
    if isinstance(left_operand, SB_EXPR):
        left_operand = UNARY_SB(left_operand)
    if isinstance(left_operand, str):
        if left_operand == "RL":
            left_operand = RL_EXPR.find_by_value(left_operand)
        else:
            left_operand = l1_reg(left_operand)
    if left_operand is SRC_EXPR.RL:
        left_operand = RL_EXPR.RL
    if isinstance(right_operand, str):
        right_operand = SRC_EXPR.find_by_value(right_operand)
    if isinstance(right_operand, SRC_EXPR):
        right_operand = UNARY_SRC(right_operand)
    if not isinstance(left_operand, (UNARY_SB, RL_EXPR)):
        raise NotImplementedError(f"Unsupported left_operand type: {left_operand}")
    if not isinstance(right_operand, (UNARY_SRC, L1_REG, L2_REG, LXRegWithOffsets)):
        raise NotImplementedError(f"Unsupported right_operand type: {right_operand}")
    return BINARY_EXPR(operator, left_operand, right_operand)


def conjoin(left_operand: Any, right_operand: Any) -> BINARY_EXPR:
    return build_binary_expr(BINOP.AND, left_operand, right_operand)


def disjoin(left_operand: Any, right_operand: Any) -> BINARY_EXPR:
    return build_binary_expr(BINOP.OR, left_operand, right_operand)


def xor(left_operand: Any, right_operand: Any) -> BINARY_EXPR:
    return build_binary_expr(BINOP.XOR, left_operand, right_operand)


def shift(register: Union[SM_REG, str], num_bits: int) -> SHIFTED_SM_REG:
    if isinstance(register, str):
        register = strip_rp_suffix(register.strip())
        register = SM_REG(register)
    return SHIFTED_SM_REG(register, num_bits)


Maskable = Union[MASK, SHIFTED_SM_REG, SM_REG, str]


def mask(mask: Maskable, negated: bool = False) -> MASK:
    if isinstance(mask, BleirSerializable):
        mask = mask.as_bleir()
    if isinstance(mask, str):
        mask = strip_rp_suffix(mask.strip())
        shift_parts = mask.split("<<")
        if len(shift_parts) == 2:
            reg_id, num_bits = (part.strip() for part in shift_parts)
            reg_id = strip_rp_suffix(reg_id)
            if reg_id.startswith("~"):
                reg_id = reg_id[1:].strip()
                mask = MASK(expression=SHIFTED_SM_REG(
                                register=SM_REG(
                                    identifier=reg_id),
                                num_bits=int(num_bits)),
                            operator=UNARY_OP.NEGATE)
            else:
                mask = MASK(expression=SHIFTED_SM_REG(
                                register=SM_REG(
                                    identifier=reg_id),
                                num_bits=int(num_bits)))
        elif mask.startswith("~"):
            mask = mask[1:].strip()
            mask = MASK(expression=SM_REG(identifier=mask),
                        operator=UNARY_OP.NEGATE)
        else:
            mask = MASK(expression=SM_REG(identifier=mask))
    if isinstance(mask, SHIFTED_SM_REG) or isinstance(mask, SM_REG):
        mask = MASK(mask)
    if negated:
        mask = invert(mask)
    return mask


def masked(mask_expr: Maskable,
           assignment: Optional[ASSIGNMENT]) -> MASKED:
    return MASKED(mask(mask_expr), assignment)


FN_or_BLEIR = Union[Callable, MASKED, SPECIAL, str]


def statement(fn_or_bleir: FN_or_BLEIR,
              comment: Optional[Comment] = None) -> Union[Callable, STATEMENT]:
    """May be used as either a decorator or factory function for STATEMENTs."""

    if callable(fn_or_bleir):
        fn = fn_or_bleir

        @wraps(fn)
        def wrapper(*args, **kwargs):
            masked = fn(*args, **kwargs)
            return statement(masked)

        return wrapper

    bleir = fn_or_bleir
    if isinstance(bleir, str):
        bleir = SPECIAL.find_by_value(bleir)
    return STATEMENT(operation=bleir, comment=comment)


# Factory function for the MultiStatement type
multi_statement = MultiStatement

# Factory function for the Fragment type
fragment = Fragment

# Factory function for the FragmentCaller type
fragment_caller = FragmentCaller

# Factory function for the FragmentCallerCall type
fragment_caller_call = FragmentCallerCall

# Factory function for the Snippet type
snippet = Snippet
