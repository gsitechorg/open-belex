r"""
By Dylon Edwards
"""

from enum import Enum
from functools import reduce
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Type

from open_belex.bleir.walkables import (camel_case_to_underscore, memoize,
                                        name_of)


class Cardinality(Enum):
    ONE = "ONE"
    MANY = "MANY"


class Field(NamedTuple):
    identifier: str
    cardinality: Cardinality
    kinds: Sequence["Kind"]
    nullable: bool


class Kind(NamedTuple):
    kind: Type
    docs: Optional[str]
    camel_case_id: str
    underscore_id: str
    fields: Sequence[Field]


def is_bleir(obj: Any) -> bool:
    return hasattr(obj, "__module__") and obj.__module__ == "open_belex.bleir.types"


def is_type_hint(obj: Any) -> bool:
    return hasattr(obj, "__module__") and obj.__module__ == "typing"


def unwrap_type_hints(hint: Any) -> Tuple[Sequence[Type], Cardinality]:
    if is_type_hint(hint):
        if hasattr(hint, "_name") and hint._name in ["Sequence", "List", "Set"]:
            hints, cardinality = unwrap_type_hints(hint.__args__[0])
            return hints, Cardinality.MANY

        if not hasattr(hint, "__args__"):
            return [hint], Cardinality.ONE

        hints = reduce(lambda lhs, rhs: lhs + rhs[0],
                       map(unwrap_type_hints, hint.__args__),
                       list())

        return hints, Cardinality.ONE

    return [hint], Cardinality.ONE


def inspect_field(bleir: Type, attr: str, hint: Any) -> Field:
    hints, cardinality = unwrap_type_hints(hint)
    kinds = [inspect_kind(kind) for kind in hints]
    nullable = None.__class__ in hints
    return Field(attr, cardinality, kinds, nullable)


@memoize
def inspect_kind(bleir: Type) -> Kind:
    docs = None
    if hasattr(bleir, "_doc"):
        docs = bleir._doc
    elif hasattr(bleir, "__doc__"):
        docs = bleir.__doc__

    if is_bleir(bleir):

        if not hasattr(bleir, "__annotations__"):  # Enum type
            return Kind(
                kind=bleir,
                docs=docs,
                camel_case_id=bleir.__name__,
                underscore_id=camel_case_to_underscore(bleir),
                fields=[inspect_field(bleir, "value", str)])

        fields = []
        for attr, hint in bleir.__annotations__.items():
            field = inspect_field(bleir, attr, hint)
            fields.append(field)

        return Kind(
            kind=bleir,
            docs=docs,
            camel_case_id=bleir.__name__,
            underscore_id=camel_case_to_underscore(bleir),
            fields=fields)

    fully_qualified_name = name_of(bleir)
    # fully_qualified_name = f"{bleir.__module__}.{fully_qualified_name}"

    return Kind(
        kind=bleir,
        docs=docs,
        camel_case_id=fully_qualified_name,
        underscore_id=camel_case_to_underscore(bleir),
        fields=[])
