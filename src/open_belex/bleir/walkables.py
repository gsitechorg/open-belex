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

from enum import Enum
from functools import wraps
from typing import Any, Dict, Optional, Union

from open_belex.bleir.types import instance_members_of
# NOTE: `memoize` and `name_of` are imported for backwards compatibility
from open_belex.utils.name_utils import camel_case_to_underscore, memoize, name_of


class BLEIRVisitor:
    """Use this type when you want to control how the BLEIR tree is walked. To understand which
    methods are supported, please see test_bleir_walkers.py::stub_visitor()."""

    @staticmethod
    def visit_name_for(kind: Any) -> Optional[str]:
        if is_bleir(kind):
            underscore_nym = camel_case_to_underscore(kind)
            return f"visit_{underscore_nym}"


class BLEIRTransformer:
    """Use this type when you need to transform BLEIR trees. Each rule is optional and returns a
    transformed BLEIR subtree. BLEIRTransformer is useful for such tasks as optimization or
    interpolation. To understand which methods are supported, please see
    test_bleir_walkers.py::stub_transformer()."""

    @staticmethod
    def transform_name_for(kind: Any) -> Optional[str]:
        if is_bleir(kind):
            underscore_nym = camel_case_to_underscore(kind)
            return f"transform_{underscore_nym}"


class BLEIRListener:
    """Use this type when you don't need to control the tree traversal but need to handle enter and
    exit events when rules are visited (e.g. analysis or inference). To understand which methods
    are supported, please see test_bleir_walkers.py::stub_listener()."""

    @staticmethod
    def enter_name_for(kind: Any) -> Optional[str]:
        if is_bleir(kind):
            underscore_nym = camel_case_to_underscore(kind)
            return f"enter_{underscore_nym}"

    @staticmethod
    def exit_name_for(kind: Any) -> Optional[str]:
        if is_bleir(kind):
            underscore_nym = camel_case_to_underscore(kind)
            return f"exit_{underscore_nym}"


Walkable = Union[BLEIRVisitor, BLEIRListener, BLEIRTransformer]


def category(walkable: Walkable) -> str:
    """Returns the category of the current visitor, which is one of the following:
        1. rewriter
        2. syntactic_validator
        3. semantic_validator
        4. optimizer
        5. interpreter
        6. generator

    For example, take an instance of bleir.optimizers.ConvergenceOptimizer:

        kind = walkable.__class__           #-> ConvergenceOptimizer
        package = kind.__module__           #-> bleir.optimizers
        package_parts = package.split(".")  #-> ("bleir", "optimizers")
        categories = package_parts[-1]      #-> "optimizers"
        category = categories[:-1]          #-> "optimizer"
    """
    kind = walkable.__class__
    package = kind.__module__
    package_parts = package.split(".")
    categories = package_parts[-1]
    category = categories[:-1]
    return category


def is_bleir(obj: Any) -> bool:
    return hasattr(obj, "__module__") and obj.__module__ == "open_belex.bleir.types"


def walkabout(fn):

    @wraps(fn)
    def wrapper(self: "BLEIRWalker",
                walkable: Walkable,
                bleir: Optional[Any]) -> Any:

        if bleir is None:
            return None

        if isinstance(walkable, BLEIRVisitor):
            visitor = walkable

            visit_nym = BLEIRVisitor.visit_name_for(bleir.__class__)
            if visit_nym is not None and hasattr(walkable, visit_nym):
                visit_fn = getattr(visitor, visit_nym)
                visit_fn(bleir)
                return bleir

            fn(self, visitor, bleir)

        if isinstance(walkable, BLEIRListener):
            listener = walkable
            enter_nym = BLEIRListener.enter_name_for(bleir.__class__)
            if enter_nym is not None and hasattr(walkable, enter_nym):
                enter_fn = getattr(listener, enter_nym)
                enter_fn(bleir)

        if isinstance(walkable, BLEIRTransformer):
            transformer = walkable

            children = fn(self, transformer, bleir)
            if bleir is None:
                return None

            if is_bleir(bleir) and not isinstance(bleir, Enum):
                construct = bleir.__class__
                parameters = {arg: children[arg] for arg in bleir.__annotations__.keys()}
                bleir = construct(**parameters)

            transform_nym = BLEIRTransformer.transform_name_for(bleir.__class__)
            if transform_nym is not None and hasattr(transformer, transform_nym):
                transform_fn = getattr(transformer, transform_nym)
                bleir = transform_fn(bleir)

        # It is possible for walkable to be both a listener and transformer, but the
        # listener methods are invoked during transformation so there's no need to invoke them
        # twice.
        elif isinstance(walkable, BLEIRListener):
            listener = walkable
            fn(self, listener, bleir)

        if isinstance(walkable, BLEIRListener):
            listener = walkable
            exit_nym = BLEIRListener.exit_name_for(bleir.__class__)
            if exit_nym is not None and hasattr(walkable, exit_nym):
                exit_fn = getattr(listener, exit_nym)
                exit_fn(bleir)

        return bleir

    return wrapper


class BLEIRWalker:

    @walkabout
    def walk(self: "BLEIRWalker",
             walkable: Walkable,
             bleir: Optional[Any]) -> Dict[str, Any]:

        children = {}

        if is_bleir(bleir):
            for attr, value in instance_members_of(bleir):
                if is_bleir(value):
                    child = self.walk(walkable, value)
                elif isinstance(value, (tuple, list)):
                    child = []
                    for item in value:
                        if is_bleir(item):
                            grandchild = self.walk(walkable, item)
                            child.append(grandchild)
                        else:
                            child.append(item)
                    child = tuple(child)
                else:
                    child = value
                children[attr] = child

        return children
