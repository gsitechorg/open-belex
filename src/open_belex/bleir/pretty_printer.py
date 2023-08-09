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

from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from sys import stdout
from typing import Any, Callable, Optional

from open_belex.bleir.inspectors import is_bleir
from open_belex.bleir.syntactic_validators import BLEIRTypeValidator


@dataclass
class PrettyPrinter:

    # `context` does not have to be a specific type, but it must contain a `write(str)` method.
    # (see: https://docs.python.org/3/library/functions.html#print)
    context: Any = stdout

    validator: BLEIRTypeValidator = field(default_factory=BLEIRTypeValidator)

    is_erroneous: bool = False

    colorize: bool = True

    def print_to_context(self: "PrettyPrinter", text: str, indent_level: int) -> None:
        indent = "  " * indent_level
        if self.is_erroneous:
            text = self.light_red(self.underline(text))
        print(f"{indent}{text}", end="", file=self.context)

    def newline(self: "PrettyPrinter") -> None:
        print("", file=self.context)

    def open_paren(self: "PrettyPrinter", indent_level: int) -> None:
        if indent_level == 0:
            self.print_to_context("'(", indent_level)
        else:
            self.print_to_context("(", indent_level)

    def close_paren(self: "PrettyPrinter", indent_level: int) -> None:
        self.print_to_context(")", 0)
        if indent_level == 0:
            self.newline()

    def underline(self: "PrettyPrinter", value: str) -> str:
        if not self.colorize:
            return value
        return f"\033[4m{value}"

    def light_red(self: "PrettyPrinter", value: str) -> str:
        if not self.colorize:
            return value
        return f"\033[91m{value}\033[0m"

    def magenta(self: "PrettyPrinter", value: str) -> str:
        if not self.colorize:
            return value
        return f"\033[35m{value}\033[0m"

    def light_blue(self: "PrettyPrinter", value: str) -> str:
        if not self.colorize:
            return value
        return f"\033[94m{value}\033[0m"

    def cyan(self: "PrettyPrinter", value: str) -> str:
        if not self.colorize:
            return value
        return f"\033[36m{value}\033[0m"

    def light_green(self: "PrettyPrinter", value: str) -> str:
        if not self.colorize:
            return value
        return f'\033[32m{value}\033[0m'

    def yellow(self: "PrettyPrinter", value: str) -> str:
        if not self.colorize:
            return value
        return f'\033[33m{value}\033[0m'

    def fmt_identifier(self: "PrettyPrinter", identifier: str) -> str:
        if self.is_erroneous:
            return identifier
        return self.magenta(identifier)

    def fmt_kind(self: "PrettyPrinter", bleir: Any) -> str:
        if self.is_erroneous:
            return bleir.__class__.__name__
        return self.cyan(bleir.__class__.__name__)

    def fmt_string(self: "PrettyPrinter", value: str) -> str:
        if self.is_erroneous:
            return f'"{value}"'
        return self.light_green(f'"{value}"')

    def fmt_enum(self: "PrettyPrinter", enumeration: str) -> str:
        if self.is_erroneous:
            return f'(enum {enumeration.name} {self.fmt_string(enumeration.value)})'
        return f'({self.yellow("enum")} {self.fmt_identifier(enumeration.name)} {self.fmt_string(enumeration.value)})'

    def fmt_literal(self: "PrettyPrinter", literal: str) -> str:
        if self.is_erroneous:
            return literal
        return self.light_blue(literal)

    def has_error(self: "PrettyPrinter", context: Any) -> bool:
        if not self.is_erroneous and is_bleir(context) and not isinstance(context, Enum):
            pass
        return False

    def pprint(self: "PrettyPrinter", context: Any,
               identifier: Optional[str] = None,
               indent_level: int = 0) -> None:

        self.open_paren(indent_level)

        if identifier is None:
            self.print_to_context(self.fmt_kind(context), 0)
        else:
            self.print_to_context(f"{self.fmt_identifier(identifier)} {self.fmt_kind(context)}", 0)

        if is_bleir(context) and not isinstance(context, Enum):
            for attr, hint in context.__class__.__annotations__.items():
                value = getattr(context, attr)

                has_valid_syntax = True
                if not self.is_erroneous:
                    has_valid_syntax = self.validator.assert_satisfies_hint(value, hint)
                    self.is_erroneous = not has_valid_syntax

                self.newline()
                self.pprint(value, identifier=attr, indent_level=(indent_level + 1))

                if not has_valid_syntax:
                    self.is_erroneous = False

        elif isinstance(context, dict):
            for key, value in context.items():
                self.newline()
                self.pprint(value, identifier=key, indent_level=(indent_level + 1))

        elif isinstance(context, str):
            self.newline()
            self.print_to_context(self.fmt_string(context), (indent_level + 1))

        elif is_iterable(context):
            for value in context:
                self.newline()
                self.pprint(value, indent_level=(indent_level + 1))

        elif isinstance(context, Enum):
            self.newline()
            self.print_to_context(self.fmt_enum(context), (indent_level + 1))

        else:
            self.newline()
            self.print_to_context(self.fmt_literal(context), (indent_level + 1))

        self.close_paren(indent_level)


Fn_or_BLEIR_or_Header = Any


def is_iterable(arg: Any) -> bool:
    return hasattr(arg, "__len__") and hasattr(arg, "__getitem__") \
        or hasattr(arg, "__iter__")


def pretty_print(fn_or_bleir_or_header: Fn_or_BLEIR_or_Header,
                 header: Optional[str] = None,
                 colorize: bool = True) -> Optional[Callable]:
    """Pretty prints the BLEIR tree of the value returned from some function.
    If a BLEIR instance is provided instead of a function, the BLEIR instance
    is pretty printed and the same instance will be returned. If the given
    parameter is a str, it is assumed to be a header to print before
    pretty-printing the decorated function's return value.

    Examples:

        @pretty_print
        def get_broadcast_expr_gl():
            return BLEIR.BROADCAST_EXPR.GL

        @pretty_print("This is a header.")
        def get_src_expr_rl():
            return BLEIR.SRC_EXPR.RL

        # snippet is an instance of BLEIR.Snippet
        pretty_print(snippet)"""

    if isinstance(fn_or_bleir_or_header, str):
        header = fn_or_bleir_or_header

        def decorator(fn):
            return pretty_print(fn, header=header)

        return decorator

    pp = PrettyPrinter(colorize=colorize)

    if is_bleir(fn_or_bleir_or_header):
        bleir = fn_or_bleir_or_header
        pp.pprint(bleir)
        return bleir

    # If it isn't a BLEIR or a function to decorate, raise an error
    if not callable(fn_or_bleir_or_header):
        raise ValueError(
            f"Unsupported fn_or_bleir_or_header type "
            f"({fn_or_bleir_or_header.__class__.__name__}): "
            f"{fn_or_bleir_or_header}")

    fn = fn_or_bleir_or_header

    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal fn, pp
        retval = fn(*args, **kwargs)
        if is_bleir(retval):
            pp.pprint(retval, identifier=header)
        elif hasattr(retval, "__caller__") and is_bleir(retval.__caller__):
            pp.pprint(retval.__caller__)
        return retval

    return wrapper
