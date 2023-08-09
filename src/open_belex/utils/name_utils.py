"""
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

import re
from functools import wraps
from typing import ForwardRef, Type, Union

RE_CAMEL_CASE_TO_UNDERSCORE: re.Pattern = \
    re.compile("(?:([a-z])([A-Z])|([A-Z0-9])([A-Z][a-z]))")


def memoize(fn):
    memo = {}

    @wraps(fn)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        value = fn(*args)
        memo[args] = value
        return value

    return wrapper


def match_pair_to_underscore(match: re.Match) -> str:
    if match.group(1):
        lhs = match.group(1)
        rhs = match.group(2)
    else:
        lhs = match.group(3)
        rhs = match.group(4)
    return f"{lhs}_{rhs}"


def name_of(kind: Type) -> str:
    if isinstance(kind, ForwardRef):
        return kind.__forward_arg__
    if hasattr(kind, "_name"):
        return kind._name
    if hasattr(kind, "__name__"):
        return kind.__name__
    raise ValueError(f"Cannot determine name of kind: {kind}")


Kind_or_Name = Union[Type, str]


@memoize
def camel_case_to_underscore(kind_or_name: Kind_or_Name) -> str:
    if isinstance(kind_or_name, ForwardRef):
        source = kind_or_name.__forward_arg__
    elif isinstance(kind_or_name, type):
        source = name_of(kind_or_name)
    elif kind_or_name.__class__ is str:
        source = kind_or_name
    elif hasattr(kind_or_name, "_name"):
        source = kind_or_name._name
    else:
        raise ValueError(
            f"Unsupported type ({kind_or_name.__class__.__name__}): "
            f"{kind_or_name}")

    target = RE_CAMEL_CASE_TO_UNDERSCORE.sub(
        match_pair_to_underscore,
        source
    )

    while source != target:
        source = target
        target = RE_CAMEL_CASE_TO_UNDERSCORE.sub(
            match_pair_to_underscore,
            source
        )

    return target.lower()
