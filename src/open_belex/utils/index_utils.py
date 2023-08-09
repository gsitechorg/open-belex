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

from typing import Any, Dict, Optional, Sequence, Union

from open_belex.common.types import Indices, Integer


def is_sequence(xs: Any) -> bool:
    return hasattr(xs, "__len__") and hasattr(xs, "__getitem__")


def is_iterator(xs: Any) -> bool:
    return hasattr(xs, "__next__")


def parse_indices(indices: Optional[Indices],
                  upper_bound: int) -> Sequence[Integer]:
    if indices is None:
        indices = list(range(upper_bound))

    elif isinstance(indices, Integer.__args__):
        indices = list(range(indices))

    elif isinstance(indices, str):
        if len(indices) >= 2 and indices.startswith("0x"):
            if indices == "0x0000":
                # Special case
                return []
            indices = parse_hex_literal(indices)
        else:
            indices = list(map(
                lambda index: parse_index(index, upper_bound),
                indices))

    elif isinstance(indices, range):
        indices = list(indices)

    elif isinstance(indices, slice):
        indices = translate_index_slice(indices, upper_bound)

    # must check for dict before seq because dicts are seqs
    elif isinstance(indices, dict):
        indices = translate_index_map(indices, upper_bound)

    elif isinstance(indices, set):
        indices = sorted(map(
            lambda index: parse_index(index, upper_bound),
            indices))

    elif is_sequence(indices) or is_iterator(indices):
        indices = list(indices)

    else:
        raise ValueError(
            f"Unsupported indices type ({indices.__class__.__name__}): "
            f"{indices}")

    if len(indices) > 0:
        if isinstance(indices[0], bool):
            indices = translate_index_mask(indices, upper_bound)
        elif isinstance(indices[0], (*Integer.__args__, str)):
            indices = list(map(
                lambda index: parse_index(index, upper_bound),
                indices))
        else:
            raise ValueError(
                f"Unsupported list type for indices "
                f"({indices[0].__class__.__name__}): {indices}")

    if len(indices) == 0:
        indices = list(range(upper_bound))

    elif len(indices) > upper_bound:
        raise ValueError(f"Too many indices ({len(indices)}): {indices}")

    return indices


def parse_hex_literal(hex_literal: str) -> Sequence[Integer]:
    if len(hex_literal) < 3:
        raise ValueError(
            f"Hex literal must have 1 to 4 hex digits: {hex_literal}")

    hex_literal = hex_literal[len("0x"):]  # drop the "0x" prefix
    hex_literal = int(hex_literal, 16)
    if hex_literal > 0xFFFF:
        raise ValueError(
            f"Hex literal may not exceed 0xFFFF ({0xFFFF}, dec): {hex_literal}")

    indices = []
    for index in range(16):
        bit = (1 << index)
        if hex_literal & bit != 0:
            indices.append(index)

    return indices


def parse_index(index: Union[str, Integer], upper_bound: int) -> Integer:
    if isinstance(index, Integer.__args__):
        if not 0 <= index < upper_bound:
            raise ValueError(
                f"Expected index to be in the range [0,{upper_bound}): {index}")
        return index

    if len(index) != 1:
        raise ValueError(
            f"Expected exactly one index but received {len(index)}: \"{index}\"")

    if index not in "0123456789abcdefABCDEF":
        raise ValueError(f"Expected index to be hex but was: \"{index}\"")

    return int(index, 16)


def translate_index_map(mapping: Dict[Union[Integer, str], bool],
                        upper_bound: int) -> Sequence[Integer]:

    index_mask = [False] * upper_bound

    for index, is_included in mapping.items():
        if not isinstance(index, (*Integer.__args__, str)):
            raise ValueError(
                f"Index {index} of indices should be an int or hex str, not "
                f"{index.__class__.__name__}: {mapping}")

        if isinstance(index, Integer.__args__) and not 0 <= index < upper_bound:
            raise ValueError(
                f"Expected index to be in the range [0,{upper_bound}): {index}")

        elif isinstance(index, str):
            index = parse_index(index, upper_bound)

        if not isinstance(is_included, bool):
            raise ValueError(
                f"Expected is_included at index {index} to be a bool but was "
                f"{is_included.__class__.__name__}: {mapping}")

        if is_included:
            index_mask[index] = True

    return translate_index_mask(index_mask, upper_bound)


def translate_index_slice(indices: slice, upper_bound: int) -> Sequence[Integer]:
    start = indices.start
    if start is None:
        start = 0

    stop = indices.stop
    if stop is None:
        stop = upper_bound

    step = indices.step
    if step is None:
        step = 1

    indices = range(start, stop, step)
    indices = list(indices)
    return indices


def translate_index_mask(index_mask: Sequence[bool],
                         upper_bound: int) -> Sequence[Integer]:

    indices = []

    if len(index_mask) != upper_bound:
        raise ValueError(
            f"Expected section mask to have {upper_bound} bools, but had "
            f"{len(index_mask)}")

    for index, is_included in enumerate(index_mask):
        if not isinstance(is_included, bool):
            raise ValueError(
                f"Index {index} of indices should be a boo, not "
                f"{is_included.__class__.__name__}: {indices}")

        if is_included:
            indices.append(index)

    return indices
