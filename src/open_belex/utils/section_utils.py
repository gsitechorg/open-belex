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

from typing import Optional, Sequence
from warnings import warn

from open_belex.common.constants import NSECTIONS
from open_belex.common.types import Indices, Integer
from open_belex.utils.index_utils import parse_indices


def parse_sections(indices: Optional[Indices] = None) -> Sequence[Integer]:
    """Parses an index or collection of indices to a sequence of integers
    representing specific sections in the range [0,16). Indices may take
    the following forms:

    1. Sequence[int] := A sequence of at most 16 integers in the range [0,16).
    2. Sequence[bool] := A sequence of 16 bools representing a section mask
    3. Sequence[str] := A sequence of at most 16 hex chars
    4. Sequence[Union[int, str]] := A sequence of at most 16 hex chars and ints
                                    in the range [0,16).
    5. Set[int] := A set of at most 16 integers in the range [0,16)
    6. Set[bool] := A set of 16 bools representing a section mask
    7. Set[str] := A set of at most 16 hex chars
    8. Set[Union[int, str]] := A set of at most 16 hex chars and ints in the
                               range [0,16).
    9. Iterator[int] := An iterator over at most 16 integers in the range
                        [0,16).
    10. Iterator[bool] := An iterator over 16 bools representing a section mask
    11. Iterator[str] := An iterator over at most 16 hex chars
    12. Iterator[Union[int, str]] := An iterator over at most 16 hex chars and
                                     ints in the range [0,16)
    13. Dict[int, bool] := A mapping of indices in the range [0,16) to whether
                           they should be included in the section mask.
    14. Dict[str, bool] := A mapping of hex indices to whether they should be
                           included in the section mask.
    15. Dict[Union[int, str], bool] := A mapping of hex and dec indices in the
                                       range [0,16) to whether they should be
                                       included in the section mask.
    16. range := A range of indices bounded by [0,16)
    17. slice := A slice of indices bounded by [0,16)
    18. int := A single index in the range [0,16)
    19. str := A sequence of at least one and at most 16 hex indices or a hex
               literal beginning with "0x" followed by 1 to 4 hex digits.

    Parameters:
        cls: Type variable representing the class owner of this method.
        indices: An index or collection of indices as described above.

    Returns:
        A sequence of integers representing specific sections, each in the
        range [0,16).
    """

    if isinstance(indices, Integer.__args__):
        if 0x000F < indices <= 0xFFFF:
            # [Heuristic] Try to detect a hex literal
            raise ValueError(
                f"An int literal index must represent a single section within "
                f"the range [0,16). Did you mean to specify a hex literal string? "
                f"{indices}")
        indices = [indices]

    indices = parse_indices(indices, upper_bound=NSECTIONS)

    if len(indices) == 0:
        # Special case
        return indices

    prev_index = indices[0]
    for i in range(1, len(indices)):
        curr_index = indices[i]
        if curr_index < prev_index:
            warn(f"indices should increase monotonically: {indices}")
            break
        prev_index = curr_index

    return indices
