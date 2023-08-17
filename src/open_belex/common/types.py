r"""
By Dylon Edwards
"""

from typing import Dict, Iterator, Sequence, Set, Union

import numpy as np


Integer = Union[int,  # Python built-in, 64-bit signed int
                np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64]


Indices = Union[
    Sequence[Union[Integer, str]],
    Sequence[bool],
    Set[Union[Integer, str]],
    Iterator[Union[Integer, str]],
    Iterator[bool],
    Dict[Union[Integer, str], bool],
    range,
    slice,
    Integer,
    str,
]
