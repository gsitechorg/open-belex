r"""By Brian Beckman.

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


# Do not import "Types" here because of circular dependencies:
# VECTOR is a Mask, Mask is a Subset.


from collections.abc import Iterable
from functools import reduce
from itertools import compress, groupby
from math import log2
from typing import List, Set, Union

import numpy as np

from open_belex.common.constants import NUM_HALF_BANKS_PER_APUC
from open_belex.common.types import Integer


class Subset(object):
    r"""A class for subsets of 'ranges'. A 'range' is a set of integers
    from 0 (inclusive) to mex (sic, exclusive). Max = mex - 1. The name
    'mex' reminds us that mex is 'ex'cluded from the range and from any
    subset of the range. The name 'max' is more familiar for the
    maximum possible value. In Charles Simonyi's classic 'Hungarian
    notation,' 'mex' was called 'mac'.

    There are several mathematical notations for a range. [0..mex), with
    round-brackets suggesting exclusion and square brackets suggesting
    inclusion; [0..max]; range(mex); {0, 1, ..., max}, standard set
    notation. "Range" is the only one that does what you expect in
    Python. {1, 2, ..., 15} yields {1, 2, 15, Ellipsis}. The others are
    just Python syntax errors. These notations are just mathematical
    abstractions at the present time.

    Examples: range(16) = [0..15] = [0..16) = {0, 1, ..., 15}
              max == 15, mex == 16;
              That's the range of section indices or section numbers.

              range(2048) = [0..2047] = [0..2048) = {0, 1, ..., 2047}
              max == 2047, mex == 2048;
              That's the range of plat indices or plat numbers, which
              we use for markers.

    We use Subsets to represent section masks, markers, and collections
    of VRs (in the future).

    The internal representation of a Subset is a numpy ndarray of
    int16s. Various properties produce and consume various other
    representations.

    TODO: Update with kwargs design from "Mask."
    """

    def __repr__(self):
        return f'0x{self.big_endian_hex}'
        # return str(self.list)

    @property
    def bit_count(self):
        result = len(self._bit_index_array)
        return result

    @staticmethod
    def bit_array_from_big_endian_bit_string(bit_string: str):
        r"""Produce little-endian bit-array from big-endian bit string."""
        bit_list_2 = list(compress(
            range(len(bit_string)),
            [int(b) for b in bit_string[::-1]]))
        # Old, inelegant way
        #
        # bit_list: List[int] = []
        # for i, c in enumerate(bit_string[::-1]):
        #     if c == '1':
        #         bit_list += [i]
        #     elif c != '0':
        #         raise ValueError(f"Bits must be 0 or 1; you gave a {c}")
        # assert all([b == b2 for b, b2 in zip(bit_list, bit_list_2)])
        bit_array = np.array(bit_list_2, dtype=np.int16)
        return bit_array

    def bit_array_from_big_endian_hex_string(self, hex_string: str):
        r"""Produce little-endian list TODO: other ops produce big-endian lists
        TODO: Northward and Southward depend on this little-endian-ness!"""
        full_integer = int(hex_string, base=16)
        # Old, inelegant
        # bit_string = bin(full_integer)[2:].zfill(self.max)
        bit_string_2 = f'{full_integer:b}'.zfill(self.max)
        # assert bit_string == bit_string_2
        bit_array = \
            self.bit_array_from_big_endian_bit_string(bit_string_2)
        return bit_array

    def bit_array_from_single_bit_index(self, bit_number: Integer):
        self.check_int_range(bit_number)
        bit_array = np.array([bit_number], dtype=np.int16)
        return bit_array

    @staticmethod
    def bit_array_from_bool_array(
            bools: Union[List[bool], np.ndarray]):
        indices_2 = compress(range(len(bools)), bools)
        indices_2_list = list(indices_2)
        return indices_2_list
        # Old, inelegant way:
        # indices = [i for i in range(len(bools)) if bools[i]]
        # assert all([i == i2 for i, i2 in zip(indices, indices_2_list)])
        # return np.array(indices_2, dtype=np.int16)

    def bit_array_from_bit_number_iterable(self, bit_number_iterable):
        for h in bit_number_iterable:
            self.check_int_range(h)
        l_unsorted = list(set(bit_number_iterable))
        # l_big_endian = sorted(l_unsorted, reverse=True)
        bit_array = np.array(l_unsorted, dtype=np.int16)
        return bit_array

    def check_int_range(self, i: int):
        if not isinstance(i, Integer.__args__):
            raise TypeError(
                f"Elements of range subsets must be ints. You gave {type(i)}")
        if (i < 0) or (i >= self.mex):
            raise ValueError(
                f"Elements of this subset must be between 0 and {self.max},"
                f" inclusive both ends. You gave {i}.")

    def random(self, seed=None):
        """Produce a random hex string."""
        prng = np.random.default_rng(seed)
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
        # That doc shows that the argument to integers must be self.max
        # and not self.mex. Also states that .integers should be used
        # for new code, not legacy .randint or .random_integers.
        rint = prng.integers(2**self.mex - 1)
        result = Subset(max=self.max, user_input=f'{rint:04X}', hex=True)
        return result

    def inv_hex(self):
        temp0 = ~self
        result = temp0.big_endian_hex
        return result

    def __iter__(self):
        r"""Return a generator comprehension."""
        return (m for m in self._bit_index_array)

    def indicator(self, item: Union[int, np.int16]) -> int:
        """Iverson's 'indicator' function, domain extended to all ints"""
        result = 1 if item in self._bit_index_array else 0
        return result

    # TODO: consider re-purposing __getitem__ to Iverson's indicator function.

    # def __getitem__(self, item):
    #     return self._bit_index_array[item]

    def __len__(self):
        return len(self._bit_index_array)

    def __eq__(self, other):
        temp0 = set(self._bit_index_array)
        temp1 = set(other._bit_index_array)
        result = temp0 == temp1
        return result

    def __getitem__(self, item):
        result = item in self._bit_index_array
        return result

    def __invert__(self):

        # TODO: make this work
        # test_0 = self.full_integer
        # test_1 = ~test_0
        # test_2 = Subset(max=self.max, user_input=[], hex=False)
        # test_2.full_integer = test_1

        bit_string = self.big_endian_binary
        temp = reduce(lambda s, c: s + ('1' if (c == '0') else '0'),
                      bit_string, "")
        result = Subset(max=self.max, user_input=temp, hex=False)
        return result

    def __or__(self, other: 'Subset'):
        bit_string = self.big_endian_binary
        other_bits = other.big_endian_binary
        ored_bits = [int(b) | int(o) for b, o in zip(bit_string, other_bits)]
        stred_bits = ''.join([str(b) for b in ored_bits])
        result = Subset(max=self.max, user_input=stred_bits, hex=False)
        return result

    def __and__(self, other: 'Subset'):
        bit_string = self.big_endian_binary
        other_bits = other.big_endian_binary
        anded_bits = [int(b) & int(o) for b, o in zip(bit_string, other_bits)]
        stred_bits = ''.join([str(b) for b in anded_bits])
        result = Subset(max=self.max, user_input=stred_bits, hex=False)
        return result

    def __xor__(self: 'Subset', other: 'Subset') -> 'Subset':
        bit_string = self.big_endian_binary
        other_bits = other.big_endian_binary
        xored_bits = [int(b) ^ int(o) for b, o in zip(bit_string, other_bits)]
        stred_bits = ''.join([str(b) for b in xored_bits])
        result = Subset(max=self.max, user_input=stred_bits, hex=False)
        return result

    def __lshift__(self, other: int):
        if (other > self.max):
            raise ValueError(f'Attempt to left-shift by {other} bits,'
                             f' more than the width {self.max}.')
        if (other < 0):
            raise ValueError(f'Attempt to left-shift by {other} bits,'
                             f' a negative number.')
        temp_0 = self.full_integer
        temp_1 = (temp_0 << other)
        temp_3 = (temp_1 & self.max_value)
        result = Subset(max=self.max)
        result.full_integer = temp_3
        return result

    @property
    def little_endian_numpy_bool_array(self) -> np.ndarray:
        r"""Make little-endian numpy array of bools for copying and for
        logic ops."""
        temp0 = self.big_endian_binary
        little_endian_binary = temp0[::-1]
        temp4 = list([int(c) for c in little_endian_binary])
        little_endian_result = np.array(temp4, dtype=bool)
        return little_endian_result

    @little_endian_numpy_bool_array.setter
    def little_endian_numpy_bool_array(self, little_endian_bool_or_bit_array):
        result = []
        for i, k in enumerate(little_endian_bool_or_bit_array):
            if bool(k):
                result.append(i)
        result_2 = np.array(result, dtype=np.int16)
        self._bit_index_array = result_2

    @property
    def sections(self):
        r"""Make numpy vector of bools for copying and for logic ops."""
        return self.little_endian_numpy_bool_array

    @property
    def array(self) -> np.ndarray:
        return self._bit_index_array

    @property
    def little_endian_slices(self) -> List[slice]:
        r"""Converts a Subset into a little-endian, run-length-encoded
        sequence of slices for the '1' bits. Trog!"""
        little_endian = self.big_endian_binary[::-1]
        grouped = groupby(little_endian)
        keys = []
        lengths = []
        for k, v in grouped:
            keys.append(k)
            lengths.append(len(list(v)))
        kls = zip(keys, lengths)
        # pairwise from more-itertools seems not to work. TROG IT!
        mex = 0
        kmms = []
        for kl in list(kls):  # we get to consume kls ONCE only
            min = mex
            mex += kl[1]
            kmms.append((kl[0], min, mex))
        result = [slice(kmm[1], kmm[2]) for kmm in kmms if kmm[0] == '1']
        return result

    @little_endian_slices.setter
    def little_endian_slices(self, slyces):
        recon = np.zeros(self.mex, dtype=bool)
        for slyce in slyces:
            recon[slyce] = '1'
        self.little_endian_numpy_bool_array = recon

    @property
    def list(self) -> List[int]:
        r"""BIG-endian? TODO: check and consider renaming."""
        return list(self._bit_index_array)

    @property
    def set(self) -> Set[int]:
        return set(self._bit_index_array)

    @property
    def full_integer(self) -> int:
        temp0 = reduce(
            lambda a, b: a + 2 ** int(b),
            self._bit_index_array,
            0)
        return temp0

    @full_integer.setter
    def full_integer(self, value: int):
        if (value < 0) or (value >= (2 ** self.mex)):
            raise ValueError(
                rf"full integer must be between 0 (inclusive) and"
                rf" {2 ** self.mex} (exclusive). You gave {value}.")
        # Pick the "ON" bits
        compressed = compress(range(self.mex),
                              [int(b) for b in f'{value:b}'[::-1]])
        compressed_list = list(compressed)
        # Old, inelegant way:
        # filtered = filter(lambda i: 1 & (value >> i), range(self.mex))
        # result = np.array(list(filtered), dtype=np.int16)
        # assert all(compressed_list == result)
        self._bit_index_array = compressed_list  # result

    @property
    def flood_filled_right(self):
        x = self.full_integer
        # See 'Hacker's Delight,' the fourth formula.
        y = 0 if x == 0 else x | (x - 1)
        result = Subset(self.max)
        result.full_integer = y
        return result

    @property
    def flood_filled_down(self):
        return self.flood_filled_left

    @property
    def flood_filled_left(self):
        # TODO: trog!
        x = self.big_endian_binary
        y = Subset(self.max, x[::-1], hex=False)
        z = y.flood_filled_right.big_endian_binary
        result = Subset(self.max, z[::-1], hex=False)
        return result

    @property
    def flood_filled_up(self):
        return self.flood_filled_right

    @property
    def big_endian_hex(self) -> str:
        temp0 = self.full_integer
        result = f'{temp0:X}'.zfill(self.hex_zfill_len)
        return result

    @property
    def little_endian_hex(self):
        temp0 = self.big_endian_hex
        temp1 = len(temp0)
        assert (temp1 % 4) == 0
        assert (temp1 <= 512 * NUM_HALF_BANKS_PER_APUC)
        # First, reverse each hex:
        temp2 = ''
        for i in range(temp1):
            batch = temp0[i:(i + 1)]
            int_ = int(batch, 16)
            rh_ = (int_ & 0xA) >> 1
            rl_ = (int_ & 0x5) << 1
            r1_ = rh_ | rl_
            r2h_ = (r1_ & 0xC) >> 2
            r2l_ = (r1_ & 0x3) << 2
            r2_ = r2h_ | r2l_
            temp2 += f'{r2_:X}'
        # Now, higher-level batches up to 256 (half of the max, 512)
        # each pair of hexes, each quad, each octuplet, ...
        # ((1 << j) == (2 ** j)) is the number of hexits in a batch.
        for j in range(1, 1 + int(log2(512 * NUM_HALF_BANKS_PER_APUC))):
            # how many f's and 0's in each half-mask
            two_p_of_j_m_1 = (1 << (j - 1))
            # how many hexits in the batch.
            two_p_of_j = two_p_of_j_m_1 << 1
            # The number of batches is the number of hexits divided
            # by the number of hexits in a batch
            number_of_batches = temp1 // two_p_of_j
            if number_of_batches == 0:
                # We're done, and none of the higher batch sizes will exist
                # either.
                break
            # half mask of all-1
            f_mask_str = 'F' * two_p_of_j_m_1
            # half mask of all-0
            z_mask_str = '0' * two_p_of_j_m_1
            # a mask that picks off the high bits
            h_mask = int(f_mask_str + z_mask_str, 16)
            # a mask that picks off the low bits
            l_mask = int(z_mask_str + f_mask_str, 16)
            # a place to put the answer as a string
            temp3 = ''
            # how many bits to shift each half-masked piece,
            # four bits for each hexit in a half-mask.
            bit_shift = (4 * two_p_of_j_m_1)
            for i in range(number_of_batches):
                # the hexits in a batch
                batch = temp2[(i * two_p_of_j):((i + 1) * two_p_of_j)]
                # as an int
                int_ = int(batch, 16)
                # swap the low and high
                nu_l = (int_ & h_mask) >> bit_shift
                nu_h = (int_ & l_mask) << bit_shift
                nu = (nu_l | nu_h)
                # pad them out for the string
                temp3 = temp3 + f'{nu:X}'.zfill(two_p_of_j)
            if temp3:
                temp2 = temp3
        result = temp2
        return result

    @property
    def big_endian_binary(self) -> str:
        temp0 = self.full_integer
        result = bin(temp0)[2:].zfill(self.mex)
        return result

    @property
    def is_zero(self):
        return len(self._bit_index_array) == 0

    @staticmethod
    def _input_is_all_bools(user_input) -> bool:

        is_list_or_array = (isinstance(user_input, list) or
                            isinstance(user_input, np.ndarray))

        is_not_empty = (len(user_input) > 0)

        types = [type(u) for u in user_input]

        # bool is different from np.bool_
        are_bools = [((t is bool) or (t is np.bool_)) for t in types]

        all_types_are_bools = np.all(np.array(are_bools))

        result = is_list_or_array \
            and is_not_empty \
            and all_types_are_bools

        return result

    def __init__(self: "Subset",
                 max: int = 1,
                 user_input: Union[int, np.int16, str, Iterable] = None,
                 hex: bool = True) -> None:
        r"""Classic SICP dispatch-on-type. The following are the
        acceptable types of input:

        Strings:
        - a hex string if "hex" is True
        - a binary bit string if "hex" is False

        Integers:
        - a single integer in [0..mex], or equivalently, [0..max)

        Iterables:
        - an array or np.ndarray of integers, each in [0..mex]
        - a set or range of integers, each in [0..mex]
        - a Subset, Mask, or Types.VECTOR
        - an array or np.ndarray of bools

        The term "mex" means "maximum value, exclusive." It is one more
        than "max". "Mex" is normally used for counts and lengths; "max"
        is normally used for indices.

        If you want to specify a subset via an integer not an exact
        power of two, that is,  with more than one bit on, you must use
        the 'full_integer' property setter on an instance of Subset
        where you have set the "max" attribute appropriately  at
        creation time. "Max" is normally a power of two less one.
        Subset is not tested for other values of "max," but it might
        work.

        If "hex" is True, an input of type str is interpreted as a big-
        endian hex string. Otherwise, an input of type str is
        interpreted as a big-endian string of binary.

        "Hex" defaults to True.
        """

        self._bit_index_array = np.array([], dtype=np.int16)

        if max <= 0:
            raise ValueError(rf'Max must be > 0; you gave {max}')

        self.max = max
        self.mex = max + 1
        self.max_value = (2 ** self.mex - 1)
        self.hex_zfill_len = 1 + max // 4

        if isinstance(user_input, str):
            if hex:
                bit_array = self.bit_array_from_big_endian_hex_string(
                    str(user_input))
            else:
                bit_array = self.bit_array_from_big_endian_bit_string(
                    str(user_input))

        elif isinstance(user_input, Integer.__args__):
            # No easy way to get rid of mypy error on int(user_input)
            bit_array = self.bit_array_from_single_bit_index(user_input)

        elif user_input is None:
            bit_array = self._bit_index_array

        elif self._input_is_all_bools(user_input):
            bit_array = self.bit_array_from_bool_array(user_input)

        elif isinstance(user_input, Iterable):
            assert isinstance(user_input, list) or \
                   isinstance(user_input, np.ndarray) or \
                   isinstance(user_input, set) or \
                   isinstance(user_input, range) or \
                   isinstance(user_input, Subset)  # catches Mask and VECTOR
            bit_array = self.bit_array_from_bit_number_iterable(user_input)

        else:
            raise TypeError(
                f"Input type must be str, int, np.int16, None, or Iterable."
                f" Your input, {user_input}, has type {type(user_input)}.")

        self._bit_index_array = bit_array

