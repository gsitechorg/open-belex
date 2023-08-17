r"""
By Brian Beckman and Dylon Edwards
"""

from typing import Iterable, Union

import numpy

from open_belex.common.constants import NSECTIONS
from open_belex.common.subset import Subset


class IllegalArgumentError(ValueError):
    pass


Maskable = Union[int, numpy.int16, str, Iterable]


class Mask(Subset):

    def __init__(self,
                 user_input: Maskable = None,
                 as_hex: bool = True,
                 **kwargs,
                 ):
        r"""Mimic the old mask, which knew that strings of length
        exactly NSECTIONS are binary strings. Also implement more
        usable 'kwargs' scheme. See test/test_Mask.py."""

        if (user_input is not None) and (len(kwargs) > 0):
            raise IllegalArgumentError(
                f'''Mask takes either one or two positional arguments,
user_input and as_hex, or one of the kwargs
bit_indices, bit_numbers, packed_integer, hex_str, binary_str.
You gave user_input={user_input}, as_hex={as_hex},
kwargs={kwargs}''')

        if user_input is not None:
            if isinstance(user_input, str) and len(user_input) == NSECTIONS:
                as_hex = False
        else:
            if len(kwargs) > 1:
                raise IllegalArgumentError(
                    f'''If there is no "user_input" in the argument list,
Mask takes exactly one of the kwargs
bit_indices, bit_numbers, packed_integer, hex_str, binary_str.
You gave user_input={user_input}, as_hex={as_hex},
kwargs={kwargs}''')

        if 'bit_indices' in kwargs:
            user_input = kwargs['bit_indices']
        elif 'bit_numbers' in kwargs:
            user_input = kwargs['bit_numbers']
        elif 'packed_integer' in kwargs:
            user_input = f'{kwargs["packed_integer"]:X}'
        elif 'hex_str' in kwargs:
            remove_possible_leading_0x = int(kwargs['hex_str'], 16)
            user_input = f'{remove_possible_leading_0x:X}'
            as_hex = True
        elif 'binary_str' in kwargs:
            remove_possible_leading_0b_and_underscores = \
                int(kwargs['binary_str'], 2)
            user_input = f'{remove_possible_leading_0b_and_underscores:b}'
            as_hex = False

        super().__init__(
            max=(NSECTIONS - 1),
            user_input=user_input,
            hex=as_hex)

    @staticmethod
    def no_sections():
        return Mask("0000")

    @staticmethod
    def all_sections():
        return Mask("FFFF")

    def __str__(self):
        return str(self.list)
        # return self.big_endian_binary
