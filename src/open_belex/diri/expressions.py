"""
By Dylon Edwards.
"""

from copy import deepcopy


def right(lhs, rhs):
    return deepcopy(rhs)


def INV_right(lhs, rhs):
    return (~rhs)


def left_OR_right(lhs, rhs):
    return lhs | rhs


def left_OR_INV_right(lhs, rhs):
    return lhs | (~rhs)


def left_AND_right(lhs, rhs):
    return lhs & rhs


def left_AND_INV_right(lhs, rhs):
    return lhs & (~rhs)


def left_XOR_right(lhs, rhs):
    return lhs ^ rhs


def left_XOR_INV_right(lhs, rhs):
    return lhs ^ (~rhs)


def INV_left_AND_right(lhs, rhs):
    return (~lhs) & rhs


def INV_left_AND_INV_right(lhs, rhs):
    return (~lhs) & (~rhs)
