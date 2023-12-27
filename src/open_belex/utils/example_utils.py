r"""
By Brian Beckman and Dylon Edwards
"""

from typing import Union

import numpy as np

from open_belex.common.constants import NUM_PLATS_PER_APUC, NSECTIONS


BITS_PER_NIBBLE: int = 4
NIBBLES_PER_SECTION: int = NUM_PLATS_PER_APUC // BITS_PER_NIBBLE


def convert_to_u16(diri_vr: np.ndarray) -> np.ndarray:
    """Transforms a DIRI VR into a representation suitable for C-sim.

    Parameters:
        diri_vr: 2-dimensional np.ndarray of booleans representing a VR from DIRI.

    Returns:
        A 1-dimensional np.ndarray of the same number of plats as the input vr,
        but with each plat represented as a 16-bit integer instead of 16
        booleans."""

    return np.packbits(diri_vr, bitorder='little').view(np.uint16)


def convert_to_f16(diri_vr: np.ndarray) -> Union[np.ndarray, np.float16]:
    if np.ndim(diri_vr) == 1:
        fraction = 0.0
        for i in range(10):
            fraction += diri_vr[i] * (2 ** (-(10 - i)))
        exponent = 0.0
        for i in range(5):
            exponent += diri_vr[i + 10] * (2 ** i)
        bias = 15  # 2 ** (5 - 1) - 1 = 2 ** 4 - 1 = 16 - 1 = 15
        biased_exponent = exponent - bias
        sgn_bit = diri_vr[15]
        f16 = ((-1) ** sgn_bit) * (2 ** biased_exponent) * (1 + fraction)
        return np.float16(f16)

    c_sim_vr = np.zeros(shape=diri_vr.shape[0], dtype=np.float16)

    for index, plat in enumerate(diri_vr):
        c_sim_vr[index] = convert_to_f16(plat)

    return c_sim_vr


def convert_to_bool(c_sim_vr: Union[np.ndarray, int],
                    nsections: int = NSECTIONS) -> np.ndarray:

    if np.ndim(c_sim_vr) == 0:
        value = c_sim_vr
        diri_vr = np.zeros(shape=nsections, dtype=bool)
        for section in range(nsections):
            diri_vr[section] = (value & (1 << section))
        return diri_vr

    if c_sim_vr.dtype == np.uint16:
        bits = np.unpackbits(c_sim_vr.view(np.uint8), bitorder="little")
        return bits.reshape(bits.shape[0] // 16, 16)[::, :nsections].view(bool)

    if c_sim_vr.dtype == np.float16:
        nplats = c_sim_vr.shape[0]
        diri_vr = np.zeros(shape=(nplats, nsections), dtype=bool)
        for i, value in enumerate(c_sim_vr):
            plat = diri_vr[i]
            for j, bit in enumerate(bin(value.view("H"))[2:].zfill(16)):
                plat[15 - j] = (bit == '1')
        return diri_vr

    raise ValueError(
        f"Unsupported c_sim_vr dtype ({c_sim_vr.dtype})")


def u16_to_vr(value: np.uint16) -> np.ndarray:
    return np.repeat(value, NUM_PLATS_PER_APUC).astype(np.uint16)


def index_vr(nplats: int) -> np.ndarray:
    """As with exercise 8, create a VR-shaped array with 0 in plat 0,
    1 in plat 1, etc., up to nplats, exclusive."""
    assert nplats <= NUM_PLATS_PER_APUC
    vr = np.zeros((NSECTIONS, NUM_PLATS_PER_APUC), dtype=bool)
    for plat in range(nplats):
        for section in range(NSECTIONS):
            vr[section, plat] = (plat >> section) & 0x0001
    return vr


def littlendian_bools_to_u16_platwise(vr : np.ndarray, nplats: int) -> np.ndarray:
    assert nplats <= NUM_PLATS_PER_APUC
    result = np.zeros(nplats, dtype=np.uint16)
    for plat in range(nplats):
        temp: int = 0
        for section in range(NSECTIONS):
            temp |= (vr[section, plat] << section)
        result[plat] = int(temp)
    return result


def u16_to_bool(value: np.uint16,
                num_plats: int = NUM_PLATS_PER_APUC) -> np.ndarray:
    plat = np.ndarray(NSECTIONS, dtype=bool)
    for section in range(NSECTIONS):
        plat[section] = (value >> section) & 0x0001
    return np.tile(plat, (num_plats, 1))


def section_wise_nibble(value: int) -> np.ndarray:
    nibble = np.ndarray(BITS_PER_NIBBLE, dtype=bool)
    for bit in range(BITS_PER_NIBBLE):
        nibble[bit] = (value >> bit) & 0x0001
    return np.tile(nibble, NIBBLES_PER_SECTION)


NIBBLES_PER_BYTE = 2
BITS_PER_BYTE    = NIBBLES_PER_BYTE * BITS_PER_NIBBLE


def littlendian_section_wise_from_i8s(values: np.ndarray) -> np.ndarray:
    len_ = min(len(values), NUM_PLATS_PER_APUC // 8)
    section = np.zeros(NUM_PLATS_PER_APUC, dtype=bool)
    for byte in range(len_):
        for bit in range(BITS_PER_BYTE):
            idx = (BITS_PER_BYTE * byte) + bit
            section[idx] = (values[byte] >> bit) & 0x0001
    return section


def littlendian_section_wise_from_bytes(values: bytes) -> np.ndarray:
    """Typically for BHV vectors of length 8192 bits, or
    1024 8-bit bytes."""
    assert NUM_PLATS_PER_APUC == 32_768
    len_ = min(len(values), NUM_PLATS_PER_APUC // 8)
    section = np.zeros(NUM_PLATS_PER_APUC, dtype=bool)
    for byte in range(len_):
        for bit in range(BITS_PER_BYTE):
            idx = (BITS_PER_BYTE * byte) + bit
            section[idx] = (values[byte] >> bit) & 0x0001
    return section


def bytes_from_section_array(arr: np.ndarray, nbytes: int) -> bytes:
    result : bytes = bytearray(nbytes)
    for byte in range(nbytes):
        value : int = 0
        for bit in range(BITS_PER_BYTE):
            idx = (BITS_PER_BYTE * byte) + bit
            value |= (arr[idx] << bit)
        result[byte] = value
    return result


def littlendian_int8s_from_section_array(arr: np.ndarray, nbytes: int) -> np.ndarray:
    result : np.ndarray = np.zeros(nbytes, dtype=int)
    for byte in range(nbytes):
        value : int = 0
        for bit in range(BITS_PER_BYTE):
            idx = (BITS_PER_BYTE * byte) + bit
            value |= (arr[idx] << bit)
        result[byte] = value
    return result
