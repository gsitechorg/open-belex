r"""
 By Brian Beckman and Dylon Edwards

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

from typing import Union

import numpy as np

from open_belex.common.constants import NUM_PLATS_PER_APUC, NSECTIONS


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


def u16_to_bool(value: np.uint16,
                num_plats: int = NUM_PLATS_PER_APUC) -> np.ndarray:
    plat = np.ndarray(NSECTIONS, dtype=bool)
    for section in range(NSECTIONS):
        plat[section] = (value >> section) & 0x0001
    return np.tile(plat, (num_plats, 1))
