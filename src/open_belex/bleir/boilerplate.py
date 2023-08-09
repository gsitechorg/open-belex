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

from typing import Sequence

from open_belex.bleir.types import (GGL, GL, NOOP, RL, RN_REG, RSP2K, RSP16,
                                    RSP32K, RSP256, RSP_END, RSP_START_RET, SB,
                                    SM_REG, AllocatedRegister, Fragment,
                                    FragmentCaller, FragmentCallerCall, assign,
                                    masked, statement)
from open_belex.common.constants import NSB

lvr_rps = [RN_REG(f"lvr_{vp}") for vp in range(16)]
fs_rp = SM_REG("fs")

clear_sb_fragment = Fragment(
    identifier="clear_sb",
    parameters=lvr_rps + [fs_rp],
    operations=[
        statement(masked(fs_rp, assign(RL, 0))),
    ] + [
        statement(masked(fs_rp, assign(SB[lvr_rp], RL)))
        for lvr_rp in lvr_rps
    ])

clear_sb_caller = FragmentCaller(
    fragment=clear_sb_fragment,
    registers=[
        AllocatedRegister(parameter=lvr_rp, register=f"RN_REG_{vp}")
        for vp, lvr_rp in enumerate(lvr_rps)
    ] + [
        AllocatedRegister(parameter=fs_rp, register="SM_REG_0")
    ])

clear_rsps_fragment = Fragment(
    identifier="clear_rsps",
    parameters=[fs_rp],
    operations=[
        statement(masked(fs_rp, assign(RL, 0))),
        statement(masked(fs_rp, assign(RSP16, RL))),
        statement(assign(RSP256, RSP16)),
        statement(assign(RSP2K, RSP256)),
        statement(assign(RSP32K, RSP2K)),
        statement(RSP_START_RET),
        statement(assign(RSP2K, RSP32K)),
        statement(assign(RSP256, RSP2K)),
        statement(assign(RSP16, RSP256)),
        statement(masked(fs_rp, assign(RL, RSP16))),
        statement(NOOP),
        statement(NOOP),
        statement(RSP_END),
    ])

clear_rsps_caller = FragmentCaller(fragment=clear_rsps_fragment)

clear_gl_fragment = Fragment(
    identifier="clear_gl",
    parameters=[fs_rp],
    operations=[
        statement(masked(fs_rp, assign(RL, 0))),
        statement(masked(fs_rp, assign(GL, RL))),
        statement(NOOP),
    ])

clear_gl_caller = FragmentCaller(fragment=clear_gl_fragment)

clear_ggl_fragment = Fragment(
    identifier="clear_ggl",
    parameters=[fs_rp],
    operations=[
        statement(masked(fs_rp, assign(RL, 0))),
        statement(masked(fs_rp, assign(GGL, RL))),
        statement(NOOP),
    ])

clear_ggl_caller = FragmentCaller(fragment=clear_ggl_fragment)

INITIALIZERS: Sequence[FragmentCallerCall] = [
    clear_rsps_caller(0xFFFF, is_initializer=True),
    clear_gl_caller(0xFFFF, is_initializer=True),
    clear_ggl_caller(0xFFFF, is_initializer=True),
] + [
    # Trog through the complete list of VRs and clear them
    clear_sb_caller(*([lvr_vp] * 16), 0xFFFF, is_initializer=True)
    for lvr_vp in range(NSB)
]
