r"""
 By Dylon Edwards and Brian Beckman

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

from open_belex.literal import (INV_RSP16, RL, RSP16, VR, apl_commands,
                                belex_apl, u16)


@belex_apl
def cpy_vr(Belex, tgt: VR, src: VR) -> None:
    RL["0xffff"] <= src()
    tgt["0xffff"] <= RL()


@belex_apl
def cpy_16(Belex, dst: VR, src: VR) -> None:
    cpy_vr(dst, src)


@belex_apl
def cpy_imm_16_to_rl(Belex, val: u16) -> None:
    with apl_commands():
        RL[val] <= 1
        RL[~val] <= 0


@belex_apl
def cpy_imm_16(Belex, tgt: VR, val: u16) -> None:
    with apl_commands():
        tgt[val] <= INV_RSP16()
        tgt[~val] <= RSP16()


@belex_apl
def cpy_rl_to_vr(Belex, tgt: VR) -> None:
    tgt[::] <= RL()


@belex_apl
def cpy_vr_to_rl(Belex, src: VR) -> None:
    RL[::] <= src()
