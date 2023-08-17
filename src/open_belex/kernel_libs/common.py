r"""
By Dylon Edwards and Brian Beckman
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
