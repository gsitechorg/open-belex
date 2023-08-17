r"""
By Dylon Edwards and Brian Beckman
"""

from open_belex.kernel_libs.constants import PE_FLAG
from open_belex.literal import (GGL, GL, INV_GL, INV_NRL, INV_SRL, L1, L2,
                                L2_ADDR_REG_0, L2_END, LGL, NRL, RL,
                                RN_REG_FLAGS, SM_0X0001, SM_0X0101, SM_0X1111,
                                SM_0X3333, SM_0XFFFF, VR, Mask, apl_commands,
                                belex_apl)
# NOTE: These are included for backwards compatibility
from open_belex.utils.memory_utils import NUM_VM_REGS, vmr_to_row

# __   ____  __ ___     _    _
# \ \ / /  \/  | _ \   | |  / |
#  \ V /| |\/| |   /_  | |__| |
#   \_/ |_|  |_|_|_( ) |____|_|
#                  |/


@belex_apl
def restore_vr(Belex, dst: VR, src: L1) -> None:

    with apl_commands():
        GGL() <= src() + 0

    with apl_commands():
        dst[SM_0X1111 << 0] <= GGL()
        GGL() <= src() + 1

    with apl_commands():
        dst[SM_0X1111 << 1] <= GGL()
        GGL() <= src() + 2

    with apl_commands():
        dst[SM_0X1111 << 2] <= GGL()
        GGL() <= src() + 3

    with apl_commands():
        dst[SM_0X1111 << 3] <= GGL()


@belex_apl
def spill_vr(Belex, dst: L1, src: VR) -> None:

    with apl_commands():
        RL[SM_0XFFFF] <= src()
        GGL[SM_0X1111 << 0] <= RL()

    with apl_commands():
        dst() + 0 <= GGL()
        GGL[SM_0X1111 << 1] <= RL()

    with apl_commands():
        dst() + 1 <= GGL()
        GGL[SM_0X1111 << 2] <= RL()

    with apl_commands():
        dst() + 2 <= GGL()
        GGL[SM_0X1111 << 3] <= RL()

    with apl_commands():
        dst() + 3 <= GGL()

@belex_apl
def restore_rl(Belex, src: L1) -> None:

    with apl_commands():
        GGL() <= src() + 0

    with apl_commands():
        RL[SM_0X1111 << 0] <= GGL()
        GGL() <= src() + 1

    with apl_commands():
        RL[SM_0X1111 << 1] <= GGL()
        GGL() <= src() + 2

    with apl_commands():
        RL[SM_0X1111 << 2] <= GGL()
        GGL() <= src() + 3

    with apl_commands():
        RL[SM_0X1111 << 3] <= GGL()


@belex_apl
def spill_rl(Belex, dst: L1) -> None:

    with apl_commands():
        GGL[SM_0X1111 << 0] <= RL()

    with apl_commands():
        dst() + 0 <= GGL()
        GGL[SM_0X1111 << 1] <= RL()

    with apl_commands():
        dst() + 1 <= GGL()
        GGL[SM_0X1111 << 2] <= RL()

    with apl_commands():
        dst() + 2 <= GGL()
        GGL[SM_0X1111 << 3] <= RL()

    with apl_commands():
        dst() + 3 <= GGL()


@belex_apl
def restore_ggl(Belex, src: L1) -> None:

    with apl_commands():
        GGL() <= src() + 0


@belex_apl
def spill_ggl(Belex, dst: L1) -> None:

    with apl_commands():
        dst() + 0 <= GGL()


@belex_apl
def load_16_t0(
        Belex,
        dst: VR,
        src: L1,
        parity_src: L1,
        parity_mask: Mask
        ) -> None:

    with apl_commands():
        RL[SM_0X0001 << PE_FLAG] <= RN_REG_FLAGS()
        GL[SM_0X0001 << PE_FLAG] <= RL()
        GGL() <= src() + 0

    with apl_commands():
        dst[SM_0X1111 << 0] <= GGL()
        RL[SM_0X3333 << 3] <= GGL()
        RL[SM_0X0001 << 1] <= INV_GL()
        GGL() <= src() + 1

    with apl_commands():
        dst[SM_0X1111 << 1] <= GGL()
        RL[parity_mask] ^= GGL()
        GGL() <= src() + 2

    with apl_commands():
        dst[SM_0X1111 << 2] <= GGL()
        RL[parity_mask] ^= GGL()
        GGL() <= src() + 3

    with apl_commands():
        dst[SM_0X1111 << 3] <= GGL()
        RL[parity_mask] ^= GGL()
        GGL() <= parity_src()

    with apl_commands():
        RL[parity_mask] ^= GGL()

    with apl_commands():
        RL[SM_0X0101 << 3] ^= INV_SRL()
        RL[SM_0X0101 << 4] ^= INV_NRL()
        GL[parity_mask] <= RL()
        GL[SM_0X0001 << 1] <= RL()

    with apl_commands():
        RN_REG_FLAGS[SM_0X0001 << PE_FLAG] <= INV_GL()


@belex_apl
def store_16_t0(
        Belex,
        dst: L1,
        parity_dst: L1,
        parity_mask: Mask,
        src: VR,
        ) -> None:

    with apl_commands("instruction 1"):
        RL[SM_0XFFFF] <= src()
        GGL[SM_0X1111 << 0] <= RL()

    with apl_commands("instruction 2"):
        dst() + 0 <= GGL()
        GGL[SM_0X1111 << 1] <= RL()

    with apl_commands("instruction 3"):
        dst() + 1 <= GGL()
        RL[SM_0X1111 << 1] ^= NRL()
        GGL[SM_0X1111 << 2] <= RL()

    with apl_commands("instruction 4"):
        dst() + 2 <= GGL()
        RL[SM_0X1111 << 2] ^= NRL()
        GGL[SM_0X1111 << 3] <= RL()

    with apl_commands("instruction 5"):
        dst() + 3 <= GGL()
        RL[SM_0X1111 << 3] ^= NRL()
        GL[SM_0X0001 << 11] <= RL()

    with apl_commands("instruction 6"):
        RL[SM_0X0001 << 15] ^= GL()
        GL[SM_0X0001 << 3] <= RL()
        GGL() <= parity_dst()

    with apl_commands("instruction 7"):
        RL[SM_0X0001 << 7] ^= GL()
        GL[SM_0X0001 << 7] <= RL()
        RL[SM_0X1111 << 0] <= GGL()

    with apl_commands("instruction 8"):
        RL[parity_mask << 0] <= GL()
        GL[SM_0X0001 << 15] <= RL()

    with apl_commands("instruction 9"):
        RL[parity_mask << 8] <= GL()
        GGL[SM_0X1111 << 0] <= RL()

    with apl_commands("instruction 10"):
        parity_dst() <= GGL()


@belex_apl(dst=L2_ADDR_REG_0)
def copy_l1_to_l2_byte(Belex, dst: L2, src: L1, parity_src: L1):
    LGL() <= src() + 0
    with apl_commands():
        dst() + 0 <= LGL()
        LGL() <= src() + 1
    with apl_commands():
        dst() + 1 <= LGL()
        LGL() <= src() + 2
    with apl_commands():
        dst() + 2 <= LGL()
        LGL() <= src() + 3
    with apl_commands():
        dst() + 3 <= LGL()
        LGL() <= src() + (1, 0)
    with apl_commands():
        dst() + 4 <= LGL()
        LGL() <= src() + (1, 1)
    with apl_commands():
        dst() + 5 <= LGL()
        LGL() <= src() + (1, 2)
    with apl_commands():
        dst() + 6 <= LGL()
        LGL() <= src() + (1, 3)
    with apl_commands():
        dst() + 7 <= LGL()
        LGL() <= parity_src()
    dst() + 8 <= LGL()


@belex_apl(src=L2_ADDR_REG_0)
def copy_l2_to_l1_byte(Belex, dst: L1, parity_dst: L1, src: L2):
    LGL() <= src() + 0
    with apl_commands():
        dst() + (0, 0) <= LGL()
        LGL() <= src() + 1
    with apl_commands():
        dst() + (0, 1) <= LGL()
        LGL() <= src() + 2
    with apl_commands():
        dst() + (0, 2) <= LGL()
        LGL() <= src() + 3
    with apl_commands():
        dst() + (0, 3) <= LGL()
        LGL() <= src() + 4
    with apl_commands():
        dst() + (1, 0) <= LGL()
        LGL() <= src() + 5
    with apl_commands():
        dst() + (1, 1) <= LGL()
        LGL() <= src() + 6
    with apl_commands():
        dst() + (1, 2) <= LGL()
        LGL() <= src() + 7
    with apl_commands():
        dst() + (1, 3) <= LGL()
        LGL() <= src() + 8
    parity_dst() <= LGL()


@belex_apl
def l2_end(Belex):
    L2_END()
