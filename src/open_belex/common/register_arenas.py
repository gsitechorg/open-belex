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

from typing import Optional, Set

from open_belex.common.constants import (NUM_EWE_REGS, NUM_L1_REGS,
                                         NUM_L2_REGS, NUM_RE_REGS, NUM_RN_REGS,
                                         NUM_SM_REGS)
from open_belex.common.string_arena import StringArena


class SmRegArena(StringArena):
    """From /usr/local/include/gsi/preproc_defs.h

        SM_REG_0 = 0,
        SM_REG_1 = 1,
        SM_REG_2 = 2,
            ...
        SM_REG_15 = 15,
    """
    def __init__(self: "SmRegArena",
                 nregs: int = NUM_SM_REGS,
                 reservations: Optional[Set[int]] = None) -> None:
        super().__init__(prefix="SM_REG_",
                         nregs=nregs,
                         reservations=reservations)


class RnRegArena(StringArena):
    """From /usr/local/include/gsi/preproc_defs.h

        RN_REG_0 = 0,
        RN_REG_1 = 1,
        RN_REG_2 = 2,
            ...
        RN_REG_15 = 15,
    """

    def __init__(self: "RnRegArena",
                 nregs: int = NUM_RN_REGS,
                 reservations: Optional[Set[int]] = None) -> None:
        super().__init__(prefix="RN_REG_",
                         nregs=nregs,
                         reservations=reservations)


class ReRegArena(StringArena):
    """From /usr/local/include/gsi/preproc_defs.h

        RE_REG_0 = 0,
        RE_REG_1 = 1,
        RE_REG_2 = 2,
        RE_REG_3 = 3,
    """

    def __init__(self: "ReRegArena",
                 nregs: int = NUM_RE_REGS,
                 reservations: Optional[Set[int]] = None) -> None:
        super().__init__(prefix="RE_REG_",
                         nregs=nregs,
                         reservations=reservations)


class EweRegArena(StringArena):
    """From /usr/local/include/gsi/preproc_defs.h

        EWE_REG_0 = 0,
        EWE_REG_1 = 1,
        EWE_REG_2 = 2,
        EWE_REG_3 = 3,
    """

    def __init__(self: "EweRegArena",
                 nregs: int = NUM_EWE_REGS,
                 reservations: Optional[Set[int]] = None):
        super().__init__(prefix="EWE_REG_",
                         nregs=nregs,
                         reservations=reservations)


class L1RegArena(StringArena):
    """From /usr/local/include/gsi/preproc_defs.h

        L1_ADDR_REG_0 = 0,
        L1_ADDR_REG_1 = 1,
        L1_ADDR_REG_2 = 2,
        L1_ADDR_REG_3 = 3,
    """

    def __init__(self: "L1RegArena",
                 nregs: int = NUM_L1_REGS,
                 reservations: Optional[Set[int]] = None) -> None:
        super().__init__(prefix="L1_ADDR_REG_",
                         nregs=nregs,
                         reservations=reservations)


class L2RegArena(StringArena):
    """From /usr/local/include/gsi/preproc_defs.h

        L2_ADDR_REG_0 = 0,
    """

    def __init__(self: "L2RegArena",
                 nregs: int = NUM_L2_REGS,
                 reservations: Optional[Set[int]] = None) -> None:
        super().__init__(prefix="L2_ADDR_REG_",
                         nregs=nregs,
                         reservations=reservations)
