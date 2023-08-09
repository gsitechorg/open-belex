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

from open_belex.kernel_libs.constants import (APL_VM_ROWS_PER_U16,
                                              GSI_L1_VA_NUM_ROWS,
                                              GSI_L1_VA_SET_ADDR_ROWS)

# Number of spillable VRs (VMRs)
NUM_VM_REGS = 48


def vmr_to_row(vm_reg: int) -> int:
    parity_set = (vm_reg >> 1)
    parity_grp = (vm_reg & 1)  # 1 or 0 (True or False)
    row = parity_set * GSI_L1_VA_SET_ADDR_ROWS
    row += (APL_VM_ROWS_PER_U16 * parity_grp)
    assert row < GSI_L1_VA_NUM_ROWS
    return row
