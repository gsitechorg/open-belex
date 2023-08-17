r"""
By Dylon Edwards and Brian Beckman
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
