r"""
By Dylon Edwards and Brian Beckman
"""

# FIXME: Parse these from:
# ./subprojects/gsi-sys-apu/gsi-device-libs/include/gal/gsi/apl_defs.h

GSI_APC_PARITY_SET_BITS = 3

GSI_L1_VA_SET_DATA_ADDR_BITS = GSI_APC_PARITY_SET_BITS
GSI_L1_VA_SET_PARITY_ADDR_BITS = 1
GSI_L1_VA_SET_ADDR_BITS = \
    GSI_L1_VA_SET_DATA_ADDR_BITS + GSI_L1_VA_SET_PARITY_ADDR_BITS
GSI_L1_VA_SET_ADDR_ROWS = 1 << GSI_L1_VA_SET_ADDR_BITS
GSI_L1_NUM_DATA_ROWS_PER_GRP = 192
GSI_L1_VA_SET_DATA_ADDR_BITS = GSI_APC_PARITY_SET_BITS
GSI_L1_VA_NUM_SETS = \
    GSI_L1_NUM_DATA_ROWS_PER_GRP / (1 << GSI_L1_VA_SET_DATA_ADDR_BITS)
GSI_L1_VA_NUM_ROWS = GSI_L1_VA_SET_ADDR_ROWS * GSI_L1_VA_NUM_SETS

# FIXME: Parse these from:
# ./subprojects/gsi-sys-apu/gsi-device-libs/include/gal/gsi/apl_defs.h

APL_VM_ROWS_PER_U16 = 4

SM_BIT_3 = 3

# FIXME: Parse these from:
# ./src/include/gvml_apl_defs.apl.h

PE_FLAG = SM_BIT_3  # Parity error
