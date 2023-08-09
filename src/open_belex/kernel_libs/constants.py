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
