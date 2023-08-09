r"""
 By John Cook and Dylon Edwards

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


RL = 24
NSB = 24
NVR = 25  # new ontology
NPLATS = 2048  # new ontology
NRSP16 = NPLATS / 16  # 128
NSECTIONS = 16
NINSTRS_PER_LANE = 4

NGGL_ROWS = 4

ALL_PARAMS_TYPE = 'u16'

NUM_L1_ROWS = 384
NUM_L2_ROWS = 128

MAX_SM_VALUE = 0xFFFF
MAX_RN_VALUE = NSB - 1
MAX_RE_VALUE = 0xFFFF00
MAX_EWE_VALUE = 0x2FF
MAX_L1_VALUE = (1 << 13) - 1
MAX_L2_VALUE = (1 << 7) - 1

NUM_PLATS_PER_HALF_BANK = NPLATS

NUM_HALF_BANKS_PER_APC = 8
NUM_PLATS_PER_APC = NUM_PLATS_PER_HALF_BANK * NUM_HALF_BANKS_PER_APC

NUM_APCS_PER_APUC = 2
NUM_HALF_BANKS_PER_APUC = NUM_HALF_BANKS_PER_APC * NUM_APCS_PER_APUC
NUM_PLATS_PER_APUC = NUM_PLATS_PER_APC * NUM_APCS_PER_APUC

NUM_APUCS_PER_APU = 4
NUM_APCS_PER_APU = NUM_APCS_PER_APUC * NUM_APUCS_PER_APU
NUM_HALF_BANKS_PER_APU = NUM_HALF_BANKS_PER_APUC * NUM_APCS_PER_APU
NUM_PLATS_PER_APU = NUM_PLATS_PER_APUC * NUM_APUCS_PER_APU

NUM_SM_REGS = 16
NUM_RN_REGS = 16
NUM_RE_REGS = 4
NUM_EWE_REGS = 4
NUM_L1_REGS = 4
NUM_L2_REGS = 1
