# Belex (Open Source)

Belex is the Bit-Engine Language of Expressions for GSI's APU.
[open-belex](https://github.com/gsitechorg/open-belex) defines a low-level
language (Python DSL) named, Belex. Also defined are a model of the APU named,
DIRI, an interpreter, a compilation pipeline, and a few other utilities for use
in Belex development.

Fragments written in Belex are parsed into an intermediate representation (IR)
named, BLEIR, or the Bit-Level Engine Intermediate Representation. This IR
facilitates syntactic analysis, semantic analysis, code transformations such as
optimizations, and code generation.

Version of 02-Feb-2024

# Initialization

At the moment, only conda environments are supported. The following shows how to set up yours:

```bash
# let $WORKSPACE be the parent working directory of open-belex
cd "$WORKSPACE"

# Clone the open-belex repositories (unless you know what you are doing,
# please choose the same branch for all repositories):
# 1. "master" -> clone latest release code
# 2. "develop" -> clone latest development code
DEFAULT_BRANCH="master"
BELEX_BRANCH="$DEFAULT_BRANCH"

git clone --branch "$BELEX_BRANCH" \
    https://github.com/gsitechorg/open-belex.git

cd open-belex

# Create the conda environment
mamba env create --force -f environment.yml
conda activate open-belex

# Tell pip to use the cloned version of open-belex
pip install -e .
```

## Application Initialization

Before running any Belex-related code, you should initialize the environment as
follows (note that this is done automatically for
[open-belex-tests](https://github.com/gsitechorg/open-belex-tests)):

```python
import logging

from open_belex.bleir.interpreters import BLEIRInterpreter
from open_belex.common.rsp_fifo import ApucRspFifo
from open_belex.common.seu_layer import SEULayer
from open_belex.diri.half_bank import DIRI
from open_belex.utils.log_utils import init_logger

LOGGER = logging.getLogger()
if len(LOGGER.handlers) == 0:
    # Let APP_NYM be the name of your application
    init_logger(LOGGER, APP_NYM, log_to_console=False)

apuc_rsp_fifo = ApucRspFifo.context()

if DIRI.has_context():
    diri = DIRI.context()
else:
    diri = DIRI.push_context(apuc_rsp_fifo=apuc_rsp_fifo)

seu = SEULayer.context()

if BLEIRInterpreter.has_context():
    interpreter = BLEIRInterpreter.context()
else:
    interpreter = BLEIRInterpreter.push_context(diri=diri)
```

# Conventions

Unless otherwise specified, all Python symbols in this documentation come from
the `open_belex.literal` package.

A fragment, in Belex, is a Python function decorated with `@belex_apl` that
represents a snippet of microcode that is executed by the APU. All parameters to
Belex fragments must be annotated with their respective types, except the first
parameter. By convention, the first parameter is named `Belex` and is
represented by an internal type that facilitates some special logic like the
creation of temporaries. Either no return type or the return type of `None`
should be specified by fragments, as native fragments have no return value.

Belex supports both implicit and explicit register allocation, as well as some
mixed use cases. Mixing implicit and explicit register allocation can occasional
yield unexpected results, so it is highly recommended to choose only one of
them.

Belex works best when allowed to implicitly allocate registers as the code can
be better optimized. You should prefer explicit register allocation if you
desire manual code optimization; this mode is recommended only for those who are
very experienced in APU programming.

There is no explicit mode switch, the allocation mode is determined by the
manner in which parameters are passed to fragments. Fragments only accept
register parameters. Passing literal register values, such as `0xFFFF`, will
require a register to be implicitly allocated and assigned the value, and the
register then passed to the fragment in the value's stead. Setting a register
value and passing the register as the parameter value requires no implicit
allocation since the register was allocated explicitly.

Belex fragments may be called with kwarg notation. By convention, when using
implicit register allocation, kwargs are optional, but when using explicit
register allocation they are recommended. Mixing standard and keyword args is
also supported in the same manner they are supported by Python functions, but it
is recommended to choose only one convention.

# Architecture

An overview of the architecture of an APU (Associative Processing Unit) follows:
- There are 4 APUCs (APU Cores) per APU.
- There is 1 32-bit ARC coprocessor per APUC.
  - The host (CPU) communicates with the ARC coprocessor (device).
  - The ARC coprocessor communicates with the APUC.
- There are 2 APCs (Associative Processing Core) per APUC.
- There are different levels of memory in the APU.
  - L4 and L3 memory are part of the ARC coprocessor.
    - Memory is not shared among ARC coprocessors.
    - L4 is used primarily for data and heap memory.
      - The host (CPU) communicates with the ARC coprocessor via L4 memory.
    - L3 is used primarily for stack memory, but it is faster than L4 and is
      therefore also used to manipulate vectors.
      - There is less L3 memory than L4 memory.
  - L2 is the first layer of memory on the APUC.
    - I/O is performed between the ARC coprocessor and APU by copying data
      to/from either L4 or L3 memory on the ARC coprocessor and L2 memory on the
      APUC.
    - There are 128 L2 addresses.
    - There is 1 L2 register.
  - L1 is the next layer of memory in the APUC.
    - Data is copied between L2 and L1 memory via the LGL (described later).
      - LGL has the same shape as an L2 buffer that holds an L2 address.
    - There are 384 L1 addresses.
      - For every 9 contiguous, valid L1 addresses (beginning at address 0),
        there are 7 contiguous, invalid addresses.
      - Invalid L1 addresses read random data regardless what is written to them.
        - Invalid L1 addresses are only useful for generating random data.
    - There are 4 L1 registers that hold L1 addresses.
  - MMB (main memory block) is the primary layer of memory in which data is
    manipulated.
    - The MMB consists of the following components:
      - 24 VRs (vector registers)
      - 1 RL (read latch)
      - 1 GL
      - 1 GGL
      - 1 RSP16
      - 1 RSP256
      - 1 RSP2K
      - 1 RSP32K
    - Data is copied between L1 and the MMB via the GGL (described later).
      - GGL has the same shape as an L1 buffer.
      - Data from L1 may be copied into either RL or a VR.
        - VRs are the primary storage mechanisms for data in the MMB.
          - VRs are used exclusively for storage and not manipulation.
        - RL is used as temporary storage for data manipulation.
          - RL has the same shape as a VR.
        - If storage for future use is desired, then data should be copied into
          a VR.
        - If it is desired to begin manipulating or analyzing the data
          immediately, then it should be copied into RL.
- There are 16 half-banks per APUC.
  - Half the half-banks (8 qty.) belong to one APC and half belong to the other.

## Vector Registers

There are 24 vector registers (`VRs`) per APUC. Each `VR` is a 2-dimensional
array of size 32,768 plats (i.e. columns, according to our orientation) and 16
sections (i.e. rows, according to our orientation). The plats are grouped
contiguously and divided evenly among the half-banks, so each half-bank contains
2,048 of the 32,768 plats -- for each `VR` -- and all 16 sections. Although
there are 24 `VRs`, no more than 16 of them may be used by a fragment.

To interact with a `VR`, its unique row number must be written to one of the 16
`RN_REGs` (row number registers). There are 24 `VRs` and their row numbers range
from 0 to 23. The 16 `RN_REGs` are named `RN_REG_0`, `RN_REG_1`, ...,
`RN_REG_15`. There are two ways to do this with Belex: (1) implicity with
fragment parameters and temporaries or (2) explicitly as follows:

```python
# Assign the vector register, row number 3 to RN_REG_0:
apl_set_rn_reg(RN_REG_0, 3)
```

If you use the implicit convention then you should pass the row number as the
parameter value when invoking a fragment (function decorated with `@belex_apl`),
as follows:

```python
@belex_apl
def vr_fragment(Belex, some_vr: VR):
    # The first parameter of a Belex fragment is an implicit instance of
    # `open_belex.literal.Belex`. It is provided by the decorator and should
    # not be passed as the first parameter to your function call.
    pass

vr_fragment(3)  # pass the vector register, row number 3 to `vr_fragment`
```

If you use the explicit convention, then the value must be set in the ARC
coprocessor code and the register literal passed as the parameter value, as
follows (by convention, kwarg notation should be used for explicit register
parameters):

```python
@belex_apl
def vr_fragment(Belex, some_vr: VR):
    # The first parameter of a Belex fragment is an implicit instance of
    # `open_belex.literal.Belex`. It is provided by the decorator and should
    # not be passed as the first parameter to your function call.
    pass

# pass the vector register, row number 3 to `vr_fragment`
apl_set_rn_reg(RN_REG_0, 3)
vr_fragment(some_vr=RN_REG_0)
```

Belex is designed to support both implicit and explicit register allocation, and
they normally play well together, but some edge cases can cause mixing
conventions to behave incorrectly. It is recommended to choose one convention
and stick with it, preferably the implicit convention because it grants Belex
more freedom in how it allocates resources.

## Section Masks

A section mask is a 16-bit integer whose bits specify a mask over sections to
manipulate. For example, `0x0001` means only section 0 should be manipulated,
`0xFFFF` means all 16 section should be manipulated, and `0x0000` means no
section should be manipulated.

To specify a section mask, an `SM_REG` (section mask register) must be used.
There are 16 `SM_REGs` ranging from `SM_REG_0` to `SM_REG_15`. Passing section
masks to fragments is handled analogously to how vector registers are passed:
either implicitly by passing the section mask literals as parameter values to
explicitly by manually setting a register value and passing the register as the
parameter value.

```python
@belex_apl
def sm_fragment(Belex, some_sm: Mask):
    a: VR = Belex.VR()  # temporary VR, do not use the VR constructor directly!
    b: VR = Belex.VR()  # temporary VR, do not use the VR constructor directly!
    c: VR = Belex.VR()  # temporary VR, do not use the VR constructor directly!
    with apl_commands():
        RL[some_sm] <= 1
        RL[~some_sm] <= 0
    a[some_sm] <= RL()
    some_sm[b, c] <= NRL()  # alternative notation when targeting up to 3 VRs

# Call `sm_fragment` with an implicit section mask register for sections 0 and 8
sm_fragment(0x0101)

# Call `sm_fragment` with an explicit section mask register for section 12
apl_set_sm_reg(SM_REG_0, 0x1000)
sm_fragment(some_sm=SM_REG_0)
```

Section masks support two operations: (1) left-shifts without rotation -- up to
15 bits -- and (2) inversion. You may invert a shifted section mask, but you may
not shift an inverted section mask.

```python
@belex_apl
def frag_w_sm_shift_and_inv(Belex, sm: Mask):
    RL[sm<<10] <= RSP16()
    a: VR = Belex.VR()  # temporary VR, do not use the VR constructor directly!
    b: VR = Belex.VR()  # temporary VR, do not use the VR constructor directly!
    a[~sm] <= RL()
    b[~(sm<<3)] <= GGL()
```

Section mask literals within fragments are also supported. An example using a
string is given below, but please reference
[Indices](docs/belex/types.md#indices) for all supported in-fragment literal
types. Please note that integer literals are treated as section numbers and not
section masks.

```python
@belex_apl
def frag_w_sm_literal(Belex):
    RL["0xFFFF"] <= 1
```

Should you wish to use an integer literal to construct a section mask, you must
create a temporary with `Belex.Mask`:

```python
@belex_apl
def frag_w_int_literal_sm(Belex):
    x = 0xF0F0
    y = 0xABCD
    z = x ^ y  # 0x5B3D
    RL[Belex.Mask(z)] <= INV_NRL()
    # `RL[z] <= INV_NRL()` would raise the error:
    # ValueError: An int literal index must represent a single section within the range [0,16). Did you mean to specify a hex literal string? 23357
```

### Sections

A `Section` is a special type of section mask that makes it easier to operate on
singletons. It accepts decimal values in the range `[0,15)`, and differs from
the `Mask` type as the value `1<<section` will be the `SM_REG` value, where
`section` is the desired section index. If the value `0` is passed as the
parameter value of a `Mask` type, it would be treated as `0x0000`, or the empty
mask; if the value `0` is passed as the parameter value of a `Section` type, it
would be treated as `0x0001`, or the singleton mask representing the section at
index 0.

```python
@belex_apl
def sec_fragment(Belex, some_sec: Section):
    RL[some_sec] <= GL()

# Call `sec_fragment` with section 3 as the value of `some_sec`
sec_fragment(3)
```

Sections support the same operations as section masks.

## RL



## GL



## GGL



## RSP

RSP functionality provides the ability to OR data over grouped plats. There are
4 RSP registers: `RSP16`, `RSP256`, `RSP2K`, and `RSP32K`.

To understand how each RSP register ORs its respective plats, it is necessary to
understand some about data layout within the APU (Associative Processing Unit).
An APU consists of 4 APUCs (APU Cores). Device code executes on 1 of the 4
APUCs, as specified by the host code which runs on your CPU. Each APUC is
independent from the others; there is no way for the APUCs to communicate
directly -- the only way they can communicate is for the host to pull messages
from one and send them to another.

Each APUC consists of of 24 VRs (vector registers). Each vector register
consists of 32,768 (32K) plats, and 16 sections. Those 32,768 plats are grouped,
contiguously by plat index, and divided evenly over 16 half-banks. Each
half-bank consists of 2,048 plats and 16 sections.

Each APUC consists of an RL (read latch) which serves as the intermediate
register when copying data among MMB registers. RL has the same shape as a
vector register. There are a number of commands that affect RL, and the RSP
registers consist of the OR'd plats of RL in varying stages of reduction.

RSP registers `RSP16`, `RSP256`, and `RSP2K` operate over each half-bank. RSP
register `RSP32K` operates over all half-banks and operations with it are a
little different than the others.

### RSP16

`RSP16` operates on each half-bank. The suffix, 16, specifies the plats in each
half-bank are to be grouped, contiguously, by 16 plats and OR'd together. Since
there are 2,048 plats per half-bank, `RSP16` has `2048/16=128` plats
(overloaded term for a column-like structure according to our orientation of the
APU).

When reading from `RL` to `RSP16`, the 128 plats of `RSP16` are defined as
follows:
- Plat 0 of `RSP16` consists of the OR'd plats of `RL` in the range `[0,16)`.
- Plat 1 of `RSP16` consists of the OR'd plats of `RL` in the range `[16,32)`.
- ...
- Plat 127 of `RSP16` consists of the OR' plats of `RL` in the range
  `[2032,2048)`.

When writing from `RSP16` to `RL`, the 2,048 plats of `RL` are defined as
follows:
- Plats in the range `[0,16)` of `RL` consist of copies of plat 0 of `RSP16`.
- Plats in the range `[16,32)` of `RL` consist of copies of plat 1 of `RSP16`.
- ...
- Plats in the range `[2032,2048)` of `RL` consist of copies of plat 127 of
  `RSP16`.

### RSP256

`RSP256` operates on each half-bank. The suffix, 256, specifies the plats in
each half-bank are to be grouped, contiguously, by 256 plats and OR'd together.
Since there are 2,048 plats per half-bank, `RSP256` has `2048/256=8` plats.

When reading from `RSP16` to `RSP256`, the 8 plats of `RSP256` are defined as
follows (in terms of `RL`):
- Plat 0 of `RSP256` consists of the OR'd plats of `RL` in the range `[0,256)`.
- Plat 1 of `RSP256` consists of the OR'd plats of `RL` in the range
  `[256,512)`.
- ...
- Plat 7 of `RSP256` consists of the OR'd plats of `RL` in the range
  `[1792,2048)`.

When reading from `RSP16` to `RSP256`, the 8 plats of `RSP256` are defined as
follows (in terms of `RSP16`):
- Plat 0 of `RSP256` consists of the OR'd plats of `RSP16` in the range
  `[0,16)`.
- Plat 1 of `RSP256` consists of the OR'd plats of `RSP16` in the range
  `[16,32)`.
- ...
- Plat 7 of `RSP256` consists of the OR'd plats of `RSP16` in the range
  `[112,128)`.

When writing from `RSP256` to `RSP16`, the 128 plats of `RSP16` are defined as
follows:
- Plats in the range `[0,16)` of `RSP16` consist of copies of plat 0 of `RSP256`.
- Plats in the range `[16,32)` of `RSP16` consist of copies of plat 1 of `RSP256`.
- ...
- Plats in the range `[112,128)` of `RSP16` consist of copies of plat 7 of
  `RSP256`.

### RSP2K

`RSP2K` operates on each half-bank. The suffix, 2K, specifies the 2,048 plats in
each half-bank are grouped and OR'd together into a single plat. Since there are
2,048 plats per half-bank, `RSP2K` has `2048/2048=1` plat.

When reading from `RSP256` to `RSP2K`, the 1 plat of `RSP2K` is defined as
follows (in terms of `RL`):
- Plat 0 of `RSP2K` consists of the OR'd plats of `RL` in the range `[0,2048)`.

When reading from `RSP256` to `RSP2K`, the 1 plat of `RSP2K` is defined as
follows (in terms of `RSP256`):
- Plat 0 of `RSP2K` consists of the OR'd plats of `RSP256` in the range
  `[0,8)`.

When writing from `RSP2K` to `RSP256`, the 8 plats of `RSP256` are defined as
follows:
- Plats in the range `[0,8)` of `RSP256` consist of copies of plat 0 of `RSP2K`.

### RSP32K

`RSP32K` operates over all half-banks. The suffix, 32K, specifies the 32,768
plats in the APUC. Rather than being the OR over plats of `RL`, each bit of the
16-bit value of `RSP32K` consists of the OR'd reduction over a specific `RSP2K`
value. There are 16 half-banks per APUC, and there are therefore 16 `RSP2K`
values. Since there are 16 `RSP2K` values, there are 16-bits in the `RSP32K`
value.

When reading from `RSP2K` to `RSP32K`, its 16 bits (of its 1 plat) are defined
as follows (note that the bits do not represent sections, despite having the
same number of bits as `RL` has sections):
- Bit 0 of `RSP32K` consists of the OR'd sections of the 1 plat of `RSP2K` for
  half-bank 0.
- Bit 1 of `RSP32K` consists of the OR'd sections of the 1 plat of `RSP2K` for
  half-bank 1.
- ...
- Bit 15 of `RSP32K` consists of the OR'd sections of the 1 plat of `RSP2K` for
  half-bank 15.

When writing from `RSP32K` to `RSP2K`, every bit of the 1 plat of `RSP2K`
reflects the bit-value of the `RSP32K` bit at the index corresponding to the
half-bank index of the `RSP2K`. To elaborate:
- Every bit of `RSP2K` for half-bank 0 is 1 (ON) if the bit at index 0 of
  `RSP32K` is on, and 0 (OFF) otherwise.
- Every bit of `RSP2K` for half-bank 1 is 1 (ON) if the bit at index 1 of
  `RSP32K` is on, and 0 (OFF) otherwise.
- ...
- Every bit of `RSP2K` for half-bank 15 is 1 (ON) if the bit at index 15 of
  `RSP32K` is on, and 0 (OFF) otherwise.

### RSP Operation Modes

There are 2 basic RSP operation modes: RSP Read Mode and RSP Write Mode.

#### RSP Read Mode

In this mode, the results of either `RSP32K` or both `RSP32K` and `RSP2K` are
written to the RSP queue. These operations are denoted, RSP32K Read and RSP2K
Read, respectively. With either operation, results for both `RSP32K` and `RSP2K`
are written to the queue, but the results for `RSP2K` will not be complete for
an RSP32K Read since reading from RSP32K completes in fewer cycles than reading
from RSP2K. This is also why an RSP2K Read will add the results from both
`RSP32K` and `RSP2K` to the RSP queue.

An RSP32K Read may be completed with the following operations:

```python
RSP16[mask] <= RL()
RSP256() <= RSP16()
RSP2K() <= RSP256()
RSP32K() <= RSP2K()
RSP_END()
```

An RSP2K Read may be completed with the following operations:

```python
RSP16[mask] <= RL()
RSP256() <= RSP16()
RSP2K() <= RSP256()
RSP32K() <= RSP2K()
NOOP()
NOOP()
RSP_END()
```

##### RSP Queues

There is 1 RSP queue per APC, and 2 APCs per APUC, so there are 2 RSP queues per
APUC. To pop a message off an RSP queue:
1. Specify an APC to read from via `apuc_rsp_fifo.rsp_rd(apc-id)`, where
   `apc_id` is either 0 or 1. This will take the first message off the queue and
   make it available to read.
2. Read the `RSP32K` result from the message via
   `apuc_rsp_fifo.rd_rsp32k_reg()`. This will be an 8-bit integer where each bit
   represents the OR'd `RSP2K` result from the respective half-bank in the same
   APC. For example, half-banks `[0,8)` are members of APC 0, while half-banks
   `[8,16)` are members of APC 1.
3. Read the `RSP2K` result from the message via
   `apuc_rsp_fifo.rd_rsp2k_reg(bank_id)`, where `bank_id` is an integer in the
   range `[0,4)`, where the result is a 32-bit integer consisting of the
   concatenated `RSP2K` results of two half-banks as follows (add 8 to the
   half-bank index for APC 1, e.g. half-bank 2 of APC 1 is half-bank `1+8=9` of
   the APUC):
   - `bank_id=0` implies the concatenated `RSP2K` results for half-banks 0 and
     4 in the respective APC, where bits `[0,16)` are the `RSP2K` result for
     half-bank 0 and bits `[16,32)` are the `RSP2K` result for half-bank 4.
   - `bank_id=1` implies the concatenated `RSP2K` results for half-banks 1 and
     5 in the respective APC, where bits `[0,16)` are the `RSP2K` result for
     half-bank 1 and bits `[16,32)` are the `RSP2K` result for half-bank 5.
   - `bank_id=2` implies the concatenated `RSP2K` results for half-banks 2 and
     6 in the respective APC, where bits `[0,16)` are the `RSP2K` result for
     half-bank 2 and bits `[16,32)` are the `RSP2K` result for half-bank 6.
   - `bank_id=3` implies the concatenated `RSP2K` result for half-banks 3 and
     7 in the respective APC, where bits `[0,16)` are the `RSP2K` result for
     half-bank 3 and bits `[16,32)` are the `RSP2K` result for half-bank 7.

#### RSP Write Mode

This mode performs the varying levels of OR-reduction over the plats of `RL` to
a specific RSP register, and then broadcasts the result back to either `RL` or a
set of vector registers using the rules above. No messages are added to the RSP
queue in this mode. There are 4 types of RSP Writes, one for each of the 4 types
of RSP registers: `RSP16` Write, `RSP256` Write, `RSP2K` Write, and `RSP32K`
Write.

An `RSP16` Write may be completed as follows, where the final READ to `RL` may
be replaced with any READ or WRITE involving `RSP16` (or `INV_RSP16`):

```python
RSP16[mask_0] <= RL()
RSP_START_RET()
RL[mask_1] <= RSP16()
RSP_END()
```

An `RSP256` Write may be completed as follows, where the final READ to `RL` may
be replaced with any READ or WRITE involving `RSP16` (or `INV_RSP16`):

```python
# A section mask is required when broadcasting to RSP16 from RL:
RSP16[mask_0] <= RL()
RSP256() <= RSP16()
RSP_START_RET()
# No section mask is required when writing to RSP16 from RSP256:
RSP16() <= RSP256()
RL[mask_1] <= RSP16()
RSP_END()
```

An `RSP2K` Write may be completed as follows, where the final READ to `RL` may
be replaced with any READ or WRITE involving `RSP16` (or `INV_RSP16`):

```python
RSP16[mask_0] <= RL()
RSP256() <= RSP16()
RSP2K() <= RSP256()
RSP_START_RET()
RSP256() <= RSP2K()
RSP16() <= RSP256()
RL[mask_1] <= RSP16()
RSP_END()
```

An `RSP32K` Write may be completed as follows, where the final READ to `RL` may
be replaced with any READ or WRITE involving `RSP16` (or `INV_RSP16`):

```python
RSP16[mask_0] <= RL()
RSP256() <= RSP16()
RSP2K() <= RSP256()
RSP32K() <= RSP2K()
RSP_START_RET()
RSP2K() <= RSP32K()
RSP256() <= RSP2K()
RSP16() <= RSP256()
RL[mask_1] <= RSP16()
RSP_END()
```

# Command Syntax

## SRC

### <SRC>

In a command, `<SRC>` may be any one of the following.

    RL
    NRL
    ERL
    WRL
    SRL
    GL
    GGL
    RSP16

    INV_RL
    INV_NRL
    INV_ERL
    INV_WRL
    INV_SRL
    INV_GL
    INV_GGL
    INV_RSP16

### ~<SRC>

In a command, `~<SRC>` denotes the symbol `~` followed by one of the `<SRC>`
symbols, as defined above.

The difference between `~<SRC>` and its `INV_<SRC>` variant lies in the semantic
rules for laning (parallelism). Lanes may contain up to 4 commands. Only one
`<SRC>` may appear among all the commands in a lane. Within a lane, `~<SRC>` and
`INV_<SRC>` (for `<SRC>` elements with inverted variants) are considered
different despite having equivalent operations on the `<SRC>` element.

For example, if you reference `RSP16` in one command in a lane, you may not also
reference `INV_RSP16` in another command in the same lane. If the command syntax
supports the `~<SRC>` notation, you may safely combine `RSP16` with `~RSP16`.
Not every command supports the `~<SRC>` syntax.

## <SB>

Up to three SB numbers (row numbers) may appear in an `<SB>` expression.
Semantically, the contents are combined Implicitly via AND when `SB` appears on
the right-hand side of a command. On the left-hand side of a command, a parallel
assignment to all SB numbers (row numbers) is implied. On the right-hand side of
a command, the row numbers may be any combination of 1 to 3 elements in the
range `[0,24)`, but on the left-hand side of a command, they must be among the
same group, where a group consists of row numbers in the range `[0,8)`,
`[8, 16)`, or `[16, 24)`.

    SB[x]
    SB[x, y]
    SB[x, y, z]

This notation is extended to support up to 16 row numbers (any combination of 0
to 16 row numbers) on the right-hand side of a command with `RE_REG` parameters,
and up to 8 row numbers (within the same group) on the left-hand side of a
command with `EWE_REG` parameters. An `EWE_REG` group consists of row numbers in
the range `[0,8)`, `[8,16)`, or `[16,24)`.

Extended SB registers support the operations of negation and left-shifting. The
operations restrict the results within their respective constraints/domains.

In the grammar rules, everywhere an `<SB>` or `SB[...]` appears, an extended
`<SB>` may appear in its place so long as the type of the extended register is
`RE_REG` on the right-hand side of the command or `EWE_REG` on the left-hand
side of the command.

## WRITE Logic

    SB[x] = <SRC>
    SB[x,y] = <SRC>
    SB[x,y,z] = <SRC>

    SB[x] = ~<SRC>
    SB[x,y] = ~<SRC>
    SB[x,y,z] = ~<SRC>

    SB[x] ?= <SRC>        # OR-equal
    SB[x,y] ?= <SRC>      # OR-equal
    SB[x,y,z] ?= <SRC>    # OR-equal

    SB[x] ?= ~<SRC>       # OR-equal
    SB[x,y] ?= ~<SRC>     # OR-equal
    SB[x,y,z] ?= ~<SRC>   # OR-equal

## READ Logic

`<BIT>` is either a literal 1 or a literal 0, with no quote marks.
`&` means AND, `|` means OR, `^` means XOR. `&=`, `=`, and `^=`
are destructive assignment operators, as in the C language.

    RL = <BIT>            # READ LOGIG #1 and #2

    RL = <SB>             # READ LOGIC #3
    RL = <SRC>            # READ LOGIC #4
    RL = <SB> & <SRC>     # READ LOGIC #5

    RL = ~<SB>
    RL = ~<SRC>

    RL |= <SB>            # READ LOGIC #10
    RL |= <SRC>           # READ LOGIC #11
    RL |= <SB> & <SRC>    # READ LOGIC #12

    RL &= <SB>            # READ LOGIC #13
    RL &= <SRC>           # READ LOGIC #14
    RL &= <SB> & <SRC>    # READ LOGIC #15

    RL ^= <SB>            # READ LOGIC #18
    RL ^= <SRC>           # READ LOGIC #19
    RL ^= ~<SRC>          # READ LOGIC #19
    RL ^= <SB> & <SRC>    # READ LOGIC #20

    RL = <SB> | <SRC>     # READ LOGIC #6
    RL = <SB> ^ <SRC>     # READ LOGIC #7

    RL = ~<SB> & <SRC>    # READ LOGIC #8
    RL = <SB> & ~<SRC>    # READ LOGIC #9
    RL = <SB> ^ ~<SRC>    # UNDOCUMENTED VARIATION OF READ LOGIC #9

    RL &= ~<SB>           # READ LOGIC #16
    RL &= ~<SRC>          # READ LOGIC #17

    RL = ~<SB> & ~<SRC>

## R-SEL (BROADCAST) Logic

    GL = RL               # R-SEL LOGIC
    GGL = RL              # R-SEL LOGIC
    RSP16 = RL            # R-SEL LOGIC

    RSP256 = RSP16        # SPECIAL ASSIGNMENT
    RSP2K = RSP256        # SPECIAL ASSIGNMENT
    RSP32K = RSP2K        # SPECIAL ASSIGNMENT

    RWINH_SET
    RWINH_RST
