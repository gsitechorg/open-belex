# Belex (Open Source)

Belex is the Bit-Engine Language of Expressions for GSI's APU.

Version of 02-Feb-2024

# Initialization


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
range `[0, 24)`, but on the left-hand side of a command, they must be among the
same group, where a group consists of row numbers in the range `[0, 8)`,
`[8, 16)`, or `[16, 24)`.

    SB[x]
    SB[x, y]
    SB[x, y, z]

This notation is extended to support up to 16 row numbers (any combination of 0
to 16 row numbers) on the right-hand side of a command with `RE_REG` parameters,
and up to 8 row numbers (within the same group) on the left-hand side of a
command with `EWE_REG` parameters. An `EWE_REG` group consists of row numbers in
the range `[0, 8)`, `[8, 16)`, or `[16, 24)`.

Extended SB registers support the operations of negation and left-shifting. The
operations restrict the results within their respective constraints/domains.

In the grammar rules, everywhere an `<SB>` or `SB[...]` appears, an extended
`<SB>` may appear in its place so long as the type of the extended register is
`RE_REG` on the right-hand side of the command or `EWE_REG` on the left-hand
side of the command.

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
there are 2,048 plats per half-bank, `RSP16` has `2048 / 16 = 128` plats
(overloaded term for a column-like structure according to our orientation of the
APU).

When reading from `RL` to `RSP16`, the 128 plats of `RSP16` are defined as
follows:
- Plat 0 of `RSP16` consists of the OR'd plats of `RL` in the range `[0, 16)`.
- Plat 1 of `RSP16` consists of the OR'd plats of `RL` in the range `[16, 32)`.
- ...
- Plat 127 of `RSP16` consists of the OR' plats of `RL` in the range
  `[2032, 2048)`.

When writing from `RSP16` to `RL`, the 2,048 plats of `RL` are defined as
follows:
- Plats `[0, 16)` of `RL` consist of copies of plat 0 of `RSP16`.
- Plats `[16, 32)` of `RL` consist of copies of plat 1 of `RSP16`.
- ...
- Plats `[2032, 2048)` of `RL` consist of copies of plat 127 of `RSP16`.

### RSP256

`RSP256` operates on each half-bank. The suffix, 256, specifies the plats in
each half-bank are to be grouped, contiguously, by 256 plats and OR'd together.
Since there are 2,048 plats per half-bank, `RSP256` has `2048 / 256 = 8` plats.

When reading from `RSP16` to `RSP256`, the 8 plats of `RSP256` are defined as
follows (in terms of `RL`):
- Plat 0 of `RSP256` consists of the OR'd plats of `RL` in the range `[0, 256)`.
- Plat 1 of `RSP256` consists of the OR'd plats of `RL` in the range
  `[256, 512)`.
- ...
- Plat 7 of `RSP256` consists of the OR'd plats of `RL` in the range
  `[1792, 2048)`.

When reading from `RSP16` to `RSP256`, the 8 plats of `RSP256` are defined as
follows (in terms of `RSP16`):
- Plat 0 of `RSP256` consists of the OR'd plats of `RSP16` in the range
  `[0, 16)`.
- Plat 1 of `RSP256` consists of the OR'd plats of `RSP16` in the range
  `[16, 32)`.
- ...
- Plat 7 of `RSP256` consists of the OR'd plats of `RSP16` in the range
  `[112, 128)`.

When writing from `RSP256` to `RSP16`, the 128 plats of `RSP16` are defined as
follows:
- Plats `[0, 16)` of `RSP16` consist of copies of plat 0 of `RSP256`.
- Plats `[16, 32)` of `RSP16` consist of copies of plat 1 of `RSP256`.
- ...
- Plats `[112, 128)` of `RSP16` consist of copies of plat 7 of `RSP256`.

### RSP2K

`RSP2K` operates on each half-bank. The suffix, 2K, specifies the 2,048 plats in
each half-bank are grouped and OR'd together into a single plat. Since there are
2,048 plats per half-bank, `RSP2K` has `2048 / 2048 = 1` plat.

When reading from `RSP256` to `RSP2K`, the 1 plat of `RSP2K` is defined as
follows (in terms of `RL`):
- Plat 0 of `RSP2K` consists of the OR'd plats of `RL` in the range `[0, 2048)`.

When reading from `RSP256` to `RSP2K`, the 1 plat of `RSP2K` is defined as
follows (in terms of `RSP256`):
- Plat 0 of `RSP2K` consists of the OR'd plats of `RSP256` in the range
  `[0, 8)`.

When writing from `RSP2K` to `RSP256`, the 8 plats of `RSP256` are defined as
follows:
- Plats `[0, 8)` of `RSP256` consist of copies of plat 0 of `RSP2K`.

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
1. Specify an APC to read from via `APC_RSP_RD(apc-id)`, where `apc_id` is
   either 0 or 1. This will take the first message off the queue and make it
   available to read.
2. Read the `RSP32K` result from the message via `APL_RD_RSP32K_REG()`. This
   will be an 8-bit integer where each bit represents the OR'd `RSP2K` result
   from the respective half-bank in the same APC. For example, half-banks
   `[0, 8)` are members of APC 0, while half-banks `[8, 16)` are members of
   APC 1.
3. Read the `RSP2K` result from the message via `APL_RD_RSP2K_REG(bank_id)`,
   where `bank_id` is an integer in the range `[0, 4)`, where the result is a
   32-bit integer consisting of the concatenated `RSP2K` results of two
   half-banks as follows (add 8 to the half-bank index for APC 1, e.g. half-bank
   2 of APC 1 is half-bank `1 + 8 = 9` of the APUC):
   - `bank_id = 0` implies the concatenated `RSP2K` results for half-banks 0 and
     4 in the respective APC, where bits `[0, 16)` are the `RSP2K` result for
     half-bank 0 and bits `[16, 32)` are the `RSP2K` result for half-bank 4.
   - `bank_id = 1` implies the concatenated `RSP2K` results for half-banks 1 and
     5 in the respective APC, where bits `[0, 16)` are the `RSP2K` result for
     half-bank 1 and bits `[16, 32)` are the `RSP2K` result for half-bank 5.
   - `bank_id = 2` implies the concatenated `RSP2K` results for half-banks 2 and
     6 in the respective APC, where bits `[0, 16)` are the `RSP2K` result for
     half-bank 2 and bits `[16, 32)` are the `RSP2K` result for half-bank 6.
   - `bank_id = 3` implies the concatenated `RSP2K` result for half-banks 3 and
     7 in the respective APC, where bits `[0, 16)` are the `RSP2K` result for
     half-bank 3 and bits `[16, 32)` are the `RSP2K` result for half-bank 7.

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
RSP16[mask_0] <= RL()
RSP256() <= RSP16()
RSP_START_RET()
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
