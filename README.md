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

### ~<SRC>

In a command, `~<SRC>` may be any one of the following.

    INV_RL
    INV_NRL
    INV_ERL
    INV_WRL
    INV_SRL
    INV_GL
    INV_GGL
    INV_RSP16

## <SB>

Up to three SB numbers (row numbers) may appear in an `<SB>` expression.
Semantically, the contents are combined Implicitly via AND when `SB` appears on
the right-hand side of a command. On the left-hand side of a command, a
parallel assignment to all SB numbers (row numbers) is implied.

    SB[x]
    SB[x, y]
    SB[x, y, z]

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
