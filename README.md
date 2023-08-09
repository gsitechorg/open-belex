# BELEX (Open Source)

Belex is the Bit-Engine Language of Expressions.

Version of 08 Aug 2023

This documentation covers the core classes and functions of Belex.

# PYTHON REQUIREMENTS

Connect to the GSI server network; VPN recommended.

Make sure you have a virtual environment of some kind that has the
necessary Python packages installed. Here is an example using
`venv`, but `conda` works just as well:

```bash
  cd ~/GSI  # or wherever you likd to keep your project directories

  # Clone the Belex repositories (choose the same branch for all repositories):
  # 1. "--branch master" -> clone latest release code
  # 2. "--branch develop" -> (recommended) clone latest development code
  git clone --branch develop git@bitbucket.org:gsitech/belex.git
  git clone --branch develop git@bitbucket.org:gsitech/belex-libs.git
  git clone --branch develop git@bitbucket.org:gsitech/belex-tests.git

  # Initialize your virtual environment
  cd ~/GSI/belex-tests  # or wherever you cloned the belex-tests repo
  python -m venv venv  # you need Python>=3.8, 3.9.7 recommended
  source venv/bin/activate

  cd ~/GSI/belex  # or wherever you cloned the belex repo
  pip install -e .

  cd ~/GSI/belex-libs  # or wherever you cloned the belex-libs repo
  pip install -e .

  cd ~/GSI/belex-tests  # or wherever you cloned the belex-tests repo
  pip install -e .

  pip install --upgrade ninja
  pip install \
      --upgrade \
      --index-url http://192.168.42.9:8081/repository/gsi-pypi/simple \
      --trusted-host 192.168.42.9 \
      meson
```

# COMMAND SYNTAX

## SRC

### \<SRC\>

In a command, `<SRC>` may be any one of the following.

    RL
    NRL
    ERL
    WRL
    SRL
    GL
    GGL
    RSP16

### ~\<SRC\>

In a command, `~<SRC>` may be any one of the following.

    INV_RL
    INV_NRL
    INV_ERL
    INV_WRL
    INV_SRL
    INV_GL
    INV_GGL
    INV_RSP16

## \<SB\>

Up to three SB numbers (row numbers) may appear in an `<SB>` expression.
Semantically, the contents are combined Implicitly via AND when `SB` appears on
the right-hand side of a command. On the left-hand side of a command, a
parallel assignment to all SB numbers (row numbers) is implied.

    SB[x]
    SB[x, y]
    SB[x, y, z]

## WRITE LOGIC

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

## READ LOGIC

`<BIT>` is either a literal 1 or a literal 0, with no quote marks.
`&` means AND, `|` means OR, `^` means XOR. `&=`, `\=`, and `^=`
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

## R-SEL (BROADCAST) LOGIC

    GL = RL               # R-SEL LOGIC
    GGL = RL              # R-SEL LOGIC
    RSP16 = RL            # R-SEL LOGIC

    RSP256 = RSP16        # SPECIAL ASSIGNMENT
    RSP2K = RSP256        # SPECIAL ASSIGNMENT
    RSP32K = RSP2K        # SPECIAL ASSIGNMENT

    RWINH_SET
    RWINH_RST

# MODULES OVERVIEW

## APL

```{eval-rst}
.. toctree::
.. automodule:: belex.apl
   :members:
```

## APL OPTIMIZATIONS

```{eval-rst}
.. toctree::
.. automodule:: belex.apl_optimizations
   :members:
```

## COMPILER

```{eval-rst}
.. toctree::
.. automodule:: belex.compiler
   :members:
```

## DECORATORS

```{eval-rst}
.. toctree::
.. automodule:: belex.decorators
   :members:
```

## DIRECTED GRAPH

```{eval-rst}
.. toctree::
.. automodule:: belex.directed_graph
   :members:
```

## EXPRESSIONS

```{eval-rst}
.. toctree::
.. automodule:: belex.expressions
   :members:
```

## INTERMEDIATE REPRESENTATION

```{eval-rst}
.. toctree::
.. automodule:: belex.intermediate_representation
   :members:
```

## LANING

```{eval-rst}
.. toctree::
.. automodule:: belex.laning
   :members:
```

## LITERAL

```{eval-rst}
.. toctree::
.. automodule:: belex.literal
   :members:
```

## REGISTER ALLOCATION

```{eval-rst}
.. toctree::
.. automodule:: belex.register_allocation
   :members:
```

## RENDERABLE

```{eval-rst}
.. toctree::
.. automodule:: belex.renderable
   :members:
```
