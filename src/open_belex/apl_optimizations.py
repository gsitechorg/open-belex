r"""open_belex.apl_optimizations: peephole optimizer functions for High-Level
Belex.

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

********************************************************

This module consists of functions from IC (intermediate code) to
IC, with a context argument, that is, rewriters for IC. IC is a
list of APL commands. APL is the lowest-level machine code of the
MMB part of the APU device. Peephole optimizers consider two or
a few adjacent IC commands.

The module also includes some helper functions, not part of the
API for peephole optimizers.

Any public peephole-optimizer function in this file the following
signature:

.. highlight:: python
.. code-block:: python

    <function-name>(apl: IC, context: Context=None)

where ``apl`` is  of  type ``IC``, which is a list of

.. highlight:: python
.. code-block:: python

    open_belex.apl.APL_masked_stmt

(TODO: IC should be a list of
  ``Union[APL_masked_stmt, APL_statement])``,
  but such fails type-checking via mypy,
  for reasons as-yet-unknown).

``Context`` is a dictionary of the form

.. highlight:: python
.. code-block:: python

    context = {"keep_alives": [register_map[Variable.symbol]],
               "alive_symbol": Variable.symbol}

where ``Variable`` is defined in ``open_belex.expressions`` and
``register_map`` is a dictionary from ``Variable.symbol``
to SB number (or row number).

All but one peephole-optimizer functions have ``peephole_`` as a
prefix to their names. The one exception is ``delete_dead_writes``.

Public Functions
    * delete_dead_writes
    * peephole_coalesce_consecutive_and_assignments
    * peephole_coalesce_shift_before_op
    * peephole_coalesce_two_consecutive_sb_from_rl
    * peephole_coalesce_two_consecutive_sb_from_src
    * peephole_eliminate_read_after_write
    * peephole_eliminate_write_read_dependence
    * peephole_merge_rl_from_src_and_rl_from_sb
    * peephole_merge_rl_from_src_and_sb_from_rl
    * peephole_replace_zero_xor

Private Helper Functions
    * all_sbs_in_same_group
    * apl_liveness_analysis
    * apply_optimizations
    * def get_lhs_sbs
    * get_rhs_sbs

Global Data
    * APL_OPTIMIZATIONS: Dict[str, Callable]

**Detailed docs**

"""

from typing import Callable, Dict, List, Sequence, Set, Union

from open_belex.apl import (APL_comment, APL_masked_stmt, APL_rl_from_sb,
                            APL_rl_from_sb_src, APL_rl_from_src,
                            APL_sb_from_src, APL_set_rl)

# FIXME: Union with APL_statement doesn't work
IC = List[Union[APL_masked_stmt]]

MaskInt = int
SB = int
Symbol = str
Context = Dict[str, Union[List[int], Symbol]]


def all_sbs_in_same_group(sbs: Set[SB]) -> bool:
    r"""Check that all SBs in a set belong to the same SB group,
    where the groups are SB[0]-SB[7], SB[8]-SB[15], and
    SB[16]-S[23].

    Parameters
    ----------
    sbs: Set[int]
        input: Contains a set of SB numbers (row-numbers).

    Returns
    -------
    bool
        true if and only if all SB numbers in the Set belong
        to the same group
    """
    groups = {sb // 8 for sb in sbs}
    return len(groups) == 1


def peephole_eliminate_read_after_write(
        apl: IC, context: Context = None) -> IC:
    r"""Eliminate a read (e.g., ``RL[mask] <= SB_9()``) immediately after
    a write to the same SB (e.g., ``SB_9[mask] <= RL()``).

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
        input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl2 = [apl[0]]

    for x, y in zip(apl[:-1], apl[1:]):
        if type(x) == type(y) == APL_masked_stmt and \
           type(x.stmt) == APL_sb_from_src and \
           type(y.stmt) == APL_rl_from_sb and \
           x.stmt.src == 'RL' and \
           x.msk.mask == y.msk.mask and \
           x.stmt.sbs == y.stmt.sbs:

            continue
        else:
            apl2.append(y)

    return apl2


def peephole_replace_zero_xor(
        apl: IC, context: Context = None) -> IC:
    r"""Replace a common trick for zero-ing RL with an explicit
    form, i.e., ``RL[:] <= SB_x; RL[:] ^= ~SB_x``, with ``RL[:] <= 0``.

    Parameters
    ----------
    apl: IC
        input: Internal representation of APL, pre optimization
    context: Dict[str, Any]
        input: Ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """
    apl2 = [apl[0]]

    for x in apl[1:]:
        if isinstance(apl2[-1].stmt, APL_rl_from_sb) \
                and apl2[-1].stmt.assign_op == '' \
                and isinstance(x.stmt, APL_rl_from_sb) \
                and x.stmt.assign_op == '^' \
                and apl2[-1].msk.mask == x.msk.mask \
                and apl2[-1].stmt.sbs[0] == x.stmt.sbs[0] \
                and len(x.stmt.sbs) == 1 \
                and len(apl2[-1].stmt.sbs) == 1:
           apl2[-1].stmt = APL_set_rl(value=0)
        else:
           apl2.append(x)

    return apl2


def peephole_eliminate_write_read_dependence(
        apl_: IC, context: Context = None) -> IC:
    r"""Remove the SB of a write from an immediately following read,

    e.g., replace

    .. highlight:: python
    .. code-block:: python

        SB_x[mask] <= RL()
        RL[mask] <= SB_x() & SB_y() (& SB_z()) ...

    with

    .. highlight:: python
    .. code-block:: python

        SB_x[mask] <= RL()
        RL[mask] &= SB_y() & (SB_z()) ...

    Parameters
    ----------
    apl_: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
        input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl = apl_.copy()
    n = len(apl)

    for i in range(1, n):
        if isinstance(apl[i - 1].stmt, APL_sb_from_src) \
                and apl[i - 1].stmt.src == 'RL' \
                and isinstance(apl[i].stmt, APL_rl_from_sb) \
                and apl[i - 1].stmt.sbs[0] in apl[i].stmt.sbs \
                and apl[i - 1].msk.mask & apl[i].msk.mask == apl[i].msk.mask \
                and len(apl[i - 1].stmt.sbs) == 1 \
                and len(apl[i].stmt.sbs) > 1 \
                and apl[i].stmt.assign_op == '':
            apl[i].stmt.sbs = [sb for sb in apl[i].stmt.sbs if sb != apl[i - 1].stmt.sbs[0]]
            apl[i].stmt.assign_op = '&'
    return apl


def peephole_merge_rl_from_src_and_sb_from_rl(
        apl: IC, context: Context = None) -> IC:
    r"""Replace

    .. highlight:: python
    .. code-block:: python

        RL[mask] <= <SRC>()
        SB[mask] <= RL()

    with

    .. highlight:: python
    .. code-block:: python

        SB_x[mask] <= <SRC>()

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
        input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl2 = [apl[0]]

    for x in apl[1:]:
        if isinstance(apl2[-1].stmt, APL_rl_from_src) \
                and apl2[-1].stmt.assign_op == '' \
                and isinstance(x.stmt, APL_sb_from_src) \
                and x.stmt.src == 'RL':
            prev = apl2[-1]
            curr = x

            new_stmt = APL_sb_from_src(sbs=curr.stmt.sbs,
                                       src=prev.stmt.src)

            apl2[-1].stmt = new_stmt
            apl2[-1].msk = curr.msk

        else:
            apl2.append(x)

    return apl2


def peephole_merge_rl_from_src_and_rl_from_sb(
        apl: IC, context: Context = None) -> IC:
    r"""Merge a read from ``<SRC>`` and read with logic from ``<SB>``
    into one of the combining forms, namely read-logic #5, #6, or #7

    e.g., replace

    .. highlight:: python
    .. code-block:: python

        RL[mask] <= <SRC>()  # READ LOGIC #4
        RL[mask] OP= <SB>()  # READ LOGIC #10, #13, #18

    with

    .. highlight:: python
    .. code-block:: python

        RL[mask] <= <SRC>() OP <SB>()  # READ LOGIC #5, %6, #7

    where ``OP`` is one of ``&``, ``|``, or ``^``.

    Parameters
    ----------
    apl_: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
        input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl2 = [apl[0]]

    for x in apl[1:]:
        if isinstance(apl2[-1].stmt, APL_rl_from_src) \
                and isinstance(x.stmt, APL_rl_from_sb) \
                and x.stmt.assign_op != '':
            apl2[-1].stmt = APL_rl_from_sb_src (sbs = x.stmt.sbs,
                                                assign_op = '',
                                                binary_op = x.stmt.assign_op,
                                                src = apl2[-1].stmt.src)

            apl2[-1].msk = x.msk
        else:
            apl2.append(x)

    return apl2


def peephole_coalesce_consecutive_and_assignments(
        apl: IC, context: Context = None) -> IC:
    r"""Replace consecutive AND reads (RL on the left-hand side)
    with a read from multiple SBs with implicit and. For example,
    replace

    .. highlight:: python
    .. code-block:: python

        RL[mask] <= SB[x]()
        RL[mask] &= SB[y]

    with

    .. highlight:: python
    .. code-block:: python

        RL[mask] <= SB[x, y]()

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
        input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl2 = [apl[0]]

    for x in apl[1:]:
        if apl2[-1].msk.mask == x.msk.mask \
                and isinstance(apl2[-1].stmt, APL_rl_from_sb) \
                and isinstance(x.stmt, APL_rl_from_sb) \
                and apl2[-1].stmt.assign_op == '' \
                and x.stmt.assign_op == '&' \
                and len(set(apl2[-1].stmt.sbs)) < 3 \
                and len(x.stmt.sbs) == 1:
            coalesced_indices = set(x.stmt.sbs) | set(apl2[-1].stmt.sbs)
            if all_sbs_in_same_group(coalesced_indices):
                apl2[-1].stmt.sbs = list(coalesced_indices)
            else:
                apl2.append(x)
        else:
            apl2.append(x)
    return apl2


def peephole_coalesce_two_consecutive_sb_from_src(
        apl: IC, context: Context = None) -> IC:
    r"""Replace consecutive writes from the same SRC with a write
    to multiple SBs. For example, replace

    .. highlight:: python
    .. code-block:: python

        SB[x][mask] <= <SRC>
        SB[y][mask] <= <SRC>

    with

    .. highlight:: python
    .. code-block:: python

        SB[x, y][mask] <= <SRC>

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
         input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl2 = [apl[0]]

    for x in apl[1:]:
        if apl2[-1].msk.mask == x.msk.mask \
           and isinstance(apl2[-1].stmt, APL_sb_from_src) \
           and isinstance(x.stmt, APL_sb_from_src) \
           and x.stmt.src == apl2[-1].stmt.src \
           and len(set(apl2[-1].stmt.sbs)) < 3 \
           and len(x.stmt.sbs) == 1:
            src = apl2[-1].stmt.src
            coalesced_indices = set(x.stmt.sbs) | set(apl2[-1].stmt.sbs)

            if all_sbs_in_same_group(coalesced_indices):
                coalesced_stmt = APL_sb_from_src(
                    sbs=list(coalesced_indices), src=src)
                apl2[-1] = APL_masked_stmt(msk=x.msk, stmt=coalesced_stmt)
            else:
                apl2.append(x)
        else:
            apl2.append(x)

    return apl2


def peephole_coalesce_two_consecutive_sb_from_rl(
        apl: IC, context: Context = None) -> IC:
    r"""Replace consecutive writes from RL with a write
    to multiple SBs. For example, replace

    .. highlight:: python
    .. code-block:: python

        SB[x][mask] <= RL()
        SB[y][mask] <= RL()

    with

    .. highlight:: python
    .. code-block:: python

        SB[x, y][mask] <= RL()

    Note: this is a special case of
    ``peephole_coalesce_two_consecutive_sb_from_src``. It is
    included to support certain experiments with compiler
    internals.

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
         input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl2 = [apl[0]]

    for x in apl[1:]:
        if apl2[-1].msk.mask == x.msk.mask \
           and isinstance(apl2[-1].stmt, APL_sb_from_src) \
           and isinstance(x.stmt, APL_sb_from_src) \
           and x.stmt.src == apl2[-1].stmt.src == 'RL' \
           and len(set(x.stmt.sbs) | set(apl2[-1].stmt.sbs)) < 4:

            xyz = list(set(x.stmt.sbs) | set(apl2[-1].stmt.sbs))
            sb_from_src = APL_sb_from_src(sbs=xyz, src='RL')
            apl2[-1] = APL_masked_stmt(msk=x.msk, stmt=sb_from_src)
        else:
            apl2.append(x)

    return apl2


def peephole_coalesce_shift_before_op(
        apl: IC, context: Context = None) -> IC:
    r"""Replace

    .. highlight:: python
    .. code-block:: python

        RL["0xFFFF"]  <= NRL()
        RL["0xFFFF"] OP= SB[x]()  # OP == one of '&', '^', '|'

    with

    .. highlight:: python
    .. code-block:: python

        RL["0xFFFF"] <= SB[x]() OP NRL()

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
         input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl2 = [apl[0]]

    for x, y in zip(apl[:-1], apl[1:]):
        if type(x.stmt) == APL_rl_from_src \
           and x.msk.mask == 0xffff \
           and x.stmt.src == 'NRL' \
           and type(y.stmt) == APL_rl_from_sb \
           and y.stmt.assign_op[0] in '&|^':

            rl_from_sb_src = APL_rl_from_sb_src(
                sb=y.stmt.sbs,
                src='NRL',
                assign_op='',
                op=y.stmt.assign_op[0])

            masked_stmt = APL_masked_stmt(msk=y.msk, stmt=rl_from_sb_src)

            apl2.pop()
            apl2.append(masked_stmt)
        else:
            apl2.append(y)

    return apl2


def get_lhs_sbs(inst: APL_masked_stmt) -> Sequence[SB]:
    r"""Get a list of SB numbers (row numbers) on the left-hand
    side of the given masked statement.

    Parameters
    ----------
    inst: APL_masked_stmt
        input: the statement to analyze

    Returns
    -------
    Sequence[SB]
        the list (as a Sequence) of SBs on the left-hand side
        of the statement
    """

    stmt  = inst.stmt

    if type(stmt) == APL_sb_from_src:
        return stmt.sbs

    return []


def get_rhs_sbs(inst: APL_masked_stmt) -> Sequence[int]:
    r"""Get a list of SB numbers (row numbers) on the right-hand
    side of the given masked statement.

    Parameters
    ----------
    inst: APL_masked_stmt
        input: the statement to analyze

    Returns
    -------
    Sequence[SB]
        the list (as a Sequence) of SBs on the right-hand side
        of the statement
    """

    stmt = inst.stmt

    if type(stmt) == APL_rl_from_sb_src or type(stmt) == APL_rl_from_sb:
        return stmt.sbs

    return []


def apl_liveness_analysis(apl: IC, context: Context) -> \
        List[Dict[Union[SB, Symbol], MaskInt]]:
    r"""Given a list of permanent keep-alives in the context, attach
    to each apl statement a list of SBs to keep alive after the
    statement. Uses classic KILL/GEN analysis.

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
         input: ignored in this case

    Returns
    -------
    List[Dict[Union[SB, Symbol], MaskInt]]
        ordered list, one-to-one with the input APL, of
        mappings from SBs or symbols (strs) to section masks
        to keep alive after the corresponding APL statement.
    """

    global LOGGER

    # list of SBs to be kept alive after the APL code finishes
    return_values = context['keep_alives']

    # same code as in intermediate_representation; however,
    # register indices are integers
    alive = [{var: 0xffff for var in return_values}]

    for inst in reversed(apl):
        curr_alive = alive[-1].copy()

        lhs_sbs = get_lhs_sbs(inst)
        rhs_sbs = get_rhs_sbs(inst)
        mask = inst.msk.mask  # hex mask stored as integer

        for lhs_sb in lhs_sbs:
            if lhs_sb in curr_alive:
                curr_alive[lhs_sb] &= ~mask
                if curr_alive[lhs_sb] == 0x0:
                    curr_alive.pop(lhs_sb)

        for rhs_sb in rhs_sbs:
            if not rhs_sb in curr_alive:
                curr_alive[rhs_sb] = mask
            else:
                curr_alive[rhs_sb] |= mask

        alive.append(curr_alive)

    return [x for x in reversed(alive)]


def delete_dead_writes(apl: IC, context: Context) -> IC:
    r"""Delete writes (SB on the left-hand side) to registers
    known to be dead (unused in later code)

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    context: Dict[str, Any]
         input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    n = len(apl)

    # Dictionaries of registers for each command in the APL
    lastwrite = {sb: n - 1 for sb in range(25)}
    lastread = {sb: n - 1 for sb in range(25)}
    write_masks = {sb: 0x0000 for sb in range(25)}

    instr_to_delete = set()

    keep_alives: Union[List[SB], Symbol] = \
        context["keep_alives"] \
        if context else []

    for i in range(n - 1, -1, -1):
        inst = apl[i]
        stmt_type = type(inst.stmt)

        write_regs = inst.stmt.sbs if type(inst.stmt) == APL_sb_from_src else []
        read_regs = inst.stmt.sbs if type(inst.stmt) == APL_rl_from_sb else []

        if write_regs and \
            all(lastwrite[reg] < lastread[reg] for reg in write_regs) and \
            all(isinstance(apl[lastwrite[reg]].stmt, stmt_type)
                for reg in write_regs) and \
            all((apl[i].msk.mask & write_masks[reg]) == apl[i].msk.mask
                for reg in write_regs):

            instr_to_delete.add(i)

        for write_reg in write_regs:
            write_masks[write_reg] = apl[i].msk.mask
            if write_reg not in keep_alives:
                lastwrite[write_reg] = i

        for read_reg in read_regs:
            lastread[read_reg] = i

    return [inst for i, inst in enumerate(apl)
            if i not in instr_to_delete]


def apply_optimizations(
        apl: IC,
        optimizations: List[Callable[[IC, Context], IC]],
        context: Context) -> IC:
    r"""Apply a list of optimizations to a stream of APL and
    a Context, producing a stream of optimized APL.

    Parameters
    ----------
    apl: IC
        input: internal representation of APL, pre optimization
    optimizations: List[Callable[[IC, Context], IC]
        input: list of optimization functions to apply
    context: Dict[str, Any]
        input: ignored in this case

    Returns
    -------
    IC
        internal representation of APL, post optimization
    """

    apl = [x for x in apl if type(x) != APL_comment]

    for f in optimizations:
        apl = f(apl, context)

    return apl


# There should be TEN of these, all different.
APL_OPTIMIZATIONS: Dict[str, Callable] = {
    "eliminate-read-after-write":      peephole_eliminate_read_after_write,
    "delete-dead-writes":              delete_dead_writes,
    "replace-zero-xor":                peephole_replace_zero_xor,
    "eliminate-write-read-dependence": peephole_eliminate_write_read_dependence,
    "coalesce-sb-from-rl":             peephole_merge_rl_from_src_and_sb_from_rl,
    "merge-rl-src-sb":                 peephole_merge_rl_from_src_and_rl_from_sb,
    "coalesce-consecutive-and-reads":  peephole_coalesce_consecutive_and_assignments,
    "coalesce-sb-from-src":            peephole_coalesce_two_consecutive_sb_from_src,
    "coalesce-consecutive-writes":     peephole_coalesce_two_consecutive_sb_from_rl,
    "merge-rl-src-sb2":                peephole_coalesce_shift_before_op,
    }
