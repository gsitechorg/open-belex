r"""
By Dylon Edwards
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set, Tuple

import open_belex.bleir.types as BLEIR
from open_belex.bleir.types import bleir_dataclass

# helpers (exported into "laning.py")
is_rl_read_sb = lambda x: isinstance(x, APL_rl_from_sb) or isinstance(x, APL_rl_from_sb_src)
is_rl_read = lambda x: isinstance(x, APL_rl_from_sb) or isinstance(x, APL_rl_from_src) or isinstance(x, APL_rl_from_sb_src)
is_sb_write = lambda x: isinstance(x, APL_sb_from_src)


DEFAULT_BLEIR_MAP: Dict[str, Any] = {
    'RL': BLEIR.RL,
    'NRL': BLEIR.NRL,
    'ERL': BLEIR.ERL,
    'WRL': BLEIR.WRL,
    'SRL': BLEIR.SRL,
    'GL': BLEIR.GL,
    'GGL': BLEIR.GGL,
    'RSP16': BLEIR.RSP16,

    'INV_RL': BLEIR.INV_RL,
    'INV_NRL': BLEIR.INV_NRL,
    'INV_ERL': BLEIR.INV_ERL,
    'INV_WRL': BLEIR.INV_WRL,
    'INV_SRL': BLEIR.INV_SRL,
    'INV_GL': BLEIR.INV_GL,
    'INV_GGL': BLEIR.INV_GGL,
    'INV_RSP16': BLEIR.INV_RSP16,

    'RSP256': BLEIR.RSP256,
    'RSP2K': BLEIR.RSP2K,
    'RSP32K': BLEIR.RSP32K,

    'NOOP': BLEIR.NOOP,
    'RSP_END': BLEIR.RSP_END,

    '=': BLEIR.assign,
    '&=': BLEIR.and_eq,
    '|=': BLEIR.or_eq,
    '^=': BLEIR.xor_eq,
    '?=': BLEIR.cond_eq,
    '&': BLEIR.conjoin,
    '|': BLEIR.disjoin,
    '^': BLEIR.xor,
}


def make_bleir_map() -> Dict[str, Any]:
    global DEFAULT_BLEIR_MAP
    return dict(DEFAULT_BLEIR_MAP)  # No need for deepcopy


# a mask is represented internally by an unsigned int
@dataclass
class Mask:
    mask: int

    def __init__(self: "Mask", indices: Sequence[int]) -> None:
        self.mask = 0x0000
        for i in indices:
            self.mask |= (0x0001 << i)

    def __hash__(self: "Mask") -> int:
        return hash((self.mask,))

    def __eq__(self: "Mask", other: Any) -> bool:
        return isinstance(other, Mask) \
            and self.mask == other.mask

    def __ne__(self: "Mask", other: Any) -> bool:
        return not self.__eq__(other)

    def __str__(self: "Mask") -> str:
        return Mask.to_hex(self.mask)

    def get_shift(self: "Mask") -> int:
        mask = self.mask
        for i in range(16):
            if mask & 0x0001 == 1:
                return i
            mask = mask >> 1
        raise RuntimeError("Mask is 0x0000 so there is no shift")

    def get_normalized(self: "Mask") -> Tuple[str, int]:
        t = self.get_shift()
        if self.mask >> t == 0xffff >> t:
            return (Mask.to_hex(0xffff), t)
        return (Mask.to_hex(self.mask >> t), t)

    @staticmethod
    def from_hex(mask: int) -> "Mask":
        t = Mask([])
        t.mask = mask
        return t

    @staticmethod
    def to_hex(mask: int) -> str:
        return f'0x{mask:04x}'


SRC = str
SB = int
SBs = List[SB]


class APL_statement(ABC):
    src: SRC
    sbs: SBs
    assign_op: str

    @abstractmethod
    def render_bleir(self: "APL_statement",
                     rn_registers: Dict[int, str],
                     bleir_map: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def defs(self: "APL_statement"):
        raise NotImplementedError

    def uses(self: "APL_statement"):
        raise NotImplementedError


def ref_overlap(refs1, refs2):
    d1 = dict(refs1)
    overlaps = []
    for ref, mask in refs2:
        if (ref in d1) and (d1[ref] & mask != 0):
            overlaps.append((ref, mask))

    return overlaps


def is_overlap(refs1, refs2):
    overlap = ref_overlap(refs1, refs2)
    return len(overlap) > 0


def data_dep(inst1, inst2):
    return is_overlap(inst1.defs(), inst2.uses())


def anti_dep(inst1, inst2):
    return is_overlap(inst1.uses(), inst2.defs())


def output_dep(inst1, inst2):
    return is_overlap(inst1.defs(), inst2.defs())


def modify_rl_ref(var, mask):
    if var == 'NRL':
        return ('RL', mask >> 1)  # this shift is apparently not sign-extending
        # that is, (0xFFFF) >> 1 == 0x7FFFF
    elif var == 'SRL':
        # better make sure we don't overflow
        return ('RL', (mask << 1) & 0xFFFF)

    return (var, mask)


@bleir_dataclass
class APL_masked_stmt():
    msk: Mask
    stmt: APL_statement

    @BLEIR.statement
    def render_bleir(self: "APL_masked_stmt",
                     rn_registers: Dict[int, str],
                     sm_registers: Dict[str, str],
                     bleir_map: Dict[str, Any]) -> Any:

        r"""Convert allocated registers and a bleir map
        to a BLEIR masked statement.

        Parameters
        ----------
        rn_registers: Dict[int, str]
            input: mapping from register numbers to APL
            symbols, e.g., ``1`` to ``_INTERNAL0``

        sm_registers: Dict[str, str]
            input: mapping from normalized section masks like
            ``0xFFFF`` to GVML pseudo-constants like
            ``SM_0XFFFF``

        bleir_map: Dict[str, Any]
            input: mapping from bleir symbols to various
            objects and functions that the bleir renderer
            interprets, e.g., ``^=`` to a function that
            renders an XOR-equals operation; ``IR`` to an
            RN_REG, etc.

        Returns
        -------
        Any

            Bleir code for the masked statement, e.g.,
            ``MASKED(mask=...)``. The bleir code is equivalent
            to an S-expression, i.e., nested, representation
            of the final machine code. The final code is, in
            turn, interpreted in nanosim and passed to a
            physical code generator.

        """

        normal, shift = self.msk.get_normalized()
        sm_register = sm_registers[normal]

        result = BLEIR.masked(
            BLEIR.shift(bleir_map[sm_register], shift),
            self.stmt.render_bleir(rn_registers, bleir_map))

        return result

    def __str__(self: "APL_masked_stmt") -> str:
        r"""Produce a human-readable string representation
        of the masked statement."""
        return f'{self.msk} : {self.stmt};'

    def defs(self: "APL_statement"):
        r"""Produce a list of register-mask tuples from an
        internal list of registers and an internal mask,
        corresponding to the KILL list for dependency
        analysis."""
        ds = self.stmt.defs()
        ms = self.msk.mask
        result = [modify_rl_ref(var, ms) for var in ds]
        return result

    def uses(self: "APL_statement"):
        r"""Produce a list of register-mask tuples from an
        internal list of registers and an internal mask,
        corresponding to the GEN list for dependency
        analysis."""
        us = self.stmt.uses()
        ms = self.msk.mask
        result = [modify_rl_ref(var, ms) for var in us]
        return result

    def conflict(self, other: APL_statement) -> bool:
        r"""Check whether two APL statements have conflicting
        sections or vector registers. Does not exclude special
        cases of read-after-write or broadcast-after-read.

        Parameters
        ----------

            other: APL_statement
        """
        ddep = data_dep(self, other)
        adep = anti_dep(self, other)
        odep = output_dep(self, other)

        return ddep or adep or odep


@bleir_dataclass
class APL_comment(APL_statement):
    comment: Any

    def render_bleir(
            self: "APL_comment",
            rn_registers: Dict[int, str],
            bleir_map: Dict[str, Any]) -> Any:
        return BLEIR.SingleLineComment(str(self.comment))

    def __str__(self: "APL_comment") -> str:
        return f'/* {self.comment} */'


@bleir_dataclass
class APL_multi_statement(APL_statement):

    stmts: Sequence[APL_statement]

    def render_bleir(
             self: "APL_multi_statement",
             rn_registers: Dict[int, str],
             sm_registers: Dict[str, str],
             bleir_map: Dict[str, Any]) -> Any:
        return BLEIR.MultiStatement(
            statements=[stmt.render_bleir(
                rn_registers, sm_registers, bleir_map)
                for stmt in self.stmts])

    def __str__(self: "APL_multi_statement") -> str:
        multi_stmt_str = '['+';'.join([str(stmt) for stmt in self.stmts]) + ']'
        return multi_stmt_str

    def is_laning_v2_compatible(self, candidate_cmd: APL_statement):

        r"""Conservatively remove all laning with NRL and SRL.
        Otherwise, this is the same as V1 laning."""

        # Any command is compatible with an empty instruction.
        if len(self.stmts) == 0:
            return True

        # broadcast after read
        if isinstance(candidate_cmd.stmt, APL_src_from_rl) \
                and isinstance(self.stmts[0].stmt, APL_rl_from_sb) \
                and len(self.stmts) < 4:
                # Restricting broadcast-after-read to doubletons creates
                # singleton broadcasts, which generate hangs (in the case
                # of GL). Solution? Follow singleton GL = ... with
                # singleton NOOP;

                # and len(self.stmts) == 1:
            return True

        # read after a write with disjoint SBs
        # generalize to take reads without sbs <--- LATER
        if is_rl_read_sb(candidate_cmd.stmt) \
                and is_sb_write(self.stmts[0].stmt) \
                and len(set(candidate_cmd.stmt.sbs) & set(self.stmts[0].stmt.sbs)) == 0 \
                and len(self.stmts) == 1:
            # two writes
            # PROVED: a read after two writes with non-disjoint masks fails
            # we tried with len(self.stmts) < 3

            # ANTI-FENCEPOST: should generate regressions if return True below

            # return False

            # AGGRESSIVE CONSERVATISM: no read-after-write when either
            # command has an NRL or SRL, irrespective of masks.

            cand_has_NRL, cand_has_SRL, mask_overlap, self_has_NRL, self_has_SRL = \
                self.potentially_shifted_masks_overlap(candidate_cmd)

            # Observed empirically that even refined mask-overlap computation
            # is not enough. Must reject any candidates when self or cand has
            # NRL or SRL.

            if (self_has_NRL or self_has_SRL or
                    cand_has_NRL or cand_has_SRL or mask_overlap):
            # if (mask_overlap):
                return False

            return True

        if isinstance(candidate_cmd.stmt, APL_rl_from_sb) \
                and isinstance(self.stmts[0].stmt, APL_sb_from_src) \
                and candidate_cmd.msk.mask & self.stmts[0].msk.mask == 0 \
                and len(self.stmts) == 1:
            return True

        if isinstance(candidate_cmd.stmt, APL_set_rl) \
                and candidate_cmd.msk.mask & self.stmts[0].msk.mask == 0 \
                and len(self.stmts) == 1:
            return True

        # read after a write with non-disjoint SBs
        # PROVED NOT TO WORK
        #elif is_rl_read(candidates_cmd.stmt) \
        #        and is_sb_write(self.stmts[0].stmt) \
        #        and len(self.stmts) == 1:
        #    return True

        # read after write with disjoint masks

        # read after read, disjoint sbs, disjoint masks
        # generalize to take reads without sbs <--- LATER
        ## [[bbeckman: breaks in the case of (e.g.) 0x1111 << 1: RL ^= SB[x];
        ##                                          0x1111 << 0: RL  = SB[INTERNAL1] & SRL;
        ##  because the masks are NOT really disjoint. The mask for the SRL is actually 0x1111 << 1
        ##  so we need to disable this case. ]]

        if is_rl_read_sb(candidate_cmd.stmt) \
                and is_rl_read_sb(self.stmts[0].stmt) \
                and len(set(candidate_cmd.stmt.sbs) & set(self.stmts[0].stmt.sbs)) == 0 \
                and candidate_cmd.msk.mask & self.stmts[0].msk.mask == 0 \
                and len(self.stmts) == 1:

            # FENCEPOST
            # return False

            cand_has_NRL, cand_has_SRL, mask_overlap, self_has_NRL, self_has_SRL = \
                self.potentially_shifted_masks_overlap(candidate_cmd)

            if (self_has_NRL or self_has_SRL or
                    cand_has_NRL or cand_has_SRL or mask_overlap):
                return False
            return True

        if is_sb_write(candidate_cmd.stmt) \
                and is_sb_write(self.stmts[0].stmt) \
                and self.all_sbs_in_same_sb_group(candidate_cmd.stmt) \
                and candidate_cmd.msk.mask & self.stmts[0].msk.mask == 0 \
                and len(self.stmts) == 1:
            return True

        return False

    def all_sbs_in_same_sb_group(self, candidate_cmd_stmt) -> bool:
        cand_sbs = set(candidate_cmd_stmt.sbs)
        self_sbs = set(self.stmts[0].stmt.sbs)
        union = cand_sbs | self_sbs
        group_0 = set(range(0, 8))
        group_1 = set(range(8, 16))
        group_2 = set(range(16, 24))
        all_in_group_0 = all([sb in group_0 for sb in union])
        all_in_group_1 = all([sb in group_1 for sb in union])
        all_in_group_2 = all([sb in group_2 for sb in union])
        result = all_in_group_0 | all_in_group_1 | all_in_group_2
        return result

    def potentially_shifted_masks_overlap(self, candidate_cmd):

        self_mask = self.stmts[0].msk.mask
        self_mask_left_shifted = ((self_mask << 1) & 0xffff)
        self_mask_rght_shifted = ((self_mask >> 1) & 0xffff)

        cand_mask = candidate_cmd.msk.mask
        cand_mask_left_shifted = ((cand_mask << 1) & 0xffff)
        cand_mask_rght_shifted = ((cand_mask >> 1) & 0xffff)

        self_has_NRL = (
                hasattr(self.stmts[0], 'src')
                and (self.stmts[0].src == 'NRL'))

        self_has_SRL = (
                hasattr(self.stmts[0], 'src')
                and (self.stmts[0].src == 'SRL'))

        cand_has_NRL = (
                hasattr(candidate_cmd.stmt, 'src')
                and (candidate_cmd.stmt.src == 'NRL'))

        cand_has_SRL = (
                hasattr(candidate_cmd.stmt, 'src')
                and (candidate_cmd.stmt.src == 'SRL'))

        if self_has_NRL:  # e.g., 0x2222 SB[5] ^= NRL; check self << 1
            if cand_has_NRL:  # both have same shift
                mask_overlap = (0 != (self_mask & cand_mask))
            elif cand_has_SRL:  # check cand >> 1
                mask_overlap = (0 != (self_mask_left_shifted &
                                      cand_mask_rght_shifted))
            else:  # don't shift cand
                mask_overlap = (0 != (self_mask_left_shifted &
                                      cand_mask))

        elif self_has_SRL:  # e.g., 0x2222 SB[5] ^= SRL; check self >> 1
            if cand_has_NRL:  # check cand << 1
                mask_overlap = (0 != (self_mask_rght_shifted &
                                      cand_mask_left_shifted))
            elif cand_has_SRL:  # both have same shift
                mask_overlap = (0 != (self_mask & cand_mask))
            else:  # don't shift cand
                mask_overlap = (0 != (self_mask_rght_shifted &
                                      cand_mask))

        else:
            if cand_has_NRL:  # check cand.NRL << 1
                mask_overlap = (0 != (self_mask & cand_mask_left_shifted))
            elif cand_has_SRL:  # check cand.SRL >> 1
                mask_overlap = (0 != (self_mask & cand_mask_rght_shifted))
            else:
                mask_overlap = (0 != (self_mask & cand_mask))
        return cand_has_NRL, cand_has_SRL, mask_overlap, self_has_NRL, self_has_SRL


@bleir_dataclass
class APL_sb_from_src(APL_statement):
    sbs: Sequence[int]
    src: str

    def __init__(self, sbs: SBs, src: str):
        super().__init__()
        self.sbs = sbs
        self.src = src

    def render_bleir(
            self: "APL_sb_from_src",
            rn_registers: Dict[int, str],
            bleir_map: Dict[str, Any]) -> Any:
        xyz = [rn_registers[sb] for sb in self.sbs]
        xyz = [bleir_map[sb] for sb in xyz]
        return BLEIR.assign(BLEIR.SB[xyz], bleir_map[self.src])

    def __str__(self: "APL_sb_from_src") -> str:
        xyz = ",".join(map(str, self.sbs))
        return f'SB[{xyz}] = {self.src}'

    def defs(self: "APL_statement"):
        return [f'SB[{x}]' for x in self.sbs]

    def uses(self: "APL_statement"):
        return [self.src]


@bleir_dataclass
class APL_set_rl(APL_statement):

    value: int

    def render_bleir(
            self: "APL_set_rl",
            rn_registers: Dict[int, str],
            bleir_map: Dict[str, Any]) -> Any:
        return BLEIR.assign(BLEIR.RL, self.value)

    def __str__(self: "APL_set_rl") -> Sequence[str]:
        return f'RL = {self.value}'

    def defs(self: "APL_statement"):
        return ['RL']

    def uses(self: "APL_statement"):
        return []


@bleir_dataclass
class APL_rl_from_sb(APL_statement):
    sbs: Sequence[int]
    assign_op: str

    def render_bleir(
            self: "APL_rl_from_sb",
            rn_registers: Dict[int, str],
            bleir_map: Dict[str, Any]) -> Any:
        assignment = bleir_map[f'{self.assign_op}=']
        xyz = [rn_registers[sb] for sb in self.sbs]
        xyz = [bleir_map[sb] for sb in xyz]
        return assignment(BLEIR.RL, BLEIR.SB[xyz])

    def __str__(self: "APL_rl_from_sb") -> str:
        xyz = ",".join(map(str, self.sbs))
        return f'RL {self.assign_op}= SB[{xyz}]'

    def defs(self: "APL_statement"):
        return ['RL']

    def uses(self: "APL_statement"):
        return [f'SB[{x}]' for x in self.sbs] + (['RL'] if self.assign_op != '' else [])


@bleir_dataclass
class APL_src_from_rl(APL_statement):
    src: str
    assign_op: str

    def render_bleir(
            self: "APL_src_from_rl",
            rn_registers: Dict[int, str],
            bleir_map: Dict[str, Any]) -> Any:
        assignment = bleir_map[f'{self.assign_op}=']
        return assignment(bleir_map[self.src], BLEIR.RL)

    def __str__(self: "APL_src_from_rl") -> str:
        return f'{self.src} {self.assign_op}= RL'

    def defs(self: "APL_statement"):
        return [self.src]

    def uses(self: "APL_statement"):
        return ['RL'] + ([self.src] if self.assign_op != '' else [])


@bleir_dataclass
class APL_rl_from_src(APL_statement):
    src: str
    assign_op: str

    def render_bleir(
            self: "APL_rl_from_src",
            rn_registers: Dict[int, str],
            bleir_map: Dict[str, Any]) -> Any:
        assignment = bleir_map[f'{self.assign_op}=']
        return assignment(BLEIR.RL, bleir_map[self.src])

    def __str__(self: "APL_rl_from_src") -> str:
        return f'RL {self.assign_op}= {self.src}'

    def defs(self: "APL_statement"):
        return ['RL']

    def uses(self: "APL_statement"):
        return [self.src] + (['RL'] if self.assign_op != '' else [])


@bleir_dataclass
class APL_rl_from_sb_src(APL_statement):
    sbs: Sequence[int]
    src: str
    assign_op: str
    binary_op: str

    def render_bleir(
            self: "APL_rl_from_sb_src",
            rn_registers: Dict[int, str],
            bleir_map: Dict[str, Any]) -> Any:
        assignment = bleir_map[f'{self.assign_op}=']
        binop = bleir_map[self.binary_op]
        xyz = [rn_registers[sb] for sb in self.sbs]
        xyz = [bleir_map[sb] for sb in xyz]
        return assignment(BLEIR.RL,
                          binop(BLEIR.SB[xyz],
                                bleir_map[self.src]))

    def __str__(self: "APL_rl_from_sb_src") -> str:
        xyz = ",".join(map(str, self.sbs))
        return f'RL {self.assign_op}= SB[{xyz}] {self.binary_op} {self.src}'

    def defs(self: "APL_statement"):
        return ['RL']

    def uses(self: "APL_statement"):
        return [f'SB[{x}]' for x in self.sbs] + [self.src] + (['RL'] if self.assign_op != '' else [])


def collect_all_normalized_masks(
            stmts: Sequence[APL_multi_statement]) -> Set[str]:
        return {cmd.msk.get_normalized()[0] for stmt in stmts for cmd in stmt.stmts}


def collect_all_normalized_masks_from_masked_stmt(
        cmds: Sequence[APL_masked_stmt]) -> Set[str]:
    return {cmd.msk.get_normalized()[0] for cmd in cmds}
