"""
By Brian Beckman and Dylon Edwards
"""

import sys
from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import wraps
from itertools import chain
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import numpy as np

from reactivex.subject import Subject

import open_belex.diri.expressions as exp
from open_belex.common.constants import (NSB, NSECTIONS, NUM_APCS_PER_APUC,
                                         NUM_HALF_BANKS_PER_APC,
                                         NUM_HALF_BANKS_PER_APUC, NUM_L1_ROWS,
                                         NUM_L2_ROWS, NUM_PLATS_PER_APC,
                                         NUM_PLATS_PER_APUC,
                                         NUM_PLATS_PER_HALF_BANK, NVR)
from open_belex.common.half_bank import HalfBankAbstractInterface as HalfBank
from open_belex.common.mask import Mask
from open_belex.common.rsp_fifo import ApucRspFifo, RspFifoMsg
from open_belex.common.stack_manager import contextual
from open_belex.common.subset import Subset
from open_belex.common.types import Integer
from open_belex.utils.index_utils import Indices, parse_indices
from open_belex.utils.memory_utils import NUM_VM_REGS, vmr_to_row
from open_belex.utils.name_utils import camel_case_to_underscore

DEFAULTSLICE_FOR_GRID: int = 4

# 1 low-order and 1 high-order half-bank per APC => 2 half-banks per APC
# 2 APCs per APUC => 4 half-banks per APUC for LGL
NUM_LGL_PLATS: int = NUM_PLATS_PER_HALF_BANK * 4

NGGL_ROWS: int = 4
NLGL_ROWS: int = 1

SECTION_SHAPE: int = NUM_PLATS_PER_APUC
VR_SHAPE: Tuple[int, int] = (NUM_PLATS_PER_APUC, NSECTIONS)
GL_SHAPE: Tuple[int] = (NUM_PLATS_PER_APUC,)
GGL_SHAPE: Tuple[int, int] = (NUM_PLATS_PER_APUC, NGGL_ROWS)

# RL_SHAPE could be NRL, ERL, WRL, SRL, RL, or their inverses
RL_SHAPE: Tuple[int, int] = (NUM_PLATS_PER_APUC, NSECTIONS)

RSP16_SHAPE: Tuple[int, int] = (NUM_PLATS_PER_APUC // 16, NSECTIONS)
RSP256_SHAPE: Tuple[int, int] = (NUM_PLATS_PER_APUC // 256, NSECTIONS)
RSP2K_SHAPE: Tuple[int, int] = (NUM_PLATS_PER_APUC // 2048, NSECTIONS)
RSP32K_SHAPE: Tuple[int, int] = (NUM_PLATS_PER_APUC // 32768, NSECTIONS)

VALID_L1_ADDRESSES: Sequence[bool] = [False] * NUM_L1_ROWS
for vm_reg in range(0, NUM_VM_REGS, 2):
    l1_addr = vmr_to_row(vm_reg)
    for offset in range(9):
        VALID_L1_ADDRESSES[l1_addr + offset] = True


class RspMode(IntEnum):
    IDLE = auto()
    RSP16_READ = auto()
    RSP256_READ = auto()
    RSP2K_READ = auto()
    RSP32K_READ = auto()
    RSP16_WRITE = auto()
    RSP256_WRITE = auto()
    RSP2K_WRITE = auto()
    RSP32K_WRITE = auto()


def plats_for_bank(bank: int) -> Sequence[int]:
    apc_0_lo = bank * NUM_PLATS_PER_HALF_BANK
    apc_0_hi = apc_0_lo + 4 * NUM_PLATS_PER_HALF_BANK
    apc_1_lo = apc_0_lo + NUM_PLATS_PER_APC
    apc_1_hi = apc_0_hi + NUM_PLATS_PER_APC
    plats = list(
        chain(
            range(apc_0_lo, apc_0_lo + NUM_PLATS_PER_HALF_BANK),
            range(apc_0_hi, apc_0_hi + NUM_PLATS_PER_HALF_BANK),
            range(apc_1_lo, apc_1_lo + NUM_PLATS_PER_HALF_BANK),
            range(apc_1_hi, apc_1_hi + NUM_PLATS_PER_HALF_BANK)))
    return plats


VectorIndices = Union[Indices,
                      Tuple[Indices],
                      Tuple[Indices, Indices],
                      Tuple[Indices, Indices, Indices]]

HalfBankIndices = Union[VectorIndices,
                        Tuple[Indices, Indices, Indices, Indices]]


def make_vector_register() -> np.ndarray:
    return np.ndarray((NUM_PLATS_PER_APUC, NSECTIONS), dtype=bool)


def index_to_list(xs):
    if isinstance(xs, Integer.__args__):
        x = xs
        xs = [x]
    return xs


def anti_glass(glass: str) -> np.ndarray:
    sections = glass.split("\n")
    vr = np.ndarray((len(sections[0]), len(sections)), dtype=bool)
    for j, section in enumerate(sections):
        for i, plat in enumerate(section):
            vr[i, j] = (plat == '1')
    return vr


def glass_to_vr(value: Union[np.ndarray, bool, str, List[str]]) \
        -> Union[np.ndarray, bool]:
    """Conditional transform that operates on types str and List[str]. This
    transform rewrites glass statements as numpy arrays compatible with portions
    of bool VRs. Values of all other types are returned unchanged."""
    if isinstance(value, str):
        value = value.split("\n")
    if hasattr(value, "__len__") \
       and len(value) > 0 \
       and isinstance(value[0], str):
        values = value
        values = list(map(anti_glass, values))
        value = np.concatenate(values, axis=1)
    return value


VR = np.ndarray
VRs = Sequence[VR]

SB_or_VR = Union[Integer, VR]
SBs_and_VRs = Union[SB_or_VR, Sequence[SB_or_VR]]

UnaryOp = Callable[[np.ndarray], np.ndarray]
BinaryOp = Callable[[np.ndarray, np.ndarray], np.ndarray]
TernaryOp = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

SectionMap = Sequence[Tuple[VR, int, np.ndarray]]

# The current way parallelism is modeled within multi-statements is that diffs
# are generated at the section level of each command and applied once the
# respective instruction is executed, so all commands see the same data (with
# respect to their half-clock semantics). It is possible that with proper
# sortation, it will no longer be necessary to generate and apply patches for
# parallelism, but experimentation is required.

PatchFn = Callable[["DIRI", Any], None]
PatchPair = Tuple[PatchFn, Any]


def patch_whole_vr(diri: "DIRI", patch: Tuple[np.ndarray, np.ndarray]) -> None:
    vr, update = patch
    vr[::] = update


def patch_sb(diri: "DIRI", patch: SectionMap) -> None:
    for (vr, section, wordline) in patch:
        vr[::, section] = wordline


def patch_gl(diri: "DIRI", patch: np.ndarray) -> None:
    diri.GL[::] = patch


def patch_ggl(diri: "DIRI", patch: np.ndarray) -> None:
    diri.GGL[::] = patch


def patch_lgl(diri: "DIRI", patch: np.ndarray) -> None:
    diri.LGL[::] = patch


def patch_whole_l1(diri: "DIRI", patch: np.ndarray) -> None:
    diri.L1[VALID_L1_ADDRESSES] = patch[VALID_L1_ADDRESSES]


def patch_l1(diri: "DIRI", patch: Tuple[int, np.ndarray]) -> None:
    lx_addr, LX = patch
    if LX.shape == (NUM_PLATS_PER_APUC, NGGL_ROWS):
        diri.L1[lx_addr] = LX
    elif LX.shape == (NUM_LGL_PLATS,):
        bank = (lx_addr >> 11) & 0b11
        group = (lx_addr >> 9) & 0b11
        row = lx_addr & 0b111111111
        plats = plats_for_bank(bank)
        diri.L1[row, plats, group] = LX
    else:
        raise ValueError(f"Unsupported L1 patch shape: {LX.shape}")


def patch_whole_l2(diri: "DIRI", patch: np.ndarray) -> None:
    diri.L2[::] = patch


def patch_l2(diri: "DIRI", patch: Tuple[int, np.ndarray]) -> None:
    lx_addr, LX = patch
    diri.L2[lx_addr] = LX


def patch_rsp16(diri: "DIRI", patch: np.ndarray) -> None:
    diri.RSP16[::] = patch


def patch_partial_rsp16(diri: "DIRI",
                        patch: Sequence[Tuple[int, np.ndarray]]) -> None:
    for (section, wordline) in patch:
        diri.RSP16[::, section] = wordline


def patch_rsp256(diri: "DIRI", patch: np.ndarray) -> None:
    diri.RSP256[::] = patch


def patch_rsp2k(diri: "DIRI", patch: np.ndarray) -> None:
    diri.RSP2K[::] = patch


def patch_rsp32k(diri: "DIRI", patch: np.ndarray) -> None:
    diri.RSP32K[::] = patch


def patch_rsps(diri: "DIRI",
               patches: Tuple[
                   Tuple[PatchFn, np.ndarray],
                   Tuple[PatchFn, np.ndarray],
                   Tuple[PatchFn, np.ndarray],
                   Tuple[PatchFn, np.ndarray],
               ]) -> None:
    for patch_fn, patch in patches:
        patch_fn(diri, patch)


def patch_noop(diri: "DIRI", patch: Any) -> None:
    pass


def patch_rsp_end(diri: "DIRI", patches: Any) -> None:
    if diri.rsp_mode == RspMode.RSP32K_READ:
        rsp2k = diri.RSP2K.data
        rsp32k = diri.RSP32K.data
        for apc_id in range(NUM_APCS_PER_APUC):
            offset = apc_id * NUM_HALF_BANKS_PER_APC
            rsp_fifo_msg = RspFifoMsg()
            for half_bank in range(NUM_HALF_BANKS_PER_APC):
                for section in range(NSECTIONS):
                    rsp_fifo_msg.rsp2k[half_bank] |= \
                        rsp2k[half_bank + offset][section] << section
            for section in range(NUM_HALF_BANKS_PER_APC):
                rsp_fifo_msg.rsp32k |= \
                    rsp32k[0][section + offset] << section
            apc_rsp_fifo = diri.apuc_rsp_fifo.queues[apc_id]
            if apc_rsp_fifo.enqueue(rsp_fifo_msg) != 0:
                raise RuntimeError(
                    f"RSP queue is full for APC {apc_id}")
    diri.rsp_mode = RspMode.IDLE
    patch_rsps(diri, patches)


def patch_rsp_start_ret(diri: "DIRI", patch: Any) -> None:
    pass


def patch_l2_end(diri: "DIRI", patch: Any) -> None:
    pass


def patch_rwinh_set(diri: "DIRI", patch: Any) -> None:
    mask = patch
    if len(mask) > 0:
        ss = list(mask)
        diri.rwinh_filter[::, ss] = diri.hb.rl.data[::, ss]
        diri.rwinh_sects |= mask


def patch_rwinh_rst(diri: "DIRI", patch: Any) -> None:
    mask, has_read = patch
    if len(mask) > 0:
        ss = list(mask)
        if not has_read:
            diri.hb.rl[::, ss] = diri.rwinh_filter[::, ss]
        diri.rwinh_filter[::, ss] = True
        diri.rwinh_sects &= ~mask


def patch_with(patch_fn: PatchFn) -> Callable:

    def decorator(fn: Callable) -> Callable:

        @wraps(fn)
        def wrapper(self: "DIRI", *args, **kwargs) -> Tuple[PatchFn, Any]:
            patch = fn(self, *args, **kwargs)
            if self.in_place:
                patch_fn(self, patch)
            return patch_fn, patch

        return wrapper

    return decorator


def apply_patches(diri: "DIRI", patches: Sequence[Tuple[PatchFn, Any]]) -> None:
    for patch_fn, patch in patches:
        patch_fn(diri, patch)


def rsp_from_rsp(fn: Callable) -> Callable:
    # e.g. "rsp16_from_rsp256".split("_from_") == ("rsp16", "rsp256")
    rsp_left_nym, rsp_right_nym = fn.__name__.split("_from_")
    rsp_left_nym = rsp_left_nym.upper()
    rsp_right_nym = rsp_right_nym.upper()

    patch_nym = f"patch_{rsp_left_nym.lower()}"
    patch_fn = getattr(sys.modules[__name__], patch_nym)

    @patch_with(patch_fn)
    @wraps(fn)
    def wrapper(self: "DIRI") -> np.ndarray:
        rsp_left = getattr(self, rsp_left_nym)
        rsp_right = getattr(self, rsp_right_nym)
        rsp_right = rsp_right.data

        rsp_left = np.zeros(rsp_left.shape, dtype=bool)

        left_width = rsp_left.shape[0]
        right_width = rsp_right.shape[0]

        if left_width < right_width:
            self.rsp_from_contraction(rsp_left,
                                      rsp_right,
                                      left_width,
                                      right_width)
        else:
            self.rsp_from_expansion(rsp_left,
                                    rsp_right,
                                    left_width,
                                    right_width)

        return rsp_left

    return wrapper


@dataclass
class SBAccessor:
    hb: np.ndarray

    def __getitem__(self: "SBAccessor", sb: int) -> np.ndarray:
        if 0 <= sb < NSB:
            return self.hb[sb]
        raise ValueError(
            f"'sb' must be between 0 and {NSB-1}, inclusive, not {sb}.")


@dataclass
class VMRAccessor:
    L1: np.ndarray

    def __getitem__(self: "VMRAccessor", vmr_addr: int) -> np.ndarray:
        if 0 <= vmr_addr < NUM_VM_REGS:
            lower_l1_addr = vmr_to_row(vmr_addr)
            upper_l1_addr = lower_l1_addr + 4
            vmr = np.zeros(VR_SHAPE, dtype=bool)
            vmr_sections = np.array([0, 4, 8, 12])
            for group, l1_addr in enumerate(range(lower_l1_addr, upper_l1_addr)):
                # Invert the GGL operations to reconstruct the stored VR
                vmr[::, group + vmr_sections] = self.L1[l1_addr]
            return vmr
        raise ValueError(
            f"Expected vmr_addr={vmr_addr} to be in the range [0, {NUM_VM_REGS}).")


Subscriber = Callable[[Any], None]


class DiriRegister:
    kind_nym: str
    data: np.ndarray
    subject: Subject
    on_update: Callable[[Union[Sequence[Indices], Indices],
                         Union[np.ndarray, bool]],
                        None]

    def __init__(self: "DiriRegister", shape: Sequence[int]) -> None:
        # FIXME: Not zeroing everything out is more realistic, but would
        # presently break the expected behavior.
        self.kind_nym = camel_case_to_underscore(self.__class__)[len("diri_"):]
        self.kind_nym = f"diri::{self.kind_nym}"
        self.data = np.zeros(shape=shape, dtype=bool)
        self.subject = Subject()
        if len(shape) == 2:
            self.on_update = self.on_2d_update
        elif len(shape) == 1:
            self.on_update = self.on_1d_update
        else:
            raise ValueError(f"Unsupported register shape: {shape}")

    def __deepcopy__(self: "DiriRegister",
                     memo: Dict[int, Any]) -> "DiriRegister":
        cls = self.__class__
        cpy = cls.__new__(cls)
        setattr(cpy, "kind_nym", self.kind_nym)
        setattr(cpy, "data", deepcopy(self.data))
        setattr(cpy, "subject", Subject())
        cpy.subject.observers = [observer
                                 for observer in self.subject.observers]
        if self.ndim == 2:
            setattr(cpy, "on_update", cpy.on_2d_update)
        elif self.ndim == 1:
            setattr(cpy, "on_update", cpy.on_1d_update)
        else:
            raise ValueError(f"Unsupported register shape: {self.shape}")
        attr_nyms = \
            set(self.__annotations__.keys()) - \
            {"data", "subject", "on_update", "kind_nym"}
        for nym in attr_nyms:
            val = getattr(self, nym)
            setattr(cpy, nym, deepcopy(val))
        memo[id(self)] = cpy
        return cpy

    def __eq__(self: "DiriRegister", other: Any) -> bool:
        if not isinstance(other, DiriRegister):
            return False
        return np.array_equal(self.data, other.data)

    def __ne__(self: "DiriRegister", other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def shape(self: "DiriRegister") -> Sequence[int]:
        return self.data.shape

    @property
    def ndim(self: "DiriRegister") -> int:
        return self.data.ndim

    def any(self: "DiriRegister") -> bool:
        return self.data.any()

    def all(self: "DiriRegister") -> bool:
        return self.data.all()

    def __len__(self: "DiriRegister") -> int:
        return len(self.data)

    def __getitem__(self: "DiriRegister",
                    indices: Union[Sequence[Indices], Indices]) \
            -> Union[np.ndarray, bool]:
        return self.data.__getitem__(indices)

    def __setitem__(self: "DiriRegister",
                    indices: Union[Sequence[Indices], Indices],
                    value: Union["DiriRegister", np.ndarray, bool]) -> None:
        if isinstance(value, DiriRegister):
            value = value.data
        self.data.__setitem__(indices, value)
        if len(self.subject.observers) > 0:
            self.on_update(indices, value)

    @staticmethod
    def is_2d_grid_entry(indices: Any, nth: int) -> bool:
        return isinstance(indices, np.ndarray) \
            and indices.ndim == 2 \
            and indices.shape[(nth + 1) % 2] == 1

    @classmethod
    def is_2d_grid(cls: Type, indices: Any) -> bool:
        return isinstance(indices, tuple) \
            and len(indices) == 2 \
            and cls.is_2d_grid_entry(indices[0], 0) \
            and cls.is_2d_grid_entry(indices[1], 1)

    def parse_2d_indices(self: "DiriRegister",
                         indices: Union[Sequence[Indices], Indices]) \
            -> Tuple[Sequence[int], Sequence[int]]:

        if isinstance(indices, tuple):
            if self.is_2d_grid(indices):
                cols = [indices[0][i][0] for i in range(len(indices[0]))]
                rows = [indices[1][0][j] for j in range(len(indices[1][0]))]
            else:
                cols = parse_diri_indices(indices[0], self.shape[0])

                if len(indices) == 1:
                    rows = list(range(self.shape[1]))

                elif len(indices) == 2:
                    rows = parse_diri_indices(indices[1], self.shape[1])

                else:
                    raise ValueError(
                        f"Too many indices ({len(indices)}): {indices}")

        else:
            cols = parse_diri_indices(indices, self.shape[0])
            rows = list(range(self.shape[1]))

        return cols, rows

    def on_2d_update(self: "DiriRegister",
                     indices: Union[Sequence[Indices], Indices],
                     value: Union[np.ndarray, bool]) -> None:
        plats, sections = self.parse_2d_indices(indices)
        if plats is not None and len(plats) == self.shape[0]:
            plats = None
        if sections is not None and len(sections) == self.shape[1]:
            sections = None
        self.subject.on_next((self.kind_nym, plats, sections, value))

    @staticmethod
    def is_1d_grid_entry(indices: Any, nth: int) -> bool:
        return isinstance(indices, np.ndarray) \
            and indices.ndim == 1

    @classmethod
    def is_1d_grid(cls: Type, indices: Any) -> bool:
        return isinstance(indices, tuple) \
            and len(indices) == 1 \
            and cls.is_1d_grid_entry(indices[0], 0)

    def parse_1d_indices(self: "DiriRegister",
                         indices: Union[Sequence[Indices], Indices]) \
            -> Tuple[Sequence[int], Sequence[int]]:

        if isinstance(indices, tuple):
            if self.is_1d_grid(indices):
                cols = [indices[0][i] for i in range(len(indices[0]))]
            else:
                cols = parse_diri_indices(indices[0], self.shape[0])

        else:
            cols = parse_diri_indices(indices, self.shape[0])

        return cols

    def on_1d_update(self: "DiriRegister",
                     indices: Union[Sequence[Indices], Indices],
                     value: Union[np.ndarray, bool]) -> None:
        plats = self.parse_1d_indices(indices)
        if plats is not None and len(plats) == self.shape[0]:
            plats = None
        self.subject.on_next((self.kind_nym, plats, value))

    def subscribe(self: "DiriRegister", subscriber: Subscriber) -> None:
        self.subject.subscribe(subscriber)

    def __invert__(self: "DiriRegister") -> np.ndarray:
        return ~self.data


class DiriRWInhFilter(DiriRegister):

    def __init__(self: "DiriRWInhFilter") -> None:
        super().__init__(VR_SHAPE)
        self.data[::] = True


class DiriVR(DiriRegister):
    row_number: Optional[int] = None

    def __init__(self: "DiriVR", row_number: int) -> None:
        super().__init__(VR_SHAPE)
        self.row_number = row_number

    def __eq__(self: "DiriVR", other: Any) -> bool:
        return super().__eq__(other) \
            and self.row_number == other.row_number

    def on_2d_update(self: "DiriVR",
                     indices: Union[Sequence[Indices], Indices],
                     value: Union[np.ndarray, bool]) -> None:
        plats, sections = self.parse_2d_indices(indices)
        if plats is not None and len(plats) == self.shape[0]:
            plats = None
        if sections is not None and len(sections) == self.shape[1]:
            sections = None
        self.subject.on_next(
            (self.kind_nym, self.row_number, plats, sections, value))


class DiriRL(DiriRegister):

    def __init__(self: "DiriRL") -> None:
        super().__init__(RL_SHAPE)


def parse_diri_indices(indices: Indices, upper_bound: int) -> List[int]:
    if isinstance(indices, Integer.__args__):
        index = indices
        indices = [index]
    return parse_indices(indices, upper_bound)


class Diri3D(ABC):
    size: int
    cols: int
    rows: int

    def __init__(self: "Diri3D", size: int, cols: int, rows: int) -> None:
        self.size = size
        self.cols = cols
        self.rows = rows

    @property
    def shape(self: "Diri3D") -> Sequence[int]:
        return (self.size, self.cols, self.rows)

    @property
    def ndim(self: "Diri3D") -> int:
        return 3

    @abstractmethod
    def any(self: "Diri3D") -> bool:
        raise NotImplementedError

    @abstractmethod
    def all(self: "Diri3D") -> bool:
        raise NotImplementedError

    def __len__(self: "Diri3D") -> int:
        return self.size

    @staticmethod
    def is_grid_entry(indices: Any, nth: int) -> bool:
        return isinstance(indices, np.ndarray) \
            and indices.ndim == 3 \
            and indices.shape[(nth + 1) % 3] == 1 \
            and indices.shape[(nth + 2) % 3] == 1

    @classmethod
    def is_grid(cls: Type, indices: Any) -> bool:
        return isinstance(indices, tuple) \
            and len(indices) == 3 \
            and cls.is_grid_entry(indices[0], 0) \
            and cls.is_grid_entry(indices[1], 1) \
            and cls.is_grid_entry(indices[2], 2)

    def parse_indices(self: "Diri3D",
                      indices: Union[Sequence[Indices], Indices]) \
            -> Tuple[Sequence[int],
                     Sequence[int],
                     Sequence[int]]:

        if isinstance(indices, tuple):
            if self.is_grid(indices):
                idxs = [indices[0][i][0][0] for i in range(len(indices[0]))]
                cols = [indices[1][0][j][0] for j in range(len(indices[1][0]))]
                rows = [indices[2][0][0][k] for k in range(len(indices[2][0][0]))]
            else:
                idxs = parse_diri_indices(indices[0], self.size)

                if len(indices) == 1:
                    cols = list(range(self.cols))
                    rows = list(range(self.rows))

                elif len(indices) == 2:
                    cols = parse_diri_indices(indices[1], self.cols)
                    rows = list(range(self.rows))

                elif len(indices) == 3:
                    cols = parse_diri_indices(indices[1], self.cols)
                    rows = parse_diri_indices(indices[2], self.rows)

                else:
                    raise ValueError(
                        f"Too many indices ({len(indices)}): {indices}")

        else:
            idxs = parse_diri_indices(indices, self.size)
            cols = list(range(self.cols))
            rows = list(range(self.rows))

        return idxs, cols, rows

    @abstractmethod
    def register_at(self: "Diri3D", index: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self: "Diri3D", other: Any) -> bool:
        raise NotImplementedError

    def __ne__(self: "Diri3D", other: Any) -> bool:
        return not self.__eq__(other)

    @staticmethod
    def should_transpose_for_numpy(
            indices: Union[Sequence[Indices], Indices]) -> bool:
        # special numpy case (why?)
        return isinstance(indices, tuple) \
            and len(indices) == 3 \
            and not isinstance(indices[0], slice) \
            and isinstance(indices[1], slice) \
            and not isinstance(indices[2], slice)

    def __getitem__(self: "Diri3D",
                    indices: Union[Sequence[Indices], Indices]) \
            -> Union[np.ndarray, bool]:

        if isinstance(indices, (list, np.ndarray)) and len(indices) == 0:
            return np.ndarray((0, self.cols, self.rows), dtype=bool)

        idxs, cols, rows = self.parse_indices(indices)

        if len(idxs) == 1 and not self.is_grid(indices):
            index = idxs[0]
            register = self.register_at(index)
            if len(cols) == self.cols and len(rows) == self.rows:
                retval = register.data
            elif len(rows) == 1:
                retval = register.data[cols, rows[0]]
            else:
                grid = np.ix_(cols, rows)
                retval = register.data[grid]
            if isinstance(indices, (list, np.ndarray)):
                retval = retval[None]
        else:
            shape = (len(idxs), len(cols), len(rows))
            data = np.ndarray(shape=shape, dtype=bool)
            grid = np.ix_(cols, rows)
            for enumeration, index in enumerate(idxs):
                register = self.register_at(index)
                data[enumeration] = register.data[grid]
            retval = data

        if self.should_transpose_for_numpy(indices):
            retval = retval.T

        return retval

    def __setitem__(self: "Diri3D",
                    indices: Union[Sequence[Indices], Indices],
                    value: Union[np.ndarray, bool]) -> None:
        idxs, cols, rows = self.parse_indices(indices)
        if len(rows) == 1:  # Special case that breaks with a grid
            for index in idxs:
                register = self.register_at(index)
                register[cols, rows] = value
        else:
            grid = np.ix_(cols, rows)
            if self.should_transpose_for_numpy(indices) \
               and isinstance(value, np.ndarray):  # special numpy case (why?)
                value = value.T
            if np.ndim(value) == 3:
                if value.shape[0] != len(idxs):
                    raise ValueError(
                        f"Shape mismatch: Cannot assign a value of shape "
                        f"{value.shape} to a region of shape ({len(idxs)}, "
                        f"{len(cols)}, {len(rows)})")
                values = value
            else:
                values = [value] * len(idxs)
            for enumeration, index in enumerate(idxs):
                register = self.register_at(index)
                register[grid] = values[enumeration]


class DiriHB(Diri3D):
    vrs: Sequence[DiriVR]
    rl: DiriRL

    def __init__(self: "DiriHB") -> None:
        super().__init__(NVR, NUM_PLATS_PER_APUC, NSECTIONS)
        self.vrs = [DiriVR(row_number) for row_number in range(NSB)]
        self.rl = DiriRL()

    def any(self: "DiriHB") -> bool:
        for vr in self.vrs:
            if vr.any():
                return True
        return self.rl.any()

    def all(self: "DiriHB") -> bool:
        for vr in self.vrs:
            if not vr.all():
                return False
        return self.rl.all()

    def register_at(self: "DiriHB", row_number: int) -> DiriRegister:
        if 0 <= row_number < NSB:
            return self.vrs[row_number]
        if row_number == NSB:
            return self.rl
        raise KeyError(
            f"Expected row_number={row_number} to be in the range [0, {NVR})")

    def __eq__(self: "DiriHB", other: Any) -> bool:
        if not isinstance(other, DiriHB):
            return False
        for row_number in range(NSB):
            if self.vrs[row_number] != other.vrs[row_number]:
                return False
        return self.rl == other.rl

    def subscribe_to_vr(self: "DiriHB",
                        row_number: int,
                        subscriber: Subscriber) -> None:
        vr = self.vrs[row_number]
        vr.subscribe(subscriber)

    def subscribe_to_vrs(self: "DiriHB", subscriber: Subscriber) -> None:
        for row_number in range(NSB):
            self.subscribe_to_vr(row_number, subscriber)

    def subscribe_to_rl(self: "DiriHB", subscriber: Subscriber) -> None:
        self.rl.subscribe(subscriber)

    def subscribe(self: "DiriHB", subscriber: Subscriber) -> None:
        self.subscribe_to_vrs(subscriber)
        self.subscribe_to_rl(subscriber)


class DiriGL(DiriRegister):

    def __init__(self: "DiriGL") -> None:
        super().__init__(GL_SHAPE)


class DiriGGL(DiriRegister):

    def __init__(self: "DiriGGL") -> None:
        super().__init__(GGL_SHAPE)


class DiriRSP16(DiriRegister):

    def __init__(self: "DiriRSP16") -> None:
        super().__init__(RSP16_SHAPE)


class DiriRSP256(DiriRegister):

    def __init__(self: "DiriRSP256") -> None:
        super().__init__(RSP256_SHAPE)


class DiriRSP2K(DiriRegister):

    def __init__(self: "DiriRSP2K") -> None:
        super().__init__(RSP2K_SHAPE)


class DiriRSP32K(DiriRegister):

    def __init__(self: "DiriRSP32K") -> None:
        super().__init__(RSP32K_SHAPE)


class DiriUsableL1Addr(DiriRegister):
    l1_addr: int

    def __init__(self: "DiriUsableL1Addr", l1_addr: int) -> None:
        super().__init__((NUM_PLATS_PER_APUC, NGGL_ROWS))
        self.l1_addr = l1_addr

    def on_2d_update(self: "DiriUsableL1Addr",
                     indices: Union[Sequence[Indices], Indices],
                     value: Union[np.ndarray, bool]) -> None:
        plats, sections = self.parse_2d_indices(indices)
        if plats is not None and len(plats) == self.shape[0]:
            plats = None
        if sections is not None and len(sections) == self.shape[1]:
            sections = None
        self.subject.on_next(
            ("diri::l1", self.l1_addr, plats, sections, value))


class DiriRandomL1Addr:
    l1_addr: int

    def __init__(self: "DiriRandomL1Addr", l1_addr: int) -> None:
        self.l1_addr = l1_addr

    @property
    def shape(self: "DiriRandomL1Addr") -> Sequence[int]:
        return (NUM_PLATS_PER_APUC, NGGL_ROWS)

    @property
    def ndim(self: "DiriRandomL1Addr") -> int:
        return 2

    def __len__(self: "DiriRandomL1Addr") -> int:
        return NUM_PLATS_PER_APUC

    def __getitem__(self: "DiriRandomL1Addr",
                    indices: Union[Sequence[Indices], Indices]) -> np.ndarray:
        return np.random.choice(a=[False, True], size=(NUM_PLATS_PER_APUC, NGGL_ROWS))

    def __setitem__(self: "DiriRandomL1Addr",
                    indices: Union[Sequence[Indices], Indices],
                    value: np.ndarray) -> None:
        raise RuntimeError(
            f"Cannot assign value to invalid l1_addr={self.l1_addr}")

    def __eq__(self: "DiriRandomL1Addr", other: Any) -> bool:
        if not isinstance(other, DiriRandomL1Addr):
            return False
        return self.l1_addr == other.l1_addr

    def __ne__(self: "DiriRandomL1Addr", other: Any) -> bool:
        return not self.__eq__(other)

    def any(self: "DiriRandomL1Addr") -> bool:
        return False

    def all(self: "DiriRandomL1Addr") -> bool:
        return False

    def subscribe(self: "DiriRandomL1Addr", subscriber: Subscriber) -> None:
        raise RuntimeError(
            f"Cannot subscribe to invalid L1 address: {self.l1_addr}")


class DiriL1(Diri3D):
    l1_regs: Sequence[Union[DiriUsableL1Addr, DiriRandomL1Addr]]

    def __init__(self: "DiriL1") -> None:
        super().__init__(NUM_L1_ROWS, NUM_PLATS_PER_APUC, NGGL_ROWS)
        self.l1_regs = []
        for _ in range(NUM_VM_REGS // 2):
            self._add_usable_batch()
            self._add_random_batch()

    def _add_usable_batch(self: "DiriL1") -> None:
        for _ in range(9):
            l1_addr = len(self.l1_regs)
            self.l1_regs.append(DiriUsableL1Addr(l1_addr))

    def _add_random_batch(self: "DiriL1") -> None:
        for _ in range(7):
            l1_addr = len(self.l1_regs)
            self.l1_regs.append(DiriRandomL1Addr(l1_addr))

    def any(self: "DiriL1") -> bool:
        for vmr in range(0, NUM_VM_REGS, 2):
            l1_addr = vmr_to_row(vmr)
            for offset in range(9):
                l1_reg = self.l1_regs[l1_addr + offset]
                if l1_reg.any():
                    return True
        return False

    def all(self: "DiriL1") -> bool:
        for vmr in range(0, NUM_VM_REGS, 2):
            l1_addr = vmr_to_row(vmr)
            for offset in range(9):
                l1_reg = self.l1_regs[l1_addr + offset]
                if not l1_reg.all():
                    return False
        return True

    def register_at(self: "DiriL1", l1_addr: int) \
            -> Union[DiriUsableL1Addr, DiriRandomL1Addr]:
        return self.l1_regs[l1_addr]

    def __eq__(self: "DiriL1", other: Any) -> bool:
        if not isinstance(other, DiriL1):
            return False
        for l1_addr in range(NUM_L1_ROWS):
            if self.l1_regs[l1_addr] != other.l1_regs[l1_addr]:
                return False
        return True

    def subscribe_to_l1_addr(self: "DiriL1",
                             l1_addr: int,
                             subscriber: Subscriber) -> None:
        l1_reg = self.l1_regs[l1_addr]
        l1_reg.subscribe(subscriber)

    def subscribe(self: "DiriL1", subscriber: Subscriber) -> None:
        for l1_addr in np.where(VALID_L1_ADDRESSES)[0]:
            self.subscribe_to_l1_addr(l1_addr, subscriber)


class DiriL2Addr(DiriRegister):
    l2_addr: int

    def __init__(self: "DiriL2Addr", l2_addr: int) -> None:
        super().__init__((NUM_LGL_PLATS,))
        self.l2_addr = l2_addr

    def on_1d_update(self: "DiriL2Addr",
                     indices: Union[Sequence[Indices], Indices],
                     value: Union[np.ndarray, bool]) -> None:
        plats = self.parse_1d_indices(indices)
        if plats is not None and len(plats) == self.shape[0]:
            plats = None
        self.subject.on_next(("diri::l2", self.l2_addr, plats, value))


class DiriL2:
    l2_regs: Sequence[DiriL2Addr]

    def __init__(self: "DiriL2") -> None:
        self.l2_regs = [DiriL2Addr(l2_addr)
                        for l2_addr in range(NUM_L2_ROWS)]

    @property
    def shape(self: "DiriL2") -> Sequence[int]:
        return (NUM_L2_ROWS, NUM_LGL_PLATS)

    @property
    def ndim(self: "DiriL2") -> int:
        return 2

    def __len__(self: "DiriL2") -> int:
        return NUM_L2_ROWS

    def __eq__(self: "DiriL2", other: Any) -> bool:
        if not isinstance(other, DiriL2):
            return False
        for l2_addr in range(NUM_L2_ROWS):
            if self.l2_regs[l2_addr] != other.l2_regs[l2_addr]:
                return False
        return True

    def __ne__(self: "DiriL2", other: Any) -> bool:
        return not self.__eq__(other)

    def any(self: "DiriL2") -> bool:
        for l2_reg in self.l2_regs:
            if l2_reg.any():
                return True
        return False

    def all(self: "DiriL2") -> bool:
        for l2_reeg in self.l2_regs:
            if not l2_reg.all():
                return False
        return True

    def parse_indices(self: "DiriL2",
                      indices: Union[Sequence[Indices], Indices]) \
            -> Tuple[Sequence[int],
                     Sequence[int]]:

        if isinstance(indices, tuple):
            l2_addrs = parse_diri_indices(indices[0], NUM_L2_ROWS)

            if len(indices) == 1:
                plats = list(range(NUM_LGL_PLATS))

            elif len(indices) == 2:
                plats = parse_diri_indices(indices[1], NUM_LGL_PLATS)

            else:
                raise ValueError(
                    f"Too many indices ({len(indices)}): {indices}")

        else:
            l2_addrs = parse_diri_indices(indices, NUM_L2_ROWS)
            plats = list(range(NUM_LGL_PLATS))

        return l2_addrs, plats

    def __getitem__(self: "DiriL2",
                    indices: Union[Sequence[Indices], Indices]) \
            -> Union[np.ndarray, bool]:

        l2_addrs, plats = self.parse_indices(indices)

        if len(l2_addrs) == 1:
            l2_addr = l2_addrs[0]
            register = self.l2_regs[l2_addr]
            if len(plats) == NUM_LGL_PLATS:
                return register.data
            return register.data[plats]

        shape = (len(l2_addrs), len(plats))
        data = np.ndarray(shape=shape, dtype=bool)
        for index, l2_addr in enumerate(l2_addrs):
            register = self.l2_regs[l2_addr]
            data[index] = register.data[plats]
        return data

    def __setitem__(self: "DiriL2",
                    indices: Union[Sequence[Indices], Indices],
                    value: Union[np.ndarray, bool]) -> None:
        l2_addrs, plats = self.parse_indices(indices)
        if np.ndim(value) == 2:
            if value.shape[0] != len(l2_addrs):
                raise ValueError(
                    f"Shape mismatch: Cannot assign a value of shape "
                    f"{value.shape} to a region of shape ({len(l2_addrs)}, "
                    f"{len(plats)})")
            values = value
        else:
            values = [value] * len(l2_addrs)
        for enumeration, l2_addr in enumerate(l2_addrs):
            register = self.l2_regs[l2_addr]
            register[plats] = values[enumeration]

    def subscribe_to_l2_addr(self: "DiriL2",
                             l2_addr: int,
                             subscriber: Subscriber) -> None:
        l2_reg = self.l2_regs[l2_addr]
        l2_reg.subscribe(subscriber)

    def subscribe(self: "DiriL2", subscriber: Subscriber) -> None:
        for l2_addr in range(NUM_L2_ROWS):
            self.subscribe_to_l2_addr(l2_addr, subscriber)


class DiriLGL(DiriRegister):

    def __init__(self: "DiriLGL") -> None:
        super().__init__((NUM_LGL_PLATS,))


@contextual(lazy_init=True)
@dataclass
class DIRI(HalfBank, metaclass=ABCMeta):
    in_place: bool = True

    apuc_rsp_fifo: ApucRspFifo = field(default_factory=ApucRspFifo)
    rsp_mode: RspMode = RspMode.IDLE

    hb: DiriHB = field(default_factory=DiriHB)
    GL: DiriGL = field(default_factory=DiriGL)
    GGL: DiriGGL = field(default_factory=DiriGGL)
    RSP16: DiriRSP16 = field(default_factory=DiriRSP16)
    RSP256: DiriRSP256 = field(default_factory=DiriRSP256)
    RSP2K: DiriRSP2K = field(default_factory=DiriRSP2K)
    RSP32K: DiriRSP32K = field(default_factory=DiriRSP32K)
    L1: DiriL1 = field(default_factory=DiriL1)
    L2: DiriL2 = field(default_factory=DiriL2)
    LGL: DiriLGL = field(default_factory=DiriLGL)

    rwinh_filter: DiriRWInhFilter = field(default_factory=DiriRWInhFilter)
    rwinh_sects: Subset = field(default_factory=lambda: Mask("0x0000"))

    def rwinh(self: "DIRI",
              data: np.ndarray,
              section: int,
              wordline: np.ndarray) -> np.ndarray:
        if len(self.rwinh_sects) > 0:
            rwinh_filter = self.rwinh_filter.data
            filtered = rwinh_filter[::, section] & wordline
            unfiltered = ~rwinh_filter[::, section] & data[::, section]
            wordline = filtered | unfiltered
        return wordline

    @property
    def SB(self: "DIRI") -> SBAccessor:
        return SBAccessor(self.hb)

    @property
    def vmr(self: "DIRI") -> VMRAccessor:
        return VMRAccessor(self.L1)

    def __post_init__(self: "DIRI", in_place: bool = True) -> None:
        HalfBank.__init__(self)

    # def subscribe_to_apc_rsp_fifo(self: "DIRI",
    #                               apc_id: int,
    #                               subscriber: Subscriber) -> None:
    #     self.apuc_rsp_fifo.subscribe_to_apc_rsp_fifo(apc_id, subscriber)

    # def subscribe_to_apuc_rsp_fifo(self: "DIRI",
    #                                subscriber: Subscriber) -> None:
    #     self.apuc_rsp_fifo.subscribe(subscriber)

    def subscribe_to_vr(self: "DIRI",
                        row_number: int,
                        subscriber: Subscriber) -> None:
        self.hb.subscribe_to_vr(row_number, subscriber)

    def subscribe_to_vrs(self: "DIRI", subscriber: Subscriber) -> None:
        self.hb.subscribe_to_vrs(subscriber)

    def subscribe_to_rl(self: "DIRI", subscriber: Subscriber) -> None:
        self.hb.subscriber_to_rl(subscriber)

    def subscribe_to_hb(self: "DIRI", subscriber: Subscriber) -> None:
        self.hb.subscribe(subscriber)

    def subscribe_to_gl(self: "DIRI", subscriber: Subscriber) -> None:
        self.GL.subscribe(subscriber)

    def subscribe_to_ggl(self: "DIRI", subscriber: Subscriber) -> None:
        self.GGL.subscribe(subscriber)

    def subscribe_to_rsp16(self: "DIRI", subscriber: Subscriber) -> None:
        self.RSP16.subscribe(subscriber)

    def subscribe_to_rsp256(self: "DIRI", subscriber: Subscriber) -> None:
        self.RSP256.subscribe(subscriber)

    def subscribe_to_rsp2k(self: "DIRI", subscriber: Subscriber) -> None:
        self.RSP2K.subscribe(subscriber)

    def subscribe_to_rsp32k(self: "DIRI", subscriber: Subscriber) -> None:
        self.RSP32K.subscribe(subscriber)

    def subscribe_to_l1_addr(self: "DIRI",
                             l1_addr: int,
                             subscriber: Subscriber) -> None:
        self.L1.subscribe_to_l1_addr(l1_addr, subscriber)

    def subscribe_to_l1(self: "DIRI", subscriber: Subscriber) -> None:
        self.L1.subscribe(subscriber)

    def subscribe_to_l2_addr(self: "DIRI",
                             l2_addr: int,
                             subscriber: Subscriber) -> None:
        self.L2.subscribe_to_l2_addr(l2_addr, subscriber)

    def subscribe_to_l2(self: "DIRI", subscriber: Subscriber) -> None:
        self.L2.subscribe(subscriber)

    def subscribe_to_lgl(self: "DIRI", subscriber: Subscriber) -> None:
        self.LGL.subscribe(subscriber)

    def subscribe_to_rwinh_filter(self: "DIRI",
                                  subscriber: Subscriber) -> None:
        self.rwinh_filter.subscribe(subscriber)

    def subscribe(self: "DIRI", subscriber: Subscriber) -> None:
        # NOTE: Subscribing to apuc_rsp_fifo should be done separately, like
        # subscribing to the seu_layer:
        # ------------------------------------------------------------------
        # self.subscribe_to_apuc_rsp_fifo(subscriber)
        self.subscribe_to_hb(subscriber)
        self.subscribe_to_gl(subscriber)
        self.subscribe_to_ggl(subscriber)
        self.subscribe_to_rsp16(subscriber)
        self.subscribe_to_rsp256(subscriber)
        self.subscribe_to_rsp2k(subscriber)
        self.subscribe_to_rsp32k(subscriber)
        self.subscribe_to_l1(subscriber)
        self.subscribe_to_l2(subscriber)
        self.subscribe_to_lgl(subscriber)
        self.subscribe_to_rwinh_filter(subscriber)

    @staticmethod
    def parse_indices(indices: HalfBankIndices) \
            -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:

        if isinstance(indices, tuple) and len(indices) == 4:
            half_banks, row_numbers, plats, sections = indices
            half_banks = index_to_list(half_banks)
            row_numbers = index_to_list(row_numbers)
            plats = index_to_list(plats)
            sections = index_to_list(sections)
            half_banks = parse_indices(half_banks, NUM_HALF_BANKS_PER_APUC)
            row_numbers = parse_indices(row_numbers, NSB)
            plats = parse_indices(plats, NUM_PLATS_PER_HALF_BANK)
            offsets = [NUM_PLATS_PER_HALF_BANK * half_bank
                       for half_bank in half_banks]
            plats = [offset + plat
                     for offset in offsets
                     for plat in plats]
            sections = parse_indices(sections, NSECTIONS)

        elif isinstance(indices, tuple) and len(indices) == 3:
            row_numbers, plats, sections = indices
            row_numbers = index_to_list(row_numbers)
            plats = index_to_list(plats)
            sections = index_to_list(sections)
            row_numbers = parse_indices(row_numbers, NSB)
            plats = parse_indices(plats, NUM_PLATS_PER_APUC)
            sections = parse_indices(sections, NSECTIONS)

        elif isinstance(indices, tuple) and len(indices) == 2:
            row_numbers, plats = indices
            row_numbers = index_to_list(row_numbers)
            plats = index_to_list(plats)
            row_numbers = parse_indices(row_numbers, NSB)
            plats = parse_indices(plats, NUM_PLATS_PER_APUC)
            sections = list(range(NSECTIONS))

        else:
            row_numbers = index_to_list(indices)
            row_numbers = parse_indices(row_numbers, NSB)
            plats = list(range(NUM_PLATS_PER_HALF_BANK))
            sections = list(range(NSECTIONS))

        grid = np.ix_(row_numbers, plats, sections)
        return grid

    def __getitem__(self: "DIRI", indices: HalfBankIndices) -> VR:
        grid = self.parse_indices(indices)
        vr = self.hb[grid]
        if vr.shape[0] == 1:
            return np.squeeze(vr)
        return vr

    def __setitem__(self: "DIRI",
                    indices: HalfBankIndices,
                    value: Union[np.ndarray, bool, str, List[str]]) -> None:
        value = glass_to_vr(value)
        grid = self.parse_indices(indices)
        self.hb[grid] = value

    def __eq__(self: "DIRI", other: Any) -> bool:
        return isinstance(other, DIRI) \
            and (self.hb == other.hb) \
            and (self.GL == other.GL) \
            and (self.GGL == other.GGL) \
            and (self.RSP16 == other.RSP16) \
            and (self.RSP256 == other.RSP256) \
            and (self.RSP2K == other.RSP2K) \
            and (self.RSP32K == other.RSP32K) \
            and (self.L1 == other.L1) \
            and (self.L2 == other.L2) \
            and (self.LGL == other.LGL) \
            and (self.rwinh_filter == other.rwinh_filter)

    def __ne__(self: "DIRI", other: Any) -> bool:
        return not self.__eq__(other)

    #   ___                      _
    #  / __|___ _ ___ _____ _ _ (_)___ _ _  __ ___
    # | (__/ _ \ ' \ V / -_) ' \| / -_) ' \/ _/ -_)
    #  \___\___/_||_\_/\___|_||_|_\___|_||_\__\___|

    def sbs_to_vrs(self: "DIRI", sbs_and_vrs: SBs_and_VRs) -> VRs:
        if not isinstance(sbs_and_vrs, (list, tuple)):
            sb_or_vr = sbs_and_vrs
            sbs_and_vrs = [sb_or_vr]

        vrs = []
        for sb_or_vr in sbs_and_vrs:
            if isinstance(sb_or_vr, Integer.__args__):
                sb = sb_or_vr
                vr = self.hb.vrs[sb]
            else:
                vr = sb_or_vr
            vrs.append(vr)

        return vrs

    @patch_with(patch_sb)
    def reset_rl_section(self: "DIRI", section: int) -> SectionMap:
        section_map = []
        wordline = np.zeros(NUM_PLATS_PER_APUC, dtype=bool)
        section_map.append((self.hb.rl, section, wordline))
        return section_map

    @patch_with(patch_sb)
    def reset_sb_section(self: "DIRI", section: int, sb: int) -> SectionMap:
        section_map = []
        wordline = np.zeros(NUM_PLATS_PER_APUC, dtype=bool)
        section_map.append((self.hb[sb], section, wordline))
        return section_map

    def build_gl(self: "DIRI") -> np.ndarray:
        return np.zeros(GL_SHAPE, dtype=bool)

    @patch_with(patch_gl)
    def reset_gl(self: "DIRI") -> np.ndarray:
        gl = self.build_gl()
        return gl

    def build_ggl(self: "DIRI") -> np.ndarray:
        return np.zeros(GGL_SHAPE, dtype=bool)

    @patch_with(patch_ggl)
    def reset_ggl(self: "DIRI") -> np.ndarray:
        ggl = self.build_ggl()
        return ggl

    def build_rl(self: "DIRI") -> np.ndarray:
        return np.zeros(RL_SHAPE, dtype=bool)

    @patch_with(patch_whole_vr)
    def reset_rl(self: "DIRI") -> np.ndarray:
        rl = self.build_rl()
        return self.hb.rl, rl

    def build_vr(self: "DIRI") -> np.ndarray:
        return np.zeros(VR_SHAPE, dtype=bool)

    @patch_with(patch_whole_vr)
    def reset_sb(self: "DIRI", sb: int) -> np.ndarray:
        vr = self.build_vr()
        return self.hb[sb], vr

    def build_rsp16(self: "DIRI") -> np.ndarray:
        return np.zeros(RSP16_SHAPE, dtype=bool)

    @patch_with(patch_rsp16)
    def reset_rsp16(self: "DIRI") -> np.ndarray:
        rsp16 = self.build_rsp16()
        return rsp16

    def build_rsp256(self: "DIRI") -> np.ndarray:
        return np.zeros(RSP256_SHAPE, dtype=bool)

    @patch_with(patch_rsp256)
    def reset_rsp256(self: "DIRI") -> np.ndarray:
        rsp256 = self.build_rsp256()
        return rsp256

    def build_rsp2k(self: "DIRI") -> np.ndarray:
        return np.zeros(RSP2K_SHAPE, dtype=bool)

    @patch_with(patch_rsp2k)
    def reset_rsp2k(self: "DIRI") -> np.ndarray:
        rsp2k = self.build_rsp2k()
        return rsp2k

    def build_rsp32k(self: "DIRI") -> np.ndarray:
        return np.zeros(RSP32K_SHAPE, dtype=bool)

    @patch_with(patch_rsp32k)
    def reset_rsp32k(self: "DIRI") -> np.ndarray:
        rsp32k = self.build_rsp32k()
        return rsp32k

    # Helper method, not a mimic ...
    def pull_rsps(self: "DIRI", sections: Subset) -> None:
        rsp16_patch = self.rsp16_from_rl(sections)
        if not self.in_place:
            patch_rsp16(self, rsp16_patch)

        rsp256_patch = self.rsp256_from_rsp16()
        if not self.in_place:
            patch_rsp256(self, rsp256_patch)

        rsp2k_patch = self.rsp2k_from_rsp256()
        if not self.in_place:
            patch_rsp2k(self, rsp2k_patch)

        rsp32k_patch = self.rsp32k_from_rsp2k()
        if not self.in_place:
            patch_rsp32k(self, rsp32k_patch)

    @patch_with(patch_whole_l1)
    def reset_l1(self: "DIRI") -> np.ndarray:
        L1 = np.zeros((NUM_L1_ROWS, NUM_PLATS_PER_APUC, NGGL_ROWS), dtype=bool)
        return L1

    @patch_with(patch_whole_l2)
    def reset_l2(self: "DIRI") -> np.ndarray:
        L2 = np.zeros((NUM_L2_ROWS, NUM_LGL_PLATS), dtype=bool)
        return L2

    @patch_with(patch_lgl)
    def reset_lgl(self: "DIRI") -> np.ndarray:
        LGL = np.zeros(NUM_LGL_PLATS, dtype=bool)
        return LGL

    #  _                   _                _   _  ___      __  ___
    # | |   _____ __ _____| |   _____ _____| | | || \ \    / / | __|_ __
    # | |__/ _ \ V  V /___| |__/ -_) V / -_) | | __ |\ \/\/ /  | _|| '  \
    # |____\___/\_/\_/    |____\___|\_/\___|_| |_||_| \_/\_/   |___|_|_|_|

    def rsp_from_contraction(self: "DIRI",
                             rsp_left: np.ndarray,
                             rsp_right: np.ndarray,
                             left_width: int,
                             right_width: int) -> None:
        step_size = right_width // left_width
        num_steps = left_width
        for step in range(num_steps):
            lower = step * step_size
            upper = lower + step_size
            rsp_left[step] = rsp_right[lower:upper].any(axis=0)

    def rsp_from_expansion(self: "DIRI",
                           rsp_left: np.ndarray,
                           rsp_right: np.ndarray,
                           left_width: int,
                           right_width: int) -> None:
        step_size = left_width // right_width
        num_steps = right_width
        for step in range(num_steps):
            lower = step * step_size
            upper = lower + step_size
            rsp_left[lower:upper] = rsp_right[step]

    @rsp_from_rsp
    def rsp16_from_rsp256(self: "DIRI") -> np.ndarray:
        """[APL] RSP16 = RSP256;"""

    @rsp_from_rsp
    def rsp256_from_rsp16(self: "DIRI") -> np.ndarray:
        """[APL] RSP256 = RSP16;"""
        self.rsp_mode = RspMode.RSP256_READ

    @rsp_from_rsp
    def rsp256_from_rsp2k(self: "DIRI") -> np.ndarray:
        """[APL] RSP256 = RSP2K;"""

    @rsp_from_rsp
    def rsp2k_from_rsp256(self: "DIRI") -> np.ndarray:
        """[APL] RSP2K = RSP256;"""
        self.rsp_mode = RspMode.RSP2K_READ

    @patch_with(patch_rsp2k)
    def rsp2k_from_rsp32k(self: "DIRI") -> np.ndarray:
        """[APL] RSP2K = RSP32K;"""
        rsp2k = self.build_rsp2k()
        rsp32k = self.RSP32K.data
        for half_bank in range(NUM_HALF_BANKS_PER_APUC):
            rsp2k[half_bank][::] = rsp32k[0][half_bank]
        return rsp2k

    @patch_with(patch_rsp32k)
    def rsp32k_from_rsp2k(self: "DIRI") -> np.ndarray:
        """[APL] RSP32K = RSP2K;"""
        # NOTE: Each section of RSP32K maps to a specific half-bank.
        self.rsp_mode = RspMode.RSP32K_READ
        rsp32k = np.logical_or.reduce(self.RSP2K, axis=1)[None]
        return rsp32k

    @patch_with(patch_noop)
    def noop(self: "DIRI") -> None:
        """[APL] NOOP;"""
        return None

    @patch_with(patch_noop)
    def fsel_noop(self: "DIRI") -> None:
        """[APL] FSEL_NOOP;"""
        return None

    @patch_with(patch_rsp_end)
    def rsp_end(self: "DIRI") -> Sequence[np.ndarray]:
        """[APL] RSP_END;"""
        return [
            (patch_rsp16, False),
            (patch_rsp256, False),
            (patch_rsp2k, False),
            (patch_rsp32k, False),
        ]

    @patch_with(patch_rsp_start_ret)
    def rsp_start_ret(self: "DIRI") -> None:
        """[APL] RSP_START_RET;"""
        if self.rsp_mode == RspMode.RSP16_READ:
            self.rsp_mode = RspMode.RSP16_WRITE
        elif self.rsp_mode == RspMode.RSP256_READ:
            self.rsp_mode = RspMode.RSP256_WRITE
        elif self.rsp_mode == RspMode.RSP2K_READ:
            self.rsp_mode = RspMode.RSP2K_WRITE
        elif self.rsp_mode == RspMode.RSP32K_READ:
            self.rsp_mode = RspMode.RSP32K_WRITE

    @patch_with(patch_l2_end)
    def l2_end(self: "DIRI") -> None:
        """[APL] L2_END;"""
        return None

    #  ___         _               _   _
    # |_ _|_ _  __| |_ _ _ _  _ __| |_(_)___ _ _  ___
    #  | || ' \(_-<  _| '_| || / _|  _| / _ \ ' \(_-<
    # |___|_||_/__/\__|_|  \_,_\__|\__|_\___/_||_/__/
    #

    # __      __   _ _         _              _
    # \ \    / / _(_) |_ ___  | |   ___  __ _(_)__
    #  \ \/\/ / '_| |  _/ -_) | |__/ _ \/ _` | / _|
    #   \_/\_/|_| |_|\__\___| |____\___/\__, |_\__|
    #                                   |___/

    def RL(self: "DIRI") -> VR:
        rl = self.hb.rl.data
        return rl

    def NRL(self: "DIRI") -> VR:
        rl = self.hb.rl.data
        nrl = np.roll(rl, 1, axis=1)
        nrl[::, 0] = False
        return nrl

    def nrl_glass(self: "DIRI") -> str:
        """Peer into NRL."""
        result = DIRI.ndarray_2d_to_bitstring(self.NRL())
        return result

    def nrl_hex_glass(self: "DIRI") -> str:
        """Peer hex-wise into NRL."""
        result = DIRI.ndarray_2d_to_hexstring(self.NRL())
        return result

    def ERL(self: "DIRI") -> VR:
        rl = self.hb.rl.data
        erl = rl.reshape((NUM_HALF_BANKS_PER_APUC, NUM_PLATS_PER_HALF_BANK, NSECTIONS))
        erl = np.roll(erl, -1, axis=1)
        erl[::, -1] = False
        erl = erl.reshape(RL_SHAPE)
        return erl

    def erl_glass(self: "DIRI") -> str:
        """Peer into ERL."""
        result = DIRI.ndarray_2d_to_bitstring(self.ERL())
        return result

    def erl_hex_glass(self: "DIRI") -> str:
        """Peer hex-wise into ERL."""
        result = DIRI.ndarray_2d_to_hexstring(self.ERL())
        return result

    def WRL(self: "DIRI") -> np.ndarray:
        rl = self.hb.rl.data
        wrl = rl.reshape((NUM_HALF_BANKS_PER_APUC, NUM_PLATS_PER_HALF_BANK, NSECTIONS))
        wrl = np.roll(wrl, 1, axis=1)
        wrl[::, 0] = False
        wrl = wrl.reshape(RL_SHAPE)
        return wrl

    def wrl_glass(self: "DIRI") -> str:
        """Peer into WRL."""
        result = DIRI.ndarray_2d_to_bitstring(self.WRL())
        return result

    def wrl_hex_glass(self: "DIRI") -> str:
        """Peer hex-wise into WRL."""
        result = DIRI.ndarray_2d_to_hexstring(self.WRL())
        return result

    def SRL(self: "DIRI") -> np.ndarray:
        rl = self.hb.rl.data
        srl = np.roll(rl, -1, axis=1)
        srl[::, -1] = False
        return srl

    def srl_glass(self: "DIRI") -> str:
        """Peer into SRL."""
        result = DIRI.ndarray_2d_to_bitstring(self.SRL())
        return result

    def srl_hex_glass(self: "DIRI") -> str:
        """Peer hex-wise into SRL."""
        result = DIRI.ndarray_2d_to_hexstring(self.SRL())
        return result

    def gl_glass(self: "DIRI") -> str:
        """Peer into GL."""
        if self.GL is None:
            raise RuntimeError("Attempt to peer into an empty glass of GL.")
        result = DIRI.ndarray_1d_to_bitstring(self.GL)
        return result

    def gl_hex_glass(self: "DIRI") -> str:
        """Peer hex-wise into GL."""
        if self.GL is None:
            raise RuntimeError("Attempt to peer into an empty glass of GL.")
        result = DIRI.ndarray_1d_to_hexstring(self.GL)
        return result

    def ggl_glass(self: "DIRI") -> str:
        """Peer into GGL."""
        if self.GGL is None:
            raise RuntimeError("Attempt to peer into an empty glass of GGL.")
        result = DIRI.ndarray_2d_to_bitstring(self.GGL)
        return result

    def ggl_hex_glass(self: "DIRI") -> str:
        """Peer hex-wise into GGL."""
        if self.GGL is None:
            raise RuntimeError("Attempt to peer into an empty glass of GGL.")
        result = DIRI.ndarray_2d_to_hexstring(self.GGL)
        return result

    @patch_with(patch_sb)
    def _sb_OPEQUALS_gl(self: "DIRI",
                        destination_section_list: Subset,
                        vrs: VRs,
                        op: BinaryOp,
                        src: np.ndarray) -> SectionMap:
        section_map = []
        src = src.data if isinstance(src, DiriRegister) else src
        for vr in vrs:
            data = vr.data if isinstance(vr, DiriVR) else vr
            for s in destination_section_list:
                wordline = self.rwinh(data, s, op(data[::, s], src))
                section_map.append((vr, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _sb_OPEQUALS_ggl(self: "DIRI",
                         destination_section_list: Subset,
                         vrs: VRs,
                         op: BinaryOp,
                         src: np.ndarray) -> SectionMap:
        section_map = []
        src = src.data if isinstance(src, DiriRegister) else src
        for vr in vrs:
            data = vr.data if isinstance(vr, DiriVR) else vr
            for s in destination_section_list:
                pseudo_GL = src[::, s // NGGL_ROWS]
                wordline = self.rwinh(data, s, op(data[::, s], pseudo_GL))
                section_map.append((vr, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _sb_OPEQUALS_rl(self: "DIRI",
                        destination_section_list: Subset,
                        vrs: VRs,
                        op: BinaryOp,
                        src: np.ndarray) -> SectionMap:
        section_map = []
        src = src.data if isinstance(src, DiriRegister) else src
        for vr in vrs:
            data = vr.data if isinstance(vr, DiriVR) else vr
            for s in destination_section_list:
                wordline = self.rwinh(data, s, op(data[::, s], src[::, s]))
                section_map.append((vr, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _sb_OPEQUALS_rsp16(self: "DIRI",
                           destination_section_list: Subset,
                           vrs: VRs,
                           op: BinaryOp,
                           src: np.ndarray) -> SectionMap:

        # IMPORTANT: It seems RSP16 is the only <SRC> affected by RWINH.

        # Needed for RWINH filter
        brsp16 = self.build_vr()
        src = src.data if isinstance(src, DiriRegister) else src
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            brsp16[start:stop] = src[index]

        section_map = []
        for vr in vrs:
            data = vr.data if isinstance(vr, DiriVR) else vr
            for s in destination_section_list:
                wordline = self.rwinh(data, s, op(data[::, s], brsp16[::, s]))
                section_map.append((vr, s, wordline))

        return section_map

    def _sb_OPEQUALS_src(self: "DIRI",
                         sections: Subset,
                         vrs: VRs,
                         op: BinaryOp,
                         src: np.ndarray) -> SectionMap:

        # assert 1 <= len(vrs) <= 8

        destination_section_mask = Mask(sections)
        destination_section_list = destination_section_mask.list

        if src.shape == GL_SHAPE:
            return self._sb_OPEQUALS_gl(destination_section_list, vrs, op, src)

        if src.shape == GGL_SHAPE:
            return self._sb_OPEQUALS_ggl(destination_section_list, vrs, op, src)

        if src.shape == RL_SHAPE:
            return self._sb_OPEQUALS_rl(destination_section_list, vrs, op, src)

        if src.shape == RSP16_SHAPE:
            return self._sb_OPEQUALS_rsp16(destination_section_list, vrs, op, src)

        raise ValueError(f"unknown shape {src.shape} for SRC")

    def sb_from_src(self: "DIRI",
                    sections: Subset,
                    sbs_and_vrs: SBs_and_VRs,
                    src: np.ndarray) -> SectionMap:
        """[APL] msk: <SB> = <SRC>;
        src can be [INV_][NEWS]RL or [INV_]GL [INV_]GGL or [INV_]RSP16."""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._sb_OPEQUALS_src(sections, vrs, exp.right, src)

    def sb_from_inv_src(self: "DIRI",
                        sections: Subset,
                        sbs_and_vrs: SBs_and_VRs,
                        src: np.ndarray) -> SectionMap:
        """[APL] msk: <SB> = ~<SRC>;
        src can be [INV_][NEWS]RL or [INV_]GL [INV_]GGL or [INV_]RSP16."""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._sb_OPEQUALS_src(sections, vrs, exp.INV_right, src)

    def sb_cond_equals_src(self: "DIRI",
                           sections: Subset,
                           sbs_and_vrs: SBs_and_VRs,
                           src: np.ndarray) -> SectionMap:
        """[APL] msk: <SB> ?= <SRC>;
        src can be [INV_][NEWS]RL or [INV_]GL [INV_]GGL or [INV_]RSP16."""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._sb_OPEQUALS_src(sections, vrs, exp.left_OR_right, src)

    def sb_cond_equals_inv_src(self: "DIRI",
                               sections: Subset,
                               sbs_and_vrs: SBs_and_VRs,
                               src: np.ndarray) -> SectionMap:
        """[APL] msk: <SB> ?= ~<SRC>;
        src can be [INV_][NEWS]RL or [INV_]GL [INV_]GGL or [INV_]RSP16."""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._sb_OPEQUALS_src(sections, vrs, exp.left_AND_INV_right, src)


    #  ___             _   _              _
    # | _ \___ __ _ __| | | |   ___  __ _(_)__
    # |   / -_) _` / _` | | |__/ _ \/ _` | / _|
    # |_|_\___\__,_\__,_| |____\___/\__, |_\__|
    #                               |___/

    #   ___    ___
    #  <  /   |_  |
    #  / /   / __/
    # /_( ) /____/
    #   |/

    # Read-logic Instructions 1 and 2

    @patch_with(patch_sb)
    def set_rl(self: "DIRI",
               mask: Subset,
               bit: int) -> SectionMap:
        """Read-logic Instructions 1 and 2
           [APL  1] msk: RL = 1
           [APL  2] msk: RL = 0
        Set all columns of RL to a given bit through a mask"""
        # TODO: Consider optimizing common case of "FFFF".
        ss = Mask(mask)
        section_map = []
        value = bool(bit)
        rl = self.hb.rl
        data = rl.data
        for s in ss.list:
            wordline = self.rwinh(data, s, value)
            section_map.append((rl, s, wordline))
        return section_map

    #    __ __    __               ___           ____     ___  ___
    #   / // /__ / /__  ___ ____  / _/__  ____  |_  /    |_  |/ _ \
    #  / _  / -_) / _ \/ -_) __/ / _/ _ \/ __/ _/_ <_ _ / __// // /
    # /_//_/\__/_/ .__/\__/_/   /_/ \___/_/   /____(_|_)____/\___/
    #           /_/

    # Helper for Instructions 3 through 20

    def _get_sb_rows(self: "DIRI", vrs: VRs, _, section: int) -> np.ndarray:
        conj = np.ones(SECTION_SHAPE, dtype=bool)
        for vr in vrs:
            data = vr.data if isinstance(vr, DiriVR) else vr
            conj &= data[::, section]
        return conj

    def ternary_expr(self: "DIRI",
                     rl: VR,
                     vr: VR,
                     src: VR,
                     op: BinaryOp,
                     op2: BinaryOp) -> np.ndarray:
        rhs = op2(vr, src)
        return op(rl, rhs)

    #   ____     ______    _______     _______    ______
    #  / / /    <  <  /   <  / / /    <  /_  /   <  / _ \
    # /_  _/    / // /    / /_  _/    / / / /    / /\_, /
    #  /_/( )  /_//_( )  /_/ /_/( )  /_/ /_( )  /_//___/
    #     |/        |/          |/         |/

    # Read-logic instructions 4, 11, 14, 17, 19
    #   |  4.  RL  =  <SRC>           | :=               <SRC>  |
    #   | 11.  RL |=  <SRC>           | |=               <SRC>  |
    #   | 14.  RL &=  <SRC>           | &=               <SRC>  |
    #   | 17.  RL &= ~<SRC>           | &=              ~<SRC>  |
    #   | 19.  RL ^=  <SRC>           | ^=               <SRC>  |

    @patch_with(patch_sb)
    def _rl_OPEQUALS_gl(self: "DIRI",
                        dest_list: Subset,
                        op: BinaryOp,
                        src: VR) -> SectionMap:
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_list:
            wordline = self.rwinh(data, s, op(data[::, s], src))
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_OPEQUALS_ggl(self: "DIRI",
                         dest_list: Subset,
                         op: BinaryOp,
                         src: VR) -> SectionMap:
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_list:
            pseudo_GL = src[::, s // NGGL_ROWS]
            wordline = self.rwinh(data, s, op(data[::, s], pseudo_GL))
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_OPEQUALS_rl(self: "DIRI",
                        dest_list: Subset,
                        op: BinaryOp,
                        src: VR) -> SectionMap:
        # could be NRL, ERL, WRL, SRL, RL, or their inverses
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_list:
            wordline = self.rwinh(data, s, op(data[::, s], src[::, s]))
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_OPEQUALS_rsp16(self: "DIRI",
                           dest_list: Subset,
                           op: BinaryOp,
                           src: VR) -> SectionMap:

        # Needed for RWINH filter
        brsp16 = self.build_vr()
        src = src.data if isinstance(src, DiriRegister) else src
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            brsp16[start:stop] = src[index]

        section_map = []
        rl = self.hb.rl
        data = rl.data
        for s in dest_list:
            wordline = self.rwinh(data, s, op(data[::, s], brsp16[::, s]))
            section_map.append((rl, s, wordline))
        return section_map

    def _rl_OPEQUALS_src(self: "DIRI",
                         sections: Subset,
                         op: BinaryOp,
                         src: VR) -> SectionMap:
        """Abstract instructions 4, 11, 14, 17, 19.
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""

        dest_mask = Mask(sections)
        dest_list = dest_mask.list

        # heuristic: reading from GL, GGL, RL, or RSP16

        if src.shape == GL_SHAPE:
            return self._rl_OPEQUALS_gl(dest_list, op, src)

        if src.shape == GGL_SHAPE:
            return self._rl_OPEQUALS_ggl(dest_list, op, src)

        if src.shape == RL_SHAPE:
            return self._rl_OPEQUALS_rl(dest_list, op, src)

        if src.shape == RSP16_SHAPE:
            return self._rl_OPEQUALS_rsp16(dest_list, op, src)

        raise ValueError(f"unknown shape {src.shape} for SRC")
#   ____
#  / / /
# /_  _/
#  /_/

    def rl_from_src(self: "DIRI",
                    sections: Subset,
                    src: VR) -> SectionMap:
        """Read-logic Instruction 4
           [APL  4] msk: RL = <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        return self._rl_OPEQUALS_src(sections, exp.right, src)

    def rl_from_inv_src(self: "DIRI",
                        sections: Subset,
                        src: VR) -> SectionMap:
        """Read-logic Instruction 4
           [APL  4] msk: RL = ~<SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        return self._rl_OPEQUALS_src(sections, exp.INV_right, src)
#   ______
#  <  <  /
#  / // /
# /_//_/

    def rl_or_equals_src(self: "DIRI",
                         sections: Subset,
                         src: VR) -> SectionMap:
        """Read-logic Instruction 11
           [APL 11] msk: RL |= <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        return self._rl_OPEQUALS_src(sections, exp.left_OR_right, src)

    def rl_or_equals_inv_src(self: "DIRI",
                             sections: Subset,
                             src: VR) -> SectionMap:
        """Read-logic Instruction 11
           [APL 11] msk: RL |= ~<SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        return self._rl_OPEQUALS_src(sections, exp.left_OR_INV_right, src)
#   _______
#  <  / / /
#  / /_  _/
# /_/ /_/

    def rl_and_equals_src(self: "DIRI",
                          sections: Subset,
                          src: VR) -> SectionMap:
        """Read-logic Instruction 14
           [APL 14] msk: RL &= <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        return self._rl_OPEQUALS_src(sections, exp.left_AND_right, src)
#   _______
#  <  /_  /
#  / / / /
# /_/ /_/

    def rl_and_equals_inv_src(self: "DIRI",
                              sections: Subset,
                              src: VR) -> SectionMap:
        """Read-logic Instruction 17
           [APL 17] msk: RL &= ~<SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        return self._rl_OPEQUALS_src(sections, exp.left_AND_INV_right, src)

#   ______
#  <  / _ \
#  / /\_, /
# /_//___/

    def rl_xor_equals_src(self: "DIRI",
                          sections: Subset,
                          src: VR) -> SectionMap:
        """Read-logic Instruction 19
           [APL 19] msk: RL ^= <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        return self._rl_OPEQUALS_src(sections, exp.left_XOR_right, src)

    def rl_xor_equals_inv_src(self: "DIRI",
                              sections: Subset,
                              src: VR) -> SectionMap:
        """Read-logic Instruction 19
           [APL 19] msk: RL ^= ~<SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        return self._rl_OPEQUALS_src(sections, exp.left_XOR_INV_right, src)

    #    ____     ______      _______     _______     ______
    #   |_  /    <  / _ \    <  /_  /    <  / __/    <  ( _ )
    #  _/_ <_    / / // /    / //_ <_    / / _ \_    / / _  |
    # /____( )  /_/\___( )  /_/____( )  /_/\___( )  /_/\___/
    #      |/          |/          |/          |/

    # Read-logic instructions 3, 10, 13, 16, 18
    #   |  3.  RL  =  <SB>            | :=  <SB>                |
    #   | 10.  RL |=  <SB>            | |=  <SB>                |
    #   | 13.  RL &=  <SB>            | &=  <SB>                |
    #   | 16.  RL &= ~<SB>            | &= ~<SB>                |
    #   | 18.  RL ^=  <SB>            | ^=  <SB>                |

    @patch_with(patch_sb)
    def _rl_OPEQUALS_sb(self: "DIRI",
                        sections: Subset,
                        rl_op: BinaryOp,
                        vrs: VRs) -> SectionMap:
        """Abstract instructions 3, 10, 13, 18."""
        section_map = []
        p = self.hb.shape[1]
        ss = Mask(sections)
        rl = self.hb.rl
        data = rl.data
        for s in ss:
            sbdata = self._get_sb_rows(vrs, p, s)
            wordline = self.rwinh(data, s, rl_op(data[::, s], sbdata))
            section_map.append((rl, s, wordline))
        return section_map  # for chaining

#    ____
#   |_  /
#  _/_ <
# /____/

    def rl_from_sb(self: "DIRI",
                   sections: Subset,
                   sbs_and_vrs: SBs_and_VRs) -> SectionMap:
        """Read-logic Instruction 3
           [APL  3] msk: RL = <SB>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb(sections, exp.right, vrs)

    def rl_from_inv_sb(self: "DIRI",
                       sections: Subset,
                       sbs_and_vrs: SBs_and_VRs) -> SectionMap:
        """Read-logic Instruction 3
           [APL  3] msk: RL = ~<SB>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb(sections, exp.INV_right, vrs)
#   ______
#  <  / _ \
#  / / // /
# /_/\___/

    def rl_or_equals_sb(self: "DIRI",
                        sections: Subset,
                        sbs_and_vrs: SBs_and_VRs) -> SectionMap:
        """Read-logic Instruction 10
           [APL 10] msk: RL |= <SB>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb(sections, exp.left_OR_right, vrs)
#   _______
#  <  /_  /
#  / //_ <
# /_/____/

    def rl_and_equals_sb(self: "DIRI",
                         sections: Subset,
                         sbs_and_vrs: SBs_and_VRs) -> SectionMap:
        """Read-logic Instruction 13
        [APL 13] msk: RL &= <SB>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb(sections, exp.left_AND_right, vrs)
#   _______
#  <  / __/
#  / / _ \
# /_/\___/

    def rl_and_equals_inv_sb(self: "DIRI",
                             sections: Subset,
                             sbs_and_vrs: SBs_and_VRs) -> SectionMap:
        """Read-logic Instruction 16
           [APL 16] msk: RL &= ~<SB>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb(sections, exp.left_AND_INV_right, vrs)

#   ______
#  <  ( _ )
#  / / _  |
# /_/\___/

    def rl_xor_equals_sb(self: "DIRI",
                         sections: Subset,
                         sbs_and_vrs: SBs_and_VRs) -> SectionMap:
        """Read-logic Instruction 18
           [APL 18] msk: RL ^= <SB>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb(sections, exp.left_XOR_right, vrs)

    #    ____     ______      _______     ___  ___
    #   / __/    <  /_  |    <  / __/    |_  |/ _ \
    #  /__ \_    / / __/_    / /__ \_   / __// // /
    # /____( )  /_/____( )  /_/____( ) /____/\___/
    #      |/          |/          |/

    # Read-logic instruction 5, 12, 15, 20

    #   |  5.  RL  =  <SB> &  <SRC>   | :=  <SB>    &    <SRC>  |
    #   | 12.  RL |=  <SB> &  <SRC>   | |=  <SB>    &    <SRC>  |
    #   | 15.  RL &=  <SB> &  <SRC>   | &=  <SB>    &    <SRC>  |
    #   | 20.  RL ^=  <SB> &  <SRC>   | ^=  <SB>    &    <SRC>  |

    @patch_with(patch_sb)
    def _rl_OPEQUALS_sb_AND_gl(self: "DIRI",
                               dest_mask: Subset,
                               op: BinaryOp,
                               op2: BinaryOp,
                               vrs: VRs,
                               src: np.ndarray) -> SectionMap:
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_mask:
            sbdata = self._get_sb_rows(vrs, NUM_PLATS_PER_APUC, s)
            wordline = self.ternary_expr(data[::, s], sbdata, src, op, op2)
            wordline = self.rwinh(data, s, wordline)
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_OPEQUALS_sb_AND_ggl(self: "DIRI",
                                dest_mask: Subset,
                                op: BinaryOp,
                                op2: BinaryOp,
                                vrs: VRs,
                                src: np.ndarray) -> SectionMap:
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_mask:
            sbdata = self._get_sb_rows(vrs, NUM_PLATS_PER_APUC, s)
            pseudo_GL = src[::, s // NGGL_ROWS]
            wordline = \
                self.ternary_expr(data[::, s], sbdata, pseudo_GL, op, op2)
            wordline = self.rwinh(data, s, wordline)
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_OPEQUALS_sb_AND_rl(self: "DIRI",
                               dest_mask: Subset,
                               op: BinaryOp,
                               op2: BinaryOp,
                               vrs: VRs,
                               src: np.ndarray) -> SectionMap:
        # could be NRL, ERL, WRL, SRL, RL, or their inverses
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_mask:
            sbdata = self._get_sb_rows(vrs, NUM_PLATS_PER_APUC, s)
            wordline = \
                self.ternary_expr(data[::, s], sbdata, src[::, s], op, op2)
            wordline = self.rwinh(data, s, wordline)
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_OPEQUALS_sb_AND_rsp16(self: "DIRI",
                                  dest_mask: Subset,
                                  op: BinaryOp,
                                  op2: BinaryOp,
                                  vrs: VRs,
                                  src: np.ndarray) -> SectionMap:

        # Needed for RWINH filter
        brsp16 = self.build_vr()
        src = src.data if isinstance(src, DiriRegister) else src
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            brsp16[start:stop] = src[index]

        section_map = []
        rl = self.hb.rl
        data = rl.data
        for s in dest_mask:
            sbdata = self._get_sb_rows(vrs, NUM_PLATS_PER_APUC, s)
            wordline = \
                self.ternary_expr(data[::, s], sbdata, brsp16[::, s], op, op2)
            wordline = self.rwinh(data, s, wordline)
            section_map.append((rl, s, wordline))

        return section_map

    def _rl_OPEQUALS_sb_AND_src(self: "DIRI",
                                sections: Subset,
                                op: BinaryOp,
                                op2: BinaryOp,
                                vrs: VRs,
                                src: np.ndarray) -> SectionMap:
        """Abstract instructions 5, 12, 15, 20."""

        dest_mask = Mask(sections)

        if src.shape == GL_SHAPE:
            return self._rl_OPEQUALS_sb_AND_gl(dest_mask, op, op2, vrs, src)

        if src.shape == GGL_SHAPE:
            return self._rl_OPEQUALS_sb_AND_ggl(dest_mask, op, op2, vrs, src)

        if src.shape == RL_SHAPE:
            return self._rl_OPEQUALS_sb_AND_rl(dest_mask, op, op2, vrs, src)

        if src.shape == RSP16_SHAPE:
            return self._rl_OPEQUALS_sb_AND_rsp16(dest_mask, op, op2, vrs, src)

        raise ValueError(f"unknown shape {src.shape} for SRC")

#    ____
#   / __/
#  /__ \
# /____/

    def rl_from_sb_and_src(self: "DIRI",
                           sections: Subset,
                           sbs_and_vrs: SBs_and_VRs,
                           src: VR) -> SectionMap:
        """Read-logic Instruction 5
           [APL  5] msk: RL = <SB> & <SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb_AND_src(
            sections,
            exp.right,
            exp.left_AND_right,
            vrs, src)

#   ______
#  <  /_  |
#  / / __/
# /_/____/

    def rl_or_equals_sb_and_src(self: "DIRI",
                                sections: Subset,
                                sbs_and_vrs: SBs_and_VRs,
                                src: VR) -> SectionMap:
        """Read-logic Instruction 12
           [APL 12] msk: RL |= <SB> & <SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb_AND_src(
            sections,
            exp.left_OR_right,
            exp.left_AND_right,
            vrs, src)

    def rl_or_equals_sb_and_inv_src(self: "DIRI",
                                    sections: Subset,
                                    sbs_and_vrs: SBs_and_VRs,
                                    src: VR) -> SectionMap:
        """Read-logic Instruction 12
           [APL 12] msk: RL |= <SB> & ~<SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb_AND_src(
            sections,
            exp.left_OR_right,
            exp.left_AND_INV_right,
            vrs, src)

#   _______
#  <  / __/
#  / /__ \
# /_/____/

    def rl_and_equals_sb_and_src(self: "DIRI",
                                 sections: Subset,
                                 sbs_and_vrs: SBs_and_VRs,
                                 src: VR) -> SectionMap:
        """Read-logic Instruction 15
           [APL 15] msk: RL &= <SB> & <SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb_AND_src(
            sections,
            exp.left_AND_right,
            exp.left_AND_right,
            vrs, src)

    def rl_and_equals_sb_and_inv_src(self: "DIRI",
                                     sections: Subset,
                                     sbs_and_vrs: SBs_and_VRs,
                                     src: VR) -> SectionMap:
        """Read-logic Instruction 15
           [APL 15] msk: RL &= <SB> & ~<SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb_AND_src(
            sections,
            exp.left_AND_right,
            exp.left_AND_INV_right,
            vrs, src)

#    ___  ___
#   |_  |/ _ \
#  / __// // /
# /____/\___/

    def rl_xor_equals_sb_and_src(self: "DIRI",
                                 sections: Subset,
                                 sbs_and_vrs: SBs_and_VRs,
                                 src: VR) -> SectionMap:
        """Read-logic Instruction 20
           [APL 20] msk: RL ^= <SB> & <SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb_AND_src(
            sections,
            exp.left_XOR_right,
            exp.left_AND_right,
            vrs, src)

    def rl_xor_equals_sb_and_inv_src(self: "DIRI",
                                     sections: Subset,
                                     sbs_and_vrs: SBs_and_VRs,
                                     src: VR) -> SectionMap:
        """Read-logic Instruction 20
           [APL 20] msk: RL ^= <SB> & ~<SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_OPEQUALS_sb_AND_src(
            sections,
            exp.left_XOR_right,
            exp.left_AND_INV_right,
            vrs, src)

    #   ____     ____    ___      ___
    #  / __/    /_  /   ( _ )    / _ \
    # / _ \_     / /   / _  |    \_, /
    # \___( )   /_( )  \___( )  /___/
    #     |/      |/       |/

    # Read-logic instruction 6, 7, 8, 9
    #   |  6.  RL  =  <SB> |  <SRC>   | :=  <SB>    |    <SRC>  |
    #   |  7.  RL  =  <SB> ^  <SRC>   | :=  <SB>    ^    <SRC>  |
    #   |  8.  RL  = ~<SB> &  <SRC>   | := ~<SB>    &    <SRC>  |
    #   |  9.  RL  =  <SB> & ~<SRC>   | :=  <SB>    &   ~<SRC>  |

    @patch_with(patch_sb)
    def _rl_from_sb_BINOP_gl(self: "DIRI",
                             dest_mask: Subset,
                             vrs: VRs,
                             binop: BinaryOp,
                             src: np.ndarray) -> SectionMap:
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_mask:
            sbdata = self._get_sb_rows(vrs, NUM_PLATS_PER_APUC, s)
            wordline = self.rwinh(data, s, binop(sbdata, src))
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_from_sb_BINOP_ggl(self: "DIRI",
                              dest_mask: Subset,
                              vrs: VRs,
                              binop: BinaryOp,
                              src: np.ndarray) -> SectionMap:
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_mask:
            pseudo_GL = src[::, s // NGGL_ROWS]
            sbdata = self._get_sb_rows(vrs, NUM_PLATS_PER_APUC, s)
            wordline = self.rwinh(data, s, binop(sbdata, pseudo_GL))
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_from_sb_BINOP_rl(self: "DIRI",
                             dest_mask: Subset,
                             vrs: VRs,
                             binop: BinaryOp,
                             src: np.ndarray) -> SectionMap:
        section_map = []
        rl = self.hb.rl
        data = rl.data
        src = src.data if isinstance(src, DiriRegister) else src
        for s in dest_mask:
            sbdata = self._get_sb_rows(vrs, NUM_PLATS_PER_APUC, s)
            wordline = self.rwinh(data, s, binop(sbdata, src[::, s]))
            section_map.append((rl, s, wordline))
        return section_map

    @patch_with(patch_sb)
    def _rl_from_sb_BINOP_rsp16(self: "DIRI",
                                dest_mask: Subset,
                                vrs: VRs,
                                binop: BinaryOp,
                                src: np.ndarray) -> SectionMap:
        # Needed for RWINH filter
        brsp16 = self.build_vr()
        src = src.data if isinstance(src, DiriRegister) else src
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            brsp16[start:stop] = src[index]

        section_map = []
        rl = self.hb.rl
        data = rl.data
        for s in dest_mask:
            sbdata = self._get_sb_rows(vrs, NUM_PLATS_PER_APUC, s)
            wordline = self.rwinh(data, s, binop(sbdata, brsp16[::, s]))
            section_map.append((rl, s, wordline))

        return section_map

    def _rl_from_sb_BINOP_src(self: "DIRI",
                              sections: Subset,
                              vrs: VRs,
                              binop: BinaryOp,
                              src: np.ndarray) -> SectionMap:
        """Abstract instructions 6, 7, 8, 9."""
        dest_mask = Mask(sections)

        assert 1 <= len(vrs) <= 3

        if src.shape == GL_SHAPE:
            return self._rl_from_sb_BINOP_gl(dest_mask, vrs, binop, src)

        if src.shape == GGL_SHAPE:
            return self._rl_from_sb_BINOP_ggl(dest_mask, vrs, binop, src)

        if src.shape == RL_SHAPE:
            return self._rl_from_sb_BINOP_rl(dest_mask, vrs, binop, src)

        if src.shape == RSP16_SHAPE:
            return self._rl_from_sb_BINOP_rsp16(dest_mask, vrs, binop, src)

        raise ValueError(f"unknown shape {src.shape} for SRC")

#   ____
#  / __/
# / _ \
# \___/

    def rl_from_sb_or_src(self: "DIRI",
                          sections: Subset,
                          sbs_and_vrs: SBs_and_VRs,
                          src: VR) -> SectionMap:
        """Read-logic Instruction 6
           [APL  6] msk: RL = <SB> | <SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_from_sb_BINOP_src(
            sections, vrs,
            exp.left_OR_right,
            src)

    def rl_from_sb_or_inv_src(self: "DIRI",
                              sections: Subset,
                              sbs_and_vrs: SBs_and_VRs,
                              src: VR) -> SectionMap:
        """Read-logic Instruction 6
           [APL  6] msk: RL = <SB> | ~<SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_from_sb_BINOP_src(
            sections, vrs,
            exp.left_OR_INV_right,
            src)
#  ____
# /_  /
#  / /
# /_/

    def rl_from_sb_xor_src(self: "DIRI",
                           sections: Subset,
                           sbs_and_vrs: SBs_and_VRs,
                           src: VR) -> SectionMap:
        """Read-logic Instruction 7
           [APL  7] msk: RL = <SB> ^ <SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_from_sb_BINOP_src(
            sections, vrs,
            exp.left_XOR_right,
            src)

    def rl_from_sb_xor_inv_src(self: "DIRI",
                               sections: Subset,
                               sbs_and_vrs: SBs_and_VRs,
                               src: VR) -> SectionMap:
        """Read-logic Instruction 7
           [APL  7] msk: RL = <SB> ^ ~<SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_from_sb_BINOP_src(
            sections, vrs,
            exp.left_XOR_INV_right,
            src)
#   ___
#  ( _ )
# / _  |
# \___/

    def rl_from_inv_sb_and_src(self: "DIRI",
                               sections: Subset,
                               sbs_and_vrs: SBs_and_VRs,
                               src: VR) -> SectionMap:
        """Read-logic Instruction 8
           [APL 8] msk: RL = ~<SB> & <SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_from_sb_BINOP_src(
            sections, vrs,
            exp.INV_left_AND_right,
            src)

    def rl_from_inv_sb_and_inv_src(self: "DIRI",
                                   sections: Subset,
                                   sbs_and_vrs: SBs_and_VRs,
                                   src: VR) -> SectionMap:
        """Read-logic Instruction 8
           [APL 8] msk: RL = ~<SB> & ~<SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_from_sb_BINOP_src(
            sections, vrs,
            exp.INV_left_AND_INV_right,
            src)
#   ___
#  / _ \
#  \_, /
# /___/

    def rl_from_sb_and_inv_src(self: "DIRI",
                               sections: Subset,
                               sbs_and_vrs: SBs_and_VRs,
                               src: VR) -> SectionMap:
        """Read-logic Instruction 9
           [APL 9] msk: RL = <SB> & ~<SRC>"""
        vrs = self.sbs_to_vrs(sbs_and_vrs)
        return self._rl_from_sb_BINOP_src(
            sections, vrs,
            exp.left_AND_INV_right,
            src)

    #  ___     ___      _   _              _
    # | _ \___/ __| ___| | | |   ___  __ _(_)__
    # |   /___\__ \/ -_) | | |__/ _ \/ _` | / _|
    # |_|_\   |___/\___|_| |____\___/\__, |_\__|
    #                                |___/

    @patch_with(patch_partial_rsp16)
    def rsp16_from_rl(self: "DIRI",
                      sections: Subset) -> np.ndarray:
        """[APL] msk: RSP16 = RL;"""
        self.rsp_mode = RspMode.RSP16_READ

        wordlines = []

        nrsp16s = NUM_PLATS_PER_APUC // 16  # RSP --16-- !
        rl = self.hb.rl.data
        for s in Mask(sections):
            wordline = rl[::, s].reshape((nrsp16s, 16))
            wordline = np.logical_or.reduce(wordline, axis=1)
            wordlines.append((s, wordline))

        return wordlines

    @patch_with(patch_gl)
    def gl_from_rl(self: "DIRI",
                   sections: Subset) -> np.ndarray:
        """[APL] msk: GL = RL;"""
        ss = Mask(sections)

        # pre-charge GL with ones
        GL = np.ones(GL_SHAPE, dtype=bool)

        rl = self.hb.rl.data
        for s in ss:
            GL &= rl[::, s]

        return GL

    @patch_with(patch_ggl)
    def ggl_from_rl(self: "DIRI",
                    sections: Subset) -> np.ndarray:
        """[APL] msk: GGL = RL;"""
        ss = Mask(sections)

        # pre-charge GGL with ones
        GGL = np.ones(GGL_SHAPE, dtype=bool)

        rl = self.hb.rl.data
        for d in range(NGGL_ROWS):
            s_min = d * NGGL_ROWS
            s_max = s_min + NGGL_ROWS
            for s in ss:
                if s in range(s_min, s_max):
                    source_1_wordline = rl[::, s]
                    # debug_1 = self.ndarray_1d_to_bitstring(source_1_wordline)
                    source_2_wordline = GGL[::, d]
                    # debug_2 = self.ndarray_1d_to_bitstring(source_2_wordline)
                    source = source_1_wordline & source_2_wordline
                    # debug_s = self.ndarray_1d_to_bitstring(source)
                    GGL[::, d] = source

        return GGL

    @patch_with(patch_l1)
    def l1_from_ggl(self: "DIRI", lx_addr: int) \
            -> Tuple[int, np.ndarray]:
        """[APL] LX = GGL;"""
        LX = deepcopy(self.GGL.data)
        return lx_addr, LX

    @patch_with(patch_lgl)
    def lgl_from_l1(self: "DIRI", lx_addr: int) -> np.ndarray:
        """[APL] LGL = <LX>;"""
        bank = (lx_addr >> 11) & 0b11
        group = (lx_addr >> 9) & 0b11
        row = lx_addr & 0b111111111
        plats = plats_for_bank(bank)
        LGL = deepcopy(self.L1[row, plats, group])
        return LGL

    @patch_with(patch_l2)
    def l2_from_lgl(self: "DIRI", lx_addr: int) \
            -> Tuple[int, np.ndarray]:
        """[APL] LX = LGL;"""
        LX = deepcopy(self.LGL.data)
        return lx_addr, LX

    @patch_with(patch_lgl)
    def lgl_from_l2(self: "DIRI", lx_addr: int) -> np.ndarray:
        """[APL] LGL = <LX>;"""
        LGL = deepcopy(self.L2[lx_addr])
        return LGL

    @patch_with(patch_l1)
    def l1_from_lgl(self: "DIRI", lx_addr: int) \
            -> Tuple[int, np.ndarray]:
        """[APL] LX = LGL;"""
        LX = deepcopy(self.LGL.data)
        return lx_addr, LX

    @patch_with(patch_ggl)
    def ggl_from_l1(self: "DIRI", lx_addr: int) -> np.ndarray:
        """[APL] GGL = <LX>;"""
        GGL = deepcopy(self.L1[lx_addr])
        return GGL

    @patch_with(patch_ggl)
    def ggl_from_rl_and_l1(self: "DIRI",
                           sections: Subset,
                           lx_addr: int) -> np.ndarray:
        """[APL] msk: GGL = RL & <LX>;"""
        _, GGL = self.ggl_from_rl(sections)
        GGL = GGL & self.L1[lx_addr]
        return GGL

    #   ___ _             _
    #  / __| |_  ___ __ _| |_ ___
    # | (__| ' \/ -_) _` |  _(_-<
    #  \___|_||_\___\__,_|\__/__/

    def repeatably_randomize_half_bank(self: "DIRI") -> None:
        r"""Fill an entire half-bank with random data according
        to fixed seeds. The test corpus depends on these fixed seeds."""

        rvr = 5
        self.randomize_sb(sb=rvr, seed=3223)

        lvr = 3
        self.randomize_sb(sb=lvr, seed=0x1337beef)

        r2vr = 8
        self.randomize_sb(sb=r2vr, seed=0xBad0B015)

        these = [rvr, lvr, r2vr]
        others = [s for s in range(NVR) if s not in these]

        those = [self.randomize_sb(sb=s, seed=0xF00BA4 + s)
                 for s in others]


    def set_marker_cheat(self: "DIRI",
                         sections: Subset,
                         v: int,
                         p: int,
                         bit: int = 1) -> "DIRI":
        ss = Mask(sections)
        for s in ss:
            self.hb[v, p, s] = bit
        return self  # for chaining

    def randomize_rl(self: "DIRI", seed: Optional[int] = None) -> "DIRI":
        r"""    See note here

        https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint

        Generator.integers should be used for new code, not legacy randint
        or random_integers.
        """
        rng = np.random.default_rng(seed)
        rl = self.hb.rl.data
        for s in range(NSECTIONS):
            rl[::, s] = rng.integers(2, size=NUM_PLATS_PER_APUC, dtype=bool)
        return self  # for chaining

    def randomize_sb(self: "DIRI",
                     sb: int,
                     seed: Optional[int] = None) -> "DIRI":
        r"""    See note here

        https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint

        Generator.integers should be used for new code, not legacy randint
        or random_integers.
        """
        rng = np.random.default_rng(seed)
        for s in range(NSECTIONS):
            self.hb[sb, ::, s] = rng.integers(2, size=NUM_PLATS_PER_APUC, dtype=bool)
        return self  # for chaining

    #  ___  _         _
    # |   \(_)____ __| |__ _ _  _
    # | |) | (_-< '_ \ / _` | || |
    # |___/|_/__/ .__/_\__,_|\_, |
    #           |_|          |__/

    @staticmethod
    def ndarray_2d_to_bitstring(nda: np.ndarray) -> str:
        r"""Transpose because numpy slices copy plats before sections.
        Remember that hb indices are in VPS order (mnemonic: GPS)."""
        temp0 = list(
            map(DIRI.ndarray_1d_to_bitstring,
                np.transpose(nda)))
        result = '\n'.join(temp0)
        return result

    @staticmethod
    def ndarray_1d_to_bitstring(nda: np.ndarray) -> str:
        return "".join(("1" if b else "0" for b in nda))

    def integer_glass(self: "DIRI", sb: int, plats: int = 4) -> Sequence[int]:
        sub_vr = self.hb[sb][:plats]
        result = [Mask(row).full_integer for row in sub_vr]
        return result

    def plat_to_int(self: "DIRI", sb: int, plat: int) -> int:
        result = Mask(
            self.hb[sb][plat:plat + 1][0]).full_integer
        return result

    def glass(self: "DIRI", sb: int, sections: int = 16, plats: int = 4) -> str:
        if sb == 24:
            result = self.ndarray_2d_to_bitstring(
                self.hb.rl.data[:plats, :sections])
        else:
            result = self.ndarray_2d_to_bitstring(
                self.hb[sb][:plats, :sections])
        return result

    @staticmethod
    def ndarray_2d_to_hexstring(nda: np.ndarray) -> str:
        temp0 = [DIRI.ndarray_1d_to_hexstring(row)
                 for row in nda.T]
        result = '\n'.join(temp0)
        return result

    @staticmethod
    def ndarray_1d_to_hexstring(nda: np.ndarray) -> str:
        l = len(nda)
        nu_nda = nda
        if (l < 4):
            # Pad nda on the left until its length is 4; supports
            # narrow half-banks in tests.
            nu_nda = np.zeros(4, dtype=bool)
            for i in range(4 - l, 4):
                nu_nda[i] = nda[4 - l - i]
            l = 4
        # Now we have multiples of 4.
        assert (l % 4) == 0
        hexes = ''
        for i in range(l // 4):
            temp0 = nu_nda[i*4 : (i+1)*4]
            temp1 = sum((((2**(3-j)) * temp0[j]) for j in range(4)))
            hexes += f'{temp1:X}'
        result = ''.join(hexes)
        return result

    def hex_glass(self: "DIRI",
                  sb: int,
                  sections: int = 16,
                  plats: int = 4) -> str:
        # must print in batches of four bits
        bump_plats = 4 * (plats // 4)
        if sb == 24:
            result = self.ndarray_2d_to_hexstring(
                self.hb.rl.data[:bump_plats, :sections])
        else:
            result = self.ndarray_2d_to_hexstring(
                self.hb[sb, :bump_plats, :sections])
        return result

    @staticmethod
    def from_split_hex_glass(split_hex_glass: List[List[str]],
                             sections: Union[str, Mask] = "FFFF",
                             glass_plats: int = 32,
                             target_plats: int = 2048):
        r"""Return a numpy array built from explicit data given in
        hexadecimal. The split hex_glass contains a sequence of
        hexadecimal rows. Each row is of length glass_plats // 4. The
        rows correspond to section numbers of the output in the
        section_mask: row 0 goes into section_mask.list[0], row 1 into
        section_mask.list[1], etc. The target ndarray may be narrower
        than 2048, as is the case with many half-banks in the tests.

        The rows contain hexits --- hexadecimal numerals. Each
        represents a bit pattern for four plats in the result. We treat
        the plats and the hexits as big-endian for the purpose of
        split_hex_glass, both incoming (as in this case) and outgoing.
        """

        section_mask = Mask(sections)

        section_mask_list_little_endian = section_mask.list

        def _check_hex_glass_preconditions() -> None:
            assert (len(section_mask_list_little_endian)) == len(split_hex_glass)
            assert ((len(split_hex_glass[i]) == (glass_plats // 16)
                     for i in range(len(split_hex_glass))))

        _check_hex_glass_preconditions()

        # Remember vps sounds like GPS; we're making up a virtual VR,
        # so the plat index, p, comes next and the section index, s,
        # comes last. Glass representations read the transpose: with
        # sections increasing downward by the row index and plats
        # increasing rightward by the column index.
        result = np.zeros((target_plats, NSECTIONS), dtype=bool)
        section_number = 0
        for i in range(NSECTIONS):
            if i in section_mask_list_little_endian:
                for j in range(glass_plats // 4):
                    hexit = split_hex_glass[section_number][j]
                    bits = f'{int(hexit, 16):b}'.zfill(4)
                    for k in range(4):
                        assert i == section_mask_list_little_endian[section_number]
                        result[4 * j + k, i] = int(bits[k])
                section_number += 1

        return result

    @staticmethod
    def _grid(bloc: np.ndarray,
              sb: int = 24,
              nrows: int = DEFAULTSLICE_FOR_GRID,
              ncols: int = DEFAULTSLICE_FOR_GRID) -> str:
        r"""so ugly, not even wrong"""
        def colHex(i):
            return '{0:0{1}X} '.format(i, 3)

        def SB(i):
            return i

        def vr_rl_label(index):
            sbil = f'SB[{index}] ' if index < 10 else f'SB[{index}]'
            return 'RL    ' if index == 24 else sbil

        def explode_bits(bits):
            return ' '.join([' ' + t + ' ' for t in bits])

        def rep_cols(s, ncols):
            return ' '.join([s for _ in range(ncols)])
        c, r = bloc.shape

        # first row of labels:
        result = f'{vr_rl_label(sb)} {rep_cols("col", ncols)} ... col\n'
        # second row of labels:
        result += '______ '
        for j in range(ncols):
            result += f'{colHex(j)}'
        result += '... '
        # doctring can't tolerate trailing space
        result += f'{colHex(c-1)}'[:-1] + '\n'
        # data:
        for i in range(nrows):
            result += f'sec {i:0X}: '
            temp0 = bloc[0:ncols, i:(i + 1)]
            temp1 = DIRI.ndarray_2d_to_bitstring(temp0)
            result += explode_bits(temp1)
            temp2 = int(bloc[-1, i:(i + 1)])
            temp3 = f' ...  {temp2}\n'
            result += temp3
        # penultimate row of ellipses:
        if nrows < 15:
            result += f' ...   {rep_cols("...", ncols)} ... ...\n'
        if nrows < 16:
            # Last row of data
            temp3 = bloc[0:ncols, -1]
            temp4 = DIRI.ndarray_1d_to_bitstring(temp3)
            result += f'sec F: ' + explode_bits(temp4)
            temp5 = int(bloc[-1, -1])
            temp6 = f' ...  {temp5}'
            result += temp6
        return result

    def grid(self: "DIRI",
             sb: int = 24,
             nrows: int = DEFAULTSLICE_FOR_GRID,
             ncols: int = DEFAULTSLICE_FOR_GRID) -> str:
        if sb == 24:
            result = self._grid(self.hb.rl.data, 24, nrows=nrows, ncols=ncols)
        else:
            result = self._grid(self.hb[sb], sb, nrows=nrows, ncols=ncols)
        return result

    def rl(self: "DIRI") -> str:
        result = self.ndarray_2d_to_bitstring(self.hb.rl.data)
        return result

    def rl_hex(self: "DIRI") -> str:
        result = self.ndarray_2d_to_hexstring(self.hb.rl.data)
        return result

    def rsp16(self: "DIRI") -> str:
        result = self.ndarray_2d_to_bitstring(self.RSP16.data)
        return result

    def rsp16_hex(self: "DIRI") -> str:
        result = self.ndarray_2d_to_hexstring(self.RSP16.data)
        return result

    def rsp256(self: "DIRI") -> str:
        result = self.ndarray_2d_to_bitstring(self.RSP256.data)
        return result

    def rsp2k(self: "DIRI") -> str:
        """bitstring rep of RSP2K"""
        result = self.ndarray_2d_to_bitstring(self.RSP2K.data)
        return result

    def check_rsp_decimal(self: "DIRI", expected: int) -> bool:
        self.pull_rsps("FFFF")
        actual = ''.join(self.rsp2k().split())
        actual = actual[::-1]  # reverse to little-endian
        actual = int(actual, 2)
        result = (actual == expected)
        return result

    def extract_plat(self: "DIRI", sb: int, plat_number: int) -> np.ndarray:
        result = self.hb[sb, plat_number]
        return result

    def plat_as_integer(self: "DIRI", sb: int, plat_number: int) -> int:
        temp0 = self.extract_plat(sb, plat_number)
        temp1 = Mask(temp0)
        temp2 = temp1.full_integer
        return temp2

    #  ___             _   ____      __   _ _          ___      _    _ _    _ _
    # | _ \___ __ _ __| | / /\ \    / / _(_) |_ ___ __|_ _|_ _ | |_ (_) |__(_) |_
    # |   / -_) _` / _` |/ /  \ \/\/ / '_| |  _/ -_)___| || ' \| ' \| | '_ \ |  _|
    # |_|_\___\__,_\__,_/_/    \_/\_/|_| |_|\__\___|  |___|_||_|_||_|_|_.__/_|\__|

    @patch_with(patch_rwinh_set)
    def rwinh_set(self: "DIRI", mask: Mask) -> None:
        """[APL] msk: RL = <SB> RWINH_SET;"""
        return mask

    @patch_with(patch_rwinh_rst)
    def rwinh_rst(self: "DIRI", mask: Mask, has_read: bool) -> None:
        """[APL] msk: RL = <SB> RWINH_RST;"""
        return mask, has_read
