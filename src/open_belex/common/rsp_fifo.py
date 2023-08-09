"""
 By Dylon Edwards

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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from reactivex.subject import Subject

from open_belex.common.constants import (NUM_APCS_PER_APUC,
                                         NUM_HALF_BANKS_PER_APC)
from open_belex.common.stack_manager import contextual

FIFO_CAPACITY: int = 16


@dataclass
class RspFifoMsg:
    rsp32k: np.uint8 = np.uint8(0)
    rsp2k: Sequence[np.uint16] = field(
        default_factory=lambda: np.zeros(NUM_HALF_BANKS_PER_APC,
                                         dtype=np.uint16))

    def __eq__(self: "RspFifoMsg", other: Any) -> bool:
        return isinstance(other, RspFifoMsg) \
            and np.array_equal(self.rsp32k, other.rsp32k) \
            and np.array_equal(self.rsp2k, other.rsp2k)


KindNym = str
MethodNym = str
ApcId = int
QueueLength = int
QueueEvent = Tuple[KindNym, MethodNym, ApcId, RspFifoMsg, QueueLength]
ApcSubscriber = Callable[[QueueEvent], None]


@dataclass
class ApcRspFifo:
    apc_id: int
    length: int = 1
    cursor: int = -1
    buffer: Sequence[RspFifoMsg] = field(
        default_factory=lambda: [None] * FIFO_CAPACITY)
    subject: Subject = field(default_factory=Subject)

    def __deepcopy__(self: "ApcRspFifo", memo: Dict[str, Any]) -> "ApcRspFifo":
        cls = self.__class__
        cpy = cls.__new__(cls)
        setattr(cpy, "apc_id", self.apc_id)
        setattr(cpy, "length", self.length)
        setattr(cpy, "cursor", self.cursor)
        setattr(cpy, "buffer", deepcopy(self.buffer))
        setattr(cpy, "subject", Subject())
        cpy.subject.observers = [deepcopy(observer)
                                 for observer in self.subject.observers]
        memo[id(self)] = cpy
        return cpy

    def __eq__(self: "ApcRspFifo", other: Any) -> bool:
        return isinstance(other, ApcRspFifo) \
            and self.apc_id == other.apc_id \
            and self.length == other.length \
            and self.cursor == other.cursor \
            and all(self.buffer[i] == other.buffer[i]
                    for i in range(FIFO_CAPACITY))

    def __ne__(self: "ApcRspFifo", other: Any) -> bool:
        return not self.__eq__(other)

    def subscribe(self: "ApcRspFifo", subscriber: ApcSubscriber) -> None:
        self.subject.subscribe(subscriber)

    def enqueue(self: "ApcRspFifo", rsp_fifo_msg: RspFifoMsg) -> int:
        # Add 1 to FIFO_CAPACITY for the initial offset
        if self.length == FIFO_CAPACITY + 1:
            return 1
        index = (self.cursor + self.length) % FIFO_CAPACITY
        self.buffer[index] = rsp_fifo_msg
        self.length += 1
        if len(self.subject.observers) > 0:
            # NOTE: Deduct 1 from the length for the initial offset
            self.subject.on_next(
                ("fifo::enqueue", self.apc_id, rsp_fifo_msg,
                 self.length - 1))
        return 0

    def rsp_rd(self: "ApcRspFifo") -> int:
        if self.length == 0:
            return 1
        self.length -= 1
        self.cursor = (self.cursor + 1) % FIFO_CAPACITY
        if len(self.subject.observers) > 0:
            # NOTE: Deduct 1 from the length for the initial offset
            self.subject.on_next(
                ("fifo::dequeue", self.apc_id, self.length - 1))
        return 0

    def rd_rsp2k_reg(self: "ApcRspFifo", bank_id: int) -> np.uint32:
        rsp_fifo_msg = self.buffer[self.cursor]
        lower_half_bank = bank_id
        upper_half_bank = lower_half_bank + 4
        value = (rsp_fifo_msg.rsp2k[upper_half_bank] << 16) \
            | rsp_fifo_msg.rsp2k[lower_half_bank]
        return np.uint32(value)

    def rd_rsp32k_reg(self: "ApcRspFifo") -> np.uint8:
        rsp_fifo_msg = self.buffer[self.cursor]
        return rsp_fifo_msg.rsp32k


@contextual(lazy_init=True)
@dataclass
class ApucRspFifo:
    queues: Sequence[ApcRspFifo] = field(
        default_factory=lambda: [ApcRspFifo(apc_id)
                                 for apc_id in range(NUM_APCS_PER_APUC)])
    active: Optional[ApcRspFifo] = None

    def __eq__(self: "ApucRspFifo", other: Any) -> bool:
        return isinstance(other, ApucRspFifo) \
            and self.queues[0] == other.queues[0] \
            and self.queues[1] == other.queues[1] \
            and (self.active is None and other.active is None
                 or
                 (self.active is not None and other.active is not None
                  and self.queues.index(self.active)
                  == other.queues.index(other.active)))

    def __ne__(self: "ApucRspFifo", other: Any) -> bool:
        return not self.__eq__(other)

    def subscribe_to_apc_rsp_fifo(self: "ApucRspFifo",
                                  apc_id: int,
                                  subscriber: ApcSubscriber) -> None:
        apc_rsp_fifo = self.queues[apc_id]
        apc_rsp_fifo.subscribe(subscriber)

    def subscribe(self: "ApucRspFifo",
                  subscriber: ApcSubscriber) -> None:
        for apc_id in range(2):
            self.subscribe_to_apc_rsp_fifo(apc_id, subscriber)

    def rsp_rd(self: "ApucRspFifo", apc_id: int) -> int:
        apc_rsp_fifo = self.queues[apc_id]
        self.active = apc_rsp_fifo
        return apc_rsp_fifo.rsp_rd()

    def rd_rsp2k_reg(self: "ApucRspFifo", bank_id: int) -> np.uint32:
        return self.active.rd_rsp2k_reg(bank_id)

    def rd_rsp32k_reg(self: "ApucRspFifo") -> np.uint8:
        return self.active.rd_rsp32k_reg()
