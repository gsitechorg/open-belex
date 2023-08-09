r"""
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

import re
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import repeat
from typing import Dict, Optional, Set, Type

from open_belex.bleir.analyzers import (RegisterParameterFinder,
                                        UserRegisterScanner)
from open_belex.bleir.types import (EWE_REG, L1_REG, L2_REG, RE_REG, RN_REG,
                                    SM_REG, AllocatedRegister, FormalParameter,
                                    Fragment, FragmentCaller, Snippet)
from open_belex.bleir.walkables import (BLEIRListener, BLEIRTransformer,
                                        BLEIRWalker)
from open_belex.common.register_arenas import (EweRegArena, L1RegArena,
                                               L2RegArena, ReRegArena,
                                               RnRegArena, SmRegArena)
from open_belex.common.seu_layer import SEULayer
from open_belex.common.string_arena import StringArena


@dataclass
class RegisterAllocator:
    # Example: reservations = {"sm_regs": set(range(4, 15 + 1))}
    reservations: Optional[Dict[str, Set[int]]] = None

    reg_arenas: Dict[Type, StringArena] = field(init=False)

    def __post_init__(self: "RegisterAllocator") -> None:
        reservations = self.reservations
        if reservations is None:
            reservations = {}

        # Reserve manual allocations ...
        seu = SEULayer.context()
        for reg_kind, reg_allocs in seu:
            if reg_kind not in reservations:
                reg_nums = set()
                reservations[reg_kind] = reg_nums
            else:
                reg_nums = reservations[reg_kind]
            for reg_num, _ in reg_allocs:
                reg_nums.add(reg_num)

        self.reg_arenas = {
            RN_REG: RnRegArena(reservations=reservations.get("rn_regs", None)),
            RE_REG: ReRegArena(reservations=reservations.get("re_regs", None)),
            EWE_REG: EweRegArena(reservations=reservations.get("ewe_regs", None)),
            L1_REG: L1RegArena(reservations=reservations.get("l1_regs", None)),
            L2_REG: L2RegArena(reservations=reservations.get("l2_regs", None)),
            SM_REG: SmRegArena(reservations=reservations.get("sm_regs", None)),
        }

    def free_all(self: "RegisterAllocator") -> None:
        for reg_arena in self.reg_arenas.values():
            reg_arena.free_all()

    def __call__(self: "RegisterAllocator",
                 parameter: FormalParameter) -> AllocatedRegister:

        register = None

        if parameter.__class__ in self.reg_arenas:
            reg_arena = self.reg_arenas[parameter.__class__]
            if parameter.register is not None:
                if isinstance(parameter, (L1_REG, L2_REG)):
                    register = f"L1_ADDR_REG_{parameter.register}"
                if isinstance(parameter, (L1_REG, L2_REG)):
                    register = f"L2_ADDR_REG_{parameter.register}"
                else:
                    type_nym = parameter.__class__.__name__
                    register = f"{type_nym}_{parameter.register}"
                reg_arena.reserve(register)
            else:
                register = reg_arena.allocate()

        else:
            raise ValueError(f"Unsupported parameter type: {type(parameter).__name__}")

        if register is None:
            raise RuntimeError(f"Exhausted {parameter.__class__.__name__} registers")

        return AllocatedRegister(parameter, register)

    def free_all(self: "RegisterAllocator") -> None:
        for reg_arena in self.reg_arenas.values():
            reg_arena.free_all()


@dataclass
class AllocateRegisters(BLEIRTransformer):
    shared_registers_by_frag: Dict[Type, Set[int]]

    # Example: reservations = {"sm_regs": set(range(4, 15 + 1))}
    reservations: Optional[Dict[str, Set[int]]] = None

    def transform_fragment_caller(
            self: "AllocateRegisters",
            fragment_caller: FragmentCaller) -> FragmentCaller:

        # Scan for lowered parameters
        walker = BLEIRWalker()
        register_parameter_finder = RegisterParameterFinder()
        walker.walk(register_parameter_finder, fragment_caller)

        allocations_by_parameter = {}
        allocate = RegisterAllocator(reservations=self.reservations)

        # Reserve the registers associated with lowered parameters so we do not
        # clobber them (e.g. so we do not overwrite SM_REG_4 when we need it for
        # the lowered SM_0XFFFF parameter).
        for lowered_register in register_parameter_finder.lowered_registers:
            if lowered_register.register is not None \
               and lowered_register.identifier not in allocations_by_parameter:
                allocation = allocate(lowered_register)
                allocations_by_parameter[lowered_register.identifier] = allocation
        fragment = fragment_caller.fragment
        frag_id = fragment.identifier

        if frag_id in self.shared_registers_by_frag:
            shared_registers = self.shared_registers_by_frag[frag_id]
            for src, dest in shared_registers.items():
                if dest.identifier not in allocations_by_parameter:
                    if dest.register is not None:
                        # NOTE: The <PLACEHOLDER> will be replaced by
                        # AllocateLoweredRegisters:
                        allocation = AllocatedRegister(
                            parameter=dest,
                            register="<PLACEHOLDER>")
                    else:
                        allocation = allocate(dest)

                    allocations_by_parameter[dest.identifier] = allocation
                else:
                    allocation = allocations_by_parameter[dest.identifier]
                allocations_by_parameter[src.identifier] = allocation.having(parameter=src)

        parameters = fragment_caller.parameters
        allocated_registers = fragment_caller.registers
        if allocated_registers is None:
            allocated_registers = repeat(None)

        registers = []
        for parameter, allocated_register in \
                zip(parameters, allocated_registers):
            if allocated_register is not None:
                allocation = allocated_register
            elif parameter.identifier in allocations_by_parameter:
                allocation = allocations_by_parameter[parameter.identifier]
            else:
                allocation = allocate(parameter)
            registers.append(allocation)

        allocate.free_all()
        return fragment_caller.having(registers=registers)


RE_RN_REG_LITERAL = re.compile(r"^RN_REG_(?:[0-9]|1[0-5])$")
RE_RE_REG_LITERAL = re.compile(r"^RE_REG_[0-3]$")
RE_EWE_REG_LITERAL = re.compile(r"^EWE_REG_[0-3]$")
RE_L1_REG_LITERAL = re.compile(r"^L1_ADDR_REG_[0-3]$")
RE_L2_REG_LITERAL = re.compile(r"^L2_ADDR_REG_0$")
RE_SM_REG_LITERAL = re.compile(r"^SM_REG_(?:[0-9]|1[0-5])$")


@dataclass
class AllocateLoweredRegisters(BLEIRListener, BLEIRTransformer):
    user_register_scanner: UserRegisterScanner
    shared_registers_by_frag: Dict[Type, Set[int]]

    # Example: reservations = {"sm_regs": set(range(4, 15 + 1))}
    reservations: Dict[str, Set[int]] = field(default_factory=dict)

    allocate: Optional[RegisterAllocator] = None
    allocations: Optional[Dict[str, int]] = None

    shared_registers: Optional[Dict[RN_REG, RN_REG]] = None

    in_snippet: bool = False
    in_fragment_caller: bool = False

    def enter_snippet(self: "AllocateLoweredRegisters",
                      snippet: Snippet) -> None:

        self.in_snippet = True

        reservations = deepcopy(self.reservations)

        reserved_rn_regs = reservations.get("rn_regs", set())
        for rn_reg_id in self.user_register_scanner.registers_by_type[RN_REG]:
            reserved_rn_regs.add(rn_reg_id)
        reservations["rn_regs"] = reserved_rn_regs

        reserved_l1_regs = reservations.get("l1_regs", set())
        for l1_reg_id in self.user_register_scanner.registers_by_type[L1_REG]:
            reserved_l1_regs.add(l1_reg_id)
        reservations["l1_regs"] = reserved_l1_regs

        reserved_sm_regs = reservations.get("sm_regs", set())
        for sm_reg_id in self.user_register_scanner.registers_by_type[SM_REG]:
            reserved_sm_regs.add(sm_reg_id)
        reservations["sm_regs"] = reserved_sm_regs

        self.allocate = RegisterAllocator(reservations=reservations)
        self.allocations = {}

        walker = BLEIRWalker()
        register_parameter_finder = RegisterParameterFinder()
        walker.walk(register_parameter_finder, snippet)
        for lowered_register in register_parameter_finder.lowered_registers:
            if lowered_register.register is not None:
                allocation = self.allocate(lowered_register)
                register = allocation.reg_id
                self.allocations[lowered_register.identifier] = register

    def exit_snippet(self: "AllocateLoweredRegisters",
                     snippet: Snippet) -> None:
        self.in_snippet = False
        self.allocate = None
        # self.allocations = None

    def enter_fragment_caller(
            self: "AllocateLoweredRegisters",
            fragment_caller: FragmentCaller) -> None:
        self.in_fragment_caller = True

    def exit_fragment_caller(
            self: "AllocateLoweredRegisters",
            fragment_caller: FragmentCaller) -> None:
        self.in_fragment_caller = False
        self.shared_registers = None

    def transform_fragment_caller(
            self: "AllocateLoweredRegisters",
            fragment_caller: FragmentCaller) -> FragmentCaller:
        allocations = []
        for allocation in fragment_caller.registers:
            if allocation.register == "<PLACEHOLDER>":
                parameter = allocation.parameter
                if self.shared_registers is not None \
                   and parameter in self.shared_registers:
                    parameter = self.shared_registers[parameter]
                if isinstance(parameter, L1_REG):
                    register = f"L1_ADDR_REG_{parameter.register}"
                elif isinstance(parameter, L2_REG):
                    register = f"L2_ADDR_REG_{parameter.register}"
                else:
                    reg_type = parameter.__class__.__name__
                    register = f"{reg_type}_{parameter.register}"
                allocation = allocation.having(register=register)
            allocations.append(allocation)
        return fragment_caller.having(registers=allocations)

    def enter_fragment(self: "AllocateLoweredRegisters",
                       fragment: Fragment) -> None:
        frag_id = fragment.original_identifier
        if frag_id in self.shared_registers_by_frag:
            self.shared_registers = self.shared_registers_by_frag[frag_id]

    def exit_fragment(self: "AllocateLoweredRegisters",
                      fragment: Fragment) -> None:
        if not self.in_fragment_caller:
            self.shared_registers = None

    def transform_rn_reg(self: "AllocateLoweredRegisters",
                         rn_reg: RN_REG) -> RN_REG:

        if not self.in_snippet \
           or not rn_reg.is_lowered \
           or (rn_reg.register is not None
               and RE_RN_REG_LITERAL.fullmatch(rn_reg.identifier)):
            return rn_reg

        if self.shared_registers is not None \
           and rn_reg in self.shared_registers:
            reg_nym = self.shared_registers[rn_reg].identifier
        else:
            reg_nym = rn_reg.identifier

        if reg_nym in self.allocations:
            register = self.allocations[reg_nym]
        else:
            allocation = self.allocate(rn_reg)
            register = allocation.reg_id
            self.allocations[reg_nym] = register

        return rn_reg.having(register=register)

    def transform_re_reg(self: "AllocateLoweredRegisters",
                         re_reg: RE_REG) -> RE_REG:

        if not self.in_snippet \
           or not re_reg.is_lowered \
           or (re_reg.register is not None
               and RE_RE_REG_LITERAL.fullmatch(re_reg.identifier)):
            return re_reg

        if re_reg.identifier in self.allocations:
            register = self.allocations[re_reg.identifier]
        else:
            allocation = self.allocate(re_reg)
            register = allocation.reg_id
            self.allocations[re_reg.identifier] = register

        return re_reg.having(register=register)

    def transform_ewe_reg(self: "AllocateLoweredRegisters",
                          ewe_reg: EWE_REG) -> EWE_REG:

        if not self.in_snippet \
           or not ewe_reg.is_lowered \
           or (ewe_reg.register is not None
               and RE_EWE_REG_LITERAL.fullmatch(ewe_reg.identifier)):
            return ewe_reg

        if ewe_reg.identifier in self.allocations:
            register = self.allocations[ewe_reg.identifier]
        else:
            allocation = self.allocate(ewe_reg)
            register = allocation.reg_id
            self.allocations[ewe_reg.identifier] = register

        return ewe_reg.having(register=register)

    def transform_l1_reg(self: "AllocateLoweredRegisters",
                         l1_reg: L1_REG) -> L1_REG:

        if not self.in_snippet \
           or not l1_reg.is_lowered \
           or (l1_reg.register is not None
               and RE_L1_REG_LITERAL.fullmatch(l1_reg.identifier)):
            return l1_reg

        if l1_reg.identifier in self.allocations:
            register = self.allocations[l1_reg.identifier]
        else:
            allocation = self.allocate(l1_reg)
            register = allocation.reg_id
            self.allocations[l1_reg.identifier] = register

        return l1_reg.having(register=register)

    def transform_l2_reg(self: "AllocateLoweredRegisters",
                         l2_reg: L2_REG) -> L2_REG:

        if not self.in_snippet \
           or not l2_reg.is_lowered \
           or (l2_reg.register is not None
               and RE_L2_REG_LITERAL.fullmatch(l2_reg.identifier)):
            return l2_reg

        if l2_reg.identifier in self.allocations:
            register = self.allocations[l2_reg.identifier]
        else:
            allocation = self.allocate(l2_reg)
            register = allocation.reg_id
            self.allocations[l2_reg.identifier] = register

        return l2_reg.having(register=register)

    def transform_sm_reg(self: "AllocateLoweredRegisters",
                         sm_reg: SM_REG) -> SM_REG:

        if not self.in_snippet \
           or not sm_reg.is_lowered \
           or (sm_reg.register is not None
               and RE_SM_REG_LITERAL.fullmatch(sm_reg.identifier)):
            return sm_reg

        if sm_reg.identifier in self.allocations:
            register = self.allocations[sm_reg.identifier]
        else:
            allocation = self.allocate(sm_reg)
            register = allocation.reg_id
            self.allocations[sm_reg.identifier] = register

        return sm_reg.having(register=register)
