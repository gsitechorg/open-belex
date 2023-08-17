r"""
By Dylon Edwards
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Sequence

from open_belex.bleir.analyzers import (CoalesceGroupedRegisters,
                                        RegisterGrouper)
from open_belex.bleir.rewriters import (AllocateTemporaries,
                                        CoalesceGroupedTemporaries,
                                        ParameterLowerizer,
                                        SpillRestoreScheduler)
from open_belex.bleir.types import (RN_REG, AllocatedRegister, CallerMetadata,
                                    FragmentCaller, FragmentCallerCall)
from open_belex.bleir.walkables import BLEIRTransformer
from open_belex.kernel_libs.common import cpy_vr
from open_belex.kernel_libs.memory import restore_vr, spill_vr

LOGGER = logging.getLogger()


@dataclass
class CollectSpillRestoreCalls(BLEIRTransformer):
    spill_restore_scheduler: SpillRestoreScheduler
    register_grouper: RegisterGrouper
    allocate_temporaries: AllocateTemporaries
    parameter_lowerizer: ParameterLowerizer
    coalesce_grouped_temporaries: CoalesceGroupedTemporaries
    coalesce_grouped_registers: CoalesceGroupedRegisters

    spill_calls_by_frag: Dict[str, Sequence[FragmentCallerCall]] = \
        field(default_factory=dict)

    restore_calls_by_frag: Dict[str, Sequence[FragmentCallerCall]] = \
        field(default_factory=dict)

    initial_active_registers_by_frag: Dict[str, Sequence[AllocatedRegister]] = \
        field(default_factory=dict)

    def transform_fragment_caller(
            self: "CollectSpillRestoreCalls",
            fragment_caller: FragmentCaller) -> FragmentCaller:

        fragment = fragment_caller.fragment
        frag_id = fragment.original_identifier

        registers = fragment_caller.registers
        initial_active_registers = registers

        spill_calls = []
        restore_calls = []
        if frag_id in self.spill_restore_scheduler.user_regs_by_frag:
            groups = self.register_grouper.groups_by_frag[frag_id]

            l1_regs_by_rn_reg = \
                self.spill_restore_scheduler \
                    .final_spilled_rn_regs_by_frag[frag_id]

            user_regs = self.spill_restore_scheduler \
                            .user_regs_by_frag[frag_id]

            initial_active_regs = \
                self.spill_restore_scheduler \
                    .initial_active_regs_by_frag[frag_id]

            initial_active_registers = \
                [register
                 for register in registers
                 if (not register.isa(RN_REG)
                     or register.parameter in initial_active_regs)]

            final_active_regs = \
                self.spill_restore_scheduler \
                    .final_active_regs_by_frag[frag_id]

            initial_spilled_rn_regs = \
                self.spill_restore_scheduler \
                    .initial_spilled_rn_regs_by_frag[frag_id]

            initial_spilled_user_vrs = user_regs & set(initial_spilled_rn_regs.keys())

            LOGGER.debug("(initial) groups = %s",
                         {str(set(reg.identifier for reg in group))
                          for group in groups})
            LOGGER.debug("(initial) active_regs = %s",
                         {reg.identifier
                          for reg in initial_active_regs})
            LOGGER.debug("(initial) spilled_user_regs = %s",
                         {reg.identifier for reg in initial_spilled_user_vrs})

            for vr in initial_spilled_user_vrs:
                l1_reg = initial_spilled_rn_regs[vr]
                spill_call = spill_vr(l1_reg.bank_group_row, vr.identifier, debug=False)
                spill_calls.append(spill_call)

            shared_registers = \
                self.coalesce_grouped_registers \
                    .shared_registers_by_frag[frag_id]

            coalesced_row_numbers = {}
            for src, dest in shared_registers.items():
                coalesced_row_numbers[src.identifier] = dest.row_number

            for allocation in list(initial_active_registers):
                parameter = allocation.parameter
                if isinstance(parameter, RN_REG) \
                   and parameter.identifier in coalesced_row_numbers:
                    row_number = \
                        coalesced_row_numbers[parameter.identifier]
                    spill_calls.append(
                        cpy_vr(row_number, parameter.identifier, debug=False))
                    initial_active_registers.remove(allocation)

            registers = list(registers)
            for index, allocation in enumerate(registers):
                parameter = allocation.parameter
                if isinstance(parameter, RN_REG) \
                   and parameter.identifier in coalesced_row_numbers:
                    shared_register = shared_registers[parameter]
                    registers[index] = allocation.having(
                        register=shared_register.identifier)
            registers = tuple(registers)

            # NOTE: The way to determine whether an RN_REG is in its appropriate
            # place is to zip the parameters with their AllocatedRegisters and check
            # whether each parameter corresponds to its AllocatedRegister

            groups_by_reg = {}
            for group in groups:
                for reg in group:
                    groups_by_reg[reg] = group

            row_numbers_by_rn_reg = \
                self.allocate_temporaries \
                    .row_numbers_by_rn_reg_by_frag[frag_id]

            lowered_parameters_by_value = \
                self.parameter_lowerizer \
                    .lowered_parameters_by_value_by_type[RN_REG]

            active_regs_by_group = {}
            for active_reg in initial_active_regs:
                group = groups_by_reg[active_reg]
                active_regs_by_group[group] = active_reg

            # register_map = fragment_caller.register_map
            group_regs_by_reg = dict()

            for group in groups:
                for reg in group:
                    if reg.identifier.startswith("_INTERNAL"):
                        temporary_reg = reg
                        row_number = row_numbers_by_rn_reg[reg.identifier]
                        lowered_reg = lowered_parameters_by_value[row_number]
                        LOGGER.debug("(final) %s -> %s",
                                     temporary_reg.identifier,
                                     lowered_reg.identifier)

            for group, active_reg in active_regs_by_group.items():
                if active_reg.identifier.startswith("_INTERNAL"):
                    row_number = row_numbers_by_rn_reg[active_reg.identifier]
                    active_reg = lowered_parameters_by_value[row_number]
                for group_reg in group:
                    if not group_reg.is_temporary \
                       and group_reg is not active_reg:
                        group_regs_by_reg[group_reg] = active_reg

            regs_to_move = set(group_regs_by_reg.keys())

            # Restore Cases:
            # ==============

            active_user_regs = user_regs & final_active_regs
            spilled_user_regs = user_regs - active_user_regs

            LOGGER.debug("(final) active_regs = %s",
                         {reg.identifier
                          for reg in (final_active_regs & user_regs)})
            LOGGER.debug("(final) spilled_user_regs = %s",
                         {reg.identifier for reg in spilled_user_regs})
            LOGGER.debug("(final) user_regs_to_move = %s",
                         {reg.identifier
                          for reg in (regs_to_move & user_regs)})

            coalesced_temporaries = self.coalesce_grouped_temporaries \
                                        .coalesced_temporaries_by_frag[frag_id]

            # Case 1: reg is active and in correct place
            # -> nothing to do

            # Case 2: reg is active and in wrong place
            # -> swap with vmr for reg that should be here
            # -> load vmr into vr for swapped reg

            for user_rn_reg in (active_user_regs & regs_to_move):
                group_rn_reg = group_regs_by_reg[user_rn_reg]
                if group_rn_reg.identifier in coalesced_temporaries:
                    group_rn_reg = \
                        coalesced_temporaries[group_rn_reg.identifier]
                if group_rn_reg.is_lowered:
                    source = group_rn_reg.row_number
                else:
                    source = group_rn_reg.identifier
                if source in coalesced_row_numbers:
                    source = coalesced_row_numbers[source]
                cpy_call = cpy_vr(user_rn_reg.identifier, source, debug=False)
                restore_calls.append(cpy_call)

            # Case 3: reg is inactive
            # -> load vmr for reg into active vr

            for user_rn_reg in list(spilled_user_regs):
                user_l1_reg = l1_regs_by_rn_reg[user_rn_reg]
                load_call = restore_vr(user_rn_reg.identifier,
                                       user_l1_reg.bank_group_row,
                                       debug=False)
                restore_calls.append(load_call)

                spilled_user_regs.remove(user_rn_reg)
                active_user_regs.add(user_rn_reg)

            # NOTE: All user regs at this point should have been restored
            assert len(spilled_user_regs) == 0
            assert active_user_regs == user_regs

        metadata = fragment_caller.metadata
        if metadata is None:
            metadata = {}
        metadata[CallerMetadata.INITIAL_ACTIVE_REGISTERS] = \
            initial_active_registers

        initializers = fragment_caller.initializers
        if initializers is None:
            initializers = spill_calls
        else:
            initializers = initializers + spill_calls

        finalizers = fragment_caller.finalizers
        if finalizers is None:
            finalizers = restore_calls
        else:
            finalizers = finalizers + restore_calls

        return fragment_caller.having(
            registers=registers,
            initializers=initializers,
            finalizers=finalizers,
            metadata=metadata)
