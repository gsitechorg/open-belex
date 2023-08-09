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

from dataclasses import dataclass
from typing import Optional, Sequence

from open_belex.bleir.analyzers import (CoalesceGroupedRegisters,
                                        RegisterGrouper)
from open_belex.bleir.rewriters import AllocateTemporaries, ParameterLowerizer
from open_belex.bleir.types import RN_REG, Fragment


@dataclass
class LocalToLoweredGroups:
    register_grouper: RegisterGrouper
    allocate_temporaries: AllocateTemporaries
    parameter_lowerizer: ParameterLowerizer

    coalesce_grouped_registers: Optional[CoalesceGroupedRegisters] = None

    def __call__(self: "LocalToLoweredGroups",
                 fragment: Fragment) -> Optional[Sequence[Sequence[RN_REG]]]:

        frag_id = fragment.original_identifier
        if frag_id in self.register_grouper.groups_by_frag:
            row_numbers_by_rn_reg = self.allocate_temporaries \
                                        .row_numbers_by_rn_reg_by_frag[frag_id]

            lowered_parameters_by_value = \
                self.parameter_lowerizer \
                    .lowered_parameters_by_value_by_type[RN_REG]

            groups = self.register_grouper.groups_by_frag[frag_id]

            _groups = []
            for group in groups:
                _group = []
                for rn_reg in group:
                    if rn_reg.identifier.startswith("_INTERNAL"):
                        row_number = row_numbers_by_rn_reg[rn_reg.identifier]
                        rn_reg = lowered_parameters_by_value[row_number]
                    _group.append(rn_reg)
                _groups.append(sorted(_group, key=lambda reg: reg.identifier))

            groups = _groups

            if self.coalesce_grouped_registers is not None \
               and frag_id in self.coalesce_grouped_registers.shared_registers_by_frag:
                shared_registers = self.coalesce_grouped_registers \
                                       .shared_registers_by_frag[frag_id]
                _groups = []
                for group in groups:
                    _group = set()
                    for src in group:
                        if src in shared_registers:
                            dest = shared_registers[src]
                            _group.add(dest)
                        else:
                            _group.add(src)
                    _groups.append(sorted(_group, key=lambda reg: reg.identifier))

                groups = _groups

            return sorted(groups, key=lambda group: group[0].identifier)

        return None
