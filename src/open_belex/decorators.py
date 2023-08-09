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

import logging
from collections import OrderedDict, defaultdict
from functools import wraps
from itertools import chain
from typing import Callable, Optional, Sequence

import numpy as np

from open_belex.apl import (APL_comment, collect_all_normalized_masks,
                            make_bleir_map)
from open_belex.bleir.types import (RN_REG, SM_REG, CallerMetadata, Example,
                                    Fragment, FragmentCaller, InlineComment,
                                    ValueParameter)
from open_belex.common.constants import NSB
from open_belex.compiler import compile_apl_from_belex

LOGGER = logging.getLogger()


def reversed_dict(d):
    return OrderedDict((v, k) for k, v in d.items())


def build_belex_examples(register_map, build_examples):

    def value_param(identifier: str, value: np.ndarray) -> ValueParameter:
        row_number = register_map[identifier]
        return ValueParameter(identifier, row_number, value)

    examples = build_examples(value_param)
    return examples


def build_examples(fragment_caller: FragmentCaller) -> Sequence[Example]:
    register_map = fragment_caller.metadata[CallerMetadata.REGISTER_MAP]
    build_examples = fragment_caller.metadata[CallerMetadata.BUILD_EXAMPLES]
    return build_belex_examples(register_map, build_examples)


def collect_formal_identifiers(examples: Sequence[Example]) -> Sequence[int]:
    example = examples[0]
    formal_identifiers = [example.expected_value.identifier]
    for value_parameter in example.parameters:
        formal_identifier = value_parameter.identifier
        formal_identifiers.append(formal_identifier)
    return formal_identifiers


def collect_actual_values(examples: Sequence[Example]) -> Sequence[int]:
    example = examples[0]
    actual_values = [example.expected_value.row_number]
    for value_parameter in example.parameters:
        actual_value = value_parameter.row_number
        actual_values.append(actual_value)
    return actual_values


def belex_block(fn_or_frag_name: Optional[str] = None,
                optimizations: Optional[Sequence[Callable]] = None,
                build_examples: Optional[Callable] = None,
                skip_tests: bool = False,
                should_fail: bool = False) \
            -> Callable:

    frag_name = fn_or_frag_name
    if callable(frag_name):
        frag_name = fn_or_frag_name.__name__

    if optimizations is None:
        optimizations = []

    def decorator(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal frag_name, optimizations, build_examples, fn

            _frag_name = frag_name
            if _frag_name is None:
                _frag_name = fn.__name__

            if _frag_name.endswith("_caller"):
                _frag_name = _frag_name[:-len("_caller")]

            _optimizations = optimizations
            if "optimizations" in kwargs:
                _optimizations = kwargs["optimizations"]

            compilation = compile_apl_from_belex(fn, _optimizations)
            apl = compilation.apl
            register_map = compilation.register_map
            args_by_reg_id = compilation.args_by_reg_id
            out_param = compilation.out_param

            # masks
            used_masks = collect_all_normalized_masks([
                x for x in apl if not isinstance(x, APL_comment)
            ])
            sm_registers = {
                used_mask: f'SM_{used_mask.upper()}'
                for used_mask in used_masks
            }

            nyms_by_reg_id = defaultdict(list)
            reg_id = register_map[out_param]
            nyms_by_reg_id[reg_id].append(out_param)
            for reg_nym, reg_id in register_map.items():
                if reg_nym != out_param:
                    nyms_by_reg_id[reg_id].append(reg_nym)

            # sbs
            sb_values = list(set(register_map.values()))
            rn_set = [f'_INTERNAL_RN_REG_T{i}' for i in range(NSB)]
            for sb_value in sb_values:
                nyms_by_reg_id[sb_value].append(rn_set[sb_value])

            rn_registers = {}
            for reg_id, reg_nyms in nyms_by_reg_id.items():
                for reg_nym in reg_nyms:
                    # Prefer non-temp nyms
                    if not reg_nym.startswith("t_"):
                        break
                    else:
                        reg_nym = f"_INTERNAL_{reg_nym}"
                rn_registers[reg_id] = reg_nym

            bleir_map = make_bleir_map()

            value_map = {}
            for reg_id, reg_nym in rn_registers.items():
                nyms = nyms_by_reg_id[reg_id]
                if len(nyms) > 1:
                    nyms = ", ".join(nyms)
                    comment = f"BELEX: {_frag_name}, arguments {nyms}"
                else:
                    nym = nyms[0]
                    comment = f"BELEX: {_frag_name}, argument {nym}"

                comment = InlineComment(comment)
                rn_reg_param = RN_REG(identifier=reg_nym,
                                      comment=comment)
                bleir_map[reg_nym] = rn_reg_param
                value_map[rn_reg_param] = reg_id

            rn_regs = rn_registers.values()
            sm_regs = sorted(sm_registers.values())
            # Reorder sm_registers according to the sorted sm_regs
            # NOTE: This is position-dependent, sm_registers must be sorted
            # before it is used such as in the for-each loop below. Moving the
            # sorting logic after sm_registers is used may cause the wrong
            # section mask values to be passed to the simulator.
            sm_str_vals_by_id = reversed_dict(sm_registers)
            sm_str_vals_by_id = OrderedDict((sm_reg, sm_str_vals_by_id[sm_reg])
                                            for sm_reg in sm_regs)
            sm_registers = reversed_dict(sm_str_vals_by_id)

            for sm_val, sm_reg in sm_registers.items():
                # comment = f"BELEX: {_frag_name}, pseudo-constant SM register"
                # comment = InlineComment(comment)
                comment = None
                sm_reg_param = SM_REG(identifier=f"_INTERNAL_{sm_reg}",
                                      comment=comment,
                                      constant_value=int(sm_val, 16))
                bleir_map[sm_reg] = sm_reg_param
                value_map[sm_reg_param] = sm_reg_param.constant_value

            all_regs = chain(rn_regs, sm_regs)
            formal_parameters = [bleir_map[reg] for reg in all_regs]
            examples = build_belex_examples(register_map, build_examples)
            formal_identifiers = collect_formal_identifiers(examples)
            formal_parameters = [formal_parameter
                                 for formal_parameter in formal_parameters
                                 if formal_parameter.identifier in formal_identifiers
                                 or formal_parameter.identifier.startswith("_INTERNAL")]
            formal_weights = defaultdict(lambda: len(formal_identifiers))
            for identifier in formal_identifiers:
                formal_weights[identifier] = len(formal_weights)
            formal_parameters = sorted(formal_parameters,
                                       key=lambda p: formal_weights[p.identifier])

            fragment = Fragment(
                identifier=_frag_name,
                parameters=formal_parameters,
                operations=[x.render_bleir(rn_registers,
                                           sm_registers,
                                           bleir_map)
                            for x in apl])

            args_by_reg_nym = {}
            for reg_id, args in args_by_reg_id.items():
                reg_nym = rn_registers[reg_id]
                args_by_reg_nym[reg_nym] = args

            fragment_caller = FragmentCaller(
                fragment=fragment,
                metadata={
                    CallerMetadata.IS_HIGH_LEVEL: True,
                    CallerMetadata.REGISTER_MAP: register_map,
                    CallerMetadata.BUILD_EXAMPLES: build_examples,
                    CallerMetadata.ARGS_BY_REG_NYM: args_by_reg_nym,
                    CallerMetadata.OUT_PARAM: out_param,
                    CallerMetadata.SHOULD_FAIL: should_fail,
                })

            return fragment_caller

        wrapper.__high_level_block__ = True

        if skip_tests:
            wrapper.__skip_tests__ = True

        return wrapper

    if callable(fn_or_frag_name):
        return decorator(fn_or_frag_name)

    return decorator
