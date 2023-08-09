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
from typing import Any, Dict, Optional, Sequence

from open_belex.bleir.analyzers import (DoubleNegativeAnalyzer,
                                        LiveParameterMarker)
from open_belex.bleir.types import (SRC_EXPR, UNARY_OP, UNARY_SRC,
                                    ActualParameter, AllocatedRegister,
                                    FormalParameter, Fragment, FragmentCaller,
                                    FragmentCallerCall, MultiStatement,
                                    Snippet)
from open_belex.bleir.walkables import (BLEIRListener, BLEIRTransformer,
                                        BLEIRWalker)


@dataclass
class ConvergenceOptimizer(BLEIRTransformer):
    """Introduces an optimization cycle into the pipeline that applies optimizations to the given
    context until it has been completely optimized (convergence) or the maximum number of
    optimization iterations is exceeded."""

    optimizers: Sequence[BLEIRTransformer]
    walker: BLEIRWalker

    max_passes: int = 100

    def transform_snippet(self: "ConvergenceOptimizer",
                          snippet: Snippet) -> Snippet:
        return self.optimize_until_convergence(snippet)

    def transform_fragment_caller_call(
            self: "ConvergenceOptimizer",
            fragment_caller_call: FragmentCallerCall) -> FragmentCallerCall:
        return self.optimize_until_convergence(fragment_caller_call)

    def single_pass(self: "ConvergenceOptimizer", target: Any) -> Any:
        for optimizer in self.optimizers:
            target = self.walker.walk(optimizer, target)
        return target

    def optimize_until_convergence(self: "ConvergenceOptimizer",
                                   target: Any) -> Any:

        source = target
        target = self.single_pass(target)

        num_passes = 1
        while source != target and num_passes < self.max_passes:
            source = target
            target = self.single_pass(target)
            num_passes += 1

        return target


INVERT_SRC_EXPR: Dict[SRC_EXPR, SRC_EXPR] = {
    SRC_EXPR.RL: SRC_EXPR.INV_RL,
    SRC_EXPR.NRL: SRC_EXPR.INV_NRL,
    SRC_EXPR.ERL: SRC_EXPR.INV_ERL,
    SRC_EXPR.WRL: SRC_EXPR.INV_WRL,
    SRC_EXPR.SRL: SRC_EXPR.INV_SRL,
    SRC_EXPR.GL: SRC_EXPR.INV_GL,
    SRC_EXPR.GGL: SRC_EXPR.INV_GGL,
    SRC_EXPR.RSP16: SRC_EXPR.INV_RSP16,

    SRC_EXPR.INV_RL: SRC_EXPR.RL,
    SRC_EXPR.INV_NRL: SRC_EXPR.NRL,
    SRC_EXPR.INV_ERL: SRC_EXPR.ERL,
    SRC_EXPR.INV_WRL: SRC_EXPR.WRL,
    SRC_EXPR.INV_SRL: SRC_EXPR.SRL,
    SRC_EXPR.INV_GL: SRC_EXPR.GL,
    SRC_EXPR.INV_GGL: SRC_EXPR.GGL,
    SRC_EXPR.INV_RSP16: SRC_EXPR.RSP16,
}


class DoubleNegativeResolver(BLEIRTransformer, BLEIRListener):
    negate_mask: bool = False
    should_transform: bool = True

    def enter_multi_statement(self: "DoubleNegativeResolver",
                              multi_statement: MultiStatement) -> None:
        double_negative_analyzer = DoubleNegativeAnalyzer()
        double_negative_analyzer.visit_multi_statement(multi_statement)
        self.should_transform = \
            double_negative_analyzer.resolve_multi_statement

    def exit_multi_statement(self: "DoubleNegativeResolver",
                             multi_statement: MultiStatement) -> None:
        self.should_transform = True

    def transform_unary_src(self: "DoubleNegativeResolver",
                            unary_src: UNARY_SRC) -> UNARY_SRC:
        if unary_src.operator is not UNARY_OP.NEGATE \
           or not self.should_transform:
            return unary_src
        return unary_src.having(
            expression=INVERT_SRC_EXPR[unary_src.expression],
            operator=None)


@dataclass
class UnusedParameterRemover(BLEIRTransformer, BLEIRListener):
    live_parameter_marker: LiveParameterMarker

    formal_parameters: Optional[Sequence[FormalParameter]] = None
    actual_parameters: Optional[Sequence[ActualParameter]] = None
    registers: Optional[Sequence[AllocatedRegister]] = None

    def enter_fragment_caller_call(
            self: "UnusedParameterRemover",
            fragment_caller_call: FragmentCallerCall) -> None:
        # Snapshot the formal parameters, actual parameters, and registers
        # before anything is transformed
        self.actual_parameters = fragment_caller_call.actual_parameters

    def exit_fragment_caller_call(
            self: "UnusedParameterRemover",
            fragment_caller_call: FragmentCallerCall) -> None:
        self.actual_parameters = None

    def transform_fragment_caller_call(
            self: "UnusedParameterRemover",
            fragment_caller_call: FragmentCallerCall) -> FragmentCallerCall:

        parameter_liveness = self.live_parameter_marker.parameter_liveness
        fragment = fragment_caller_call.fragment

        if fragment.identifier not in parameter_liveness:
            return fragment_caller_call

        parameter_is_live = parameter_liveness[fragment.identifier]

        actual_parameters = []
        for index, actual_parameter in enumerate(self.actual_parameters):
            # Loop over the snapshot parameters in case the Fragment is optimized first, which
            # would truncate the list of parameters in the FragmentCallerCall, yielding unreliable
            # behavior.
            if parameter_is_live[index]:
                actual_parameters.append(actual_parameter)

        return fragment_caller_call.having(parameters=actual_parameters)

    def enter_fragment_caller(
            self: "UnusedParameterRemover",
            fragment_caller: FragmentCaller) -> FragmentCaller:
        # Snapshot the formal parameters, actual parameters, and registers
        # before anything is transformed
        if fragment_caller.registers is not None:
            self.registers = fragment_caller.registers

    def exit_fragment_caller(
            self: "UnusedParameterRemover",
            fragment_caller: FragmentCaller) -> FragmentCaller:
        self.registers = None

    def transform_fragment_caller(
            self: "UnusedParameterRemover",
            fragment_caller: FragmentCaller) -> FragmentCaller:

        registers = None

        if self.registers is not None:
            parameter_liveness = self.live_parameter_marker.parameter_liveness
            fragment = fragment_caller.fragment

            if fragment.identifier not in parameter_liveness:
                return fragment_caller

            parameter_is_live = parameter_liveness[fragment.identifier]

            registers = []
            for index, register in enumerate(self.registers):
                if parameter_is_live[index]:
                    registers.append(register)

        return fragment_caller.having(registers=registers)

    def enter_fragment(self: "UnusedParameterRemover",
                       fragment: Fragment) -> Fragment:
        if fragment.children is None:
            self.formal_parameters = fragment.parameters

    def exit_fragment(self: "UnusedParameterRemover",
                      fragment: Fragment) -> Fragment:
        if fragment.children is None:
            self.formal_parameters = None

    def transform_fragment(self: "UnusedParameterRemover",
                           fragment: Fragment) -> Fragment:

        if fragment.children is not None:
            return fragment

        parameter_liveness = self.live_parameter_marker.parameter_liveness

        if fragment.identifier not in parameter_liveness:
            return fragment

        parameter_is_live = parameter_liveness[fragment.identifier]

        formal_parameters = []
        for index, formal_parameter in enumerate(self.formal_parameters):
            if parameter_is_live[index]:
                formal_parameters.append(formal_parameter)

        return fragment.having(parameters=formal_parameters)
