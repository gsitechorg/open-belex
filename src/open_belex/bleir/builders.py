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

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Optional, Sequence

from open_belex.bleir.types import (Call, Example, FragmentCaller,
                                    FragmentCallerCall, Snippet,
                                    SnippetMetadata)
from open_belex.bleir.virtual_machines import BLEIRVirtualMachine


def local_spill_restore(*attributes: Sequence[str]):

    def decorator(fn):

        @wraps(fn)
        def wrapper(self: "SnippetBuilder", *args, **kwargs):
            values = []
            for attribute in attributes:
                value = getattr(self.vm, attribute)
                values.append(value)
                if attribute not in kwargs or kwargs[attribute] is None:
                    kwargs[attribute] = value
                elif kwargs[attribute] is not None:
                    setattr(self.vm, attribute, kwargs[attribute])

            try:
                retval = fn(self, *args, **kwargs)
            finally:
                for attribute, value in zip(attributes, values):
                    setattr(self.vm, attribute, value)

            return retval

        return wrapper

    return decorator


@dataclass
class SnippetBuilder:
    vm: BLEIRVirtualMachine

    calls: Sequence[Call] = field(default_factory=list)

    @staticmethod
    def with_options(*args, **kwargs) -> "SnippetBuilder":
        vm = BLEIRVirtualMachine(*args, **kwargs)
        return SnippetBuilder(vm)

    def compile_and_append(
            self: "SnippetBuilder",
            fragment_caller_call: FragmentCallerCall) -> FragmentCallerCall:
        # Don't return the compiled call because the annotators may annotate it
        # twice (e.g. the instruction counter)
        # ---------------------------------------------------------------------
        # fragment_caller_call = self.vm.compile(fragment_caller_call)
        self.vm.compile(fragment_caller_call)  # necessary for @parameterized_belex_test
        return self.append(fragment_caller_call)

    def append(self: "SnippetBuilder", call: Call) -> Call:
        self.calls.append(call)
        return call

    @local_spill_restore("interpret", "generate_code", "output_dir")
    def build(self: "SnippetBuilder",
              name: str,
              examples: Sequence[Example],
              calls: Optional[Sequence[FragmentCallerCall]] = None,
              library_callers: Optional[Sequence[FragmentCaller]] = None,
              interpret: Optional[bool] = None,
              generate_code: Optional[bool] = None,
              output_dir: Optional[Path] = None) -> Snippet:

        if calls is None:
            calls = self.calls

        target = self.vm.target

        if target == "baryon":
            source_file = f"{name}-funcs.c"
            header_file = f"{name}-funcs.h"
        else:
            raise RuntimeError(f"Unsupported target: {target}")

        metadata = {
            SnippetMetadata.HEADER_FILE: header_file,
            SnippetMetadata.SOURCE_FILE: source_file,
            SnippetMetadata.TARGET: target,
        }

        snippet = Snippet(
            name=name,
            examples=examples,
            calls=calls,
            library_callers=library_callers,
            metadata=metadata)

        snippet = self.vm.compile(snippet)

        self.vm.assert_no_interpreter_failures()
        return snippet
