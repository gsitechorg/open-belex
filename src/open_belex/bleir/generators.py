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
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from itertools import chain
from pathlib import Path
from typing import ClassVar, Dict, Optional, Sequence, Set, Tuple, Type, Union

import open_belex.bleir.types as BLEIR
from open_belex.bleir.analyzers import (CommandDeclarationScanner,
                                        CountNumInstructionsAndCommands,
                                        RegisterParameterFinder)
from open_belex.bleir.commands import (ApplyPatch, Command, CommandScheduler,
                                       IncrementInstructions, LoadRegister,
                                       LoadRegisters, LoadSrc, MimicFSelNoOp,
                                       MimicGGLFromL1, MimicGGLFromRL,
                                       MimicGGLFromRLAndL1, MimicGLFromRL,
                                       MimicL1FromGGL, MimicL1FromLGL,
                                       MimicL2End, MimicL2FromLGL,
                                       MimicLGLFromL1, MimicLGLFromL2,
                                       MimicNoOp, MimicRLAndEqInvSB,
                                       MimicRLAndEqSB, MimicRLAndEqSBAndInvSrc,
                                       MimicRLAndEqSBAndSrc, MimicRLAndEqSrc,
                                       MimicRLFromInvSB,
                                       MimicRLFromInvSBAndInvSrc,
                                       MimicRLFromInvSBAndSrc,
                                       MimicRLFromInvSrc, MimicRLFromSB,
                                       MimicRLFromSBAndInvSrc,
                                       MimicRLFromSBAndSrc,
                                       MimicRLFromSBOrInvSrc,
                                       MimicRLFromSBOrSrc,
                                       MimicRLFromSBXorInvSrc,
                                       MimicRLFromSBXorSrc, MimicRLFromSrc,
                                       MimicRLOrEqInvSrc, MimicRLOrEqSB,
                                       MimicRLOrEqSBAndInvSrc,
                                       MimicRLOrEqSBAndSrc, MimicRLOrEqSrc,
                                       MimicRLXorEqInvSrc, MimicRLXorEqSB,
                                       MimicRLXorEqSBAndInvSrc,
                                       MimicRLXorEqSBAndSrc, MimicRLXorEqSrc,
                                       MimicRSP2KFromRSP32K,
                                       MimicRSP2KFromRSP256, MimicRSP16FromRL,
                                       MimicRSP16FromRSP256,
                                       MimicRSP32KFromRSP2K,
                                       MimicRSP256FromRSP2K,
                                       MimicRSP256FromRSP16, MimicRSPEnd,
                                       MimicRSPStartRet, MimicRWInhRst,
                                       MimicRWInhSet, MimicSBCondEqInvSrc,
                                       MimicSBCondEqSrc, MimicSBFromInvSrc,
                                       MimicSBFromSrc, MimicSetRL,
                                       RegisterKind, SetInPlace, UnifySMRegs)
from open_belex.bleir.inspectors import (Cardinality, Field, Kind,
                                         inspect_kind, is_bleir)
from open_belex.bleir.template_accessors import (AplTemplateAccessor,
                                                 BaryonTemplateAccessor,
                                                 MarkdownTemplateAccessor,
                                                 MesonTemplateAccessor)
from open_belex.bleir.types import (EWE_REG, L1_REG, L2_REG, RE_REG, RN_REG,
                                    SM_REG, SRC_EXPR, CallerMetadata,
                                    CFunctionCall, Fragment, FragmentCaller,
                                    FragmentCallerCall, MultiLineComment,
                                    SingleLineComment, Snippet,
                                    SnippetMetadata, TrailingComment)
from open_belex.bleir.walkables import BLEIRVisitor, camel_case_to_underscore
from open_belex.utils.path_utils import path_wrt_root

LOGGER = logging.getLogger()


@dataclass
class GvmlModuleGenerator(BLEIRVisitor):
    apl_template_accessor: AplTemplateAccessor

    # State variables
    reg_nyms_by_id: Optional[Dict[int, str]] = None

    def visit_snippet(self: "GvmlModuleGenerator", snippet: Snippet) -> str:
        example = snippet.examples[0]

        row_numbers = []
        reg_nyms = []
        self.reg_nyms_by_id = {}
        for value_parameter in example:
            reg_nym = value_parameter.identifier
            row_number = value_parameter.row_number
            self.reg_nyms_by_id[row_number] = reg_nym
            reg_nyms.append(reg_nym)
            row_numbers.append(row_number)

        calls = []

        for call in chain(snippet.initializers, snippet.body):
            if isinstance(call, FragmentCallerCall):
                call = self.visit_fragment_caller_call(call)
            elif isinstance(call, CFunctionCall):
                call = self.visit_c_function_call(call)
            else:
                raise ValueError(
                    f"Unsupported call type ({call.__class__.__name__}: {call}")
            calls.append(call)

        if snippet.fragment_caller_calls[-1].caller.has_metadata(CallerMetadata.IS_HIGH_LEVEL, True):
            prefix = "hlb"
        else:
            prefix = "llb"

        return self.apl_template_accessor.emit_gvml_app_module(
            name=snippet.name,
            header_file=str(snippet.header_file),
            calls=calls,
            reg_nyms=reg_nyms,
            row_numbers=row_numbers,
            prefix=prefix,
            target=snippet.target)

    def visit_c_function_call(
            self: "GvmlModuleGenerator",
            c_function_call: CFunctionCall) -> str:
        c_function = c_function_call.c_function
        identifier = c_function.identifier
        parameter_list = ", ".join(formal_parameter.identifier
                                   for formal_parameter
                                   in c_function.formal_parameters)
        return f"{identifier}({parameter_list});"

    def visit_fragment_caller_call(
            self: "GvmlModuleGenerator",
            fragment_caller_call: FragmentCallerCall) -> str:

        parameters = []

        parameter_map = fragment_caller_call.parameter_map
        for formal_parameter, actual_parameter in parameter_map.items():
            if isinstance(formal_parameter, (RN_REG, L1_REG, L2_REG)):
                if actual_parameter in self.reg_nyms_by_id:
                    reg_nym = self.reg_nyms_by_id[actual_parameter]
                    parameter = f"vreg_{reg_nym}"
                    parameters.append(parameter)
                else:
                    parameters.append(str(actual_parameter))
            elif isinstance(formal_parameter, RE_REG):
                parameters.append(f"0x{actual_parameter:06X}")
            elif isinstance(formal_parameter, EWE_REG):
                parameters.append(f"0x{actual_parameter:03X}")
            elif isinstance(formal_parameter, SM_REG):
                parameters.append(f"0x{actual_parameter:04X}")
            else:
                raise ValueError(
                    f"Unsupported formal_parameter type: "
                    f"{formal_parameter.__class__.__name__}")

        fragment_caller = fragment_caller_call.caller
        template_out = self.apl_template_accessor.emit_fragment_caller_call(
            identifier=fragment_caller.identifier,
            parameters=parameters)

        comment = fragment_caller_call.comment

        if comment is None:
            return template_out

        if isinstance(comment, MultiLineComment):
            return self.apl_template_accessor \
                       .emit_fragment_caller_call_with_multi_line_comment(
                           fragment_caller_call=template_out,
                           comments=comment.lines)

        if isinstance(comment, SingleLineComment):
            return self.apl_template_accessor \
                       .emit_fragment_caller_call_with_single_line_comment(
                           fragment_caller_call=template_out,
                           comment=comment.line)

        if isinstance(comment, TrailingComment):
            return self.apl_template_accessor \
                       .emit_fragment_caller_call_with_trailing_comment(
                           fragment_caller_call=template_out,
                           comment=comment.line)

        raise ValueError(
            f"Unsupported comment type: {comment.__class__.__name__}")


@dataclass
class FileWriter(ABC, BLEIRVisitor):
    output_dir: Path

    @abstractmethod
    def file_name_for(self: "FileWriter", snippet: Snippet) -> str:
        raise NotImplementedError

    @abstractmethod
    def file_body_for(self: "FileWriter", snippet: Snippet) -> str:
        raise NotImplementedError

    def write_to_file(self: "FileWriter",
                      file_name: Union[str, Path],
                      file_body: str) -> None:

        if isinstance(file_name, Path):
            output_dir = file_name.parent
            output_path = file_name
        else:
            output_dir = self.output_dir
            output_path = output_dir / file_name

        output_dir.mkdir(parents=True, exist_ok=True)

        file_body = f"{file_body}\n"

        if output_path.exists():
            with open(output_path, "rt") as f:
                if file_body == f.read():
                    LOGGER.info(f"No changes, skipping file: {output_path} ...")
                    return

        LOGGER.info(f"Writing to file: {output_path} ...")
        with open(output_path, "wt") as f:
            f.write(file_body)

    def visit_snippet(self: "FileWriter", snippet: Snippet) -> None:
        file_name = self.file_name_for(snippet)
        file_body = self.file_body_for(snippet)
        self.write_to_file(file_name, file_body)


@dataclass
class AplFileWriter(FileWriter):
    apl_template_accessor: AplTemplateAccessor


@dataclass
class MesonFileWriter(FileWriter):
    meson_template_accessor: MesonTemplateAccessor
    target: str = "baryon"

    def file_name_for(self: "MesonFileWriter", snippet: Snippet) -> str:
        return "meson.build"

    def file_body_for(self: "MesonFileWriter", snippet: Snippet) -> str:
        is_high_level = \
            snippet.fragment_caller_calls[-1] \
                   .caller.has_metadata(
                       key=CallerMetadata.IS_HIGH_LEVEL,
                       value=True)

        if is_high_level:
            prefix = "hlb"
        else:
            prefix = "llb"

        return self.meson_template_accessor.emit_meson_build(
            snippet_name=snippet.name,
            source_file=str(snippet.source_file),
            prefix=prefix,
            target=self.target)


@dataclass
class ConstantsFileWriter(AplFileWriter):

    def file_name_for(self: "ConstantsFileWriter", snippet: Snippet) -> str:
        return f"{snippet.name}-constants.h"

    def file_body_for(self: "ConstantsFileWriter", snippet: Snippet) -> str:
        return self.apl_template_accessor \
                   .emit_belex_constants(
                       name=snippet.name)


@dataclass
class UtilsSourceFileWriter(AplFileWriter):

    def file_name_for(self: "UtilsSourceFileWriter", snippet: Snippet) -> str:
        return f"{snippet.name}-utils.c"

    def file_body_for(self: "UtilsSourceFileWriter", snippet: Snippet) -> str:
        header_file = snippet.get_metadata(
            SnippetMetadata.HEADER_FILE,
            default_value=None)

        if isinstance(header_file, Path):
            header_file = header_file.name

        return self.apl_template_accessor \
                   .emit_belex_utils_source(
                       name=snippet.name,
                       header_file=header_file)


@dataclass
class UtilsHeaderFileWriter(AplFileWriter):

    def file_name_for(self: "UtilsHeaderFileWriter", snippet: Snippet) -> str:
        return f"{snippet.name}-utils.h"

    def file_body_for(self: "UtilsHeaderFileWriter", snippet: Snippet) -> str:
        return self.apl_template_accessor \
                   .emit_belex_utils_header(
                       name=snippet.name,
                       target=snippet.target)


@dataclass
class InternHeaderFileWriter(AplFileWriter):

    def file_name_for(self: "InternHeaderFileWriter", snippet: Snippet) -> str:
        return f"{snippet.name}-intern.h"

    def file_body_for(self: "InternHeaderFileWriter", snippet: Snippet) -> str:
        example = snippet.examples[0]
        reg_nyms = [value_param.identifier for value_param in example]
        return self.apl_template_accessor \
                   .emit_gvml_app_intern_header(
                       name=snippet.name,
                       reg_nyms=reg_nyms,
                       target=snippet.target)


@dataclass
class ExampleHeaderFileWriter(AplFileWriter):

    def file_name_for(self: "ExampleHeaderFileWriter",
                      snippet: Snippet) -> str:
        return f"{snippet.name}-examples.h"

    def file_body_for(self: "ExampleHeaderFileWriter",
                      snippet: Snippet) -> str:
        example = snippet.examples[0]
        reg_nyms = [value_param.identifier
                    for value_param in example]
        return self.apl_template_accessor \
                   .emit_belex_examples(
                       name=snippet.name,
                       examples=snippet.examples,
                       reg_nyms=reg_nyms)


@dataclass
class DeviceMainFileWriter(FileWriter):
    gvml_module_generator: GvmlModuleGenerator

    def file_name_for(self: "DeviceMainFileWriter",
                      snippet: Snippet) -> str:
        return f"{snippet.name}-module.c"

    def file_body_for(self: "DeviceMainFileWriter",
                      snippet: Snippet) -> str:
        return self.gvml_module_generator \
                   .visit_snippet(snippet)


@dataclass
class TestHeaderFileWriter(AplFileWriter):

    def file_name_for(self: "TestHeaderFileWriter",
                      snippet: Snippet) -> str:
        return f"test_{snippet.name}.h"

    def file_body_for(self: "TestHeaderFileWriter",
                      snippet: Snippet) -> str:
        return self.apl_template_accessor \
                   .emit_test_gvml_app_header(
                       name=snippet.name,
                       target=snippet.target)


@dataclass
class TestSourceFileWriter(AplFileWriter):
    count_num_instructions_and_commands: CountNumInstructionsAndCommands

    print_params: bool = False

    def file_name_for(self: "TestSourceFileWriter",
                      snippet: Snippet) -> str:
        return f"test_{snippet.name}.c"

    def file_body_for(self: "TestSourceFileWriter",
                      snippet: Snippet) -> str:

        num_instructions, num_commands = \
            self.count_num_instructions_and_commands(snippet)

        example = snippet.examples[0]
        reg_nyms = [value_param.identifier
                    for value_param in example]

        caller = snippet.callers[-1]
        should_fail = caller.get_metadata(CallerMetadata.SHOULD_FAIL,
                                          default_value=False)

        return self.apl_template_accessor \
                   .emit_test_gvml_app(
                       name=snippet.name,
                       examples=snippet.examples,
                       reg_nyms=reg_nyms,
                       should_fail=should_fail,
                       num_instructions=num_instructions,
                       num_commands=num_commands,
                       print_params=self.print_params,
                       target=snippet.target)


@dataclass
class HostMainFileWriter(AplFileWriter):
    count_num_instructions_and_commands: CountNumInstructionsAndCommands

    def file_name_for(self: "HostMainFileWriter",
                      snippet: Snippet) -> str:
        return f"test_{snippet.name}_main.c"

    def file_body_for(self: "HostMainFileWriter",
                      snippet: Snippet) -> str:

        num_instructions, num_commands = \
            self.count_num_instructions_and_commands(snippet)

        return self.apl_template_accessor \
                   .emit_test_gvml_app_main(
                       name=snippet.name,
                       num_instructions=num_instructions,
                       num_commands=num_commands,
                       target=snippet.target)


@dataclass
class GRAILGenerator(BLEIRVisitor):
    """Generates an EBNF from the given BLEIR type using GRAIL syntax.
    See: https://www2.cs.sfu.ca/~cameron/Teaching/384/99-3/GRAIL.html

    Example:
        ebnf_generator = GRAILGenerator()
        ebnf = ebnf_generator.visit_bleir(BLEIR.Snippet)"""

    rules: Dict[Type, str] = field(default_factory=dict)

    def visit_snippet(self: "GRAILGenerator", snippet: Snippet) -> str:
        return self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "GRAILGenerator", bleir: Type) -> str:
        kind = inspect_kind(bleir)
        self.visit_kind(kind)
        return "\n\n\n".join(self.rules.values())

    def visit_kind(self: "GRAILGenerator", kind: Kind) -> None:
        if kind.kind in self.rules:
            return self.rules[kind.kind]

        docs = kind.docs

        # if it is not a bleir type ...
        if len(kind.fields) == 0:
            union = "<see relevant docs>"
            rule = f"<{kind.camel_case_id}> ::= {union}"

        elif kind.kind.__class__ is EnumMeta:
            union = "\n    | ".join(f"<{field.identifier}:{self.visit_field(field)}>"
                               for field in kind.fields)
            rule = f"<{kind.camel_case_id}> ::=\n    ( {union} )"

            docs += f"\n\n{kind.kind.__name__} values:\n" + \
                "\n".join(f'    {1 + index}. {kind.kind.__name__}.{field.name} == "{field.value}"'
                          for index, field in enumerate(kind.kind))

        else:
            union = "\n    ".join(f"<{field.identifier}:{self.visit_field(field)}>"
                                  for field in kind.fields)
            rule = f"<{kind.camel_case_id}> ::=\n    {union}"

        if docs is not None:
            if is_bleir(kind.kind):
                visit_nym = f"visit_{kind.underscore_id}"
                transform_nym = f"transform_{kind.underscore_id}"
                enter_nym = f"enter_{kind.underscore_id}"
                exit_nym = f"exit_{kind.underscore_id}"
                docs += "\n".join([
                    "",
                    "",
                    "Relevant bleir.walkables methods:",
                    f"    1. BLEIRVisitor.{visit_nym}(",
                    f"           self: BLEIRVisitor,",
                    f"           {kind.underscore_id}: {kind.camel_case_id}) -> Any",
                    f"    2. BLEIRTransformer.{transform_nym}(",
                    f"           self: BLEIRTransformer,",
                    f"           {kind.underscore_id}: {kind.camel_case_id}) -> {kind.camel_case_id}",
                    f"    3. BLEIRListener.{enter_nym}(",
                    f"           self: BLEIRListener,",
                    f"           {kind.underscore_id}: {kind.camel_case_id}) -> None",
                    f"    4. BLEIRListener.{exit_nym}(",
                    f"           self: BLEIRListener,",
                    f"           {kind.underscore_id}: {kind.camel_case_id}) -> None",
                ])

            lines = docs.split("\n")
            docstring = "\n".join(f" * {line}" for line in lines)
            docstring = f"(**\n{docstring}\n *)"
            rule = f"{docstring}\n{rule}"

        self.rules[kind.kind] = rule

        for field in kind.fields:
            for kind in field.kinds:
                self.visit_kind(kind)

    def visit_field(self: "GRAILGenerator", field: Field) -> str:
        union = " | ".join(kind.camel_case_id for kind in field.kinds)
        if len(field.kinds) > 1:
            union = f"( {union} )"
        if field.cardinality is Cardinality.MANY:
            union = f"{union}*"
        return union


@dataclass
class ANTLR4Generator(BLEIRVisitor):
    """Generates an EBNF from the given BLEIR type using ANTLR4 syntax.

    Example:
        ebnf_generator = ANTLR4Generator()
        ebnf = ebnf_generator.visit_bleir(BLEIR.Snippet)"""

    rules: Dict[Type, str] = field(default_factory=dict)

    def visit_snippet(self: "ANTLR4Generator", snippet: Snippet) -> str:
        return self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "ANTLR4Generator", bleir: Type) -> str:
        kind = inspect_kind(bleir)
        self.visit_kind(kind)
        return "\n\n\n".join(self.rules.values())

    def visit_kind(self: "ANTLR4Generator", kind: Kind) -> None:
        if kind.kind in self.rules:
            return self.rules[kind.kind]

        docs = kind.docs

        # if it is not a bleir type ...
        if len(kind.fields) == 0:
            union = "/* see relevant docs */"
            rule = f"{kind.camel_case_id}: {union};"

        elif kind.kind.__class__ is EnumMeta:
            union = "\n    | ".join(f'{field.name}="{field.value}"'
                                    for field in kind.kind)
            rule = f"{kind.camel_case_id}:\n    ( {union} );"

        else:
            fields = []
            for field in kind.fields:
                field_grammar = self.visit_field(field)
                if field.cardinality is Cardinality.MANY:
                    field_grammar = f"( {field.identifier}+={field_grammar} )*"
                elif field.nullable:
                    field_grammar = f"( {field.identifier}={field_grammar} )?"
                else:
                    field_grammar = f"{field.identifier}={field_grammar}"
                fields.append(field_grammar)
            union = "\n    ".join(fields)
            rule = f"{kind.camel_case_id}:\n    {union};"

        if docs is not None:
            lines = docs.split("\n")

            # Determine the min number of leading spaces to drop from comment
            # lines following the first one.
            min_num_leading_spaces = -1
            for index in range(1, len(lines)):
                line = lines[index]

                num_leading_spaces = 0
                for c in line:
                    if c == ' ' or c == '\t':
                        num_leading_spaces += 1
                    else:
                        break

                # Only consider non-empty lines
                if num_leading_spaces == len(line):
                    continue

                if num_leading_spaces < min_num_leading_spaces \
                        or min_num_leading_spaces == -1:
                    min_num_leading_spaces = num_leading_spaces

            # Because of indentation, drop the leading spaces from the comment
            # lines following the first line.
            for index in range(1, len(lines)):
                line = lines[index]
                lines[index] = line[min_num_leading_spaces:]

            if is_bleir(kind.kind):
                visit_nym = f"visit_{kind.underscore_id}"
                transform_nym = f"transform_{kind.underscore_id}"
                enter_nym = f"enter_{kind.underscore_id}"
                exit_nym = f"exit_{kind.underscore_id}"
                lines += [
                    "",
                    "Relevant bleir.walkables methods:",
                    f"    1. BLEIRVisitor.{visit_nym}(",
                    f"           self: BLEIRVisitor,",
                    f"           {kind.underscore_id}: {kind.camel_case_id}) -> Any",
                    f"    2. BLEIRTransformer.{transform_nym}(",
                    f"           self: BLEIRTransformer,",
                    f"           {kind.underscore_id}: {kind.camel_case_id}) -> {kind.camel_case_id}",
                    f"    3. BLEIRListener.{enter_nym}(",
                    f"           self: BLEIRListener,",
                    f"           {kind.underscore_id}: {kind.camel_case_id}) -> None",
                    f"    4. BLEIRListener.{exit_nym}(",
                    f"           self: BLEIRListener,",
                    f"           {kind.underscore_id}: {kind.camel_case_id}) -> None",
                ]

            docstring = "\n".join(f"// {line}" if len(line) > 0 else "//"
                                  for line in lines)
            # docstring = f"/**\n{docstring}\n */"
            rule = f"{docstring}\n{rule}"

        self.rules[kind.kind] = rule

        for field in kind.fields:
            for kind in field.kinds:
                self.visit_kind(kind)

    def visit_field(self: "ANTLR4Generator", field: Field) -> str:
        kinds = [kind.camel_case_id for kind in field.kinds
                 if kind.kind is not None.__class__]
        union = " | ".join(kinds)
        if len(kinds) > 1:
            union = f"( {union} )"
        return union


@dataclass
class IndexMarkdownGenerator(BLEIRVisitor):
    markdown_accessor: MarkdownTemplateAccessor = field(default_factory=MarkdownTemplateAccessor)
    ebnf_generator: ANTLR4Generator = field(default_factory=ANTLR4Generator)

    def visit_snippet(self: "IndexMarkdownGenerator", snippet: Snippet) -> None:
        self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "IndexMarkdownGenerator", bleir: Type) -> None:
        grammar = self.ebnf_generator.visit_bleir(bleir)
        content = self.markdown_accessor.emit_bleir_index(grammar=grammar)
        with open(path_wrt_root("docs/bleir/index.md"), "wt") as f:
            f.write(content)
            f.write("\n")


@dataclass
class BLEIRVisitorGenerator(BLEIRVisitor):
    apl_template_accessor: AplTemplateAccessor = field(default_factory=AplTemplateAccessor)
    definitions: Dict[Type, str] = field(default_factory=dict)

    def visit_snippet(self: "BLEIRVisitorGenerator", snippet: Snippet) -> str:
        return self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "BLEIRVisitorGenerator", bleir: Type) -> str:
        kind = inspect_kind(bleir)
        self.visit_kind(kind)
        imports = [kind.__name__ for kind in self.definitions.keys()]
        return self.apl_template_accessor.emit_visitor(
            imports=imports,
            definitions=self.definitions.values())

    def visit_kind(self: "BLEIRVisitorGenerator", kind: Kind) -> None:
        if is_bleir(kind.kind):
            if kind.kind in self.definitions:
                return self.definitions[kind.kind]

            visit_fn = self.apl_template_accessor.emit_visit_fn_definition(
                camel_case_id=kind.camel_case_id,
                underscore_id=kind.underscore_id)

            self.definitions[kind.kind] = visit_fn

        for field in kind.fields:
            self.visit_field(field)

    def visit_field(self: "BLEIRVisitorGenerator", field: Field) -> str:
        for kind in field.kinds:
            self.visit_kind(kind)


@dataclass
class VisitorMarkdownGenerator(BLEIRVisitor):
    markdown_accessor: MarkdownTemplateAccessor = field(default_factory=MarkdownTemplateAccessor)
    visitor_generator: BLEIRVisitorGenerator = field(default_factory=BLEIRVisitorGenerator)

    def visit_snippet(self: "VisitorMarkdownGenerator", snippet: Snippet) -> None:
        self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "VisitorMarkdownGenerator", bleir: Type) -> None:
        pkginfo = BLEIR.__doc__
        visitor_definition = self.visitor_generator.visit_bleir(bleir)
        content = self.markdown_accessor.emit_bleir_visitor(
            visitor_definition=visitor_definition,
            pkginfo=pkginfo)
        with open(path_wrt_root("docs/bleir/visitor.md"), "wt") as f:
            f.write(content)
            f.write("\n")


@dataclass
class BLEIRListenerGenerator(BLEIRVisitor):
    apl_template_accessor: AplTemplateAccessor = field(default_factory=AplTemplateAccessor)
    definitions: Dict[Type, Tuple[str, str]] = field(default_factory=dict)

    def visit_snippet(self: "BLEIRListenerGenerator", snippet: Snippet) -> str:
        return self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "BLEIRListenerGenerator", bleir: Type) -> str:
        kind = inspect_kind(bleir)
        self.visit_kind(kind)
        imports = [kind.__name__ for kind in self.definitions.keys()]
        return self.apl_template_accessor.emit_listener(
            imports=imports,
            definitions=self.definitions.values())

    def visit_kind(self: "BLEIRListenerGenerator", kind: Kind) -> None:
        if is_bleir(kind.kind):
            if kind.kind in self.definitions:
                return self.definitions[kind.kind]

            enter_fn = self.apl_template_accessor.emit_enter_fn_definition(
                camel_case_id=kind.camel_case_id,
                underscore_id=kind.underscore_id)

            exit_fn = self.apl_template_accessor.emit_exit_fn_definition(
                camel_case_id=kind.camel_case_id,
                underscore_id=kind.underscore_id)

            self.definitions[kind.kind] = (enter_fn, exit_fn)

        for field in kind.fields:
            self.visit_field(field)

    def visit_field(self: "BLEIRListenerGenerator", field: Field) -> str:
        for kind in field.kinds:
            self.visit_kind(kind)


@dataclass
class ListenerMarkdownGenerator(BLEIRVisitor):
    markdown_accessor: MarkdownTemplateAccessor = field(default_factory=MarkdownTemplateAccessor)
    listener_generator: BLEIRListenerGenerator = field(default_factory=BLEIRListenerGenerator)

    def visit_snippet(self: "ListenerMarkdownGenerator", snippet: Snippet) -> None:
        self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "ListenerMarkdownGenerator", bleir: Type) -> None:
        pkginfo = BLEIR.__doc__
        listener_definition = self.listener_generator.visit_bleir(bleir)
        content = self.markdown_accessor.emit_bleir_listener(
            listener_definition=listener_definition,
            pkginfo=pkginfo)
        with open(path_wrt_root("docs/bleir/listener.md"), "wt") as f:
            f.write(content)
            f.write("\n")


@dataclass
class BLEIRTransformerGenerator(BLEIRVisitor):
    apl_template_accessor: AplTemplateAccessor = field(default_factory=AplTemplateAccessor)
    definitions: Dict[Type, str] = field(default_factory=dict)
    kind_id: Optional[str] = None
    kind: Optional[Type] = None

    def visit_snippet(self: "BLEIRTransformerGenerator", snippet: Snippet) -> str:
        return self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "BLEIRTransformerGenerator", bleir: Type) -> str:
        kind = inspect_kind(bleir)
        self.visit_kind(kind)
        imports = [kind.__name__ for kind in self.definitions.keys()]
        return self.apl_template_accessor.emit_transformer(
            imports=imports,
            definitions=self.definitions.values())

    def visit_kind(self: "BLEIRTransformerGenerator", kind: Kind) -> None:
        if is_bleir(kind.kind):
            if kind.kind in self.definitions:
                return self.definitions[kind.kind]

            calls = []
            field_ids = []
            self.is_enum = issubclass(kind.kind, Enum)
            self.kind_id = kind.underscore_id

            for field in kind.fields:
                call = self.visit_field(field)
                calls.append(call)
                field_ids.append(field.identifier)

            transform_fn = self.apl_template_accessor.emit_transform_fn_definition(
                camel_case_id=kind.camel_case_id,
                underscore_id=kind.underscore_id,
                calls=calls,
                field_ids=field_ids,
                is_enum=self.is_enum)

            self.definitions[kind.kind] = transform_fn

        for field in kind.fields:
            for kind in field.kinds:
                self.visit_kind(kind)

    def visit_field(self: "BLEIRTransformerGenerator", field: Field) -> Optional[str]:
        kinds = []

        if not self.is_enum:
            for kind in field.kinds:
                kinds.append((
                    kind.camel_case_id,
                    kind.underscore_id,
                    is_bleir(kind.kind),
                ))

        return self.apl_template_accessor.emit_transform_fn_call(
            kind_id=self.kind_id,
            field_id=field.identifier,
            kinds=kinds,
            cardinality=field.cardinality.value,
            nullable=field.nullable)


@dataclass
class TransformerMarkdownGenerator(BLEIRVisitor):
    markdown_accessor: MarkdownTemplateAccessor = field(default_factory=MarkdownTemplateAccessor)
    transformer_generator: BLEIRTransformerGenerator = field(default_factory=BLEIRTransformerGenerator)

    def visit_snippet(self: "TransformerMarkdownGenerator", snippet: Snippet) -> None:
        self.visit_bleir(snippet.__class__)

    def visit_bleir(self: "TransformerMarkdownGenerator", bleir: Type) -> None:
        pkginfo = BLEIR.__doc__
        transformer_definition = self.transformer_generator.visit_bleir(bleir)
        content = self.markdown_accessor.emit_bleir_transformer(
            transformer_definition=transformer_definition,
            pkginfo=pkginfo)
        with open(path_wrt_root("docs/bleir/transformer.md"), "wt") as f:
            f.write(content)
            f.write("\n")


@dataclass
class BaryonHeaderGenerator(BLEIRVisitor):
    baryon_template_accessor: BaryonTemplateAccessor
    apl_template_accessor: AplTemplateAccessor
    register_parameter_finder: RegisterParameterFinder

    explicit_frags_only: bool = False
    target: str = "baryon"

    frag_decls: Optional[Sequence[str]] = None

    def visit_snippet(self: "BaryonHeaderGenerator", snippet: Snippet) -> str:
        self.frag_decls = []
        for fragment in snippet.fragments:
            self.visit_fragment(fragment)
        lowered_registers = self.register_parameter_finder \
                                .lowered_registers_by_type()
        # baryon_header = self.baryon_template_accessor \
        #                     .emit_baryon_header(name=snippet.name,
        #                                         declarations=self.frag_decls)
        # return baryon_header
        caller_decls = list(map(self.visit_fragment_caller, snippet.callers))
        return self.apl_template_accessor.emit_apl_header(
            name=snippet.name,
            fragments=self.frag_decls,
            declarations=caller_decls,
            lowered_registers=lowered_registers,
            explicit_frags_only=self.explicit_frags_only,
            target=self.target)

    def visit_fragment_caller(self: "BaryonHeaderGenerator",
                              fragment_caller: FragmentCaller) -> str:

        doc_comment = None
        if fragment_caller.doc_comment is not None:
            doc_comment = fragment_caller.doc_comment.lines

        # Collect the formatted formal parameters for the BELEX caller
        # signature and the BELEX parameter names for use within the generated
        # function. The latter is not the same as the former because the former
        # includes type information as well as inline comments.
        formal_parameters = []
        for formal_parameter in fragment_caller.parameters:
            parameter = f"uint32_t {formal_parameter.identifier}"
            if formal_parameter.comment is not None:
                comment = formal_parameter.comment.value
                parameter = f"{parameter} /* {comment} */"
            formal_parameters.append(parameter)

        return self.apl_template_accessor.emit_belex_declaration(
            identifier=fragment_caller.identifier,
            parameters=formal_parameters,
            doc_comment=doc_comment,
            is_declaration=True)

    def visit_fragment(self: "BaryonHeaderGenerator",
                       fragment: Fragment) -> None:
        if fragment.children is not None:
            for child in fragment.children:
                self.visit_fragment(child)

        frag_nym = fragment.identifier
        parameters = []
        for formal_parameter in fragment.parameters:
            parameter = formal_parameter.identifier
            parameters.append(parameter)
        declaration = self.baryon_template_accessor \
                          .emit_function_declaration(frag_nym=frag_nym,
                                                     parameters=parameters)
        self.frag_decls.append(f"{declaration};")


@dataclass
class BaryonHeaderFileWriter(FileWriter):
    baryon_header_generator: BaryonHeaderGenerator

    def file_name_for(self: "BaryonHeaderFileWriter", snippet: Snippet) -> str:
        return snippet.header_file

    def file_body_for(self: "BaryonHeaderFileWriter", snippet: Snippet) -> str:
        return self.baryon_header_generator \
                   .visit_snippet(snippet)


@dataclass
class BaryonSourceGenerator(BLEIRVisitor):
    re_mimic_from_src: ClassVar[re.Pattern] = re.compile("(.*?_src).*")

    types_by_kind: ClassVar[Dict[str, str]] = {
        # Register types
        "sm": "baryon_sm_t",
        "rn": "size_t",
        "re": "baryon_re_t",
        "ewe": "baryon_ewe_t",
        "l1": "size_t",
        "l2": "size_t",

        # SRC types
        "rl": "baryon_rl_t *",
        "nrl": "baryon_rl_t *",
        "erl": "baryon_rl_t *",
        "wrl": "baryon_rl_t *",
        "srl": "baryon_rl_t *",

        "gl": "baryon_gl_t *",
        "ggl": "baryon_ggl_t *",
        "rsp16": "baryon_rsp16_t *",

        "inv_rl": "baryon_rl_t *",
        "inv_nrl": "baryon_rl_t *",
        "inv_erl": "baryon_rl_t *",
        "inv_wrl": "baryon_rl_t *",
        "inv_srl": "baryon_rl_t *",

        "inv_gl": "baryon_gl_t *",
        "inv_ggl": "baryon_ggl_t *",
        "inv_rsp16": "baryon_rsp16_t *",

        # Command types
        "f_sel_no_op": "void *",
        "ggl_from_l1": "baryon_ggl_t *",
        "ggl_from_rl": "baryon_ggl_t *",
        "ggl_from_rl_and_l1": "baryon_ggl_t *",
        "gl_from_rl": "baryon_gl_t *",
        "l1_from_ggl": "baryon_l1_patch_t *",
        "l1_from_lgl": "baryon_l1_patch_t *",
        "l2_end": "void *",
        "l2_from_lgl": "baryon_l2_patch_t *",
        "lgl_from_l1": "baryon_lgl_t *",
        "lgl_from_l2": "baryon_lgl_t *",
        "no_op": "void *",
        "rl_and_eq_inv_sb": "baryon_wordline_map_t *",
        "rl_and_eq_sb": "baryon_wordline_map_t *",
        "rl_and_eq_sb_and_inv_src": "baryon_wordline_map_t *",
        "rl_and_eq_sb_and_src": "baryon_wordline_map_t *",
        "rl_and_eq_src": "baryon_wordline_map_t *",
        "rl_from_inv_sb": "baryon_wordline_map_t *",
        "rl_from_inv_sb_and_inv_src": "baryon_wordline_map_t *",
        "rl_from_inv_sb_and_src": "baryon_wordline_map_t *",
        "rl_from_inv_src": "baryon_wordline_map_t *",
        "rl_from_sb": "baryon_wordline_map_t *",
        "rl_from_sb_and_inv_src": "baryon_wordline_map_t *",
        "rl_from_sb_and_src": "baryon_wordline_map_t *",
        "rl_from_sb_or_inv_src": "baryon_wordline_map_t *",
        "rl_from_sb_or_src": "baryon_wordline_map_t *",
        "rl_from_sb_xor_inv_src": "baryon_wordline_map_t *",
        "rl_from_sb_xor_src": "baryon_wordline_map_t *",
        "rl_from_src": "baryon_wordline_map_t *",
        "rl_or_eq_inv_src": "baryon_wordline_map_t *",
        "rl_or_eq_sb": "baryon_wordline_map_t *",
        "rl_or_eq_sb_and_inv_src": "baryon_wordline_map_t *",
        "rl_or_eq_sb_and_src": "baryon_wordline_map_t *",
        "rl_or_eq_src": "baryon_wordline_map_t *",
        "rl_xor_eq_inv_src": "baryon_wordline_map_t *",
        "rl_xor_eq_sb": "baryon_wordline_map_t *",
        "rl_xor_eq_sb_and_inv_src": "baryon_wordline_map_t *",
        "rl_xor_eq_sb_and_src": "baryon_wordline_map_t *",
        "rl_xor_eq_src": "baryon_wordline_map_t *",
        "rsp16_from_rl": "baryon_rsp16_section_map_t *",
        "rsp16_from_rsp256": "baryon_rsp16_t *",
        "rsp256_from_rsp16": "baryon_rsp256_t *",
        "rsp256_from_rsp2k": "baryon_rsp256_t *",
        "rsp2k_from_rsp256": "baryon_rsp2k_t *",
        "rsp2k_from_rsp32k": "baryon_rsp2k_t *",
        "rsp32k_from_rsp2k": "baryon_rsp32k_t *",
        "rsp_end": "baryon_rsp_patches_t *",
        "rsp_start_ret": "void *",
        "rw_inh_rst": "baryon_rwinh_rst_patch_t *",
        "rw_inh_set": "size_t",
        "sb_cond_eq_inv_src": "baryon_wordline_map_t *",
        "sb_cond_eq_src": "baryon_wordline_map_t *",
        "sb_from_inv_src": "baryon_wordline_map_t *",
        "sb_from_src": "baryon_wordline_map_t *",
        "set_rl": "baryon_wordline_map_t *",
    }

    patch_fns_by_kind: ClassVar[Dict[str, str]] = {
        "f_sel_no_op": "patch_noop",
        "ggl_from_l1": "baryon_patch_ggl",
        "ggl_from_rl": "baryon_patch_ggl",
        "ggl_from_rl_and_l1": "baryon_patch_ggl",
        "gl_from_rl": "baryon_patch_gl",
        "l1_from_ggl": "baryon_patch_l1",
        "l1_from_lgl": "baryon_patch_l1",
        "l2_end": "baryon_patch_l2_end",
        "l2_from_lgl": "baryon_patch_l2",
        "lgl_from_l1": "baryon_patch_lgl",
        "lgl_from_l2": "baryon_patch_lgl",
        "no_op": "baryon_patch_noop",
        "rl_and_eq_inv_sb": "baryon_patch_sb",
        "rl_and_eq_sb": "baryon_patch_sb",
        "rl_and_eq_sb_and_inv_src": "baryon_patch_sb",
        "rl_and_eq_sb_and_src": "baryon_patch_sb",
        "rl_and_eq_src": "baryon_patch_sb",
        "rl_from_inv_sb": "baryon_patch_sb",
        "rl_from_inv_sb_and_inv_src": "baryon_patch_sb",
        "rl_from_inv_sb_and_src": "baryon_patch_sb",
        "rl_from_inv_src": "baryon_patch_sb",
        "rl_from_sb": "baryon_patch_sb",
        "rl_from_sb_and_inv_src": "baryon_patch_sb",
        "rl_from_sb_and_src": "baryon_patch_sb",
        "rl_from_sb_or_inv_src": "baryon_patch_sb",
        "rl_from_sb_or_src": "baryon_patch_sb",
        "rl_from_sb_xor_inv_src": "baryon_patch_sb",
        "rl_from_sb_xor_src": "baryon_patch_sb",
        "rl_from_src": "baryon_patch_sb",
        "rl_or_eq_inv_src": "baryon_patch_sb",
        "rl_or_eq_sb": "baryon_patch_sb",
        "rl_or_eq_sb_and_inv_src": "baryon_patch_sb",
        "rl_or_eq_sb_and_src": "baryon_patch_sb",
        "rl_or_eq_src": "baryon_patch_sb",
        "rl_xor_eq_inv_src": "baryon_patch_sb",
        "rl_xor_eq_sb": "baryon_patch_sb",
        "rl_xor_eq_sb_and_inv_src": "baryon_patch_sb",
        "rl_xor_eq_sb_and_src": "baryon_patch_sb",
        "rl_xor_eq_src": "baryon_patch_sb",
        "rsp16_from_rl": "baryon_patch_partial_rsp16",
        "rsp16_from_rsp256": "baryon_patch_rsp16",
        "rsp256_from_rsp16": "baryon_patch_rsp256",
        "rsp256_from_rsp2k": "baryon_patch_rsp256",
        "rsp2k_from_rsp256": "baryon_patch_rsp2k",
        "rsp2k_from_rsp32k": "baryon_patch_rsp2k",
        "rsp32k_from_rsp2k": "baryon_patch_rsp32k",
        "rsp_end": "baryon_patch_rsps",
        "rsp_start_ret": "baryon_patch_rsp_start_ret",
        "rw_inh_rst": "baryon_patch_rwinh_rst",
        "rw_inh_set": "baryon_patch_rwinh_set",
        "sb_cond_eq_inv_src": "baryon_patch_sb",
        "sb_cond_eq_src": "baryon_patch_sb",
        "sb_from_inv_src": "baryon_patch_sb",
        "sb_from_src": "baryon_patch_sb",
        "set_rl": "baryon_patch_sb",
    }

    baryon_template_accessor: BaryonTemplateAccessor
    command_scheduler: CommandScheduler

    apl_template_accessor: AplTemplateAccessor
    register_parameter_finder: RegisterParameterFinder

    generate_belex_callers: bool = True
    explicit_frags_only: bool = False

    target: str = "baryon"

    command_declaration_scanner: Optional[CommandDeclarationScanner] = None
    definitions: Optional[Sequence[str]] = None
    loaded_registers: Optional[Set[Union[LoadRegister, UnifySMRegs]]] = None

    patches_by_command: Optional[Dict[Command, ApplyPatch]] = None

    loads_xe_reg: bool = False
    loads_vrs: bool = False
    loads_src: bool = False

    @staticmethod
    def free_fn_by_type(baryon_type: str) -> Optional[str]:
        if baryon_type.startswith("baryon_"):
            basename = baryon_type[len("baryon_"):-len("_t *")]
            return f"baryon_free_{basename}"
        return None

    def type_by_kind(self: "BaryonSourceGenerator", kind: str) -> str:
        match = self.re_mimic_from_src.fullmatch(kind)
        if match is not None:
            kind = match.group(1)
        return self.types_by_kind[kind]

    def patch_fn_by_kind(self: "BaryonSourceGenerator", kind: str) -> str:
        match = self.re_mimic_from_src.fullmatch(kind)
        if match is not None:
            kind = match.group(1)
        return self.patch_fns_by_kind[kind]

    def visit_snippet(self: "BaryonSourceGenerator", snippet: Snippet) -> str:
        self.target = snippet.target

        self.definitions = []
        for fragment in snippet.fragments:
            self.visit_fragment(fragment)

        header_file = snippet.get_metadata(
            SnippetMetadata.HEADER_FILE,
            default_value=None)

        if isinstance(header_file, Path):
            header_file = header_file.name

        # baryon_source = self.baryon_template_accessor \
        #                     .emit_baryon_source(name=snippet.name,
        #                                         definitions=self.definitions,
        #                                         header_file=header_file)
        # return baryon_source

        lowered_registers = self.register_parameter_finder \
                                .lowered_registers_by_type()

        if self.generate_belex_callers:
            callers = list(map(self.visit_fragment_caller, snippet.callers))
        else:
            callers = []

        return self.apl_template_accessor.emit_apl_source(
            name=snippet.name,
            callers=callers,
            header_file=header_file,
            lowered_registers=lowered_registers,
            definitions=self.definitions,
            explicit_frags_only=self.explicit_frags_only,
            target=snippet.target)

    def visit_fragment_caller(self: "BaryonSourceGenerator",
                              fragment_caller: FragmentCaller) -> str:

        doc_comment = None
        if fragment_caller.doc_comment is not None:
            doc_comment = fragment_caller.doc_comment.lines

        fragment = fragment_caller.fragment

        # Collect the formatted formal parameters for the BELEX caller
        # signature and the BELEX parameter names for use within the generated
        # function. The latter is not the same as the former because the former
        # includes type information as well as inline comments.
        formal_parameters = []
        actual_parameters = []
        for formal_parameter in fragment_caller.formal_parameters:
            parameter = f"uint32_t {formal_parameter.identifier}"

            if formal_parameter.comment is not None:
                comment = formal_parameter.comment.value
                parameter = f"{parameter} /* {comment} */"

            formal_parameters.append(parameter)
            actual_parameters.append(formal_parameter.identifier)

        aggregate_caller_id = fragment_caller.identifier

        if fragment.children is not None:
            fragment_ids = [child.identifier for child in fragment.children]
        else:
            fragment_ids = [fragment.identifier]

        registers = fragment_caller.registers

        initial_active_registers = \
            fragment_caller.get_metadata(
                CallerMetadata.INITIAL_ACTIVE_REGISTERS,
                default_value=None)

        if fragment_caller.initializers is not None:
            spill_calls = list(map(self.visit_fragment_caller_call,
                                   fragment_caller.initializers))
        else:
            spill_calls = []

        if fragment_caller.finalizers is not None:
            restore_calls = list(map(self.visit_fragment_caller_call,
                                     fragment_caller.finalizers))
        else:
            restore_calls = []

        return self.apl_template_accessor.emit_belex_caller(
            doc_comment=doc_comment,
            belex_caller_id=aggregate_caller_id,
            fragment_ids=fragment_ids,
            formal_parameters=formal_parameters,
            actual_parameters=actual_parameters,
            registers=registers,
            initial_active_registers=initial_active_registers,
            spill_calls=spill_calls,
            restore_calls=restore_calls,
            target=self.target)

    def visit_fragment_caller_call(
            self: "BaryonSourceGenerator",
            fragment_caller_call: FragmentCallerCall) -> str:
        return self.apl_template_accessor.emit_fragment_caller_call(
            identifier=fragment_caller_call.identifier,
            parameters=fragment_caller_call.actual_parameters)

    def is_patched(self: "BaryonSourceGenerator", command: Command) -> bool:
        return command in self.patches_by_command

    def visit_fragment(self: "BaryonSourceGenerator",
                       fragment: Fragment) -> None:

        if fragment.children is not None:
            for child in fragment.children:
                self.visit_fragment(child)

        frag_nym = fragment.identifier

        patches_by_command_by_frag_nym = \
            self.command_scheduler.patches_by_command_by_frag_nym
        if frag_nym not in patches_by_command_by_frag_nym:
            return

        self.loaded_registers = set()
        self.patches_by_command = patches_by_command_by_frag_nym[frag_nym]
        self.loads_xe_reg = False
        self.loads_vrs = False
        self.loads_src = False

        parameters = []
        for formal_parameter in fragment.parameters:
            parameter = formal_parameter.identifier
            parameters.append(parameter)

        commands = self.command_scheduler.commands_by_frag_nym[frag_nym]

        self.command_declaration_scanner = CommandDeclarationScanner()
        self.command_declaration_scanner.visit_commands(commands)
        command_declarations = \
            self.command_declaration_scanner.command_declarations

        decls_by_kind = defaultdict(list)
        for command, (kind, decl) in command_declarations.items():
            if self.is_patched(command) or \
               isinstance(command, (LoadRegister, UnifySMRegs)):
                decls_by_kind[kind].append(decl)

        declarations = []
        for kind in sorted(decls_by_kind.keys()):
            baryon_type = self.type_by_kind(kind)
            for variable in decls_by_kind[kind]:
                declaration = f"    {baryon_type} {variable};"
                declarations.append(declaration)

        instructions = []
        for command in commands:
            instruction = self.visit_command(command);
            if instruction is not None:
                instructions.append(instruction)

        definition = self.baryon_template_accessor \
                         .emit_function_definition(frag_nym=frag_nym,
                                                   parameters=parameters,
                                                   declarations=declarations,
                                                   instructions=instructions,
                                                   loads_xe_reg=self.loads_xe_reg,
                                                   loads_vrs=self.loads_vrs,
                                                   loads_src=self.loads_src)
        self.definitions.append(definition)

    def visit_command(self: "BaryonSourceGenerator",
                      command: Command) -> str:
        underscore_nym = camel_case_to_underscore(command.__class__)
        visit_nym = f"visit_{underscore_nym}"
        if hasattr(self, visit_nym):
            visit_fn = getattr(self, visit_nym)
            return visit_fn(command)

    def visit_set_in_place(
            self: "BaryonSourceGenerator",
            set_in_place: SetInPlace) -> str:
        return self.baryon_template_accessor \
                   .emit_set_in_place(in_place=set_in_place.in_place)

    def visit_apply_patch(
            self: "BaryonSourceGenerator",
            apply_patch: ApplyPatch) -> str:
        mimic = apply_patch.mimic
        patch_kind, patch_nym = self.command_declaration_scanner \
                                    .command_declarations[mimic]
        baryon_type = self.type_by_kind(patch_kind)
        patch_fn = self.patch_fn_by_kind(patch_kind)
        free_fn = self.free_fn_by_type(baryon_type)
        return self.baryon_template_accessor \
                   .emit_apply_patch(patch_nym=patch_nym,
                                     patch_fn=patch_fn,
                                     free_fn=free_fn)

    def visit_increment_instructions(
            self: "BaryonSourceGenerator",
            increment_instructions: IncrementInstructions) -> str:
        num_instructions = increment_instructions.num_instructions
        return self.baryon_template_accessor \
                   .emit_increment_instructions(
                       num_instructions=num_instructions)

    def visit_mimic_rsp16_from_rsp256(
            self: "BaryonSourceGenerator",
            mimic_rsp16_from_rsp256: MimicRSP16FromRSP256) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp16_from_rsp256]
        is_patched = self.is_patched(mimic_rsp16_from_rsp256)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp16_from_rsp256(patch_nym=patch_nym,
                                                 is_patched=is_patched)

    def visit_mimic_rsp256_from_rsp16(
            self: "BaryonSourceGenerator",
            mimic_rsp256_from_rsp16: MimicRSP256FromRSP16) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp256_from_rsp16]
        is_patched = self.is_patched(mimic_rsp256_from_rsp16)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp256_from_rsp16(patch_nym=patch_nym,
                                                 is_patched=is_patched)

    def visit_mimic_rsp256_from_rsp2k(
            self: "BaryonSourceGenerator",
            mimic_rsp256_from_rsp2k: MimicRSP256FromRSP2K) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp256_from_rsp2k]
        is_patched = self.is_patched(mimic_rsp256_from_rsp2k)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp256_from_rsp2k(patch_nym=patch_nym,
                                                 is_patched=is_patched)

    def visit_mimic_rsp2k_from_rsp256(
            self: "BaryonSourceGenerator",
            mimic_rsp2k_from_rsp256: MimicRSP2KFromRSP256) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp2k_from_rsp256]
        is_patched = self.is_patched(mimic_rsp2k_from_rsp256)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp2k_from_rsp256(patch_nym=patch_nym,
                                                 is_patched=is_patched)

    def visit_mimic_rsp2k_from_rsp32k(
            self: "BaryonSourceGenerator",
            mimic_rsp2k_from_rsp32k: MimicRSP2KFromRSP32K) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp2k_from_rsp32k]
        is_patched = self.is_patched(mimic_rsp2k_from_rsp32k)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp2k_from_rsp32k(patch_nym=patch_nym,
                                                 is_patched=is_patched)

    def visit_mimic_rsp32k_from_rsp2k(
            self: "BaryonSourceGenerator",
            mimic_rsp32k_from_rsp2k: MimicRSP32KFromRSP2K) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp32k_from_rsp2k]
        is_patched = self.is_patched(mimic_rsp32k_from_rsp2k)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp32k_from_rsp2k(patch_nym=patch_nym,
                                                 is_patched=is_patched)

    def visit_mimic_no_op(
            self: "BaryonSourceGenerator",
            mimic_no_op: MimicNoOp) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_no_op]
        is_patched = self.is_patched(mimic_no_op)
        return self.baryon_template_accessor \
                   .emit_mimic_no_op(patch_nym=patch_nym,
                                     is_patched=is_patched)

    def visit_mimic_f_sel_no_op(
            self: "BaryonSourceGenerator",
            mimic_f_sel_no_op: MimicFSelNoOp) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_f_sel_no_op]
        is_patched = self.is_patched(mimic_f_sel_no_op)
        return self.baryon_template_accessor \
                   .emit_mimic_f_sel_no_op(patch_nym=patch_nym,
                                           is_patched=is_patched)

    def visit_mimic_rsp_end(
            self: "BaryonSourceGenerator",
            mimic_rsp_end: MimicRSPEnd) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp_end]
        is_patched = self.is_patched(mimic_rsp_end)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp_end(patch_nym=patch_nym,
                                       is_patched=is_patched)

    def visit_mimic_rsp_start_ret(
            self: "BaryonSourceGenerator",
            mimic_rsp_start_ret: MimicRSPStartRet) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp_start_ret]
        is_patched = self.is_patched(mimic_rsp_start_ret)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp_start_ret(patch_nym=patch_nym,
                                             is_patched=is_patched)

    def visit_mimic_l2_end(
            self: "BaryonSourceGenerator",
            mimic_l2_end: MimicL2End) -> str:
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_l2_end]
        is_patched = self.is_patched(mimic_l2_end)
        return self.baryon_template_accessor \
                   .emit_mimic_l2_end(patch_nym=patch_nym,
                                      is_patched=is_patched)

    def visit_mimic_sb_from_src(
            self: "BaryonSourceGenerator",
            mimic_sb_from_src: MimicSBFromSrc) -> str:
        load_mask = self.visit_load_register(mimic_sb_from_src.mask)
        load_vrs = self.visit_load_registers(mimic_sb_from_src.vrs)
        load_src, free_src = self.visit_load_src(mimic_sb_from_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_sb_from_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_sb_from_src.mask]
        is_patched = self.is_patched(mimic_sb_from_src)
        return self.baryon_template_accessor \
                   .emit_mimic_sb_from_src(load_mask=load_mask,
                                           load_vrs=load_vrs,
                                           load_src=load_src,
                                           free_src=free_src,
                                           patch_nym=patch_nym,
                                           mask_nym=mask_nym,
                                           is_patched=is_patched)

    def visit_mimic_sb_from_inv_src(
            self: "BaryonSourceGenerator",
            mimic_sb_from_inv_src: MimicSBFromInvSrc) -> str:
        load_mask = self.visit_load_register(mimic_sb_from_inv_src.mask)
        load_vrs = self.visit_load_registers(mimic_sb_from_inv_src.vrs)
        load_src, free_src = self.visit_load_src(mimic_sb_from_inv_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_sb_from_inv_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_sb_from_inv_src.mask]
        is_patched = self.is_patched(mimic_sb_from_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_sb_from_inv_src(load_mask=load_mask,
                                               load_vrs=load_vrs,
                                               load_src=load_src,
                                               free_src=free_src,
                                               patch_nym=patch_nym,
                                               mask_nym=mask_nym,
                                               is_patched=is_patched)

    def visit_mimic_sb_cond_eq_src(
            self: "BaryonSourceGenerator",
            mimic_sb_cond_eq_src: MimicSBCondEqSrc) -> str:
        load_mask = self.visit_load_register(mimic_sb_cond_eq_src.mask)
        load_vrs = self.visit_load_registers(mimic_sb_cond_eq_src.vrs)
        load_src, free_src = self.visit_load_src(mimic_sb_cond_eq_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_sb_cond_eq_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_sb_cond_eq_src.mask]
        is_patched = self.is_patched(mimic_sb_cond_eq_src)
        return self.baryon_template_accessor \
                   .emit_mimic_sb_cond_eq_src(load_mask=load_mask,
                                              load_vrs=load_vrs,
                                              load_src=load_src,
                                              free_src=free_src,
                                              patch_nym=patch_nym,
                                              mask_nym=mask_nym,
                                              is_patched=is_patched)

    def visit_mimic_sb_cond_eq_inv_src(
            self: "BaryonSourceGenerator",
            mimic_sb_cond_eq_inv_src: MimicSBCondEqInvSrc) -> str:
        load_mask = self.visit_load_register(mimic_sb_cond_eq_inv_src.mask)
        load_vrs = self.visit_load_registers(mimic_sb_cond_eq_inv_src.vrs)
        load_src, free_src = self.visit_load_src(mimic_sb_cond_eq_inv_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_sb_cond_eq_inv_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_sb_cond_eq_inv_src.mask]
        is_patched = self.is_patched(mimic_sb_cond_eq_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_sb_cond_eq_inv_src(load_mask=load_mask,
                                                  load_vrs=load_vrs,
                                                  load_src=load_src,
                                                  free_src=free_src,
                                                  patch_nym=patch_nym,
                                                  mask_nym=mask_nym,
                                                  is_patched=is_patched)

    def visit_mimic_set_rl(
            self: "BaryonSourceGenerator",
            mimic_set_rl: MimicSetRL) -> str:
        load_mask = self.visit_load_register(mimic_set_rl.mask)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_set_rl]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_set_rl.mask]
        bit = mimic_set_rl.bit
        is_patched = self.is_patched(mimic_set_rl)
        return self.baryon_template_accessor \
                   .emit_mimic_set_rl(load_mask=load_mask,
                                      patch_nym=patch_nym,
                                      mask_nym=mask_nym,
                                      bit=bit,
                                      is_patched=is_patched)

    def visit_mimic_rl_from_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_src: MimicRLFromSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_src.mask)
        load_src, free_src = self.visit_load_src(mimic_rl_from_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_from_src.mask]
        is_patched = self.is_patched(mimic_rl_from_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_src(load_mask=load_mask,
                                           load_src=load_src,
                                           free_src=free_src,
                                           patch_nym=patch_nym,
                                           mask_nym=mask_nym,
                                           is_patched=is_patched)

    def visit_mimic_rl_from_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_inv_src: MimicRLFromInvSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_inv_src.mask)
        load_src, free_src = self.visit_load_src(mimic_rl_from_inv_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_inv_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_from_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_from_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_inv_src(load_mask=load_mask,
                                               load_src=load_src,
                                               free_src=free_src,
                                               patch_nym=patch_nym,
                                               mask_nym=mask_nym,
                                               is_patched=is_patched)

    def visit_mimic_rl_or_eq_src(
            self: "BaryonSourceGenerator",
            mimic_rl_or_eq_src: MimicRLOrEqSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_or_eq_src.mask)
        load_src, free_src = self.visit_load_src(mimic_rl_or_eq_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_or_eq_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_or_eq_src.mask]
        is_patched = self.is_patched(mimic_rl_or_eq_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_or_eq_src(load_mask=load_mask,
                                            load_src=load_src,
                                            free_src=free_src,
                                            patch_nym=patch_nym,
                                            mask_nym=mask_nym,
                                            is_patched=is_patched)

    def visit_mimic_rl_or_eq_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_or_eq_inv_src: MimicRLOrEqInvSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_or_eq_inv_src.mask)
        load_src, free_src = self.visit_load_src(mimic_rl_or_eq_inv_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_or_eq_inv_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_or_eq_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_or_eq_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_or_eq_inv_src(load_mask=load_mask,
                                                load_src=load_src,
                                                free_src=free_src,
                                                patch_nym=patch_nym,
                                                mask_nym=mask_nym,
                                                is_patched=is_patched)

    def visit_mimic_rl_and_eq_src(
            self: "BaryonSourceGenerator",
            mimic_rl_and_eq_src: MimicRLAndEqSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_and_eq_src.mask)
        load_src, free_src = self.visit_load_src(mimic_rl_and_eq_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_and_eq_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_and_eq_src.mask]
        is_patched = self.is_patched(mimic_rl_and_eq_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_and_eq_src(load_mask=load_mask,
                                             load_src=load_src,
                                             free_src=free_src,
                                             patch_nym=patch_nym,
                                             mask_nym=mask_nym,
                                             is_patched=is_patched)

    def visit_mimic_rl_xor_eq_src(
            self: "BaryonSourceGenerator",
            mimic_rl_xor_eq_src: MimicRLXorEqSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_xor_eq_src.mask)
        load_src, free_src = self.visit_load_src(mimic_rl_xor_eq_src.src)
        _, patch_nym = self.command_declaration_scanner \
                            .command_declarations[mimic_rl_xor_eq_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_xor_eq_src.mask]
        is_patched = self.is_patched(mimic_rl_xor_eq_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_xor_eq_src(load_mask=load_mask,
                                             load_src=load_src,
                                             free_src=free_src,
                                             patch_nym=patch_nym,
                                             mask_nym=mask_nym,
                                             is_patched=is_patched)

    def visit_mimic_rl_xor_eq_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_xor_eq_inv_src: MimicRLXorEqInvSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_xor_eq_inv_src.mask)
        load_src, free_src = self.visit_load_src(mimic_rl_xor_eq_inv_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_xor_eq_inv_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_xor_eq_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_xor_eq_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_xor_eq_inv_src(load_mask=load_mask,
                                                 load_src=load_src,
                                                 free_src=free_src,
                                                 patch_nym=patch_nym,
                                                 mask_nym=mask_nym,
                                                 is_patched=is_patched)

    def visit_mimic_rl_from_sb(
            self: "BaryonSourceGenerator",
            mimic_rl_from_sb: MimicRLFromSB) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_sb.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_sb.vrs)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_sb]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_from_sb.mask]
        is_patched = self.is_patched(mimic_rl_from_sb)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_sb(load_mask=load_mask,
                                          load_vrs=load_vrs,
                                          patch_nym=patch_nym,
                                          mask_nym=mask_nym,
                                          is_patched=is_patched)

    def visit_mimic_rl_from_inv_sb(
            self: "BaryonSourceGenerator",
            mimic_rl_from_inv_sb: MimicRLFromInvSB) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_inv_sb.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_inv_sb.vrs)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_inv_sb]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_from_inv_sb.mask]
        is_patched = self.is_patched(mimic_rl_from_inv_sb)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_inv_sb(load_mask=load_mask,
                                              load_vrs=load_vrs,
                                              patch_nym=patch_nym,
                                              mask_nym=mask_nym,
                                              is_patched=is_patched)

    def visit_mimic_rl_or_eq_sb(
            self: "BaryonSourceGenerator",
            mimic_rl_or_eq_sb: MimicRLOrEqSB) -> str:
        load_mask = self.visit_load_register(mimic_rl_or_eq_sb.mask)
        load_vrs = self.visit_load_registers(mimic_rl_or_eq_sb.vrs)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_or_eq_sb]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_or_eq_sb.mask]
        is_patched = self.is_patched(mimic_rl_or_eq_sb)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_or_eq_sb(load_mask=load_mask,
                                           load_vrs=load_vrs,
                                           patch_nym=patch_nym,
                                           mask_nym=mask_nym,
                                           is_patched=is_patched)

    def visit_mimic_rl_and_eq_sb(
            self: "BaryonSourceGenerator",
            mimic_rl_and_eq_sb: MimicRLAndEqSB) -> str:
        load_mask = self.visit_load_register(mimic_rl_and_eq_sb.mask)
        load_vrs = self.visit_load_registers(mimic_rl_and_eq_sb.vrs)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_and_eq_sb]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_and_eq_sb.mask]
        is_patched = self.is_patched(mimic_rl_and_eq_sb)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_and_eq_sb(load_mask=load_mask,
                                            load_vrs=load_vrs,
                                            patch_nym=patch_nym,
                                            mask_nym=mask_nym,
                                            is_patched=is_patched)

    def visit_mimic_rl_and_eq_inv_sb(
            self: "BaryonSourceGenerator",
            mimic_rl_and_eq_inv_sb: MimicRLAndEqInvSB) -> str:
        load_mask = self.visit_load_register(mimic_rl_and_eq_inv_sb.mask)
        load_vrs = self.visit_load_registers(mimic_rl_and_eq_inv_sb.vrs)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_and_eq_inv_sb]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_and_eq_inv_sb.mask]
        is_patched = self.is_patched(mimic_rl_and_eq_inv_sb)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_and_eq_inv_sb(load_mask=load_mask,
                                                load_vrs=load_vrs,
                                                patch_nym=patch_nym,
                                                mask_nym=mask_nym,
                                                is_patched=is_patched)

    def visit_mimic_rl_xor_eq_sb(
            self: "BaryonSourceGenerator",
            mimic_rl_xor_eq_sb: MimicRLXorEqSB) -> str:
        load_mask = self.visit_load_register(mimic_rl_xor_eq_sb.mask)
        load_vrs = self.visit_load_registers(mimic_rl_xor_eq_sb.vrs)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_xor_eq_sb]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_xor_eq_sb.mask]
        is_patched = self.is_patched(mimic_rl_xor_eq_sb)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_xor_eq_sb(load_mask=load_mask,
                                            load_vrs=load_vrs,
                                            patch_nym=patch_nym,
                                            mask_nym=mask_nym,
                                            is_patched=is_patched)

    def visit_mimic_rl_from_sb_and_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_sb_and_src: MimicRLFromSBAndSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_sb_and_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_sb_and_src.vrs)
        load_src, free_src = self.visit_load_src(mimic_rl_from_sb_and_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_sb_and_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_from_sb_and_src.mask]
        is_patched = self.is_patched(mimic_rl_from_sb_and_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_sb_and_src(load_mask=load_mask,
                                                  load_vrs=load_vrs,
                                                  load_src=load_src,
                                                  free_src=free_src,
                                                  patch_nym=patch_nym,
                                                  mask_nym=mask_nym,
                                                  is_patched=is_patched)

    def visit_mimic_rl_or_eq_sb_and_src(
            self: "BaryonSourceGenerator",
            mimic_rl_or_eq_sb_and_src: MimicRLOrEqSBAndSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_or_eq_sb_and_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_or_eq_sb_and_src.vrs)
        load_src, free_src = self.visit_load_src(mimic_rl_or_eq_sb_and_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_or_eq_sb_and_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_or_eq_sb_and_src.mask]
        is_patched = self.is_patched(mimic_rl_or_eq_sb_and_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_or_eq_sb_and_src(load_mask=load_mask,
                                                   load_vrs=load_vrs,
                                                   load_src=load_src,
                                                   free_src=free_src,
                                                   patch_nym=patch_nym,
                                                   mask_nym=mask_nym,
                                                   is_patched=is_patched)

    def visit_mimic_rl_or_eq_sb_and_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_or_eq_sb_and_inv_src: MimicRLOrEqSBAndInvSrc) -> str:
        load_mask = \
            self.visit_load_register(mimic_rl_or_eq_sb_and_inv_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_or_eq_sb_and_inv_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_or_eq_sb_and_inv_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_or_eq_sb_and_inv_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_or_eq_sb_and_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_or_eq_sb_and_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_or_eq_sb_and_inv_src(load_mask=load_mask,
                                                       load_vrs=load_vrs,
                                                       load_src=load_src,
                                                       free_src=free_src,
                                                       patch_nym=patch_nym,
                                                       mask_nym=mask_nym,
                                                       is_patched=is_patched)

    def visit_mimic_rl_and_eq_sb_and_src(
            self: "BaryonSourceGenerator",
            mimic_rl_and_eq_sb_and_src: MimicRLAndEqSBAndSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_and_eq_sb_and_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_and_eq_sb_and_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_and_eq_sb_and_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_and_eq_sb_and_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_and_eq_sb_and_src.mask]
        is_patched = self.is_patched(mimic_rl_and_eq_sb_and_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_and_eq_sb_and_src(load_mask=load_mask,
                                                    load_vrs=load_vrs,
                                                    load_src=load_src,
                                                    free_src=free_src,
                                                    patch_nym=patch_nym,
                                                    mask_nym=mask_nym,
                                                    is_patched=is_patched)

    def visit_mimic_rl_and_eq_sb_and_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_and_eq_sb_and_inv_src: MimicRLAndEqSBAndInvSrc) -> str:
        load_mask = \
            self.visit_load_register(mimic_rl_and_eq_sb_and_inv_src.mask)
        load_vrs = \
            self.visit_load_registers(mimic_rl_and_eq_sb_and_inv_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_and_eq_sb_and_inv_src.src)
        _, patch_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_and_eq_sb_and_inv_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_and_eq_sb_and_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_and_eq_sb_and_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_and_eq_sb_and_inv_src(load_mask=load_mask,
                                                        load_vrs=load_vrs,
                                                        load_src=load_src,
                                                        free_src=free_src,
                                                        patch_nym=patch_nym,
                                                        mask_nym=mask_nym,
                                                        is_patched=is_patched)

    def visit_mimic_rl_xor_eq_sb_and_src(
            self: "BaryonSourceGenerator",
            mimic_rl_xor_eq_sb_and_src: MimicRLXorEqSBAndSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_xor_eq_sb_and_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_xor_eq_sb_and_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_xor_eq_sb_and_src.src)
        _, patch_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_xor_eq_sb_and_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_xor_eq_sb_and_src.mask]
        is_patched = self.is_patched(mimic_rl_xor_eq_sb_and_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_xor_eq_sb_and_src(load_mask=load_mask,
                                                    load_vrs=load_vrs,
                                                    load_src=load_src,
                                                    free_src=free_src,
                                                    patch_nym=patch_nym,
                                                    mask_nym=mask_nym,
                                                    is_patched=is_patched)

    def visit_mimic_rl_xor_eq_sb_and_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_xor_eq_sb_and_inv_src: MimicRLXorEqSBAndInvSrc) -> str:
        load_mask = \
            self.visit_load_register(mimic_rl_xor_eq_sb_and_inv_src.mask)
        load_vrs = \
            self.visit_load_registers(mimic_rl_xor_eq_sb_and_inv_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_xor_eq_sb_and_inv_src.src)
        _, patch_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_xor_eq_sb_and_inv_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_xor_eq_sb_and_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_xor_eq_sb_and_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_xor_eq_sb_and_inv_src(load_mask=load_mask,
                                                        load_vrs=load_vrs,
                                                        load_src=load_src,
                                                        free_src=free_src,
                                                        patch_nym=patch_nym,
                                                        mask_nym=mask_nym,
                                                        is_patched=is_patched)

    def visit_mimic_rl_from_sb_or_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_sb_or_src: MimicRLFromSBOrSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_sb_or_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_sb_or_src.vrs)
        load_src, free_src = self.visit_load_src(mimic_rl_from_sb_or_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_sb_or_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_from_sb_or_src.mask]
        is_patched = self.is_patched(mimic_rl_from_sb_or_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_sb_or_src(load_mask=load_mask,
                                                 load_vrs=load_vrs,
                                                 load_src=load_src,
                                                 free_src=free_src,
                                                 patch_nym=patch_nym,
                                                 mask_nym=mask_nym,
                                                 is_patched=is_patched)

    def visit_mimic_rl_from_sb_or_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_sb_or_inv_src: MimicRLFromSBOrInvSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_sb_or_inv_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_sb_or_inv_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_from_sb_or_inv_src.src)
        _, patch_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_from_sb_or_inv_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_from_sb_or_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_from_sb_or_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_sb_or_inv_src(load_mask=load_mask,
                                                     load_vrs=load_vrs,
                                                     load_src=load_src,
                                                     free_src=free_src,
                                                     patch_nym=patch_nym,
                                                     mask_nym=mask_nym,
                                                     is_patched=is_patched)

    def visit_mimic_rl_from_sb_xor_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_sb_xor_src: MimicRLFromSBXorSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_sb_xor_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_sb_xor_src.vrs)
        load_src, free_src = self.visit_load_src(mimic_rl_from_sb_xor_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_sb_xor_src]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rl_from_sb_xor_src.mask]
        is_patched = self.is_patched(mimic_rl_from_sb_xor_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_sb_xor_src(load_mask=load_mask,
                                                  load_vrs=load_vrs,
                                                  load_src=load_src,
                                                  free_src=free_src,
                                                  patch_nym=patch_nym,
                                                  mask_nym=mask_nym,
                                                  is_patched=is_patched)

    def visit_mimic_rl_from_sb_xor_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_sb_xor_inv_src: MimicRLFromSBXorInvSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_sb_xor_inv_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_sb_xor_inv_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_from_sb_xor_inv_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_sb_xor_inv_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_from_sb_xor_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_from_sb_xor_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_sb_xor_inv_src(load_mask=load_mask,
                                                      load_vrs=load_vrs,
                                                      load_src=load_src,
                                                      free_src=free_src,
                                                      patch_nym=patch_nym,
                                                      mask_nym=mask_nym,
                                                      is_patched=is_patched)

    def visit_mimic_rl_from_inv_sb_and_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_inv_sb_and_src: MimicRLFromInvSBAndSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_inv_sb_and_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_inv_sb_and_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_from_inv_sb_and_src.src)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rl_from_inv_sb_and_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_from_inv_sb_and_src.mask]
        is_patched = self.is_patched(mimic_rl_from_inv_sb_and_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_inv_sb_and_src(load_mask=load_mask,
                                                      load_vrs=load_vrs,
                                                      load_src=load_src,
                                                      free_src=free_src,
                                                      patch_nym=patch_nym,
                                                      mask_nym=mask_nym,
                                                      is_patched=is_patched)

    def visit_mimic_rl_from_inv_sb_and_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_inv_sb_and_inv_src: MimicRLFromInvSBAndInvSrc) -> str:
        load_mask = \
            self.visit_load_register(mimic_rl_from_inv_sb_and_inv_src.mask)
        load_vrs = \
            self.visit_load_registers(mimic_rl_from_inv_sb_and_inv_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_from_inv_sb_and_inv_src.src)
        _, patch_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_from_inv_sb_and_inv_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_from_inv_sb_and_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_from_inv_sb_and_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_inv_sb_and_inv_src(load_mask=load_mask,
                                                          load_vrs=load_vrs,
                                                          load_src=load_src,
                                                          free_src=free_src,
                                                          patch_nym=patch_nym,
                                                          mask_nym=mask_nym,
                                                          is_patched=is_patched)

    def visit_mimic_rl_from_sb_and_inv_src(
            self: "BaryonSourceGenerator",
            mimic_rl_from_sb_and_inv_src: MimicRLFromSBAndInvSrc) -> str:
        load_mask = self.visit_load_register(mimic_rl_from_sb_and_inv_src.mask)
        load_vrs = self.visit_load_registers(mimic_rl_from_sb_and_inv_src.vrs)
        load_src, free_src = \
            self.visit_load_src(mimic_rl_from_sb_and_inv_src.src)
        _, patch_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_from_sb_and_inv_src]
        _, mask_nym = \
            self.command_declaration_scanner \
                .command_declarations[mimic_rl_from_sb_and_inv_src.mask]
        is_patched = self.is_patched(mimic_rl_from_sb_and_inv_src)
        return self.baryon_template_accessor \
                   .emit_mimic_rl_from_sb_and_inv_src(load_mask=load_mask,
                                                      load_vrs=load_vrs,
                                                      load_src=load_src,
                                                      free_src=free_src,
                                                      patch_nym=patch_nym,
                                                      mask_nym=mask_nym,
                                                      is_patched=is_patched)

    def visit_mimic_rsp16_from_rl(
            self: "BaryonSourceGenerator",
            mimic_rsp16_from_rl: MimicRSP16FromRL) -> str:
        if isinstance(mimic_rsp16_from_rl.mask, LoadRegister):
            load_mask = self.visit_load_register(mimic_rsp16_from_rl.mask)
        elif isinstance(mimic_rsp16_from_rl.mask, UnifySMRegs):
            load_mask = self.visit_unify_sm_regs(mimic_rsp16_from_rl.mask)
        else:
            raise ValueError(
                f"Unsupported mask type "
                f"({mimic_rsp16_from_rl.mask.__class__.__name__}): "
                f"{mimic_rsp16_from_rl.mask}")
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rsp16_from_rl]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rsp16_from_rl.mask]
        is_patched = self.is_patched(mimic_rsp16_from_rl)
        return self.baryon_template_accessor \
                   .emit_mimic_rsp16_from_rl(load_mask=load_mask,
                                             patch_nym=patch_nym,
                                             mask_nym=mask_nym,
                                             is_patched=is_patched)

    def visit_mimic_gl_from_rl(
            self: "BaryonSourceGenerator",
            mimic_gl_from_rl: MimicGLFromRL) -> str:
        if isinstance(mimic_gl_from_rl.mask, LoadRegister):
            load_mask = self.visit_load_register(mimic_gl_from_rl.mask)
        elif isinstance(mimic_gl_from_rl.mask, UnifySMRegs):
            load_mask = self.visit_unify_sm_regs(mimic_gl_from_rl.mask)
        else:
            raise ValueError(
                f"Unsupported mask type "
                f"({mimic_gl_from_rl.mask.__class__.__name__}): "
                f"{mimic_gl_from_rl.mask}")
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_gl_from_rl]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_gl_from_rl.mask]
        is_patched = self.is_patched(mimic_gl_from_rl)
        return self.baryon_template_accessor \
                   .emit_mimic_gl_from_rl(load_mask=load_mask,
                                          patch_nym=patch_nym,
                                          mask_nym=mask_nym,
                                          is_patched=is_patched)

    def visit_mimic_ggl_from_rl(
            self: "BaryonSourceGenerator",
            mimic_ggl_from_rl: MimicGGLFromRL) -> str:
        if isinstance(mimic_ggl_from_rl.mask, LoadRegister):
            load_mask = self.visit_load_register(mimic_ggl_from_rl.mask)
        elif isinstance(mimic_ggl_from_rl.mask, UnifySMRegs):
            load_mask = self.visit_unify_sm_regs(mimic_ggl_from_rl.mask)
        else:
            raise ValueError(
                f"Unsupported mask type "
                f"({mimic_ggl_from_rl.mask.__class__.__name__}): "
                f"{mimic_ggl_from_rl.mask}")
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_ggl_from_rl]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_ggl_from_rl.mask]
        is_patched = self.is_patched(mimic_ggl_from_rl)
        return self.baryon_template_accessor \
                   .emit_mimic_ggl_from_rl(load_mask=load_mask,
                                           patch_nym=patch_nym,
                                           mask_nym=mask_nym,
                                           is_patched=is_patched)

    def visit_mimic_l1_from_ggl(
            self: "BaryonSourceGenerator",
            mimic_l1_from_ggl: MimicL1FromGGL) -> str:
        load_l1 = self.visit_load_register(mimic_l1_from_ggl.l1_addr)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_l1_from_ggl]
        _, l1_nym = self.command_declaration_scanner \
                        .command_declarations[mimic_l1_from_ggl.l1_addr]
        is_patched = self.is_patched(mimic_l1_from_ggl)
        return self.baryon_template_accessor \
                   .emit_mimic_l1_from_ggl(load_l1=load_l1,
                                           patch_nym=patch_nym,
                                           l1_nym=l1_nym,
                                           is_patched=is_patched)

    def visit_mimic_lgl_from_l1(
            self: "BaryonSourceGenerator",
            mimic_lgl_from_l1: MimicLGLFromL1) -> str:
        load_l1 = self.visit_load_register(mimic_lgl_from_l1.l1_addr)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_lgl_from_l1]
        _, l1_nym = self.command_declaration_scanner \
                        .command_declarations[mimic_lgl_from_l1.l1_addr]
        is_patched = self.is_patched(mimic_lgl_from_l1)
        return self.baryon_template_accessor \
                   .emit_mimic_lgl_from_l1(load_l1=load_l1,
                                           patch_nym=patch_nym,
                                           l1_nym=l1_nym,
                                           is_patched=is_patched)

    def visit_mimic_l2_from_lgl(
            self: "BaryonSourceGenerator",
            mimic_l2_from_lgl: MimicL2FromLGL) -> str:
        load_l2 = self.visit_load_register(mimic_l2_from_lgl.l2_addr)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_l2_from_lgl]
        _, l2_nym = self.command_declaration_scanner \
                        .command_declarations[mimic_l2_from_lgl.l2_addr]
        is_patched = self.is_patched(mimic_l2_from_lgl)
        return self.baryon_template_accessor \
                   .emit_mimic_l2_from_lgl(load_l2=load_l2,
                                           patch_nym=patch_nym,
                                           l2_nym=l2_nym,
                                           is_patched=is_patched)

    def visit_mimic_lgl_from_l2(
            self: "BaryonSourceGenerator",
            mimic_lgl_from_l2: MimicLGLFromL2) -> str:
        load_l2 = self.visit_load_register(mimic_lgl_from_l2.l2_addr)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_lgl_from_l2]
        _, l2_nym = self.command_declaration_scanner \
                        .command_declarations[mimic_lgl_from_l2.l2_addr]
        is_patched = self.is_patched(mimic_lgl_from_l2)
        return self.baryon_template_accessor \
                   .emit_mimic_lgl_from_l2(load_l2=load_l2,
                                           patch_nym=patch_nym,
                                           l2_nym=l2_nym,
                                           is_patched=is_patched)

    def visit_mimic_l1_from_lgl(
            self: "BaryonSourceGenerator",
            mimic_l1_from_lgl: MimicL1FromLGL) -> str:
        load_l1 = self.visit_load_register(mimic_l1_from_lgl.l1_addr)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_l1_from_lgl]
        _, l1_nym = self.command_declaration_scanner \
                        .command_declarations[mimic_l1_from_lgl.l1_addr]
        is_patched = self.is_patched(mimic_l1_from_lgl)
        return self.baryon_template_accessor \
                   .emit_mimic_l1_from_lgl(load_l1=load_l1,
                                           patch_nym=patch_nym,
                                           l1_nym=l1_nym,
                                           is_patched=is_patched)

    def visit_mimic_ggl_from_l1(
            self: "BaryonSourceGenerator",
            mimic_ggl_from_l1: MimicGGLFromL1) -> str:
        load_l1 = self.visit_load_register(mimic_ggl_from_l1.l1_addr)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_ggl_from_l1]
        _, l1_nym = self.command_declaration_scanner \
                        .command_declarations[mimic_ggl_from_l1.l1_addr]
        is_patched = self.is_patched(mimic_ggl_from_l1)
        return self.baryon_template_accessor \
                   .emit_mimic_ggl_from_l1(load_l1=load_l1,
                                           patch_nym=patch_nym,
                                           l1_nym=l1_nym,
                                           is_patched=is_patched)

    def visit_mimic_ggl_from_rl_and_l1(
            self: "BaryonSourceGenerator",
            mimic_ggl_from_rl_and_l1: MimicGGLFromRLAndL1) -> str:
        if isinstance(mimic_ggl_from_rl_and_l1.mask, LoadRegister):
            load_mask = self.visit_load_register(mimic_ggl_from_rl_and_l1.mask)
        elif isinstance(mimic_ggl_from_rl_and_l1.mask, UnifySMRegs):
            load_mask = self.visit_unify_sm_regs(mimic_ggl_from_rl_and_l1.mask)
        else:
            raise ValueError(
                f"Unsupported mask type "
                f"({mimic_ggl_from_rl_and_l1.mask.__class__.__name__}): "
                f"{mimic_ggl_from_rl_and_l1.mask}")
        load_l1 = self.visit_load_register(mimic_ggl_from_rl_and_l1.l1_addr)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_ggl_from_rl_and_l1]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_ggl_from_rl_and_l1.mask]
        _, l1_nym = self.command_declaration_scanner \
                        .command_declarations[mimic_ggl_from_rl_and_l1.l1_addr]
        is_patched = self.is_patched(mimic_ggl_from_rl_and_l1)
        return self.baryon_template_accessor \
                   .emit_mimic_ggl_from_rl_and_l1(load_mask=load_mask,
                                                  load_l1=load_l1,
                                                  patch_nym=patch_nym,
                                                  mask_nym=mask_nym,
                                                  l1_nym=l1_nym,
                                                  is_patched=is_patched)

    def visit_mimic_rw_inh_set(
            self: "BaryonSourceGenerator",
            mimic_rw_inh_set: MimicRWInhSet) -> str:
        load_mask = self.visit_load_register(mimic_rw_inh_set.mask)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rw_inh_set]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rw_inh_set.mask]
        is_patched = self.is_patched(mimic_rw_inh_set)
        return self.baryon_template_accessor \
                   .emit_mimic_rw_inh_set(load_mask=load_mask,
                                          patch_nym=patch_nym,
                                          mask_nym=mask_nym,
                                          is_patched=is_patched)

    def visit_mimic_rw_inh_rst(
            self: "BaryonSourceGenerator",
            mimic_rw_inh_rst: MimicRWInhRst) -> str:
        load_mask = self.visit_load_register(mimic_rw_inh_rst.mask)
        _, patch_nym = self.command_declaration_scanner \
                           .command_declarations[mimic_rw_inh_rst]
        _, mask_nym = self.command_declaration_scanner \
                          .command_declarations[mimic_rw_inh_rst.mask]
        is_patched = self.is_patched(mimic_rw_inh_rst)
        return self.baryon_template_accessor \
                   .emit_mimic_rw_inh_rst(load_mask=load_mask,
                                          patch_nym=patch_nym,
                                          mask_nym=mask_nym,
                                          has_read=mimic_rw_inh_rst.has_read,
                                          is_patched=is_patched)

    def visit_unify_sm_regs(self: "BaryonSourceGenerator",
                            unify_sm_regs: UnifySMRegs) -> Optional[str]:
        if unify_sm_regs in self.loaded_registers:
            return None
        self.loaded_registers.add(unify_sm_regs)
        loaded_registers = []
        for load_register in unify_sm_regs:
            loaded_register = self.visit_load_register(load_register)
            if loaded_register is not None:
                loaded_registers.append(loaded_register)
        reg_nyms = []
        for load_register in unify_sm_regs:
            _, reg_nym = self.command_declaration_scanner \
                             .command_declarations[load_register]
            reg_nyms.append(reg_nym)
        _, unify_nym = self.command_declaration_scanner \
                           .command_declarations[unify_sm_regs]
        unify_cmd = self.baryon_template_accessor \
                        .emit_unify_sm_regs(unify_nym=unify_nym,
                                            reg_nyms=reg_nyms)
        loaded_registers.append(unify_cmd)
        return "\n".join(loaded_registers)

    def visit_load_registers(self: "BaryonSourceGenerator",
                             load_registers: LoadRegisters) -> str:

        loaded_registers = []
        for load_register in load_registers:
            loaded_register = self.visit_load_register(load_register)
            if loaded_register is not None:
                loaded_registers.append(loaded_register)

        if len(load_registers) > 1 \
           or load_registers[0].kind is RegisterKind.RN_REG:
            reg_nyms = []
            for load_register in load_registers:
                _, reg_nym = self.command_declaration_scanner \
                                 .command_declarations[load_register]
                reg_nyms.append(reg_nym)
            load_vrs = self.baryon_template_accessor \
                           .emit_load_vrs(reg_nyms=reg_nyms)
            loaded_registers.append(load_vrs)

        return "\n".join(loaded_registers)

    def visit_load_register(self: "BaryonSourceGenerator",
                            load_register: LoadRegister) -> Optional[str]:

        if load_register in self.loaded_registers:
            return None

        self.loaded_registers.add(load_register)

        reg_kind, reg_nym = self.command_declaration_scanner \
                                .command_declarations[load_register]

        if load_register.kind is RegisterKind.SM_REG:
            return self.baryon_template_accessor \
                       .emit_load_sm_reg(reg_nym=reg_nym,
                                         reg_id=load_register.symbol,
                                         shift_width=load_register.shift_width,
                                         invert=load_register.invert)

        if load_register.kind is RegisterKind.RN_REG:
            self.loads_vrs = True
            return self.baryon_template_accessor \
                       .emit_load_rn_reg(reg_nym=reg_nym,
                                         reg_id=load_register.symbol)

        if load_register.kind is RegisterKind.RE_REG:
            self.loads_xe_reg = True
            self.loads_vrs = True
            return self.baryon_template_accessor \
                       .emit_load_re_reg(reg_nym=reg_nym,
                                         reg_id=load_register.symbol,
                                         shift_width=load_register.shift_width,
                                         invert=load_register.invert)

        if load_register.kind is RegisterKind.EWE_REG:
            self.loads_xe_reg = True
            self.loads_vrs = True
            return self.baryon_template_accessor \
                       .emit_load_ewe_reg(reg_nym=reg_nym,
                                          reg_id=load_register.symbol,
                                          shift_width=load_register.shift_width,
                                          invert=load_register.invert)

        if load_register.kind is RegisterKind.L1_REG:
            return self.baryon_template_accessor \
                       .emit_load_l1_reg(reg_nym=reg_nym,
                                         reg_id=load_register.symbol,
                                         offset=load_register.offset)

        if load_register.kind is RegisterKind.L2_REG:
            return self.baryon_template_accessor \
                       .emit_load_l2_reg(reg_nym=reg_nym,
                                         reg_id=load_register.symbol,
                                         offset=load_register.offset)

        raise NotImplementedError

    def visit_load_src(self: "BaryonSourceGenerator",
                       load_src: LoadSrc) -> Tuple[str, Optional[str]]:
        src_kind, _ = self.command_declaration_scanner \
                          .command_declarations[load_src]
        self.loads_src = True

        if load_src.src is SRC_EXPR.RL:
            load_cmd = self.baryon_template_accessor.emit_load_rl()
        elif load_src.src is SRC_EXPR.NRL:
            load_cmd = self.baryon_template_accessor.emit_load_nrl()
        elif load_src.src is SRC_EXPR.ERL:
            load_cmd = self.baryon_template_accessor.emit_load_erl()
        elif load_src.src is SRC_EXPR.WRL:
            load_cmd = self.baryon_template_accessor.emit_load_wrl()
        elif load_src.src is SRC_EXPR.SRL:
            load_cmd = self.baryon_template_accessor.emit_load_srl()
        elif load_src.src is SRC_EXPR.INV_RL:
            load_cmd = self.baryon_template_accessor.emit_load_inv_rl()
        elif load_src.src is SRC_EXPR.INV_NRL:
            load_cmd = self.baryon_template_accessor.emit_load_inv_nrl()
        elif load_src.src is SRC_EXPR.INV_ERL:
            load_cmd = self.baryon_template_accessor.emit_load_inv_erl()
        elif load_src.src is SRC_EXPR.INV_WRL:
            load_cmd = self.baryon_template_accessor.emit_load_inv_wrl()
        elif load_src.src is SRC_EXPR.INV_SRL:
            load_cmd = self.baryon_template_accessor.emit_load_inv_srl()
        elif load_src.src is SRC_EXPR.GL:
            load_cmd = self.baryon_template_accessor.emit_load_gl()
        elif load_src.src is SRC_EXPR.GGL:
            load_cmd = self.baryon_template_accessor.emit_load_ggl()
        elif load_src.src is SRC_EXPR.RSP16:
            load_cmd = self.baryon_template_accessor.emit_load_rsp16()
        elif load_src.src is SRC_EXPR.INV_GL:
            load_cmd = self.baryon_template_accessor.emit_load_inv_gl()
        elif load_src.src is SRC_EXPR.INV_GGL:
            load_cmd = self.baryon_template_accessor.emit_load_inv_ggl()
        elif load_src.src is SRC_EXPR.INV_RSP16:
            load_cmd = self.baryon_template_accessor.emit_load_inv_rsp16()
        else:
            raise NotImplementedError

        free_cmd = None
        # NOTE: This included all INV_SRCs but overlooked [NEWS]_RL. I'm
        # switching to static allocation for SRC variants so this is no longer
        # needed, but may become useful in the future.
        # --------------------------------------------------------------------
        # if load_src.src.name.startswith("INV_"):
        #     src_type = self.type_by_kind(src_kind)
        #     free_cmd = self.free_fn_by_type(src_type)

        return load_cmd, free_cmd


@dataclass
class BaryonSourceFileWriter(FileWriter):
    baryon_source_generator: BaryonSourceGenerator

    def file_name_for(self: "BaryonSourceFileWriter", snippet: Snippet) -> str:
        return snippet.source_file

    def file_body_for(self: "BaryonSourceFileWriter", snippet: Snippet) -> str:
        return self.baryon_source_generator \
                   .visit_snippet(snippet)
