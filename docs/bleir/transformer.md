```python
import builtins

from BLEIR import (
    Snippet,
    Example,
    ValueParameter,
    FragmentCallerCall,
    FragmentCaller,
    Fragment,
    RN_REG,
    InlineComment,
    RE_REG,
    EWE_REG,
    L1_REG,
    L2_REG,
    SM_REG,
    MultiStatement,
    STATEMENT,
    MASKED,
    MASK,
    SHIFTED_SM_REG,
    UNARY_OP,
    ReadWriteInhibit,
    ASSIGNMENT,
    READ,
    ASSIGN_OP,
    UNARY_EXPR,
    UNARY_SB,
    SB_EXPR,
    UNARY_SRC,
    SRC_EXPR,
    BIT_EXPR,
    BINARY_EXPR,
    BINOP,
    RL_EXPR,
    LXRegWithOffsets,
    MultiLineComment,
    SingleLineComment,
    TrailingComment,
    WRITE,
    BROADCAST,
    BROADCAST_EXPR,
    RSP16_ASSIGNMENT,
    RSP16_RVALUE,
    RSP256_ASSIGNMENT,
    RSP256_RVALUE,
    RSP2K_ASSIGNMENT,
    RSP2K_RVALUE,
    RSP32K_ASSIGNMENT,
    RSP32K_RVALUE,
    SPECIAL,
    GGL_ASSIGNMENT,
    LGL_ASSIGNMENT,
    LX_ASSIGNMENT,
    GGL_EXPR,
    LGL_EXPR,
    GlassStatement,
    RSP256_EXPR,
    RSP2K_EXPR,
    RSP32K_EXPR,
    GlassFormat,
    GlassOrder,
    FragmentMetadata,
    AllocatedRegister,
    CallerMetadata,
    CallMetadata,
    SnippetMetadata,
    SyntacticError,
)

class BLEIRTransformer:

    def transform_snippet(self: "BLEIRTransformer", snippet: Snippet) -> Snippet:
        if not isinstance(snippet, Snippet):
            raise SyntacticError(f"Unsupported snippet type ({snippet.__class__.__name__}): {snippet}")

        if isinstance(snippet.name, str):
            name = snippet.name
        else:
            raise SyntacticError(f"Unsupported name type ({snippet.name.__class__.__name__}): {snippet.name}")

        examples = []
        for example in snippet.examples:
            if isinstance(example, Example):
                example = self.visit_example(example)
            else:
                raise SyntacticError(f"Unsupported examples type ({example.__class__.__name__}): {example}")
            examples.append(example)

        calls = []
        for call in snippet.calls:
            if isinstance(call, FragmentCallerCall):
                call = self.visit_fragment_caller_call(call)
            elif isinstance(call, MultiLineComment):
                call = self.visit_multi_line_comment(call)
            elif isinstance(call, SingleLineComment):
                call = self.visit_single_line_comment(call)
            else:
                raise SyntacticError(f"Unsupported calls type ({call.__class__.__name__}): {call}")
            calls.append(call)

        if isinstance(snippet.doc_comment, MultiLineComment):
            doc_comment = self.visit_multi_line_comment(snippet.doc_comment)
        elif isinstance(snippet.doc_comment, NoneType):
            doc_comment = snippet.doc_comment
        elif snippet.doc_comment is None:
            doc_comment = None
        else:
            raise SyntacticError(f"Unsupported doc_comment type ({snippet.doc_comment.__class__.__name__}): {snippet.doc_comment}")

        if isinstance(snippet.metadata, SnippetMetadata):
            metadata = self.visit_snippet_metadata(snippet.metadata)
        elif isinstance(snippet.metadata, Any):
            metadata = snippet.metadata
        elif isinstance(snippet.metadata, NoneType):
            metadata = snippet.metadata
        elif snippet.metadata is None:
            metadata = None
        else:
            raise SyntacticError(f"Unsupported metadata type ({snippet.metadata.__class__.__name__}): {snippet.metadata}")

        if isinstance(snippet.library_callers, FragmentCaller):
            library_callers = self.visit_fragment_caller(snippet.library_callers)
        elif isinstance(snippet.library_callers, NoneType):
            library_callers = snippet.library_callers
        elif snippet.library_callers is None:
            library_callers = None
        else:
            raise SyntacticError(f"Unsupported library_callers type ({snippet.library_callers.__class__.__name__}): {snippet.library_callers}")

        return Snippet(
            name=name,
            examples=examples,
            calls=calls,
            doc_comment=doc_comment,
            metadata=metadata,
            library_callers=library_callers)

    def transform_example(self: "BLEIRTransformer", example: Example) -> Example:
        if not isinstance(example, Example):
            raise SyntacticError(f"Unsupported example type ({example.__class__.__name__}): {example}")

        if isinstance(example.expected_value, ValueParameter):
            expected_value = self.visit_value_parameter(example.expected_value)
        else:
            raise SyntacticError(f"Unsupported expected_value type ({example.expected_value.__class__.__name__}): {example.expected_value}")

        parameters = []
        for parameter in example.parameters:
            if isinstance(parameter, ValueParameter):
                parameter = self.visit_value_parameter(parameter)
            else:
                raise SyntacticError(f"Unsupported parameters type ({parameter.__class__.__name__}): {parameter}")
            parameters.append(parameter)

        return Example(
            expected_value=expected_value,
            parameters=parameters)

    def transform_value_parameter(self: "BLEIRTransformer", value_parameter: ValueParameter) -> ValueParameter:
        if not isinstance(value_parameter, ValueParameter):
            raise SyntacticError(f"Unsupported value_parameter type ({value_parameter.__class__.__name__}): {value_parameter}")

        if isinstance(value_parameter.identifier, str):
            identifier = value_parameter.identifier
        else:
            raise SyntacticError(f"Unsupported identifier type ({value_parameter.identifier.__class__.__name__}): {value_parameter.identifier}")

        if isinstance(value_parameter.row_number, int):
            row_number = value_parameter.row_number
        else:
            raise SyntacticError(f"Unsupported row_number type ({value_parameter.row_number.__class__.__name__}): {value_parameter.row_number}")

        if isinstance(value_parameter.value, ndarray):
            value = value_parameter.value
        else:
            raise SyntacticError(f"Unsupported value type ({value_parameter.value.__class__.__name__}): {value_parameter.value}")

        return ValueParameter(
            identifier=identifier,
            row_number=row_number,
            value=value)

    def transform_fragment_caller_call(self: "BLEIRTransformer", fragment_caller_call: FragmentCallerCall) -> FragmentCallerCall:
        if not isinstance(fragment_caller_call, FragmentCallerCall):
            raise SyntacticError(f"Unsupported fragment_caller_call type ({fragment_caller_call.__class__.__name__}): {fragment_caller_call}")

        if isinstance(fragment_caller_call.caller, FragmentCaller):
            caller = self.visit_fragment_caller(fragment_caller_call.caller)
        else:
            raise SyntacticError(f"Unsupported caller type ({fragment_caller_call.caller.__class__.__name__}): {fragment_caller_call.caller}")

        parameters = []
        for parameter in fragment_caller_call.parameters:
            if isinstance(parameter, int):
                pass
            elif isinstance(parameter, str):
                pass
            else:
                raise SyntacticError(f"Unsupported parameters type ({parameter.__class__.__name__}): {parameter}")
            parameters.append(parameter)

        if isinstance(fragment_caller_call.metadata, CallMetadata):
            metadata = self.visit_call_metadata(fragment_caller_call.metadata)
        elif isinstance(fragment_caller_call.metadata, Any):
            metadata = fragment_caller_call.metadata
        elif isinstance(fragment_caller_call.metadata, NoneType):
            metadata = fragment_caller_call.metadata
        elif fragment_caller_call.metadata is None:
            metadata = None
        else:
            raise SyntacticError(f"Unsupported metadata type ({fragment_caller_call.metadata.__class__.__name__}): {fragment_caller_call.metadata}")

        if isinstance(fragment_caller_call.comment, MultiLineComment):
            comment = self.visit_multi_line_comment(fragment_caller_call.comment)
        elif isinstance(fragment_caller_call.comment, SingleLineComment):
            comment = self.visit_single_line_comment(fragment_caller_call.comment)
        elif isinstance(fragment_caller_call.comment, TrailingComment):
            comment = self.visit_trailing_comment(fragment_caller_call.comment)
        elif isinstance(fragment_caller_call.comment, NoneType):
            comment = fragment_caller_call.comment
        elif fragment_caller_call.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({fragment_caller_call.comment.__class__.__name__}): {fragment_caller_call.comment}")

        return FragmentCallerCall(
            caller=caller,
            parameters=parameters,
            metadata=metadata,
            comment=comment)

    def transform_fragment_caller(self: "BLEIRTransformer", fragment_caller: FragmentCaller) -> FragmentCaller:
        if not isinstance(fragment_caller, FragmentCaller):
            raise SyntacticError(f"Unsupported fragment_caller type ({fragment_caller.__class__.__name__}): {fragment_caller}")

        if isinstance(fragment_caller.fragment, Fragment):
            fragment = self.visit_fragment(fragment_caller.fragment)
        else:
            raise SyntacticError(f"Unsupported fragment type ({fragment_caller.fragment.__class__.__name__}): {fragment_caller.fragment}")

        if isinstance(fragment_caller.registers, AllocatedRegister):
            registers = self.visit_allocated_register(fragment_caller.registers)
        elif isinstance(fragment_caller.registers, MultiLineComment):
            registers = self.visit_multi_line_comment(fragment_caller.registers)
        elif isinstance(fragment_caller.registers, SingleLineComment):
            registers = self.visit_single_line_comment(fragment_caller.registers)
        elif isinstance(fragment_caller.registers, TrailingComment):
            registers = self.visit_trailing_comment(fragment_caller.registers)
        elif isinstance(fragment_caller.registers, NoneType):
            registers = fragment_caller.registers
        elif fragment_caller.registers is None:
            registers = None
        else:
            raise SyntacticError(f"Unsupported registers type ({fragment_caller.registers.__class__.__name__}): {fragment_caller.registers}")

        if isinstance(fragment_caller.metadata, CallerMetadata):
            metadata = self.visit_caller_metadata(fragment_caller.metadata)
        elif isinstance(fragment_caller.metadata, Any):
            metadata = fragment_caller.metadata
        elif isinstance(fragment_caller.metadata, NoneType):
            metadata = fragment_caller.metadata
        elif fragment_caller.metadata is None:
            metadata = None
        else:
            raise SyntacticError(f"Unsupported metadata type ({fragment_caller.metadata.__class__.__name__}): {fragment_caller.metadata}")

        return FragmentCaller(
            fragment=fragment,
            registers=registers,
            metadata=metadata)

    def transform_fragment(self: "BLEIRTransformer", fragment: Fragment) -> Fragment:
        if not isinstance(fragment, Fragment):
            raise SyntacticError(f"Unsupported fragment type ({fragment.__class__.__name__}): {fragment}")

        if isinstance(fragment.identifier, str):
            identifier = fragment.identifier
        else:
            raise SyntacticError(f"Unsupported identifier type ({fragment.identifier.__class__.__name__}): {fragment.identifier}")

        parameters = []
        for parameter in fragment.parameters:
            if isinstance(parameter, RN_REG):
                parameter = self.visit_rn_reg(parameter)
            elif isinstance(parameter, RE_REG):
                parameter = self.visit_re_reg(parameter)
            elif isinstance(parameter, EWE_REG):
                parameter = self.visit_ewe_reg(parameter)
            elif isinstance(parameter, L1_REG):
                parameter = self.visit_l1_reg(parameter)
            elif isinstance(parameter, L2_REG):
                parameter = self.visit_l2_reg(parameter)
            elif isinstance(parameter, SM_REG):
                parameter = self.visit_sm_reg(parameter)
            else:
                raise SyntacticError(f"Unsupported parameters type ({parameter.__class__.__name__}): {parameter}")
            parameters.append(parameter)

        operations = []
        for operation in fragment.operations:
            if isinstance(operation, MultiStatement):
                operation = self.visit_multi_statement(operation)
            elif isinstance(operation, STATEMENT):
                operation = self.visit_statement(operation)
            elif isinstance(operation, MultiLineComment):
                operation = self.visit_multi_line_comment(operation)
            elif isinstance(operation, SingleLineComment):
                operation = self.visit_single_line_comment(operation)
            else:
                raise SyntacticError(f"Unsupported operations type ({operation.__class__.__name__}): {operation}")
            operations.append(operation)

        if isinstance(fragment.doc_comment, MultiLineComment):
            doc_comment = self.visit_multi_line_comment(fragment.doc_comment)
        elif isinstance(fragment.doc_comment, NoneType):
            doc_comment = fragment.doc_comment
        elif fragment.doc_comment is None:
            doc_comment = None
        else:
            raise SyntacticError(f"Unsupported doc_comment type ({fragment.doc_comment.__class__.__name__}): {fragment.doc_comment}")

        if isinstance(fragment.metadata, FragmentMetadata):
            metadata = self.visit_fragment_metadata(fragment.metadata)
        elif isinstance(fragment.metadata, Any):
            metadata = fragment.metadata
        elif isinstance(fragment.metadata, NoneType):
            metadata = fragment.metadata
        elif fragment.metadata is None:
            metadata = None
        else:
            raise SyntacticError(f"Unsupported metadata type ({fragment.metadata.__class__.__name__}): {fragment.metadata}")

        if isinstance(fragment.children, Fragment):
            children = fragment.children
        elif isinstance(fragment.children, NoneType):
            children = fragment.children
        elif fragment.children is None:
            children = None
        else:
            raise SyntacticError(f"Unsupported children type ({fragment.children.__class__.__name__}): {fragment.children}")

        return Fragment(
            identifier=identifier,
            parameters=parameters,
            operations=operations,
            doc_comment=doc_comment,
            metadata=metadata,
            children=children)

    def transform_rn_reg(self: "BLEIRTransformer", rn_reg: RN_REG) -> RN_REG:
        if not isinstance(rn_reg, RN_REG):
            raise SyntacticError(f"Unsupported rn_reg type ({rn_reg.__class__.__name__}): {rn_reg}")

        if isinstance(rn_reg.identifier, str):
            identifier = rn_reg.identifier
        else:
            raise SyntacticError(f"Unsupported identifier type ({rn_reg.identifier.__class__.__name__}): {rn_reg.identifier}")

        if isinstance(rn_reg.comment, InlineComment):
            comment = self.visit_inline_comment(rn_reg.comment)
        elif isinstance(rn_reg.comment, NoneType):
            comment = rn_reg.comment
        elif rn_reg.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({rn_reg.comment.__class__.__name__}): {rn_reg.comment}")

        if isinstance(rn_reg.initial_value, int):
            initial_value = rn_reg.initial_value
        elif isinstance(rn_reg.initial_value, NoneType):
            initial_value = rn_reg.initial_value
        elif rn_reg.initial_value is None:
            initial_value = None
        else:
            raise SyntacticError(f"Unsupported initial_value type ({rn_reg.initial_value.__class__.__name__}): {rn_reg.initial_value}")

        if isinstance(rn_reg.register, int):
            register = rn_reg.register
        elif isinstance(rn_reg.register, NoneType):
            register = rn_reg.register
        elif rn_reg.register is None:
            register = None
        else:
            raise SyntacticError(f"Unsupported register type ({rn_reg.register.__class__.__name__}): {rn_reg.register}")

        if isinstance(rn_reg.row_number, int):
            row_number = rn_reg.row_number
        elif isinstance(rn_reg.row_number, NoneType):
            row_number = rn_reg.row_number
        elif rn_reg.row_number is None:
            row_number = None
        else:
            raise SyntacticError(f"Unsupported row_number type ({rn_reg.row_number.__class__.__name__}): {rn_reg.row_number}")

        if isinstance(rn_reg.is_lowered, bool):
            is_lowered = rn_reg.is_lowered
        else:
            raise SyntacticError(f"Unsupported is_lowered type ({rn_reg.is_lowered.__class__.__name__}): {rn_reg.is_lowered}")

        if isinstance(rn_reg.is_literal, bool):
            is_literal = rn_reg.is_literal
        else:
            raise SyntacticError(f"Unsupported is_literal type ({rn_reg.is_literal.__class__.__name__}): {rn_reg.is_literal}")

        if isinstance(rn_reg.is_temporary, bool):
            is_temporary = rn_reg.is_temporary
        else:
            raise SyntacticError(f"Unsupported is_temporary type ({rn_reg.is_temporary.__class__.__name__}): {rn_reg.is_temporary}")

        return RN_REG(
            identifier=identifier,
            comment=comment,
            initial_value=initial_value,
            register=register,
            row_number=row_number,
            is_lowered=is_lowered,
            is_literal=is_literal,
            is_temporary=is_temporary)

    def transform_inline_comment(self: "BLEIRTransformer", inline_comment: InlineComment) -> InlineComment:
        if not isinstance(inline_comment, InlineComment):
            raise SyntacticError(f"Unsupported inline_comment type ({inline_comment.__class__.__name__}): {inline_comment}")

        if isinstance(inline_comment.value, str):
            value = inline_comment.value
        else:
            raise SyntacticError(f"Unsupported value type ({inline_comment.value.__class__.__name__}): {inline_comment.value}")

        return InlineComment(
            value=value)

    def transform_re_reg(self: "BLEIRTransformer", re_reg: RE_REG) -> RE_REG:
        if not isinstance(re_reg, RE_REG):
            raise SyntacticError(f"Unsupported re_reg type ({re_reg.__class__.__name__}): {re_reg}")

        if isinstance(re_reg.identifier, str):
            identifier = re_reg.identifier
        else:
            raise SyntacticError(f"Unsupported identifier type ({re_reg.identifier.__class__.__name__}): {re_reg.identifier}")

        if isinstance(re_reg.comment, InlineComment):
            comment = self.visit_inline_comment(re_reg.comment)
        elif isinstance(re_reg.comment, NoneType):
            comment = re_reg.comment
        elif re_reg.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({re_reg.comment.__class__.__name__}): {re_reg.comment}")

        if isinstance(re_reg.initial_value, int):
            initial_value = re_reg.initial_value
        elif isinstance(re_reg.initial_value, NoneType):
            initial_value = re_reg.initial_value
        elif re_reg.initial_value is None:
            initial_value = None
        else:
            raise SyntacticError(f"Unsupported initial_value type ({re_reg.initial_value.__class__.__name__}): {re_reg.initial_value}")

        if isinstance(re_reg.register, int):
            register = re_reg.register
        elif isinstance(re_reg.register, NoneType):
            register = re_reg.register
        elif re_reg.register is None:
            register = None
        else:
            raise SyntacticError(f"Unsupported register type ({re_reg.register.__class__.__name__}): {re_reg.register}")

        if isinstance(re_reg.row_mask, int):
            row_mask = re_reg.row_mask
        elif isinstance(re_reg.row_mask, NoneType):
            row_mask = re_reg.row_mask
        elif re_reg.row_mask is None:
            row_mask = None
        else:
            raise SyntacticError(f"Unsupported row_mask type ({re_reg.row_mask.__class__.__name__}): {re_reg.row_mask}")

        if isinstance(re_reg.is_lowered, bool):
            is_lowered = re_reg.is_lowered
        else:
            raise SyntacticError(f"Unsupported is_lowered type ({re_reg.is_lowered.__class__.__name__}): {re_reg.is_lowered}")

        if isinstance(re_reg.is_literal, bool):
            is_literal = re_reg.is_literal
        else:
            raise SyntacticError(f"Unsupported is_literal type ({re_reg.is_literal.__class__.__name__}): {re_reg.is_literal}")

        return RE_REG(
            identifier=identifier,
            comment=comment,
            initial_value=initial_value,
            register=register,
            row_mask=row_mask,
            is_lowered=is_lowered,
            is_literal=is_literal)

    def transform_ewe_reg(self: "BLEIRTransformer", ewe_reg: EWE_REG) -> EWE_REG:
        if not isinstance(ewe_reg, EWE_REG):
            raise SyntacticError(f"Unsupported ewe_reg type ({ewe_reg.__class__.__name__}): {ewe_reg}")

        if isinstance(ewe_reg.identifier, str):
            identifier = ewe_reg.identifier
        else:
            raise SyntacticError(f"Unsupported identifier type ({ewe_reg.identifier.__class__.__name__}): {ewe_reg.identifier}")

        if isinstance(ewe_reg.comment, InlineComment):
            comment = self.visit_inline_comment(ewe_reg.comment)
        elif isinstance(ewe_reg.comment, NoneType):
            comment = ewe_reg.comment
        elif ewe_reg.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({ewe_reg.comment.__class__.__name__}): {ewe_reg.comment}")

        if isinstance(ewe_reg.initial_value, int):
            initial_value = ewe_reg.initial_value
        elif isinstance(ewe_reg.initial_value, NoneType):
            initial_value = ewe_reg.initial_value
        elif ewe_reg.initial_value is None:
            initial_value = None
        else:
            raise SyntacticError(f"Unsupported initial_value type ({ewe_reg.initial_value.__class__.__name__}): {ewe_reg.initial_value}")

        if isinstance(ewe_reg.register, int):
            register = ewe_reg.register
        elif isinstance(ewe_reg.register, NoneType):
            register = ewe_reg.register
        elif ewe_reg.register is None:
            register = None
        else:
            raise SyntacticError(f"Unsupported register type ({ewe_reg.register.__class__.__name__}): {ewe_reg.register}")

        if isinstance(ewe_reg.wordline_mask, int):
            wordline_mask = ewe_reg.wordline_mask
        elif isinstance(ewe_reg.wordline_mask, NoneType):
            wordline_mask = ewe_reg.wordline_mask
        elif ewe_reg.wordline_mask is None:
            wordline_mask = None
        else:
            raise SyntacticError(f"Unsupported wordline_mask type ({ewe_reg.wordline_mask.__class__.__name__}): {ewe_reg.wordline_mask}")

        if isinstance(ewe_reg.is_lowered, bool):
            is_lowered = ewe_reg.is_lowered
        else:
            raise SyntacticError(f"Unsupported is_lowered type ({ewe_reg.is_lowered.__class__.__name__}): {ewe_reg.is_lowered}")

        if isinstance(ewe_reg.is_literal, bool):
            is_literal = ewe_reg.is_literal
        else:
            raise SyntacticError(f"Unsupported is_literal type ({ewe_reg.is_literal.__class__.__name__}): {ewe_reg.is_literal}")

        return EWE_REG(
            identifier=identifier,
            comment=comment,
            initial_value=initial_value,
            register=register,
            wordline_mask=wordline_mask,
            is_lowered=is_lowered,
            is_literal=is_literal)

    def transform_l1_reg(self: "BLEIRTransformer", l1_reg: L1_REG) -> L1_REG:
        if not isinstance(l1_reg, L1_REG):
            raise SyntacticError(f"Unsupported l1_reg type ({l1_reg.__class__.__name__}): {l1_reg}")

        if isinstance(l1_reg.identifier, str):
            identifier = l1_reg.identifier
        else:
            raise SyntacticError(f"Unsupported identifier type ({l1_reg.identifier.__class__.__name__}): {l1_reg.identifier}")

        if isinstance(l1_reg.comment, InlineComment):
            comment = self.visit_inline_comment(l1_reg.comment)
        elif isinstance(l1_reg.comment, NoneType):
            comment = l1_reg.comment
        elif l1_reg.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({l1_reg.comment.__class__.__name__}): {l1_reg.comment}")

        if isinstance(l1_reg.register, int):
            register = l1_reg.register
        elif isinstance(l1_reg.register, NoneType):
            register = l1_reg.register
        elif l1_reg.register is None:
            register = None
        else:
            raise SyntacticError(f"Unsupported register type ({l1_reg.register.__class__.__name__}): {l1_reg.register}")

        if isinstance(l1_reg.bank_group_row, int):
            bank_group_row = l1_reg.bank_group_row
        elif isinstance(l1_reg.bank_group_row, int):
            bank_group_row = l1_reg.bank_group_row
        elif isinstance(l1_reg.bank_group_row, int):
            bank_group_row = l1_reg.bank_group_row
        elif isinstance(l1_reg.bank_group_row, int):
            bank_group_row = l1_reg.bank_group_row
        elif isinstance(l1_reg.bank_group_row, NoneType):
            bank_group_row = l1_reg.bank_group_row
        elif l1_reg.bank_group_row is None:
            bank_group_row = None
        else:
            raise SyntacticError(f"Unsupported bank_group_row type ({l1_reg.bank_group_row.__class__.__name__}): {l1_reg.bank_group_row}")

        if isinstance(l1_reg.is_lowered, bool):
            is_lowered = l1_reg.is_lowered
        else:
            raise SyntacticError(f"Unsupported is_lowered type ({l1_reg.is_lowered.__class__.__name__}): {l1_reg.is_lowered}")

        if isinstance(l1_reg.is_literal, bool):
            is_literal = l1_reg.is_literal
        else:
            raise SyntacticError(f"Unsupported is_literal type ({l1_reg.is_literal.__class__.__name__}): {l1_reg.is_literal}")

        return L1_REG(
            identifier=identifier,
            comment=comment,
            register=register,
            bank_group_row=bank_group_row,
            is_lowered=is_lowered,
            is_literal=is_literal)

    def transform_l2_reg(self: "BLEIRTransformer", l2_reg: L2_REG) -> L2_REG:
        if not isinstance(l2_reg, L2_REG):
            raise SyntacticError(f"Unsupported l2_reg type ({l2_reg.__class__.__name__}): {l2_reg}")

        if isinstance(l2_reg.identifier, str):
            identifier = l2_reg.identifier
        else:
            raise SyntacticError(f"Unsupported identifier type ({l2_reg.identifier.__class__.__name__}): {l2_reg.identifier}")

        if isinstance(l2_reg.comment, InlineComment):
            comment = self.visit_inline_comment(l2_reg.comment)
        elif isinstance(l2_reg.comment, NoneType):
            comment = l2_reg.comment
        elif l2_reg.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({l2_reg.comment.__class__.__name__}): {l2_reg.comment}")

        if isinstance(l2_reg.register, int):
            register = l2_reg.register
        elif isinstance(l2_reg.register, NoneType):
            register = l2_reg.register
        elif l2_reg.register is None:
            register = None
        else:
            raise SyntacticError(f"Unsupported register type ({l2_reg.register.__class__.__name__}): {l2_reg.register}")

        if isinstance(l2_reg.value, int):
            value = l2_reg.value
        elif isinstance(l2_reg.value, NoneType):
            value = l2_reg.value
        elif l2_reg.value is None:
            value = None
        else:
            raise SyntacticError(f"Unsupported value type ({l2_reg.value.__class__.__name__}): {l2_reg.value}")

        if isinstance(l2_reg.is_lowered, bool):
            is_lowered = l2_reg.is_lowered
        else:
            raise SyntacticError(f"Unsupported is_lowered type ({l2_reg.is_lowered.__class__.__name__}): {l2_reg.is_lowered}")

        if isinstance(l2_reg.is_literal, bool):
            is_literal = l2_reg.is_literal
        else:
            raise SyntacticError(f"Unsupported is_literal type ({l2_reg.is_literal.__class__.__name__}): {l2_reg.is_literal}")

        return L2_REG(
            identifier=identifier,
            comment=comment,
            register=register,
            value=value,
            is_lowered=is_lowered,
            is_literal=is_literal)

    def transform_sm_reg(self: "BLEIRTransformer", sm_reg: SM_REG) -> SM_REG:
        if not isinstance(sm_reg, SM_REG):
            raise SyntacticError(f"Unsupported sm_reg type ({sm_reg.__class__.__name__}): {sm_reg}")

        if isinstance(sm_reg.identifier, str):
            identifier = sm_reg.identifier
        else:
            raise SyntacticError(f"Unsupported identifier type ({sm_reg.identifier.__class__.__name__}): {sm_reg.identifier}")

        if isinstance(sm_reg.comment, InlineComment):
            comment = self.visit_inline_comment(sm_reg.comment)
        elif isinstance(sm_reg.comment, NoneType):
            comment = sm_reg.comment
        elif sm_reg.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({sm_reg.comment.__class__.__name__}): {sm_reg.comment}")

        if isinstance(sm_reg.negated, bool):
            negated = sm_reg.negated
        elif isinstance(sm_reg.negated, NoneType):
            negated = sm_reg.negated
        elif sm_reg.negated is None:
            negated = None
        else:
            raise SyntacticError(f"Unsupported negated type ({sm_reg.negated.__class__.__name__}): {sm_reg.negated}")

        if isinstance(sm_reg.constant_value, int):
            constant_value = sm_reg.constant_value
        elif isinstance(sm_reg.constant_value, NoneType):
            constant_value = sm_reg.constant_value
        elif sm_reg.constant_value is None:
            constant_value = None
        else:
            raise SyntacticError(f"Unsupported constant_value type ({sm_reg.constant_value.__class__.__name__}): {sm_reg.constant_value}")

        if isinstance(sm_reg.register, int):
            register = sm_reg.register
        elif isinstance(sm_reg.register, NoneType):
            register = sm_reg.register
        elif sm_reg.register is None:
            register = None
        else:
            raise SyntacticError(f"Unsupported register type ({sm_reg.register.__class__.__name__}): {sm_reg.register}")

        if isinstance(sm_reg.is_section, bool):
            is_section = sm_reg.is_section
        else:
            raise SyntacticError(f"Unsupported is_section type ({sm_reg.is_section.__class__.__name__}): {sm_reg.is_section}")

        if isinstance(sm_reg.is_lowered, bool):
            is_lowered = sm_reg.is_lowered
        else:
            raise SyntacticError(f"Unsupported is_lowered type ({sm_reg.is_lowered.__class__.__name__}): {sm_reg.is_lowered}")

        if isinstance(sm_reg.is_literal, bool):
            is_literal = sm_reg.is_literal
        else:
            raise SyntacticError(f"Unsupported is_literal type ({sm_reg.is_literal.__class__.__name__}): {sm_reg.is_literal}")

        return SM_REG(
            identifier=identifier,
            comment=comment,
            negated=negated,
            constant_value=constant_value,
            register=register,
            is_section=is_section,
            is_lowered=is_lowered,
            is_literal=is_literal)

    def transform_multi_statement(self: "BLEIRTransformer", multi_statement: MultiStatement) -> MultiStatement:
        if not isinstance(multi_statement, MultiStatement):
            raise SyntacticError(f"Unsupported multi_statement type ({multi_statement.__class__.__name__}): {multi_statement}")

        statements = []
        for statement in multi_statement.statements:
            if isinstance(statement, STATEMENT):
                statement = self.visit_statement(statement)
            elif isinstance(statement, MultiLineComment):
                statement = self.visit_multi_line_comment(statement)
            elif isinstance(statement, SingleLineComment):
                statement = self.visit_single_line_comment(statement)
            else:
                raise SyntacticError(f"Unsupported statements type ({statement.__class__.__name__}): {statement}")
            statements.append(statement)

        if isinstance(multi_statement.comment, MultiLineComment):
            comment = self.visit_multi_line_comment(multi_statement.comment)
        elif isinstance(multi_statement.comment, SingleLineComment):
            comment = self.visit_single_line_comment(multi_statement.comment)
        elif isinstance(multi_statement.comment, TrailingComment):
            comment = self.visit_trailing_comment(multi_statement.comment)
        elif isinstance(multi_statement.comment, NoneType):
            comment = multi_statement.comment
        elif multi_statement.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({multi_statement.comment.__class__.__name__}): {multi_statement.comment}")

        return MultiStatement(
            statements=statements,
            comment=comment)

    def transform_statement(self: "BLEIRTransformer", statement: STATEMENT) -> STATEMENT:
        if not isinstance(statement, STATEMENT):
            raise SyntacticError(f"Unsupported statement type ({statement.__class__.__name__}): {statement}")

        if isinstance(statement.operation, MASKED):
            operation = self.visit_masked(statement.operation)
        elif isinstance(statement.operation, SPECIAL):
            operation = self.visit_special(statement.operation)
        elif isinstance(statement.operation, RSP16_ASSIGNMENT):
            operation = self.visit_rsp16_assignment(statement.operation)
        elif isinstance(statement.operation, RSP256_ASSIGNMENT):
            operation = self.visit_rsp256_assignment(statement.operation)
        elif isinstance(statement.operation, RSP2K_ASSIGNMENT):
            operation = self.visit_rsp2k_assignment(statement.operation)
        elif isinstance(statement.operation, RSP32K_ASSIGNMENT):
            operation = self.visit_rsp32k_assignment(statement.operation)
        elif isinstance(statement.operation, GGL_ASSIGNMENT):
            operation = self.visit_ggl_assignment(statement.operation)
        elif isinstance(statement.operation, LGL_ASSIGNMENT):
            operation = self.visit_lgl_assignment(statement.operation)
        elif isinstance(statement.operation, LX_ASSIGNMENT):
            operation = self.visit_lx_assignment(statement.operation)
        elif isinstance(statement.operation, GlassStatement):
            operation = self.visit_glass_statement(statement.operation)
        else:
            raise SyntacticError(f"Unsupported operation type ({statement.operation.__class__.__name__}): {statement.operation}")

        if isinstance(statement.comment, MultiLineComment):
            comment = self.visit_multi_line_comment(statement.comment)
        elif isinstance(statement.comment, SingleLineComment):
            comment = self.visit_single_line_comment(statement.comment)
        elif isinstance(statement.comment, TrailingComment):
            comment = self.visit_trailing_comment(statement.comment)
        elif isinstance(statement.comment, NoneType):
            comment = statement.comment
        elif statement.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({statement.comment.__class__.__name__}): {statement.comment}")

        return STATEMENT(
            operation=operation,
            comment=comment)

    def transform_masked(self: "BLEIRTransformer", masked: MASKED) -> MASKED:
        if not isinstance(masked, MASKED):
            raise SyntacticError(f"Unsupported masked type ({masked.__class__.__name__}): {masked}")

        if isinstance(masked.mask, MASK):
            mask = self.visit_mask(masked.mask)
        else:
            raise SyntacticError(f"Unsupported mask type ({masked.mask.__class__.__name__}): {masked.mask}")

        if isinstance(masked.assignment, ASSIGNMENT):
            assignment = self.visit_assignment(masked.assignment)
        elif isinstance(masked.assignment, NoneType):
            assignment = masked.assignment
        elif masked.assignment is None:
            assignment = None
        else:
            raise SyntacticError(f"Unsupported assignment type ({masked.assignment.__class__.__name__}): {masked.assignment}")

        if isinstance(masked.read_write_inhibit, ReadWriteInhibit):
            read_write_inhibit = self.visit_read_write_inhibit(masked.read_write_inhibit)
        elif isinstance(masked.read_write_inhibit, NoneType):
            read_write_inhibit = masked.read_write_inhibit
        elif masked.read_write_inhibit is None:
            read_write_inhibit = None
        else:
            raise SyntacticError(f"Unsupported read_write_inhibit type ({masked.read_write_inhibit.__class__.__name__}): {masked.read_write_inhibit}")

        return MASKED(
            mask=mask,
            assignment=assignment,
            read_write_inhibit=read_write_inhibit)

    def transform_mask(self: "BLEIRTransformer", mask: MASK) -> MASK:
        if not isinstance(mask, MASK):
            raise SyntacticError(f"Unsupported mask type ({mask.__class__.__name__}): {mask}")

        if isinstance(mask.expression, SM_REG):
            expression = self.visit_sm_reg(mask.expression)
        elif isinstance(mask.expression, SHIFTED_SM_REG):
            expression = self.visit_shifted_sm_reg(mask.expression)
        else:
            raise SyntacticError(f"Unsupported expression type ({mask.expression.__class__.__name__}): {mask.expression}")

        if isinstance(mask.operator, UNARY_OP):
            operator = self.visit_unary_op(mask.operator)
        elif isinstance(mask.operator, NoneType):
            operator = mask.operator
        elif mask.operator is None:
            operator = None
        else:
            raise SyntacticError(f"Unsupported operator type ({mask.operator.__class__.__name__}): {mask.operator}")

        if isinstance(mask.read_write_inhibit, ReadWriteInhibit):
            read_write_inhibit = self.visit_read_write_inhibit(mask.read_write_inhibit)
        elif isinstance(mask.read_write_inhibit, NoneType):
            read_write_inhibit = mask.read_write_inhibit
        elif mask.read_write_inhibit is None:
            read_write_inhibit = None
        else:
            raise SyntacticError(f"Unsupported read_write_inhibit type ({mask.read_write_inhibit.__class__.__name__}): {mask.read_write_inhibit}")

        return MASK(
            expression=expression,
            operator=operator,
            read_write_inhibit=read_write_inhibit)

    def transform_shifted_sm_reg(self: "BLEIRTransformer", shifted_sm_reg: SHIFTED_SM_REG) -> SHIFTED_SM_REG:
        if not isinstance(shifted_sm_reg, SHIFTED_SM_REG):
            raise SyntacticError(f"Unsupported shifted_sm_reg type ({shifted_sm_reg.__class__.__name__}): {shifted_sm_reg}")

        if isinstance(shifted_sm_reg.register, SM_REG):
            register = self.visit_sm_reg(shifted_sm_reg.register)
        elif isinstance(shifted_sm_reg.register, SHIFTED_SM_REG):
            register = shifted_sm_reg.register
        else:
            raise SyntacticError(f"Unsupported register type ({shifted_sm_reg.register.__class__.__name__}): {shifted_sm_reg.register}")

        if isinstance(shifted_sm_reg.num_bits, int):
            num_bits = shifted_sm_reg.num_bits
        else:
            raise SyntacticError(f"Unsupported num_bits type ({shifted_sm_reg.num_bits.__class__.__name__}): {shifted_sm_reg.num_bits}")

        if isinstance(shifted_sm_reg.negated, bool):
            negated = shifted_sm_reg.negated
        elif isinstance(shifted_sm_reg.negated, NoneType):
            negated = shifted_sm_reg.negated
        elif shifted_sm_reg.negated is None:
            negated = None
        else:
            raise SyntacticError(f"Unsupported negated type ({shifted_sm_reg.negated.__class__.__name__}): {shifted_sm_reg.negated}")

        return SHIFTED_SM_REG(
            register=register,
            num_bits=num_bits,
            negated=negated)

    def transform_unary_op(self: "BLEIRTransformer", unary_op: UNARY_OP) -> UNARY_OP:
        if not isinstance(unary_op, UNARY_OP):
            raise SyntacticError(f"Unsupported unary_op type ({unary_op.__class__.__name__}): {unary_op}")
        return unary_op

    def transform_read_write_inhibit(self: "BLEIRTransformer", read_write_inhibit: ReadWriteInhibit) -> ReadWriteInhibit:
        if not isinstance(read_write_inhibit, ReadWriteInhibit):
            raise SyntacticError(f"Unsupported read_write_inhibit type ({read_write_inhibit.__class__.__name__}): {read_write_inhibit}")
        return read_write_inhibit

    def transform_assignment(self: "BLEIRTransformer", assignment: ASSIGNMENT) -> ASSIGNMENT:
        if not isinstance(assignment, ASSIGNMENT):
            raise SyntacticError(f"Unsupported assignment type ({assignment.__class__.__name__}): {assignment}")

        if isinstance(assignment.operation, READ):
            operation = self.visit_read(assignment.operation)
        elif isinstance(assignment.operation, WRITE):
            operation = self.visit_write(assignment.operation)
        elif isinstance(assignment.operation, BROADCAST):
            operation = self.visit_broadcast(assignment.operation)
        elif isinstance(assignment.operation, RSP16_ASSIGNMENT):
            operation = self.visit_rsp16_assignment(assignment.operation)
        elif isinstance(assignment.operation, RSP256_ASSIGNMENT):
            operation = self.visit_rsp256_assignment(assignment.operation)
        elif isinstance(assignment.operation, RSP2K_ASSIGNMENT):
            operation = self.visit_rsp2k_assignment(assignment.operation)
        elif isinstance(assignment.operation, RSP32K_ASSIGNMENT):
            operation = self.visit_rsp32k_assignment(assignment.operation)
        else:
            raise SyntacticError(f"Unsupported operation type ({assignment.operation.__class__.__name__}): {assignment.operation}")

        return ASSIGNMENT(
            operation=operation)

    def transform_read(self: "BLEIRTransformer", read: READ) -> READ:
        if not isinstance(read, READ):
            raise SyntacticError(f"Unsupported read type ({read.__class__.__name__}): {read}")

        if isinstance(read.operator, ASSIGN_OP):
            operator = self.visit_assign_op(read.operator)
        else:
            raise SyntacticError(f"Unsupported operator type ({read.operator.__class__.__name__}): {read.operator}")

        if isinstance(read.rvalue, UNARY_EXPR):
            rvalue = self.visit_unary_expr(read.rvalue)
        elif isinstance(read.rvalue, BINARY_EXPR):
            rvalue = self.visit_binary_expr(read.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({read.rvalue.__class__.__name__}): {read.rvalue}")

        return READ(
            operator=operator,
            rvalue=rvalue)

    def transform_assign_op(self: "BLEIRTransformer", assign_op: ASSIGN_OP) -> ASSIGN_OP:
        if not isinstance(assign_op, ASSIGN_OP):
            raise SyntacticError(f"Unsupported assign_op type ({assign_op.__class__.__name__}): {assign_op}")
        return assign_op

    def transform_unary_expr(self: "BLEIRTransformer", unary_expr: UNARY_EXPR) -> UNARY_EXPR:
        if not isinstance(unary_expr, UNARY_EXPR):
            raise SyntacticError(f"Unsupported unary_expr type ({unary_expr.__class__.__name__}): {unary_expr}")

        if isinstance(unary_expr.expression, UNARY_SB):
            expression = self.visit_unary_sb(unary_expr.expression)
        elif isinstance(unary_expr.expression, UNARY_SRC):
            expression = self.visit_unary_src(unary_expr.expression)
        elif isinstance(unary_expr.expression, BIT_EXPR):
            expression = self.visit_bit_expr(unary_expr.expression)
        else:
            raise SyntacticError(f"Unsupported expression type ({unary_expr.expression.__class__.__name__}): {unary_expr.expression}")

        return UNARY_EXPR(
            expression=expression)

    def transform_unary_sb(self: "BLEIRTransformer", unary_sb: UNARY_SB) -> UNARY_SB:
        if not isinstance(unary_sb, UNARY_SB):
            raise SyntacticError(f"Unsupported unary_sb type ({unary_sb.__class__.__name__}): {unary_sb}")

        if isinstance(unary_sb.expression, SB_EXPR):
            expression = self.visit_sb_expr(unary_sb.expression)
        else:
            raise SyntacticError(f"Unsupported expression type ({unary_sb.expression.__class__.__name__}): {unary_sb.expression}")

        if isinstance(unary_sb.operator, UNARY_OP):
            operator = self.visit_unary_op(unary_sb.operator)
        elif isinstance(unary_sb.operator, NoneType):
            operator = unary_sb.operator
        elif unary_sb.operator is None:
            operator = None
        else:
            raise SyntacticError(f"Unsupported operator type ({unary_sb.operator.__class__.__name__}): {unary_sb.operator}")

        return UNARY_SB(
            expression=expression,
            operator=operator)

    def transform_sb_expr(self: "BLEIRTransformer", sb_expr: SB_EXPR) -> SB_EXPR:
        if not isinstance(sb_expr, SB_EXPR):
            raise SyntacticError(f"Unsupported sb_expr type ({sb_expr.__class__.__name__}): {sb_expr}")

        if isinstance(sb_expr.parameters, RN_REG):
            parameters = self.visit_rn_reg(sb_expr.parameters)
        elif isinstance(sb_expr.parameters, RN_REG):
            parameters = self.visit_rn_reg(sb_expr.parameters)
        elif isinstance(sb_expr.parameters, RN_REG):
            parameters = self.visit_rn_reg(sb_expr.parameters)
        elif isinstance(sb_expr.parameters, RN_REG):
            parameters = self.visit_rn_reg(sb_expr.parameters)
        elif isinstance(sb_expr.parameters, RN_REG):
            parameters = self.visit_rn_reg(sb_expr.parameters)
        elif isinstance(sb_expr.parameters, RN_REG):
            parameters = self.visit_rn_reg(sb_expr.parameters)
        elif isinstance(sb_expr.parameters, RE_REG):
            parameters = self.visit_re_reg(sb_expr.parameters)
        elif isinstance(sb_expr.parameters, EWE_REG):
            parameters = self.visit_ewe_reg(sb_expr.parameters)
        else:
            raise SyntacticError(f"Unsupported parameters type ({sb_expr.parameters.__class__.__name__}): {sb_expr.parameters}")

        return SB_EXPR(
            parameters=parameters)

    def transform_unary_src(self: "BLEIRTransformer", unary_src: UNARY_SRC) -> UNARY_SRC:
        if not isinstance(unary_src, UNARY_SRC):
            raise SyntacticError(f"Unsupported unary_src type ({unary_src.__class__.__name__}): {unary_src}")

        if isinstance(unary_src.expression, SRC_EXPR):
            expression = self.visit_src_expr(unary_src.expression)
        else:
            raise SyntacticError(f"Unsupported expression type ({unary_src.expression.__class__.__name__}): {unary_src.expression}")

        if isinstance(unary_src.operator, UNARY_OP):
            operator = self.visit_unary_op(unary_src.operator)
        elif isinstance(unary_src.operator, NoneType):
            operator = unary_src.operator
        elif unary_src.operator is None:
            operator = None
        else:
            raise SyntacticError(f"Unsupported operator type ({unary_src.operator.__class__.__name__}): {unary_src.operator}")

        return UNARY_SRC(
            expression=expression,
            operator=operator)

    def transform_src_expr(self: "BLEIRTransformer", src_expr: SRC_EXPR) -> SRC_EXPR:
        if not isinstance(src_expr, SRC_EXPR):
            raise SyntacticError(f"Unsupported src_expr type ({src_expr.__class__.__name__}): {src_expr}")
        return src_expr

    def transform_bit_expr(self: "BLEIRTransformer", bit_expr: BIT_EXPR) -> BIT_EXPR:
        if not isinstance(bit_expr, BIT_EXPR):
            raise SyntacticError(f"Unsupported bit_expr type ({bit_expr.__class__.__name__}): {bit_expr}")
        return bit_expr

    def transform_binary_expr(self: "BLEIRTransformer", binary_expr: BINARY_EXPR) -> BINARY_EXPR:
        if not isinstance(binary_expr, BINARY_EXPR):
            raise SyntacticError(f"Unsupported binary_expr type ({binary_expr.__class__.__name__}): {binary_expr}")

        if isinstance(binary_expr.operator, BINOP):
            operator = self.visit_binop(binary_expr.operator)
        else:
            raise SyntacticError(f"Unsupported operator type ({binary_expr.operator.__class__.__name__}): {binary_expr.operator}")

        if isinstance(binary_expr.left_operand, UNARY_SB):
            left_operand = self.visit_unary_sb(binary_expr.left_operand)
        elif isinstance(binary_expr.left_operand, RL_EXPR):
            left_operand = self.visit_rl_expr(binary_expr.left_operand)
        else:
            raise SyntacticError(f"Unsupported left_operand type ({binary_expr.left_operand.__class__.__name__}): {binary_expr.left_operand}")

        if isinstance(binary_expr.right_operand, UNARY_SRC):
            right_operand = self.visit_unary_src(binary_expr.right_operand)
        elif isinstance(binary_expr.right_operand, L1_REG):
            right_operand = self.visit_l1_reg(binary_expr.right_operand)
        elif isinstance(binary_expr.right_operand, L2_REG):
            right_operand = self.visit_l2_reg(binary_expr.right_operand)
        elif isinstance(binary_expr.right_operand, LXRegWithOffsets):
            right_operand = self.visit_lx_reg_with_offsets(binary_expr.right_operand)
        else:
            raise SyntacticError(f"Unsupported right_operand type ({binary_expr.right_operand.__class__.__name__}): {binary_expr.right_operand}")

        return BINARY_EXPR(
            operator=operator,
            left_operand=left_operand,
            right_operand=right_operand)

    def transform_binop(self: "BLEIRTransformer", binop: BINOP) -> BINOP:
        if not isinstance(binop, BINOP):
            raise SyntacticError(f"Unsupported binop type ({binop.__class__.__name__}): {binop}")
        return binop

    def transform_rl_expr(self: "BLEIRTransformer", rl_expr: RL_EXPR) -> RL_EXPR:
        if not isinstance(rl_expr, RL_EXPR):
            raise SyntacticError(f"Unsupported rl_expr type ({rl_expr.__class__.__name__}): {rl_expr}")
        return rl_expr

    def transform_lx_reg_with_offsets(self: "BLEIRTransformer", lx_reg_with_offsets: LXRegWithOffsets) -> LXRegWithOffsets:
        if not isinstance(lx_reg_with_offsets, LXRegWithOffsets):
            raise SyntacticError(f"Unsupported lx_reg_with_offsets type ({lx_reg_with_offsets.__class__.__name__}): {lx_reg_with_offsets}")

        if isinstance(lx_reg_with_offsets.parameter, L1_REG):
            parameter = self.visit_l1_reg(lx_reg_with_offsets.parameter)
        elif isinstance(lx_reg_with_offsets.parameter, L2_REG):
            parameter = self.visit_l2_reg(lx_reg_with_offsets.parameter)
        elif isinstance(lx_reg_with_offsets.parameter, LXRegWithOffsets):
            parameter = lx_reg_with_offsets.parameter
        else:
            raise SyntacticError(f"Unsupported parameter type ({lx_reg_with_offsets.parameter.__class__.__name__}): {lx_reg_with_offsets.parameter}")

        if isinstance(lx_reg_with_offsets.row_id, int):
            row_id = lx_reg_with_offsets.row_id
        else:
            raise SyntacticError(f"Unsupported row_id type ({lx_reg_with_offsets.row_id.__class__.__name__}): {lx_reg_with_offsets.row_id}")

        if isinstance(lx_reg_with_offsets.group_id, int):
            group_id = lx_reg_with_offsets.group_id
        else:
            raise SyntacticError(f"Unsupported group_id type ({lx_reg_with_offsets.group_id.__class__.__name__}): {lx_reg_with_offsets.group_id}")

        if isinstance(lx_reg_with_offsets.bank_id, int):
            bank_id = lx_reg_with_offsets.bank_id
        else:
            raise SyntacticError(f"Unsupported bank_id type ({lx_reg_with_offsets.bank_id.__class__.__name__}): {lx_reg_with_offsets.bank_id}")

        if isinstance(lx_reg_with_offsets.comment, MultiLineComment):
            comment = self.visit_multi_line_comment(lx_reg_with_offsets.comment)
        elif isinstance(lx_reg_with_offsets.comment, SingleLineComment):
            comment = self.visit_single_line_comment(lx_reg_with_offsets.comment)
        elif isinstance(lx_reg_with_offsets.comment, TrailingComment):
            comment = self.visit_trailing_comment(lx_reg_with_offsets.comment)
        elif isinstance(lx_reg_with_offsets.comment, NoneType):
            comment = lx_reg_with_offsets.comment
        elif lx_reg_with_offsets.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({lx_reg_with_offsets.comment.__class__.__name__}): {lx_reg_with_offsets.comment}")

        return LXRegWithOffsets(
            parameter=parameter,
            row_id=row_id,
            group_id=group_id,
            bank_id=bank_id,
            comment=comment)

    def transform_multi_line_comment(self: "BLEIRTransformer", multi_line_comment: MultiLineComment) -> MultiLineComment:
        if not isinstance(multi_line_comment, MultiLineComment):
            raise SyntacticError(f"Unsupported multi_line_comment type ({multi_line_comment.__class__.__name__}): {multi_line_comment}")

        lines = []
        for line in multi_line_comment.lines:
            if isinstance(line, str):
                pass
            else:
                raise SyntacticError(f"Unsupported lines type ({line.__class__.__name__}): {line}")
            lines.append(line)

        return MultiLineComment(
            lines=lines)

    def transform_single_line_comment(self: "BLEIRTransformer", single_line_comment: SingleLineComment) -> SingleLineComment:
        if not isinstance(single_line_comment, SingleLineComment):
            raise SyntacticError(f"Unsupported single_line_comment type ({single_line_comment.__class__.__name__}): {single_line_comment}")

        if isinstance(single_line_comment.line, str):
            line = single_line_comment.line
        else:
            raise SyntacticError(f"Unsupported line type ({single_line_comment.line.__class__.__name__}): {single_line_comment.line}")

        return SingleLineComment(
            line=line)

    def transform_trailing_comment(self: "BLEIRTransformer", trailing_comment: TrailingComment) -> TrailingComment:
        if not isinstance(trailing_comment, TrailingComment):
            raise SyntacticError(f"Unsupported trailing_comment type ({trailing_comment.__class__.__name__}): {trailing_comment}")

        if isinstance(trailing_comment.value, str):
            value = trailing_comment.value
        else:
            raise SyntacticError(f"Unsupported value type ({trailing_comment.value.__class__.__name__}): {trailing_comment.value}")

        return TrailingComment(
            value=value)

    def transform_write(self: "BLEIRTransformer", write: WRITE) -> WRITE:
        if not isinstance(write, WRITE):
            raise SyntacticError(f"Unsupported write type ({write.__class__.__name__}): {write}")

        if isinstance(write.operator, ASSIGN_OP):
            operator = self.visit_assign_op(write.operator)
        else:
            raise SyntacticError(f"Unsupported operator type ({write.operator.__class__.__name__}): {write.operator}")

        if isinstance(write.lvalue, SB_EXPR):
            lvalue = self.visit_sb_expr(write.lvalue)
        else:
            raise SyntacticError(f"Unsupported lvalue type ({write.lvalue.__class__.__name__}): {write.lvalue}")

        if isinstance(write.rvalue, UNARY_SRC):
            rvalue = self.visit_unary_src(write.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({write.rvalue.__class__.__name__}): {write.rvalue}")

        return WRITE(
            operator=operator,
            lvalue=lvalue,
            rvalue=rvalue)

    def transform_broadcast(self: "BLEIRTransformer", broadcast: BROADCAST) -> BROADCAST:
        if not isinstance(broadcast, BROADCAST):
            raise SyntacticError(f"Unsupported broadcast type ({broadcast.__class__.__name__}): {broadcast}")

        if isinstance(broadcast.lvalue, BROADCAST_EXPR):
            lvalue = self.visit_broadcast_expr(broadcast.lvalue)
        else:
            raise SyntacticError(f"Unsupported lvalue type ({broadcast.lvalue.__class__.__name__}): {broadcast.lvalue}")

        if isinstance(broadcast.rvalue, RL_EXPR):
            rvalue = self.visit_rl_expr(broadcast.rvalue)
        elif isinstance(broadcast.rvalue, L1_REG):
            rvalue = self.visit_l1_reg(broadcast.rvalue)
        elif isinstance(broadcast.rvalue, L2_REG):
            rvalue = self.visit_l2_reg(broadcast.rvalue)
        elif isinstance(broadcast.rvalue, LXRegWithOffsets):
            rvalue = self.visit_lx_reg_with_offsets(broadcast.rvalue)
        elif isinstance(broadcast.rvalue, BINARY_EXPR):
            rvalue = self.visit_binary_expr(broadcast.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({broadcast.rvalue.__class__.__name__}): {broadcast.rvalue}")

        return BROADCAST(
            lvalue=lvalue,
            rvalue=rvalue)

    def transform_broadcast_expr(self: "BLEIRTransformer", broadcast_expr: BROADCAST_EXPR) -> BROADCAST_EXPR:
        if not isinstance(broadcast_expr, BROADCAST_EXPR):
            raise SyntacticError(f"Unsupported broadcast_expr type ({broadcast_expr.__class__.__name__}): {broadcast_expr}")
        return broadcast_expr

    def transform_rsp16_assignment(self: "BLEIRTransformer", rsp16_assignment: RSP16_ASSIGNMENT) -> RSP16_ASSIGNMENT:
        if not isinstance(rsp16_assignment, RSP16_ASSIGNMENT):
            raise SyntacticError(f"Unsupported rsp16_assignment type ({rsp16_assignment.__class__.__name__}): {rsp16_assignment}")

        if isinstance(rsp16_assignment.rvalue, RSP16_RVALUE):
            rvalue = self.visit_rsp16_rvalue(rsp16_assignment.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({rsp16_assignment.rvalue.__class__.__name__}): {rsp16_assignment.rvalue}")

        return RSP16_ASSIGNMENT(
            rvalue=rvalue)

    def transform_rsp16_rvalue(self: "BLEIRTransformer", rsp16_rvalue: RSP16_RVALUE) -> RSP16_RVALUE:
        if not isinstance(rsp16_rvalue, RSP16_RVALUE):
            raise SyntacticError(f"Unsupported rsp16_rvalue type ({rsp16_rvalue.__class__.__name__}): {rsp16_rvalue}")
        return rsp16_rvalue

    def transform_rsp256_assignment(self: "BLEIRTransformer", rsp256_assignment: RSP256_ASSIGNMENT) -> RSP256_ASSIGNMENT:
        if not isinstance(rsp256_assignment, RSP256_ASSIGNMENT):
            raise SyntacticError(f"Unsupported rsp256_assignment type ({rsp256_assignment.__class__.__name__}): {rsp256_assignment}")

        if isinstance(rsp256_assignment.rvalue, RSP256_RVALUE):
            rvalue = self.visit_rsp256_rvalue(rsp256_assignment.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({rsp256_assignment.rvalue.__class__.__name__}): {rsp256_assignment.rvalue}")

        return RSP256_ASSIGNMENT(
            rvalue=rvalue)

    def transform_rsp256_rvalue(self: "BLEIRTransformer", rsp256_rvalue: RSP256_RVALUE) -> RSP256_RVALUE:
        if not isinstance(rsp256_rvalue, RSP256_RVALUE):
            raise SyntacticError(f"Unsupported rsp256_rvalue type ({rsp256_rvalue.__class__.__name__}): {rsp256_rvalue}")
        return rsp256_rvalue

    def transform_rsp2k_assignment(self: "BLEIRTransformer", rsp2k_assignment: RSP2K_ASSIGNMENT) -> RSP2K_ASSIGNMENT:
        if not isinstance(rsp2k_assignment, RSP2K_ASSIGNMENT):
            raise SyntacticError(f"Unsupported rsp2k_assignment type ({rsp2k_assignment.__class__.__name__}): {rsp2k_assignment}")

        if isinstance(rsp2k_assignment.rvalue, RSP2K_RVALUE):
            rvalue = self.visit_rsp2k_rvalue(rsp2k_assignment.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({rsp2k_assignment.rvalue.__class__.__name__}): {rsp2k_assignment.rvalue}")

        return RSP2K_ASSIGNMENT(
            rvalue=rvalue)

    def transform_rsp2k_rvalue(self: "BLEIRTransformer", rsp2k_rvalue: RSP2K_RVALUE) -> RSP2K_RVALUE:
        if not isinstance(rsp2k_rvalue, RSP2K_RVALUE):
            raise SyntacticError(f"Unsupported rsp2k_rvalue type ({rsp2k_rvalue.__class__.__name__}): {rsp2k_rvalue}")
        return rsp2k_rvalue

    def transform_rsp32k_assignment(self: "BLEIRTransformer", rsp32k_assignment: RSP32K_ASSIGNMENT) -> RSP32K_ASSIGNMENT:
        if not isinstance(rsp32k_assignment, RSP32K_ASSIGNMENT):
            raise SyntacticError(f"Unsupported rsp32k_assignment type ({rsp32k_assignment.__class__.__name__}): {rsp32k_assignment}")

        if isinstance(rsp32k_assignment.rvalue, RSP32K_RVALUE):
            rvalue = self.visit_rsp32k_rvalue(rsp32k_assignment.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({rsp32k_assignment.rvalue.__class__.__name__}): {rsp32k_assignment.rvalue}")

        return RSP32K_ASSIGNMENT(
            rvalue=rvalue)

    def transform_rsp32k_rvalue(self: "BLEIRTransformer", rsp32k_rvalue: RSP32K_RVALUE) -> RSP32K_RVALUE:
        if not isinstance(rsp32k_rvalue, RSP32K_RVALUE):
            raise SyntacticError(f"Unsupported rsp32k_rvalue type ({rsp32k_rvalue.__class__.__name__}): {rsp32k_rvalue}")
        return rsp32k_rvalue

    def transform_special(self: "BLEIRTransformer", special: SPECIAL) -> SPECIAL:
        if not isinstance(special, SPECIAL):
            raise SyntacticError(f"Unsupported special type ({special.__class__.__name__}): {special}")
        return special

    def transform_ggl_assignment(self: "BLEIRTransformer", ggl_assignment: GGL_ASSIGNMENT) -> GGL_ASSIGNMENT:
        if not isinstance(ggl_assignment, GGL_ASSIGNMENT):
            raise SyntacticError(f"Unsupported ggl_assignment type ({ggl_assignment.__class__.__name__}): {ggl_assignment}")

        if isinstance(ggl_assignment.rvalue, L1_REG):
            rvalue = self.visit_l1_reg(ggl_assignment.rvalue)
        elif isinstance(ggl_assignment.rvalue, L2_REG):
            rvalue = self.visit_l2_reg(ggl_assignment.rvalue)
        elif isinstance(ggl_assignment.rvalue, LXRegWithOffsets):
            rvalue = self.visit_lx_reg_with_offsets(ggl_assignment.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({ggl_assignment.rvalue.__class__.__name__}): {ggl_assignment.rvalue}")

        return GGL_ASSIGNMENT(
            rvalue=rvalue)

    def transform_lgl_assignment(self: "BLEIRTransformer", lgl_assignment: LGL_ASSIGNMENT) -> LGL_ASSIGNMENT:
        if not isinstance(lgl_assignment, LGL_ASSIGNMENT):
            raise SyntacticError(f"Unsupported lgl_assignment type ({lgl_assignment.__class__.__name__}): {lgl_assignment}")

        if isinstance(lgl_assignment.rvalue, L1_REG):
            rvalue = self.visit_l1_reg(lgl_assignment.rvalue)
        elif isinstance(lgl_assignment.rvalue, L2_REG):
            rvalue = self.visit_l2_reg(lgl_assignment.rvalue)
        elif isinstance(lgl_assignment.rvalue, LXRegWithOffsets):
            rvalue = self.visit_lx_reg_with_offsets(lgl_assignment.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({lgl_assignment.rvalue.__class__.__name__}): {lgl_assignment.rvalue}")

        return LGL_ASSIGNMENT(
            rvalue=rvalue)

    def transform_lx_assignment(self: "BLEIRTransformer", lx_assignment: LX_ASSIGNMENT) -> LX_ASSIGNMENT:
        if not isinstance(lx_assignment, LX_ASSIGNMENT):
            raise SyntacticError(f"Unsupported lx_assignment type ({lx_assignment.__class__.__name__}): {lx_assignment}")

        if isinstance(lx_assignment.lvalue, L1_REG):
            lvalue = self.visit_l1_reg(lx_assignment.lvalue)
        elif isinstance(lx_assignment.lvalue, L2_REG):
            lvalue = self.visit_l2_reg(lx_assignment.lvalue)
        elif isinstance(lx_assignment.lvalue, LXRegWithOffsets):
            lvalue = self.visit_lx_reg_with_offsets(lx_assignment.lvalue)
        else:
            raise SyntacticError(f"Unsupported lvalue type ({lx_assignment.lvalue.__class__.__name__}): {lx_assignment.lvalue}")

        if isinstance(lx_assignment.rvalue, GGL_EXPR):
            rvalue = self.visit_ggl_expr(lx_assignment.rvalue)
        elif isinstance(lx_assignment.rvalue, LGL_EXPR):
            rvalue = self.visit_lgl_expr(lx_assignment.rvalue)
        else:
            raise SyntacticError(f"Unsupported rvalue type ({lx_assignment.rvalue.__class__.__name__}): {lx_assignment.rvalue}")

        return LX_ASSIGNMENT(
            lvalue=lvalue,
            rvalue=rvalue)

    def transform_ggl_expr(self: "BLEIRTransformer", ggl_expr: GGL_EXPR) -> GGL_EXPR:
        if not isinstance(ggl_expr, GGL_EXPR):
            raise SyntacticError(f"Unsupported ggl_expr type ({ggl_expr.__class__.__name__}): {ggl_expr}")
        return ggl_expr

    def transform_lgl_expr(self: "BLEIRTransformer", lgl_expr: LGL_EXPR) -> LGL_EXPR:
        if not isinstance(lgl_expr, LGL_EXPR):
            raise SyntacticError(f"Unsupported lgl_expr type ({lgl_expr.__class__.__name__}): {lgl_expr}")
        return lgl_expr

    def transform_glass_statement(self: "BLEIRTransformer", glass_statement: GlassStatement) -> GlassStatement:
        if not isinstance(glass_statement, GlassStatement):
            raise SyntacticError(f"Unsupported glass_statement type ({glass_statement.__class__.__name__}): {glass_statement}")

        if isinstance(glass_statement.subject, EWE_REG):
            subject = self.visit_ewe_reg(glass_statement.subject)
        elif isinstance(glass_statement.subject, L1_REG):
            subject = self.visit_l1_reg(glass_statement.subject)
        elif isinstance(glass_statement.subject, L2_REG):
            subject = self.visit_l2_reg(glass_statement.subject)
        elif isinstance(glass_statement.subject, LGL_EXPR):
            subject = self.visit_lgl_expr(glass_statement.subject)
        elif isinstance(glass_statement.subject, LXRegWithOffsets):
            subject = self.visit_lx_reg_with_offsets(glass_statement.subject)
        elif isinstance(glass_statement.subject, RE_REG):
            subject = self.visit_re_reg(glass_statement.subject)
        elif isinstance(glass_statement.subject, RN_REG):
            subject = self.visit_rn_reg(glass_statement.subject)
        elif isinstance(glass_statement.subject, RSP256_EXPR):
            subject = self.visit_rsp256_expr(glass_statement.subject)
        elif isinstance(glass_statement.subject, RSP2K_EXPR):
            subject = self.visit_rsp2k_expr(glass_statement.subject)
        elif isinstance(glass_statement.subject, RSP32K_EXPR):
            subject = self.visit_rsp32k_expr(glass_statement.subject)
        elif isinstance(glass_statement.subject, SRC_EXPR):
            subject = self.visit_src_expr(glass_statement.subject)
        else:
            raise SyntacticError(f"Unsupported subject type ({glass_statement.subject.__class__.__name__}): {glass_statement.subject}")

        sections = []
        for section in glass_statement.sections:
            if isinstance(section, int):
                pass
            else:
                raise SyntacticError(f"Unsupported sections type ({section.__class__.__name__}): {section}")
            sections.append(section)

        plats = []
        for plat in glass_statement.plats:
            if isinstance(plat, int):
                pass
            else:
                raise SyntacticError(f"Unsupported plats type ({plat.__class__.__name__}): {plat}")
            plats.append(plat)

        if isinstance(glass_statement.fmt, GlassFormat):
            fmt = self.visit_glass_format(glass_statement.fmt)
        else:
            raise SyntacticError(f"Unsupported fmt type ({glass_statement.fmt.__class__.__name__}): {glass_statement.fmt}")

        if isinstance(glass_statement.order, GlassOrder):
            order = self.visit_glass_order(glass_statement.order)
        else:
            raise SyntacticError(f"Unsupported order type ({glass_statement.order.__class__.__name__}): {glass_statement.order}")

        return GlassStatement(
            subject=subject,
            sections=sections,
            plats=plats,
            fmt=fmt,
            order=order)

    def transform_rsp256_expr(self: "BLEIRTransformer", rsp256_expr: RSP256_EXPR) -> RSP256_EXPR:
        if not isinstance(rsp256_expr, RSP256_EXPR):
            raise SyntacticError(f"Unsupported rsp256_expr type ({rsp256_expr.__class__.__name__}): {rsp256_expr}")
        return rsp256_expr

    def transform_rsp2k_expr(self: "BLEIRTransformer", rsp2k_expr: RSP2K_EXPR) -> RSP2K_EXPR:
        if not isinstance(rsp2k_expr, RSP2K_EXPR):
            raise SyntacticError(f"Unsupported rsp2k_expr type ({rsp2k_expr.__class__.__name__}): {rsp2k_expr}")
        return rsp2k_expr

    def transform_rsp32k_expr(self: "BLEIRTransformer", rsp32k_expr: RSP32K_EXPR) -> RSP32K_EXPR:
        if not isinstance(rsp32k_expr, RSP32K_EXPR):
            raise SyntacticError(f"Unsupported rsp32k_expr type ({rsp32k_expr.__class__.__name__}): {rsp32k_expr}")
        return rsp32k_expr

    def transform_glass_format(self: "BLEIRTransformer", glass_format: GlassFormat) -> GlassFormat:
        if not isinstance(glass_format, GlassFormat):
            raise SyntacticError(f"Unsupported glass_format type ({glass_format.__class__.__name__}): {glass_format}")
        return glass_format

    def transform_glass_order(self: "BLEIRTransformer", glass_order: GlassOrder) -> GlassOrder:
        if not isinstance(glass_order, GlassOrder):
            raise SyntacticError(f"Unsupported glass_order type ({glass_order.__class__.__name__}): {glass_order}")
        return glass_order

    def transform_fragment_metadata(self: "BLEIRTransformer", fragment_metadata: FragmentMetadata) -> FragmentMetadata:
        if not isinstance(fragment_metadata, FragmentMetadata):
            raise SyntacticError(f"Unsupported fragment_metadata type ({fragment_metadata.__class__.__name__}): {fragment_metadata}")
        return fragment_metadata

    def transform_allocated_register(self: "BLEIRTransformer", allocated_register: AllocatedRegister) -> AllocatedRegister:
        if not isinstance(allocated_register, AllocatedRegister):
            raise SyntacticError(f"Unsupported allocated_register type ({allocated_register.__class__.__name__}): {allocated_register}")

        if isinstance(allocated_register.parameter, RN_REG):
            parameter = self.visit_rn_reg(allocated_register.parameter)
        elif isinstance(allocated_register.parameter, RE_REG):
            parameter = self.visit_re_reg(allocated_register.parameter)
        elif isinstance(allocated_register.parameter, EWE_REG):
            parameter = self.visit_ewe_reg(allocated_register.parameter)
        elif isinstance(allocated_register.parameter, L1_REG):
            parameter = self.visit_l1_reg(allocated_register.parameter)
        elif isinstance(allocated_register.parameter, L2_REG):
            parameter = self.visit_l2_reg(allocated_register.parameter)
        elif isinstance(allocated_register.parameter, SM_REG):
            parameter = self.visit_sm_reg(allocated_register.parameter)
        else:
            raise SyntacticError(f"Unsupported parameter type ({allocated_register.parameter.__class__.__name__}): {allocated_register.parameter}")

        if isinstance(allocated_register.register, str):
            register = allocated_register.register
        else:
            raise SyntacticError(f"Unsupported register type ({allocated_register.register.__class__.__name__}): {allocated_register.register}")

        if isinstance(allocated_register.comment, MultiLineComment):
            comment = self.visit_multi_line_comment(allocated_register.comment)
        elif isinstance(allocated_register.comment, SingleLineComment):
            comment = self.visit_single_line_comment(allocated_register.comment)
        elif isinstance(allocated_register.comment, TrailingComment):
            comment = self.visit_trailing_comment(allocated_register.comment)
        elif isinstance(allocated_register.comment, NoneType):
            comment = allocated_register.comment
        elif allocated_register.comment is None:
            comment = None
        else:
            raise SyntacticError(f"Unsupported comment type ({allocated_register.comment.__class__.__name__}): {allocated_register.comment}")

        return AllocatedRegister(
            parameter=parameter,
            register=register,
            comment=comment)

    def transform_caller_metadata(self: "BLEIRTransformer", caller_metadata: CallerMetadata) -> CallerMetadata:
        if not isinstance(caller_metadata, CallerMetadata):
            raise SyntacticError(f"Unsupported caller_metadata type ({caller_metadata.__class__.__name__}): {caller_metadata}")
        return caller_metadata

    def transform_call_metadata(self: "BLEIRTransformer", call_metadata: CallMetadata) -> CallMetadata:
        if not isinstance(call_metadata, CallMetadata):
            raise SyntacticError(f"Unsupported call_metadata type ({call_metadata.__class__.__name__}): {call_metadata}")
        return call_metadata

    def transform_snippet_metadata(self: "BLEIRTransformer", snippet_metadata: SnippetMetadata) -> SnippetMetadata:
        if not isinstance(snippet_metadata, SnippetMetadata):
            raise SyntacticError(f"Unsupported snippet_metadata type ({snippet_metadata.__class__.__name__}): {snippet_metadata}")
        return snippet_metadata
```
