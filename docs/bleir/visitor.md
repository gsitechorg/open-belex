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

class BLEIRVisitor:

    def visit_snippet(self: "BLEIRVisitor", snippet: Snippet) -> None:
        raise NotImplementedError

    def visit_example(self: "BLEIRVisitor", example: Example) -> None:
        raise NotImplementedError

    def visit_value_parameter(self: "BLEIRVisitor", value_parameter: ValueParameter) -> None:
        raise NotImplementedError

    def visit_fragment_caller_call(self: "BLEIRVisitor", fragment_caller_call: FragmentCallerCall) -> None:
        raise NotImplementedError

    def visit_fragment_caller(self: "BLEIRVisitor", fragment_caller: FragmentCaller) -> None:
        raise NotImplementedError

    def visit_fragment(self: "BLEIRVisitor", fragment: Fragment) -> None:
        raise NotImplementedError

    def visit_rn_reg(self: "BLEIRVisitor", rn_reg: RN_REG) -> None:
        raise NotImplementedError

    def visit_inline_comment(self: "BLEIRVisitor", inline_comment: InlineComment) -> None:
        raise NotImplementedError

    def visit_re_reg(self: "BLEIRVisitor", re_reg: RE_REG) -> None:
        raise NotImplementedError

    def visit_ewe_reg(self: "BLEIRVisitor", ewe_reg: EWE_REG) -> None:
        raise NotImplementedError

    def visit_l1_reg(self: "BLEIRVisitor", l1_reg: L1_REG) -> None:
        raise NotImplementedError

    def visit_l2_reg(self: "BLEIRVisitor", l2_reg: L2_REG) -> None:
        raise NotImplementedError

    def visit_sm_reg(self: "BLEIRVisitor", sm_reg: SM_REG) -> None:
        raise NotImplementedError

    def visit_multi_statement(self: "BLEIRVisitor", multi_statement: MultiStatement) -> None:
        raise NotImplementedError

    def visit_statement(self: "BLEIRVisitor", statement: STATEMENT) -> None:
        raise NotImplementedError

    def visit_masked(self: "BLEIRVisitor", masked: MASKED) -> None:
        raise NotImplementedError

    def visit_mask(self: "BLEIRVisitor", mask: MASK) -> None:
        raise NotImplementedError

    def visit_shifted_sm_reg(self: "BLEIRVisitor", shifted_sm_reg: SHIFTED_SM_REG) -> None:
        raise NotImplementedError

    def visit_unary_op(self: "BLEIRVisitor", unary_op: UNARY_OP) -> None:
        raise NotImplementedError

    def visit_read_write_inhibit(self: "BLEIRVisitor", read_write_inhibit: ReadWriteInhibit) -> None:
        raise NotImplementedError

    def visit_assignment(self: "BLEIRVisitor", assignment: ASSIGNMENT) -> None:
        raise NotImplementedError

    def visit_read(self: "BLEIRVisitor", read: READ) -> None:
        raise NotImplementedError

    def visit_assign_op(self: "BLEIRVisitor", assign_op: ASSIGN_OP) -> None:
        raise NotImplementedError

    def visit_unary_expr(self: "BLEIRVisitor", unary_expr: UNARY_EXPR) -> None:
        raise NotImplementedError

    def visit_unary_sb(self: "BLEIRVisitor", unary_sb: UNARY_SB) -> None:
        raise NotImplementedError

    def visit_sb_expr(self: "BLEIRVisitor", sb_expr: SB_EXPR) -> None:
        raise NotImplementedError

    def visit_unary_src(self: "BLEIRVisitor", unary_src: UNARY_SRC) -> None:
        raise NotImplementedError

    def visit_src_expr(self: "BLEIRVisitor", src_expr: SRC_EXPR) -> None:
        raise NotImplementedError

    def visit_bit_expr(self: "BLEIRVisitor", bit_expr: BIT_EXPR) -> None:
        raise NotImplementedError

    def visit_binary_expr(self: "BLEIRVisitor", binary_expr: BINARY_EXPR) -> None:
        raise NotImplementedError

    def visit_binop(self: "BLEIRVisitor", binop: BINOP) -> None:
        raise NotImplementedError

    def visit_rl_expr(self: "BLEIRVisitor", rl_expr: RL_EXPR) -> None:
        raise NotImplementedError

    def visit_lx_reg_with_offsets(self: "BLEIRVisitor", lx_reg_with_offsets: LXRegWithOffsets) -> None:
        raise NotImplementedError

    def visit_multi_line_comment(self: "BLEIRVisitor", multi_line_comment: MultiLineComment) -> None:
        raise NotImplementedError

    def visit_single_line_comment(self: "BLEIRVisitor", single_line_comment: SingleLineComment) -> None:
        raise NotImplementedError

    def visit_trailing_comment(self: "BLEIRVisitor", trailing_comment: TrailingComment) -> None:
        raise NotImplementedError

    def visit_write(self: "BLEIRVisitor", write: WRITE) -> None:
        raise NotImplementedError

    def visit_broadcast(self: "BLEIRVisitor", broadcast: BROADCAST) -> None:
        raise NotImplementedError

    def visit_broadcast_expr(self: "BLEIRVisitor", broadcast_expr: BROADCAST_EXPR) -> None:
        raise NotImplementedError

    def visit_rsp16_assignment(self: "BLEIRVisitor", rsp16_assignment: RSP16_ASSIGNMENT) -> None:
        raise NotImplementedError

    def visit_rsp16_rvalue(self: "BLEIRVisitor", rsp16_rvalue: RSP16_RVALUE) -> None:
        raise NotImplementedError

    def visit_rsp256_assignment(self: "BLEIRVisitor", rsp256_assignment: RSP256_ASSIGNMENT) -> None:
        raise NotImplementedError

    def visit_rsp256_rvalue(self: "BLEIRVisitor", rsp256_rvalue: RSP256_RVALUE) -> None:
        raise NotImplementedError

    def visit_rsp2k_assignment(self: "BLEIRVisitor", rsp2k_assignment: RSP2K_ASSIGNMENT) -> None:
        raise NotImplementedError

    def visit_rsp2k_rvalue(self: "BLEIRVisitor", rsp2k_rvalue: RSP2K_RVALUE) -> None:
        raise NotImplementedError

    def visit_rsp32k_assignment(self: "BLEIRVisitor", rsp32k_assignment: RSP32K_ASSIGNMENT) -> None:
        raise NotImplementedError

    def visit_rsp32k_rvalue(self: "BLEIRVisitor", rsp32k_rvalue: RSP32K_RVALUE) -> None:
        raise NotImplementedError

    def visit_special(self: "BLEIRVisitor", special: SPECIAL) -> None:
        raise NotImplementedError

    def visit_ggl_assignment(self: "BLEIRVisitor", ggl_assignment: GGL_ASSIGNMENT) -> None:
        raise NotImplementedError

    def visit_lgl_assignment(self: "BLEIRVisitor", lgl_assignment: LGL_ASSIGNMENT) -> None:
        raise NotImplementedError

    def visit_lx_assignment(self: "BLEIRVisitor", lx_assignment: LX_ASSIGNMENT) -> None:
        raise NotImplementedError

    def visit_ggl_expr(self: "BLEIRVisitor", ggl_expr: GGL_EXPR) -> None:
        raise NotImplementedError

    def visit_lgl_expr(self: "BLEIRVisitor", lgl_expr: LGL_EXPR) -> None:
        raise NotImplementedError

    def visit_glass_statement(self: "BLEIRVisitor", glass_statement: GlassStatement) -> None:
        raise NotImplementedError

    def visit_rsp256_expr(self: "BLEIRVisitor", rsp256_expr: RSP256_EXPR) -> None:
        raise NotImplementedError

    def visit_rsp2k_expr(self: "BLEIRVisitor", rsp2k_expr: RSP2K_EXPR) -> None:
        raise NotImplementedError

    def visit_rsp32k_expr(self: "BLEIRVisitor", rsp32k_expr: RSP32K_EXPR) -> None:
        raise NotImplementedError

    def visit_glass_format(self: "BLEIRVisitor", glass_format: GlassFormat) -> None:
        raise NotImplementedError

    def visit_glass_order(self: "BLEIRVisitor", glass_order: GlassOrder) -> None:
        raise NotImplementedError

    def visit_fragment_metadata(self: "BLEIRVisitor", fragment_metadata: FragmentMetadata) -> None:
        raise NotImplementedError

    def visit_allocated_register(self: "BLEIRVisitor", allocated_register: AllocatedRegister) -> None:
        raise NotImplementedError

    def visit_caller_metadata(self: "BLEIRVisitor", caller_metadata: CallerMetadata) -> None:
        raise NotImplementedError

    def visit_call_metadata(self: "BLEIRVisitor", call_metadata: CallMetadata) -> None:
        raise NotImplementedError

    def visit_snippet_metadata(self: "BLEIRVisitor", snippet_metadata: SnippetMetadata) -> None:
        raise NotImplementedError
```
