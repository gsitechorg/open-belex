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

class BLEIRListener:

    def enter_snippet(self: "BLEIRListener", snippet: Snippet) -> None:
        raise NotImplementedError

    def exit_snippet(self: "BLEIRListener", snippet: Snippet) -> None:
        raise NotImplementedError

    def enter_example(self: "BLEIRListener", example: Example) -> None:
        raise NotImplementedError

    def exit_example(self: "BLEIRListener", example: Example) -> None:
        raise NotImplementedError

    def enter_value_parameter(self: "BLEIRListener", value_parameter: ValueParameter) -> None:
        raise NotImplementedError

    def exit_value_parameter(self: "BLEIRListener", value_parameter: ValueParameter) -> None:
        raise NotImplementedError

    def enter_fragment_caller_call(self: "BLEIRListener", fragment_caller_call: FragmentCallerCall) -> None:
        raise NotImplementedError

    def exit_fragment_caller_call(self: "BLEIRListener", fragment_caller_call: FragmentCallerCall) -> None:
        raise NotImplementedError

    def enter_fragment_caller(self: "BLEIRListener", fragment_caller: FragmentCaller) -> None:
        raise NotImplementedError

    def exit_fragment_caller(self: "BLEIRListener", fragment_caller: FragmentCaller) -> None:
        raise NotImplementedError

    def enter_fragment(self: "BLEIRListener", fragment: Fragment) -> None:
        raise NotImplementedError

    def exit_fragment(self: "BLEIRListener", fragment: Fragment) -> None:
        raise NotImplementedError

    def enter_rn_reg(self: "BLEIRListener", rn_reg: RN_REG) -> None:
        raise NotImplementedError

    def exit_rn_reg(self: "BLEIRListener", rn_reg: RN_REG) -> None:
        raise NotImplementedError

    def enter_inline_comment(self: "BLEIRListener", inline_comment: InlineComment) -> None:
        raise NotImplementedError

    def exit_inline_comment(self: "BLEIRListener", inline_comment: InlineComment) -> None:
        raise NotImplementedError

    def enter_re_reg(self: "BLEIRListener", re_reg: RE_REG) -> None:
        raise NotImplementedError

    def exit_re_reg(self: "BLEIRListener", re_reg: RE_REG) -> None:
        raise NotImplementedError

    def enter_ewe_reg(self: "BLEIRListener", ewe_reg: EWE_REG) -> None:
        raise NotImplementedError

    def exit_ewe_reg(self: "BLEIRListener", ewe_reg: EWE_REG) -> None:
        raise NotImplementedError

    def enter_l1_reg(self: "BLEIRListener", l1_reg: L1_REG) -> None:
        raise NotImplementedError

    def exit_l1_reg(self: "BLEIRListener", l1_reg: L1_REG) -> None:
        raise NotImplementedError

    def enter_l2_reg(self: "BLEIRListener", l2_reg: L2_REG) -> None:
        raise NotImplementedError

    def exit_l2_reg(self: "BLEIRListener", l2_reg: L2_REG) -> None:
        raise NotImplementedError

    def enter_sm_reg(self: "BLEIRListener", sm_reg: SM_REG) -> None:
        raise NotImplementedError

    def exit_sm_reg(self: "BLEIRListener", sm_reg: SM_REG) -> None:
        raise NotImplementedError

    def enter_multi_statement(self: "BLEIRListener", multi_statement: MultiStatement) -> None:
        raise NotImplementedError

    def exit_multi_statement(self: "BLEIRListener", multi_statement: MultiStatement) -> None:
        raise NotImplementedError

    def enter_statement(self: "BLEIRListener", statement: STATEMENT) -> None:
        raise NotImplementedError

    def exit_statement(self: "BLEIRListener", statement: STATEMENT) -> None:
        raise NotImplementedError

    def enter_masked(self: "BLEIRListener", masked: MASKED) -> None:
        raise NotImplementedError

    def exit_masked(self: "BLEIRListener", masked: MASKED) -> None:
        raise NotImplementedError

    def enter_mask(self: "BLEIRListener", mask: MASK) -> None:
        raise NotImplementedError

    def exit_mask(self: "BLEIRListener", mask: MASK) -> None:
        raise NotImplementedError

    def enter_shifted_sm_reg(self: "BLEIRListener", shifted_sm_reg: SHIFTED_SM_REG) -> None:
        raise NotImplementedError

    def exit_shifted_sm_reg(self: "BLEIRListener", shifted_sm_reg: SHIFTED_SM_REG) -> None:
        raise NotImplementedError

    def enter_unary_op(self: "BLEIRListener", unary_op: UNARY_OP) -> None:
        raise NotImplementedError

    def exit_unary_op(self: "BLEIRListener", unary_op: UNARY_OP) -> None:
        raise NotImplementedError

    def enter_read_write_inhibit(self: "BLEIRListener", read_write_inhibit: ReadWriteInhibit) -> None:
        raise NotImplementedError

    def exit_read_write_inhibit(self: "BLEIRListener", read_write_inhibit: ReadWriteInhibit) -> None:
        raise NotImplementedError

    def enter_assignment(self: "BLEIRListener", assignment: ASSIGNMENT) -> None:
        raise NotImplementedError

    def exit_assignment(self: "BLEIRListener", assignment: ASSIGNMENT) -> None:
        raise NotImplementedError

    def enter_read(self: "BLEIRListener", read: READ) -> None:
        raise NotImplementedError

    def exit_read(self: "BLEIRListener", read: READ) -> None:
        raise NotImplementedError

    def enter_assign_op(self: "BLEIRListener", assign_op: ASSIGN_OP) -> None:
        raise NotImplementedError

    def exit_assign_op(self: "BLEIRListener", assign_op: ASSIGN_OP) -> None:
        raise NotImplementedError

    def enter_unary_expr(self: "BLEIRListener", unary_expr: UNARY_EXPR) -> None:
        raise NotImplementedError

    def exit_unary_expr(self: "BLEIRListener", unary_expr: UNARY_EXPR) -> None:
        raise NotImplementedError

    def enter_unary_sb(self: "BLEIRListener", unary_sb: UNARY_SB) -> None:
        raise NotImplementedError

    def exit_unary_sb(self: "BLEIRListener", unary_sb: UNARY_SB) -> None:
        raise NotImplementedError

    def enter_sb_expr(self: "BLEIRListener", sb_expr: SB_EXPR) -> None:
        raise NotImplementedError

    def exit_sb_expr(self: "BLEIRListener", sb_expr: SB_EXPR) -> None:
        raise NotImplementedError

    def enter_unary_src(self: "BLEIRListener", unary_src: UNARY_SRC) -> None:
        raise NotImplementedError

    def exit_unary_src(self: "BLEIRListener", unary_src: UNARY_SRC) -> None:
        raise NotImplementedError

    def enter_src_expr(self: "BLEIRListener", src_expr: SRC_EXPR) -> None:
        raise NotImplementedError

    def exit_src_expr(self: "BLEIRListener", src_expr: SRC_EXPR) -> None:
        raise NotImplementedError

    def enter_bit_expr(self: "BLEIRListener", bit_expr: BIT_EXPR) -> None:
        raise NotImplementedError

    def exit_bit_expr(self: "BLEIRListener", bit_expr: BIT_EXPR) -> None:
        raise NotImplementedError

    def enter_binary_expr(self: "BLEIRListener", binary_expr: BINARY_EXPR) -> None:
        raise NotImplementedError

    def exit_binary_expr(self: "BLEIRListener", binary_expr: BINARY_EXPR) -> None:
        raise NotImplementedError

    def enter_binop(self: "BLEIRListener", binop: BINOP) -> None:
        raise NotImplementedError

    def exit_binop(self: "BLEIRListener", binop: BINOP) -> None:
        raise NotImplementedError

    def enter_rl_expr(self: "BLEIRListener", rl_expr: RL_EXPR) -> None:
        raise NotImplementedError

    def exit_rl_expr(self: "BLEIRListener", rl_expr: RL_EXPR) -> None:
        raise NotImplementedError

    def enter_lx_reg_with_offsets(self: "BLEIRListener", lx_reg_with_offsets: LXRegWithOffsets) -> None:
        raise NotImplementedError

    def exit_lx_reg_with_offsets(self: "BLEIRListener", lx_reg_with_offsets: LXRegWithOffsets) -> None:
        raise NotImplementedError

    def enter_multi_line_comment(self: "BLEIRListener", multi_line_comment: MultiLineComment) -> None:
        raise NotImplementedError

    def exit_multi_line_comment(self: "BLEIRListener", multi_line_comment: MultiLineComment) -> None:
        raise NotImplementedError

    def enter_single_line_comment(self: "BLEIRListener", single_line_comment: SingleLineComment) -> None:
        raise NotImplementedError

    def exit_single_line_comment(self: "BLEIRListener", single_line_comment: SingleLineComment) -> None:
        raise NotImplementedError

    def enter_trailing_comment(self: "BLEIRListener", trailing_comment: TrailingComment) -> None:
        raise NotImplementedError

    def exit_trailing_comment(self: "BLEIRListener", trailing_comment: TrailingComment) -> None:
        raise NotImplementedError

    def enter_write(self: "BLEIRListener", write: WRITE) -> None:
        raise NotImplementedError

    def exit_write(self: "BLEIRListener", write: WRITE) -> None:
        raise NotImplementedError

    def enter_broadcast(self: "BLEIRListener", broadcast: BROADCAST) -> None:
        raise NotImplementedError

    def exit_broadcast(self: "BLEIRListener", broadcast: BROADCAST) -> None:
        raise NotImplementedError

    def enter_broadcast_expr(self: "BLEIRListener", broadcast_expr: BROADCAST_EXPR) -> None:
        raise NotImplementedError

    def exit_broadcast_expr(self: "BLEIRListener", broadcast_expr: BROADCAST_EXPR) -> None:
        raise NotImplementedError

    def enter_rsp16_assignment(self: "BLEIRListener", rsp16_assignment: RSP16_ASSIGNMENT) -> None:
        raise NotImplementedError

    def exit_rsp16_assignment(self: "BLEIRListener", rsp16_assignment: RSP16_ASSIGNMENT) -> None:
        raise NotImplementedError

    def enter_rsp16_rvalue(self: "BLEIRListener", rsp16_rvalue: RSP16_RVALUE) -> None:
        raise NotImplementedError

    def exit_rsp16_rvalue(self: "BLEIRListener", rsp16_rvalue: RSP16_RVALUE) -> None:
        raise NotImplementedError

    def enter_rsp256_assignment(self: "BLEIRListener", rsp256_assignment: RSP256_ASSIGNMENT) -> None:
        raise NotImplementedError

    def exit_rsp256_assignment(self: "BLEIRListener", rsp256_assignment: RSP256_ASSIGNMENT) -> None:
        raise NotImplementedError

    def enter_rsp256_rvalue(self: "BLEIRListener", rsp256_rvalue: RSP256_RVALUE) -> None:
        raise NotImplementedError

    def exit_rsp256_rvalue(self: "BLEIRListener", rsp256_rvalue: RSP256_RVALUE) -> None:
        raise NotImplementedError

    def enter_rsp2k_assignment(self: "BLEIRListener", rsp2k_assignment: RSP2K_ASSIGNMENT) -> None:
        raise NotImplementedError

    def exit_rsp2k_assignment(self: "BLEIRListener", rsp2k_assignment: RSP2K_ASSIGNMENT) -> None:
        raise NotImplementedError

    def enter_rsp2k_rvalue(self: "BLEIRListener", rsp2k_rvalue: RSP2K_RVALUE) -> None:
        raise NotImplementedError

    def exit_rsp2k_rvalue(self: "BLEIRListener", rsp2k_rvalue: RSP2K_RVALUE) -> None:
        raise NotImplementedError

    def enter_rsp32k_assignment(self: "BLEIRListener", rsp32k_assignment: RSP32K_ASSIGNMENT) -> None:
        raise NotImplementedError

    def exit_rsp32k_assignment(self: "BLEIRListener", rsp32k_assignment: RSP32K_ASSIGNMENT) -> None:
        raise NotImplementedError

    def enter_rsp32k_rvalue(self: "BLEIRListener", rsp32k_rvalue: RSP32K_RVALUE) -> None:
        raise NotImplementedError

    def exit_rsp32k_rvalue(self: "BLEIRListener", rsp32k_rvalue: RSP32K_RVALUE) -> None:
        raise NotImplementedError

    def enter_special(self: "BLEIRListener", special: SPECIAL) -> None:
        raise NotImplementedError

    def exit_special(self: "BLEIRListener", special: SPECIAL) -> None:
        raise NotImplementedError

    def enter_ggl_assignment(self: "BLEIRListener", ggl_assignment: GGL_ASSIGNMENT) -> None:
        raise NotImplementedError

    def exit_ggl_assignment(self: "BLEIRListener", ggl_assignment: GGL_ASSIGNMENT) -> None:
        raise NotImplementedError

    def enter_lgl_assignment(self: "BLEIRListener", lgl_assignment: LGL_ASSIGNMENT) -> None:
        raise NotImplementedError

    def exit_lgl_assignment(self: "BLEIRListener", lgl_assignment: LGL_ASSIGNMENT) -> None:
        raise NotImplementedError

    def enter_lx_assignment(self: "BLEIRListener", lx_assignment: LX_ASSIGNMENT) -> None:
        raise NotImplementedError

    def exit_lx_assignment(self: "BLEIRListener", lx_assignment: LX_ASSIGNMENT) -> None:
        raise NotImplementedError

    def enter_ggl_expr(self: "BLEIRListener", ggl_expr: GGL_EXPR) -> None:
        raise NotImplementedError

    def exit_ggl_expr(self: "BLEIRListener", ggl_expr: GGL_EXPR) -> None:
        raise NotImplementedError

    def enter_lgl_expr(self: "BLEIRListener", lgl_expr: LGL_EXPR) -> None:
        raise NotImplementedError

    def exit_lgl_expr(self: "BLEIRListener", lgl_expr: LGL_EXPR) -> None:
        raise NotImplementedError

    def enter_glass_statement(self: "BLEIRListener", glass_statement: GlassStatement) -> None:
        raise NotImplementedError

    def exit_glass_statement(self: "BLEIRListener", glass_statement: GlassStatement) -> None:
        raise NotImplementedError

    def enter_rsp256_expr(self: "BLEIRListener", rsp256_expr: RSP256_EXPR) -> None:
        raise NotImplementedError

    def exit_rsp256_expr(self: "BLEIRListener", rsp256_expr: RSP256_EXPR) -> None:
        raise NotImplementedError

    def enter_rsp2k_expr(self: "BLEIRListener", rsp2k_expr: RSP2K_EXPR) -> None:
        raise NotImplementedError

    def exit_rsp2k_expr(self: "BLEIRListener", rsp2k_expr: RSP2K_EXPR) -> None:
        raise NotImplementedError

    def enter_rsp32k_expr(self: "BLEIRListener", rsp32k_expr: RSP32K_EXPR) -> None:
        raise NotImplementedError

    def exit_rsp32k_expr(self: "BLEIRListener", rsp32k_expr: RSP32K_EXPR) -> None:
        raise NotImplementedError

    def enter_glass_format(self: "BLEIRListener", glass_format: GlassFormat) -> None:
        raise NotImplementedError

    def exit_glass_format(self: "BLEIRListener", glass_format: GlassFormat) -> None:
        raise NotImplementedError

    def enter_glass_order(self: "BLEIRListener", glass_order: GlassOrder) -> None:
        raise NotImplementedError

    def exit_glass_order(self: "BLEIRListener", glass_order: GlassOrder) -> None:
        raise NotImplementedError

    def enter_fragment_metadata(self: "BLEIRListener", fragment_metadata: FragmentMetadata) -> None:
        raise NotImplementedError

    def exit_fragment_metadata(self: "BLEIRListener", fragment_metadata: FragmentMetadata) -> None:
        raise NotImplementedError

    def enter_allocated_register(self: "BLEIRListener", allocated_register: AllocatedRegister) -> None:
        raise NotImplementedError

    def exit_allocated_register(self: "BLEIRListener", allocated_register: AllocatedRegister) -> None:
        raise NotImplementedError

    def enter_caller_metadata(self: "BLEIRListener", caller_metadata: CallerMetadata) -> None:
        raise NotImplementedError

    def exit_caller_metadata(self: "BLEIRListener", caller_metadata: CallerMetadata) -> None:
        raise NotImplementedError

    def enter_call_metadata(self: "BLEIRListener", call_metadata: CallMetadata) -> None:
        raise NotImplementedError

    def exit_call_metadata(self: "BLEIRListener", call_metadata: CallMetadata) -> None:
        raise NotImplementedError

    def enter_snippet_metadata(self: "BLEIRListener", snippet_metadata: SnippetMetadata) -> None:
        raise NotImplementedError

    def exit_snippet_metadata(self: "BLEIRListener", snippet_metadata: SnippetMetadata) -> None:
        raise NotImplementedError
```
