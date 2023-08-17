r"""
By Dylon Edwards
"""

from datetime import datetime
from typing import Any, Union

from jinja2 import Environment
from jinja2.ext import Extension

from open_belex.bleir.types import (EWE_REG, L1_REG, L2_REG, RE_REG, RN_REG,
                                    SM_REG, ActualParameter, AllocatedRegister,
                                    FormalParameter, MultiLineComment,
                                    SingleLineComment, TrailingComment)
from open_belex.common.gvml import GVML_SM_REG_VALS


class TemplateExtensions(Extension):

    def __init__(self: "TemplateExtensions", environment: Environment) -> None:
        super().__init__(environment)

        # Predicates
        environment.tests.update({
            "single_line_commented": self.has_single_line_comment,
            "multi_line_commented": self.has_multi_line_comment,
            "trailing_commented": self.has_trailing_comment,
        })

        # Filters
        environment.filters.update({
            "dec": self.fmt_dec,
            "hex": self.fmt_hex,
        })

        # Globals
        environment.globals.update({
            "apl_set_reg": self.apl_set_reg,
            "condef_register": self.condef_register,
            "emit_copyright": self.emit_copyright,
            "restore_reg": self.restore_reg,
        })

    ## ========== ##
    ## Predicates ##
    ## ========== ##

    def has_single_line_comment(self: "TemplateExtensions", context: Any) -> bool:
        return isinstance(context.comment, SingleLineComment)

    def has_multi_line_comment(self: "TemplateExtensions", context: Any) -> bool:
        return isinstance(context.comment, MultiLineComment)

    def has_trailing_comment(self: "TemplateExtensions", context: Any) -> bool:
        return isinstance(context.comment, TrailingComment)

    ## ======= ##
    ## Filters ##
    ## ======= ##

    def fmt_hex(self: "TemplateExtensions", parameter: ActualParameter) -> str:
        return f"0x{parameter:04X}"

    def fmt_dec(self: "TemplateExtensions", parameter: ActualParameter) -> str:
        return f"{parameter}"

    ## ======= ##
    ## Globals ##
    ## ======= ##

    def apl_set_reg(
            self: "TemplateExtensions",
            register_or_allocated_register: Union[FormalParameter,
                                                  AllocatedRegister]) -> str:

        if isinstance(register_or_allocated_register, FormalParameter.__args__):
            register = register_or_allocated_register

            if isinstance(register, RN_REG):
                value = register.row_number
            elif isinstance(register, RE_REG):
                value = f"0x{register.row_mask:06X}"
            elif isinstance(register, L1_REG):
                value = register.bank_group_row
            elif isinstance(register, SM_REG):
                value = self.fmt_hex(register.constant_value)
            else:
                raise ValueError(
                    f"Unsupported register type "
                    f"({register.__class__.__name__}): {register}")

            register_type = register.__class__.__name__
            apl_set_reg = f"apl_set_{register_type.lower()}"
            return f"{apl_set_reg}({register.identifier}, {value});"

        statements = []

        allocated_register = register_or_allocated_register
        register = allocated_register.parameter
        register_type = register.__class__.__name__
        apl_set_reg = f"apl_set_{register_type.lower()}"

        if allocated_register.isa(RN_REG):
            if register.initial_value is not None:
                if register.comment is not None:
                    statements[-1] += f" /* {register.comment.value} */"
                statements.append(
                    f"belex_cpy_imm_16({allocated_register.value_param}, {register.initial_value});")

        elif allocated_register.isa(L1_REG):
            if register.bank_group_row is not None:
                statements.append(
                    f"u32 {register.value_param} = {register.bank_group_row};")
                if register.comment is not None:
                    statements[-1] += f" /* {register.comment.value} */"

        elif allocated_register.isa(SM_REG):
            if register.constant_value is not None:
                statements.append(
                    f"u32 {allocated_register.value_param} = 0x{register.constant_value:04X};")
                if register.comment is not None:
                    statements[-1] += f" /* {register.comment.value} */"

            if register.is_section:
                vp_nym = allocated_register.identifier
                statements.append("/* Coerce section value to mask value: */")
                statements.append(f"{vp_nym} = (0x0001 << {vp_nym});")

        elif not allocated_register.isa((L1_REG, L2_REG, RE_REG, EWE_REG)):
            raise ValueError(
                f"Unsupported register type: {allocated_register.register_type}")

        # statements.append(
        #     f"{apl_set_reg}({allocated_register.register}, {allocated_register.value_param});")
        statements.append(
            f"{apl_set_reg}({allocated_register.register}, {allocated_register.identifier});")
        apl_set_reg = "\n    ".join(statements)

        if self.has_multi_line_comment(allocated_register):
            template = self.environment.get_template(
                "partials/apl_set_reg_with_multi_line_comment.jinja")
            return template.render(comments=allocated_register.comment.lines,
                                   apl_set_reg=apl_set_reg)
        elif self.has_single_line_comment(allocated_register):
            template = self.environment.get_template(
                "partials/apl_set_reg_with_single_line_comment.jinja")
            return template.render(comment=allocated_register.comment.line,
                                   apl_set_reg=apl_set_reg)
        elif self.has_trailing_comment(allocated_register):
            template = self.environment.get_template(
                "partials/apl_set_reg_with_trailing_comment.jinja")
            return template.render(comment=allocated_register.comment.value,
                                   apl_set_reg=apl_set_reg)
        else:
            return apl_set_reg

    def condef_register(self: "TemplateExtensions",
                        register: FormalParameter) -> str:

        if isinstance(register, (RN_REG, RE_REG, EWE_REG, SM_REG)):
            prefix = f"{register.__class__.__name__}_"
        elif isinstance(register, L1_REG):
            prefix = "L1_ADDR_REG_"
        else:
            raise ValueError(
                f"Unsupported register type "
                f"({register.__class__.__name__}): {register}")

        return "\n".join([
            f"#ifndef {register.identifier}",
            f"#define {register.identifier} {prefix}{register.register}",
            f"#endif // {register.identifier}",
        ])

    def restore_reg(self: "TemplateExtensions",
                    allocated_register: AllocatedRegister) -> str:
        if allocated_register.isa(SM_REG):
            sm_reg = allocated_register.register
            if sm_reg in GVML_SM_REG_VALS:
                sm_reg_val = GVML_SM_REG_VALS[sm_reg]
                return f"\n    apl_set_sm_reg({sm_reg}, {sm_reg_val});"
        return ""

    def emit_copyright(self: "TemplateExtensions", lower_year: int = 2019) -> str:
        timestamp = datetime.now()
        upper_year = timestamp.year
        template = self.environment.get_template("partials/copyright.jinja")
        return template.render(lower_year=lower_year, upper_year=upper_year)
