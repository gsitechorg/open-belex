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

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from open_belex.bleir.template_extensions import TemplateExtensions
from open_belex.bleir.types import AllocatedRegister, Example, FormalParameter
from open_belex.utils.path_utils import path_wrt_root


class TemplateAccessor(Environment):

    def __init__(self: "TemplateAccessor",
                 templates_path: Path,
                 *args: Sequence[Any],
                 **kwargs: Dict[str, Any]) -> None:
        opts = {
            "loader": FileSystemLoader(templates_path),
            "keep_trailing_newline": False,
            "undefined": StrictUndefined,
            "extensions": [TemplateExtensions],
        }
        opts.update(kwargs)
        super().__init__(*args, **opts)

    def emit(self: "TemplateAccessor", template_path: str, **kwargs) -> str:
        template = self.get_template(template_path)
        return template.render(**kwargs)


class BaryonTemplateAccessor(TemplateAccessor):

    def __init__(self: "BaryonTemplateAccessor",
                 *args: Sequence[Any],
                 templates_path: Path = path_wrt_root("templates/baryon"),
                 **kwargs: Dict[str, Any]) -> None:
        env_opts = {
            "block_start_string": "%{",
            "block_end_string": "}%",
            "variable_start_string": "${",
            "variable_end_string": "}$",
            "comment_start_string": "#{",
            "comment_end_string": "}#",
            "line_statement_prefix": "##",
            "line_comment_prefix": "###",
        }
        env_opts.update(kwargs)
        super().__init__(*args, templates_path, **env_opts)

    def emit_baryon_header(self: "BaryonTemplateAccessor",
                           name: str,
                           declarations: Sequence[str]) -> str:
        return self.emit("baryon_header.jinja",
                         name=name,
                         declarations=declarations)

    def emit_baryon_source(self: "BaryonTemplateAccessor",
                           name: str,
                           definitions: Sequence[str],
                           header_file: str) -> str:
        return self.emit("baryon_source.jinja",
                         name=name,
                         definitions=definitions,
                         header_file=header_file)

    def emit_function_definition(self: "BaryonTemplateAccessor",
                                 frag_nym: str,
                                 parameters: Sequence[str],
                                 declarations: Sequence[str],
                                 instructions: Sequence[str],
                                 loads_xe_reg: bool,
                                 loads_vrs: bool,
                                 loads_src: bool) -> str:
        function_declaration = \
            self.emit_function_declaration(frag_nym=frag_nym,
                                           parameters=parameters)
        function_body = self.emit_function_body(declarations=declarations,
                                                instructions=instructions,
                                                loads_xe_reg=loads_xe_reg,
                                                loads_vrs=loads_vrs,
                                                loads_src=loads_src)
        return self.emit("partials/function_definition.jinja",
                         declaration=function_declaration,
                         body=function_body)

    def emit_function_declaration(self: "BaryonTemplateAccessor",
                                  frag_nym: str,
                                  parameters: Sequence[str]) -> str:
        return self.emit("partials/function_declaration.jinja",
                         frag_nym=frag_nym,
                         parameters=parameters)

    def emit_function_body(self: "BaryonTemplateAccessor",
                           declarations: Sequence[str],
                           instructions: Sequence[str],
                           loads_xe_reg: bool,
                           loads_vrs: bool,
                           loads_src: bool) -> str:
        return self.emit("partials/function_body.jinja",
                         declarations=declarations,
                         instructions=instructions,
                         loads_xe_reg=loads_xe_reg,
                         loads_vrs=loads_vrs,
                         loads_src=loads_src)

    def emit_set_in_place(self: "BaryonTemplateAccessor",
                          in_place: bool) -> str:
        return self.emit("partials/set_in_place.jinja",
                         in_place=in_place)

    def emit_mimic_f_sel_no_op(self: "BaryonTemplateAccessor",
                               patch_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_f_sel_no_op.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_ggl_from_l1(self: "BaryonTemplateAccessor",
                               load_l1: Optional[str],
                               patch_nym: str,
                               l1_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_ggl_from_l1.jinja",
                         load_l1=load_l1,
                         patch_nym=patch_nym,
                         l1_nym=l1_nym,
                         is_patched=is_patched)

    def emit_mimic_ggl_from_rl(self: "BaryonTemplateAccessor",
                               load_mask: Optional[str],
                               patch_nym: str,
                               mask_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_ggl_from_rl.jinja",
                         load_mask=load_mask,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_ggl_from_rl_and_l1(self: "BaryonTemplateAccessor",
                                      load_mask: Optional[str],
                                      load_l1: Optional[str],
                                      patch_nym: str,
                                      mask_nym: str,
                                      l1_nym: str,
                                      is_patched: bool) -> str:
        return self.emit("partials/mimic_ggl_from_rl_and_l1.jinja",
                         load_mask=load_mask,
                         load_l1=load_l1,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         l1_nym=l1_nym,
                         is_patched=is_patched)

    def emit_mimic_gl_from_rl(self: "BaryonTemplateAccessor",
                              load_mask: Optional[str],
                              patch_nym: str,
                              mask_nym: str,
                              is_patched: bool) -> str:
        return self.emit("partials/mimic_gl_from_rl.jinja",
                         load_mask=load_mask,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_l1_from_ggl(self: "BaryonTemplateAccessor",
                               load_l1: Optional[str],
                               patch_nym: str,
                               l1_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_l1_from_ggl.jinja",
                         load_l1=load_l1,
                         patch_nym=patch_nym,
                         l1_nym=l1_nym,
                         is_patched=is_patched)

    def emit_mimic_l1_from_lgl(self: "BaryonTemplateAccessor",
                               load_l1: Optional[str],
                               patch_nym: str,
                               l1_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_l1_from_lgl.jinja",
                         load_l1=load_l1,
                         patch_nym=patch_nym,
                         l1_nym=l1_nym,
                         is_patched=is_patched)

    def emit_mimic_l2_end(self: "BaryonTemplateAccessor",
                          patch_nym: str,
                          is_patched: bool) -> str:
        return self.emit("partials/mimic_l2_end.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_l2_from_lgl(self: "BaryonTemplateAccessor",
                               load_l2: Optional[str],
                               patch_nym: str,
                               l2_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_l2_from_lgl.jinja",
                         load_l2=load_l2,
                         patch_nym=patch_nym,
                         l2_nym=l2_nym,
                         is_patched=is_patched)

    def emit_mimic_lgl_from_l1(self: "BaryonTemplateAccessor",
                               load_l1: Optional[str],
                               patch_nym: str,
                               l1_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_lgl_from_l1.jinja",
                         load_l1=load_l1,
                         patch_nym=patch_nym,
                         l1_nym=l1_nym,
                         is_patched=is_patched)

    def emit_mimic_lgl_from_l2(self: "BaryonTemplateAccessor",
                               load_l2: Optional[str],
                               patch_nym: str,
                               l2_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_lgl_from_l2.jinja",
                         load_l2=load_l2,
                         patch_nym=patch_nym,
                         l2_nym=l2_nym,
                         is_patched=is_patched)

    def emit_mimic_no_op(self: "BaryonTemplateAccessor",
                         patch_nym: str,
                         is_patched: bool) -> str:
        return self.emit("partials/mimic_no_op.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_and_eq_inv_sb(self: "BaryonTemplateAccessor",
                                    load_mask: Optional[str],
                                    load_vrs: str,
                                    patch_nym: str,
                                    mask_nym: str,
                                    is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_and_eq_inv_sb.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_and_eq_sb(self: "BaryonTemplateAccessor",
                                load_mask: Optional[str],
                                load_vrs: str,
                                patch_nym: str,
                                mask_nym: str,
                                is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_and_eq_sb.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_and_eq_sb_and_inv_src(self: "BaryonTemplateAccessor",
                                            load_mask: Optional[str],
                                            load_vrs: str,
                                            load_src: str,
                                            free_src: Optional[str],
                                            patch_nym: str,
                                            mask_nym: str,
                                            is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_and_eq_sb_and_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_and_eq_sb_and_src(self: "BaryonTemplateAccessor",
                                        load_mask: Optional[str],
                                        load_vrs: str,
                                        load_src: str,
                                        free_src: Optional[str],
                                        patch_nym: str,
                                        mask_nym: str,
                                        is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_and_eq_sb_and_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_and_eq_src(self: "BaryonTemplateAccessor",
                                 load_mask: Optional[str],
                                 load_src: str,
                                 free_src: Optional[str],
                                 patch_nym: str,
                                 mask_nym: str,
                                 is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_and_eq_src.jinja",
                         load_mask=load_mask,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_inv_sb(self: "BaryonTemplateAccessor",
                                  load_mask: Optional[str],
                                  load_vrs: str,
                                  patch_nym: str,
                                  mask_nym: str,
                                  is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_inv_sb.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_inv_sb_and_inv_src(self: "BaryonTemplateAccessor",
                                              load_mask: Optional[str],
                                              load_vrs: str,
                                              load_src: str,
                                              free_src: Optional[str],
                                              patch_nym: str,
                                              mask_nym: str,
                                              is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_inv_sb_and_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_inv_sb_and_src(self: "BaryonTemplateAccessor",
                                          load_mask: Optional[str],
                                          load_vrs: str,
                                          load_src: str,
                                          free_src: Optional[str],
                                          patch_nym: str,
                                          mask_nym: str,
                                          is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_inv_sb_and_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_inv_src(self: "BaryonTemplateAccessor",
                                   load_mask: Optional[str],
                                   load_src: str,
                                   free_src: Optional[str],
                                   patch_nym: str,
                                   mask_nym: str,
                                   is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_inv_src.jinja",
                         load_mask=load_mask,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_sb(self: "BaryonTemplateAccessor",
                              load_mask: Optional[str],
                              load_vrs: str,
                              patch_nym: str,
                              mask_nym: str,
                              is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_sb.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_sb_and_inv_src(self: "BaryonTemplateAccessor",
                                          load_mask: Optional[str],
                                          load_vrs: str,
                                          load_src: str,
                                          free_src: Optional[str],
                                          patch_nym: str,
                                          mask_nym: str,
                                          is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_sb_and_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_sb_and_src(self: "BaryonTemplateAccessor",
                                      load_mask: Optional[str],
                                      load_vrs: str,
                                      load_src: str,
                                      free_src: Optional[str],
                                      patch_nym: str,
                                      mask_nym: str,
                                      is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_sb_and_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_sb_xor_inv_src(self: "BaryonTemplateAccessor",
                                          load_mask: Optional[str],
                                          load_vrs: str,
                                          load_src: str,
                                          free_src: Optional[str],
                                          patch_nym: str,
                                          mask_nym: str,
                                          is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_sb_xor_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_sb_xor_src(self: "BaryonTemplateAccessor",
                                      load_mask: Optional[str],
                                      load_vrs: str,
                                      load_src: str,
                                      free_src: Optional[str],
                                      patch_nym: str,
                                      mask_nym: str,
                                      is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_sb_xor_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_sb_or_inv_src(self: "BaryonTemplateAccessor",
                                         load_mask: Optional[str],
                                         load_vrs: str,
                                         load_src: str,
                                         free_src: Optional[str],
                                         patch_nym: str,
                                         mask_nym: str,
                                         is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_sb_or_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)


    def emit_mimic_rl_from_sb_or_src(self: "BaryonTemplateAccessor",
                                     load_mask: Optional[str],
                                     load_vrs: str,
                                     load_src: str,
                                     free_src: Optional[str],
                                     patch_nym: str,
                                     mask_nym: str,
                                     is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_sb_or_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_from_src(self: "BaryonTemplateAccessor",
                               load_mask: Optional[str],
                               load_src: str,
                               free_src: Optional[str],
                               patch_nym: str,
                               mask_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_from_src.jinja",
                         load_mask=load_mask,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_or_eq_inv_src(self: "BaryonTemplateAccessor",
                                    load_mask: Optional[str],
                                    load_src: str,
                                    free_src: Optional[str],
                                    patch_nym: str,
                                    mask_nym: str,
                                    is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_or_eq_inv_src.jinja",
                         load_mask=load_mask,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_or_eq_sb(self: "BaryonTemplateAccessor",
                               load_mask: Optional[str],
                               load_vrs: str,
                               patch_nym: str,
                               mask_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_or_eq_sb.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_or_eq_sb_and_inv_src(self: "BaryonTemplateAccessor",
                                           load_mask: Optional[str],
                                           load_vrs: str,
                                           load_src: str,
                                           free_src: Optional[str],
                                           patch_nym: str,
                                           mask_nym: str,
                                           is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_or_eq_sb_and_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_or_eq_sb_and_src(self: "BaryonTemplateAccessor",
                                       load_mask: Optional[str],
                                       load_vrs: str,
                                       load_src: str,
                                       free_src: Optional[str],
                                       patch_nym: str,
                                       mask_nym: str,
                                       is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_or_eq_sb_and_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_or_eq_src(self: "BaryonTemplateAccessor",
                                load_mask: Optional[str],
                                load_src: str,
                                free_src: Optional[str],
                                patch_nym: str,
                                mask_nym: str,
                                is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_or_eq_src.jinja",
                         load_mask=load_mask,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_xor_eq_inv_src(self: "BaryonTemplateAccessor",
                                     load_mask: Optional[str],
                                     load_src: str,
                                     free_src: Optional[str],
                                     patch_nym: str,
                                     mask_nym: str,
                                     is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_xor_eq_inv_src.jinja",
                         load_mask=load_mask,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_xor_eq_sb(self: "BaryonTemplateAccessor",
                                load_mask: Optional[str],
                                load_vrs: str,
                                patch_nym: str,
                                mask_nym: str,
                                is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_xor_eq_sb.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_xor_eq_sb_and_inv_src(self: "BaryonTemplateAccessor",
                                            load_mask: Optional[str],
                                            load_vrs: str,
                                            load_src: str,
                                            free_src: Optional[str],
                                            patch_nym: str,
                                            mask_nym: str,
                                            is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_xor_eq_sb_and_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_xor_eq_sb_and_src(self: "BaryonTemplateAccessor",
                                        load_mask: Optional[str],
                                        load_vrs: str,
                                        load_src: str,
                                        free_src: Optional[str],
                                        patch_nym: str,
                                        mask_nym: str,
                                        is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_xor_eq_sb_and_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rl_xor_eq_src(self: "BaryonTemplateAccessor",
                                 load_mask: Optional[str],
                                 load_src: str,
                                 free_src: Optional[str],
                                 patch_nym: str,
                                 mask_nym: str,
                                 is_patched: bool) -> str:
        return self.emit("partials/mimic_rl_xor_eq_src.jinja",
                         load_mask=load_mask,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp16_from_rl(self: "BaryonTemplateAccessor",
                                 load_mask: Optional[str],
                                 patch_nym: str,
                                 mask_nym: str,
                                 is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp16_from_rl.jinja",
                         load_mask=load_mask,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp16_from_rsp256(self: "BaryonTemplateAccessor",
                                     patch_nym: str,
                                     is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp16_from_rsp256.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp256_from_rsp16(self: "BaryonTemplateAccessor",
                                     patch_nym: str,
                                     is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp256_from_rsp16.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp256_from_rsp2k(self: "BaryonTemplateAccessor",
                                     patch_nym: str,
                                     is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp256_from_rsp2k.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp2k_from_rsp256(self: "BaryonTemplateAccessor",
                                     patch_nym: str,
                                     is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp2k_from_rsp256.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp2k_from_rsp32k(self: "BaryonTemplateAccessor",
                                     patch_nym: str,
                                     is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp2k_from_rsp32k.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp32k_from_rsp2k(self: "BaryonTemplateAccessor",
                                     patch_nym: str,
                                     is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp32k_from_rsp2k.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp_end(self: "BaryonTemplateAccessor",
                           patch_nym: str,
                           is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp_end.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rsp_start_ret(self: "BaryonTemplateAccessor",
                                 patch_nym: str,
                                 is_patched: bool) -> str:
        return self.emit("partials/mimic_rsp_start_ret.jinja",
                         patch_nym=patch_nym,
                         is_patched=is_patched)

    def emit_mimic_rw_inh_rst(self: "BaryonTemplateAccessor",
                              load_mask: Optional[str],
                              patch_nym: str,
                              mask_nym: str,
                              has_read: bool,
                              is_patched: bool) -> str:
        return self.emit("partials/mimic_rw_inh_rst.jinja",
                         load_mask=load_mask,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         has_read=("true" if has_read else "false"),
                         is_patched=is_patched)

    def emit_mimic_rw_inh_set(self: "BaryonTemplateAccessor",
                              load_mask: Optional[str],
                              patch_nym: str,
                              mask_nym: str,
                              is_patched: bool) -> str:
        return self.emit("partials/mimic_rw_inh_set.jinja",
                         load_mask=load_mask,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_sb_cond_eq_inv_src(self: "BaryonTemplateAccessor",
                                      load_mask: Optional[str],
                                      load_vrs: str,
                                      load_src: str,
                                      free_src: Optional[str],
                                      patch_nym: str,
                                      mask_nym: str,
                                      is_patched: bool) -> str:
        return self.emit("partials/mimic_sb_cond_eq_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_sb_cond_eq_src(self: "BaryonTemplateAccessor",
                                  load_mask: Optional[str],
                                  load_vrs: str,
                                  load_src: str,
                                  free_src: Optional[str],
                                  patch_nym: str,
                                  mask_nym: str,
                                  is_patched: bool) -> str:
        return self.emit("partials/mimic_sb_cond_eq_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_sb_from_inv_src(self: "BaryonTemplateAccessor",
                                   load_mask: Optional[str],
                                   load_vrs: str,
                                   load_src: str,
                                   free_src: Optional[str],
                                   patch_nym: str,
                                   mask_nym: str,
                                   is_patched: bool) -> str:
        return self.emit("partials/mimic_sb_from_inv_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_sb_from_src(self: "BaryonTemplateAccessor",
                               load_mask: Optional[str],
                               load_vrs: str,
                               load_src: str,
                               free_src: Optional[str],
                               patch_nym: str,
                               mask_nym: str,
                               is_patched: bool) -> str:
        return self.emit("partials/mimic_sb_from_src.jinja",
                         load_mask=load_mask,
                         load_vrs=load_vrs,
                         load_src=load_src,
                         free_src=free_src,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         is_patched=is_patched)

    def emit_mimic_set_rl(self: "BaryonTemplateAccessor",
                          load_mask: Optional[str],
                          patch_nym: str,
                          mask_nym: str,
                          bit: bool,
                          is_patched: bool) -> str:
        return self.emit("partials/mimic_set_rl.jinja",
                         load_mask=load_mask,
                         patch_nym=patch_nym,
                         mask_nym=mask_nym,
                         bit="true" if bit else "false",
                         is_patched=is_patched)

    def emit_unify_sm_regs(self: "BaryonTemplateAccessor",
                           unify_nym: str,
                           reg_nyms: Sequence[str]) -> str:
        return self.emit("partials/unify_sm_regs.jinja",
                         unify_nym=unify_nym,
                         reg_nyms=reg_nyms)

    def emit_load_vrs(self: "BaryonTemplateAccessor",
                      reg_nyms: Sequence[str]) -> str:
        return self.emit("partials/load_vrs.jinja",
                         reg_nyms=reg_nyms)

    def emit_load_sm_reg(self: "BaryonTemplateAccessor",
                         reg_nym: str,
                         reg_id: str,
                         shift_width: int,
                         invert: bool) -> str:
        return self.emit("partials/load_sm_reg.jinja",
                         reg_nym=reg_nym,
                         reg_id=reg_id,
                         shift_width=shift_width,
                         invert=invert)

    def emit_load_rn_reg(self: "BaryonTemplateAccessor",
                         reg_nym: str,
                         reg_id: str) -> str:
        return self.emit("partials/load_rn_reg.jinja",
                         reg_nym=reg_nym,
                         reg_id=reg_id)

    def emit_load_re_reg(self: "BaryonTemplateAccessor",
                         reg_nym: str,
                         reg_id: str,
                         shift_width: int,
                         invert: bool) -> str:
        return self.emit("partials/load_re_reg.jinja",
                         reg_nym=reg_nym,
                         reg_id=reg_id,
                         shift_width=shift_width,
                         invert=invert)

    def emit_load_ewe_reg(self: "BaryonTemplateAccessor",
                          reg_nym: str,
                          reg_id: str,
                          shift_width: int,
                          invert: bool) -> str:
        return self.emit("partials/load_ewe_reg.jinja",
                         reg_nym=reg_nym,
                         reg_id=reg_id,
                         shift_width=shift_width,
                         invert=invert)

    def emit_load_l1_reg(self: "BaryonTemplateAccessor",
                         reg_nym: str,
                         reg_id: str,
                         offset: int) -> str:
        return self.emit("partials/load_l1_reg.jinja",
                         reg_nym=reg_nym,
                         reg_id=reg_id,
                         offset=offset)

    def emit_load_l2_reg(self: "BaryonTemplateAccessor",
                         reg_nym: str,
                         reg_id: str,
                         offset: int) -> str:
        return self.emit("partials/load_l2_reg.jinja",
                         reg_nym=reg_nym,
                         reg_id=reg_id,
                         offset=offset)

    def emit_apply_patch(self: "BaryonTemplateAccessor",
                         patch_nym: str,
                         patch_fn: str,
                         free_fn: Optional[str]) -> str:
        return self.emit("partials/apply_patch.jinja",
                         patch_nym=patch_nym,
                         patch_fn=patch_fn,
                         free_fn=free_fn)

    def emit_increment_instructions(self: "BaryonTemplateAccessor",
                                    num_instructions: int) -> str:
        return self.emit("partials/increment_instructions.jinja",
                         num_instructions=num_instructions)

    def emit_load_rl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_rl.jinja")

    def emit_load_nrl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_nrl.jinja")

    def emit_load_erl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_erl.jinja")

    def emit_load_wrl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_wrl.jinja")

    def emit_load_srl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_srl.jinja")

    def emit_load_inv_rl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_inv_rl.jinja")

    def emit_load_inv_nrl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_inv_nrl.jinja")

    def emit_load_inv_erl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_inv_erl.jinja")

    def emit_load_inv_wrl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_inv_wrl.jinja")

    def emit_load_inv_srl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_inv_srl.jinja")

    def emit_load_gl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_gl.jinja")

    def emit_load_ggl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_ggl.jinja")

    def emit_load_rsp16(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_rsp16.jinja")

    def emit_load_inv_gl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_inv_gl.jinja")

    def emit_load_inv_ggl(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_inv_ggl.jinja")

    def emit_load_inv_rsp16(self: "BaryonTemplateAccessor") -> str:
        return self.emit("partials/load_inv_rsp16.jinja")


class AplTemplateAccessor(TemplateAccessor):

    def __init__(self: "AplTemplateAccessor",
                 *args: Sequence[Any],
                 templates_path: Path = path_wrt_root("templates/apl"),
                 **kwargs: Dict[str, Any]) -> None:
        env_opts = {
            "block_start_string": "%{",
            "block_end_string": "}%",
            "variable_start_string": "${",
            "variable_end_string": "}$",
            "comment_start_string": "#{",
            "comment_end_string": "}#",
            "line_statement_prefix": "##",
            "line_comment_prefix": "###",
        }
        env_opts.update(kwargs)
        super().__init__(*args, templates_path, **env_opts)

    def emit_belex_examples(self: "AplTemplateAccessor",
                            name: str,
                            examples: Sequence[Example],
                            reg_nyms: Sequence[str]) -> str:

        example = examples[0]
        out_plats = example.expected_value.value
        num_plats_per_param = len(out_plats)

        example_reprs = []
        for example in examples:
            example_repr = []
            for value_parameter in example:
                parameter = self.emit_value_parameter_definition(value_parameter.value)
                example_repr.append(parameter)
            example_reprs.append(example_repr)

        return self.emit("belex_examples.jinja",
                         name=name,
                         examples=example_reprs,
                         reg_nyms=reg_nyms,
                         num_plats_per_param=num_plats_per_param)

    def emit_test_gvml_app_header(self: "AplTemplateAccessor",
                                  name: str,
                                  target: str = "baryon") -> str:
        return self.emit("test_gvml_app_header.jinja",
                         name=name,
                         target=target)

    def emit_test_gvml_app(self: "AplTemplateAccessor",
                           name: str,
                           examples: Sequence[Example],
                           reg_nyms: Sequence[str],
                           should_fail: bool,
                           num_instructions: int,
                           num_commands: int,
                           print_params: bool = False,
                           target: str = "baryon") -> str:

        example_ids = list(range(len(examples)))

        return self.emit("test_gvml_app.jinja",
                         name=name,
                         example_ids=example_ids,
                         reg_nyms=reg_nyms,
                         should_fail=should_fail,
                         num_instructions=num_instructions,
                         num_commands=num_commands,
                         print_params=print_params,
                         target=target)

    def emit_test_gvml_app_main(self: "AplTemplateAccessor",
                                name: str,
                                num_instructions: int,
                                num_commands: int,
                                target: str = "baryon") -> str:
        return self.emit("test_gvml_app_main.jinja",
                         name=name,
                         num_instructions=num_instructions,
                         num_commands=num_commands,
                         target=target)

    def emit_gvml_app_intern_header(self: "AplTemplateAccessor",
                                    name: str,
                                    reg_nyms: Sequence[str],
                                    target: str = "baryon") -> str:
        return self.emit("gvml_app_intern_header.jinja",
                         name=name,
                         reg_nyms=reg_nyms,
                         target=target)

    def emit_gvml_app_module(self: "AplTemplateAccessor",
                             name: str,
                             header_file: str,
                             calls: Sequence[str],
                             reg_nyms: Sequence[str],
                             row_numbers: Sequence[int],
                             prefix: str,
                             target: str = "baryon") -> str:
        return self.emit("gvml_app_module.jinja",
                         name=name,
                         header_file=header_file,
                         calls=calls,
                         reg_nyms=reg_nyms,
                         row_numbers=row_numbers,
                         prefix=prefix,
                         target=target)

    def emit_visit_fn_definition(self: "AplTemplateAccessor",
                                 camel_case_id: str,
                                 underscore_id: str) -> str:
        return self.emit("partials/visit_fn_definition.jinja",
                         camel_case_id=camel_case_id,
                         underscore_id=underscore_id)

    def emit_enter_fn_definition(self: "AplTemplateAccessor",
                                 camel_case_id: str,
                                 underscore_id: str) -> str:
        return self.emit("partials/enter_fn_definition.jinja",
                         camel_case_id=camel_case_id,
                         underscore_id=underscore_id)

    def emit_exit_fn_definition(self: "AplTemplateAccessor",
                                camel_case_id: str,
                                underscore_id: str) -> str:
        return self.emit("partials/exit_fn_definition.jinja",
                         camel_case_id=camel_case_id,
                         underscore_id=underscore_id)

    def emit_transform_fn_definition(self: "AplTemplateAccessor",
                                     camel_case_id: str,
                                     underscore_id: str,
                                     calls: Sequence[str],
                                     field_ids: Sequence[str],
                                     is_enum: bool) -> str:
        return self.emit("partials/transform_fn_definition.jinja",
                         camel_case_id=camel_case_id,
                         underscore_id=underscore_id,
                         calls=calls,
                         field_ids=field_ids,
                         is_enum=is_enum)

    def emit_transform_fn_call(self: "AplTemplateAccessor",
                               kind_id: str,
                               field_id: str,
                               kinds: Sequence[Tuple[str, str]],
                               cardinality: str,
                               nullable: bool) -> str:
        return self.emit("partials/transform_fn_call.jinja",
                         kind_id=kind_id,
                         field_id=field_id,
                         kinds=kinds,
                         cardinality=cardinality,
                         nullable=nullable)

    def emit_visitor(self: "AplTemplateAccessor",
                     imports: Sequence[str],
                     definitions: Sequence[str]) -> str:
        return self.emit("visitor.jinja",
                         imports=imports,
                         definitions=definitions)

    def emit_listener(self: "AplTemplateAccessor",
                      imports: Sequence[str],
                      definitions: Sequence[Tuple[str, str]]) -> str:
        return self.emit("listener.jinja",
                         imports=imports,
                         definitions=definitions)

    def emit_transformer(self: "AplTemplateAccessor",
                         imports: Sequence[str],
                         definitions: Sequence[str]) -> str:
        return self.emit("transformer.jinja",
                         imports=imports,
                         definitions=definitions)

    def emit_value_parameter_definition(self: "AplTemplateAccessor",
                                        plats: np.array) -> str:
        return self.emit("partials/value_parameter_definition.jinja",
                         plats=plats)

    def emit_fragment_caller_call_with_multi_line_comment(
            self: "AplTemplateAccessor",
            fragment_caller_call: str,
            comments: Sequence[str]) -> str:
        return self.emit("partials/fragment_caller_call_with_multi_line_comment.jinja",
                         fragment_caller_call=fragment_caller_call,
                         comments=comments)

    def emit_fragment_caller_call_with_single_line_comment(
            self: "AplTemplateAccessor",
            fragment_caller_call: str,
            comment: str) -> str:
        return self.emit("partials/fragment_caller_call_with_single_line_comment.jinja",
                         fragment_caller_call=fragment_caller_call,
                         comment=comment)

    def emit_fragment_caller_call_with_trailing_comment(
            self: "AplTemplateAccessor",
            fragment_caller_call: str,
            comment: str) -> str:
        return self.emit("partials/fragment_caller_call_with_trailing_comment.jinja",
                         fragment_caller_call=fragment_caller_call,
                         comment=comment)

    def emit_fragment_caller_call(self: "AplTemplateAccessor",
                                  identifier: str,
                                  parameters: Sequence[str]) -> str:
        return self.emit("partials/fragment_caller_call.jinja",
                         identifier=identifier,
                         parameters=parameters)

    def emit_fragment_caller_declaration(self: "AplTemplateAccessor",
                                         caller_id: str,
                                         parameters: Sequence[str],
                                         doc_comment: Optional[Sequence[str]]) -> str:
        return self.emit("partials/fragment_caller_declaration.jinja",
                         caller_id=caller_id,
                         parameters=parameters,
                         doc_comment=doc_comment)

    def emit_belex_body(self: "AplTemplateAccessor",
                        actual_parameters: Sequence[str],
                        fragment_ids: Sequence[str],
                        registers: Sequence[AllocatedRegister],
                        initial_active_registers: Optional[Sequence[AllocatedRegister]] = None,
                        spill_calls: Optional[Sequence[str]] = None,
                        restore_calls: Optional[Sequence[str]] = None,
                        target: str = "baryon") -> str:
        if initial_active_registers is None:
            initial_active_registers = registers
        return self.emit("partials/belex_body.jinja",
                         actual_parameters=actual_parameters,
                         fragment_ids=fragment_ids,
                         registers=registers,
                         initial_active_registers=initial_active_registers,
                         spill_calls=spill_calls,
                         restore_calls=restore_calls,
                         target=target)

    def emit_belex_caller(self: "AplTemplateAccessor",
                          doc_comment: Optional[Sequence[str]],
                          belex_caller_id: str,
                          fragment_ids: Sequence[str],
                          formal_parameters: Sequence[str],
                          actual_parameters: Sequence[str],
                          registers: Sequence[AllocatedRegister],
                          initial_active_registers: Sequence[AllocatedRegister],
                          spill_calls: Optional[Sequence[str]] = None,
                          restore_calls: Optional[Sequence[str]] = None,
                          target: str = "baryon") -> str:
        belex_declaration = self.emit_belex_declaration(
            identifier=belex_caller_id,
            parameters=formal_parameters,
            doc_comment=doc_comment)
        belex_body = self.emit_belex_body(
            actual_parameters=actual_parameters,
            fragment_ids=fragment_ids,
            registers=registers,
            initial_active_registers=initial_active_registers,
            spill_calls=spill_calls,
            restore_calls=restore_calls,
            target=target)

        return "\n".join([
            belex_declaration,
            belex_body,
        ])

    def emit_fragment_caller_body(self: "AplTemplateAccessor",
                                  fragment_id: str,
                                  registers: Sequence[AllocatedRegister],
                                  target: str = "baryon") -> str:
        return self.emit("partials/fragment_caller_body.jinja",
                         fragment_id=fragment_id,
                         registers=registers,
                         target=target)

    def emit_fragment_caller(self: "AplTemplateAccessor",
                             doc_comment: Optional[Sequence[str]],
                             caller_id: str,
                             parameters: Sequence[str],
                             fragment_id: str,
                             registers: Sequence[AllocatedRegister],
                             target: str = "baryon") -> str:
        fragment_caller_declaration = self.emit_fragment_caller_declaration(
            caller_id=caller_id,
            parameters=parameters,
            doc_comment=doc_comment)
        fragment_caller_body = self.emit_fragment_caller_body(
            fragment_id=fragment_id,
            registers=registers,
            target=target)
        return "\n".join([
            fragment_caller_declaration,
            fragment_caller_body,
        ])

    def emit_belex_declaration(self: "AplTemplateAccessor",
                               identifier: str,
                               parameters: Sequence[str],
                               doc_comment: Optional[Sequence[str]],
                               is_declaration: bool = False):
        if len(parameters) == 0:
            parameters = ["void"]
        return self.emit("partials/belex_declaration.jinja",
                         identifier=identifier,
                         parameters=parameters,
                         doc_comment=doc_comment)

    def emit_belex_constants(self: "AplTemplateAccessor",
                             name: str) -> str:
        return self.emit("belex_constants.jinja",
                         name=name)

    def emit_belex_utils_source(self: "AplTemplateAccessor",
                                name: str,
                                header_file: Optional[str] = None) -> str:
        return self.emit("belex_utils_source.jinja",
                         name=name,
                         header_file=header_file)

    def emit_belex_utils_header(self: "AplTemplateAccessor",
                                name: str,
                                target: str = "baryon") -> str:
        return self.emit("belex_utils_header.jinja",
                         name=name,
                         target=target)

    def emit_apl_header(self: "AplTemplateAccessor",
                        name: str,
                        fragments: Sequence[str],
                        declarations: Sequence[str],
                        lowered_registers: Sequence[Sequence[FormalParameter]],
                        explicit_frags_only: bool = False,
                        target: str = "baryon") -> str:
        return self.emit("apl_header.jinja",
                         name=name,
                         fragments=fragments,
                         declarations=declarations,
                         lowered_registers=lowered_registers,
                         explicit_frags_only=explicit_frags_only,
                         target=target)

    def emit_apl_source(self: "AplTemplateAccessor",
                        name: str,
                        callers: Sequence[Tuple[str, str]],
                        lowered_registers: Sequence[Sequence[FormalParameter]],
                        header_file: Optional[str] = None,
                        definitions: Optional[Sequence[str]] = None,
                        explicit_frags_only: Optional[bool] = False,
                        target: str = "baryon") -> str:
        if definitions is None:
            definitions = []
        return self.emit("apl_source.jinja",
                         name=name,
                         callers=callers,
                         header_file=header_file,
                         lowered_registers=lowered_registers,
                         definitions=definitions,
                         explicit_frags_only=explicit_frags_only,
                         target=target)


class MesonTemplateAccessor(TemplateAccessor):

    def __init__(self: "MesonTemplateAccessor",
                 *args: Sequence[Any],
                 templates_path: Path = path_wrt_root("templates/meson"),
                 **kwargs: Dict[str, Any]) -> None:
        env_opts = {
            "block_start_string": "{%",
            "block_end_string": "%}",
            "variable_start_string": "{{",
            "variable_end_string": "}}",
            "line_statement_prefix": "##",
            "line_comment_prefix": "###",
        }
        env_opts.update(kwargs)
        super().__init__(*args, templates_path, **env_opts)

    def emit_meson_build(self: "MesonTemplateAccessor",
                         snippet_name: str,
                         source_file: str,
                         prefix: str,
                         target: str = "baryon") -> str:
        return self.emit("meson_build.jinja",
                         snippet_name=snippet_name,
                         source_file=source_file,
                         prefix=prefix,
                         target=target)

    def emit_belex_examples_meson_build(self: "MesonTemplateAccessor",
                                        username: str,
                                        timestamp: str,
                                        example_dirs: Sequence[Path],
                                        manifest: Dict[str, Path]) -> str:
        return self.emit("belex_examples_meson_build.jinja",
                         username=username,
                         timestamp=timestamp,
                         example_dirs=example_dirs,
                         manifest=manifest)


class MarkdownTemplateAccessor(TemplateAccessor):

    def __init__(self: "MarkdownTemplateAccessor",
                 *args: Sequence[Any],
                 templates_path: Path = path_wrt_root("templates/markdown"),
                 **kwargs: Dict[str, Any]) -> None:
        env_opts = {
            "block_start_string": "{%",
            "block_end_string": "%}",
            "variable_start_string": "{{",
            "variable_end_string": "}}",
        }
        env_opts.update(kwargs)
        super().__init__(*args, templates_path, **env_opts)

    def emit_bleir_index(self: "MarkdownTemplateAccessor", grammar: str) -> str:
        return self.emit("bleir_index.jinja", grammar=grammar)

    def emit_bleir_visitor(self: "MarkdownTemplateAccessor",
                           visitor_definition: str,
                           pkginfo: str) -> str:
        return self.emit("bleir_visitor.jinja",
                         visitor_definition=visitor_definition,
                         pkginfo=pkginfo)

    def emit_bleir_listener(self: "MarkdownTemplateAccessor",
                            listener_definition: str,
                            pkginfo: str) -> str:
        return self.emit("bleir_listener.jinja",
                         listener_definition=listener_definition,
                         pkginfo=pkginfo)

    def emit_bleir_transformer(self: "MarkdownTemplateAccessor",
                               transformer_definition: str,
                               pkginfo: str) -> str:
        return self.emit("bleir_transformer.jinja",
                         transformer_definition=transformer_definition,
                         pkginfo=pkginfo)
