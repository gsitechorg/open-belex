r"""
 By Brian Beckman and Dylon Edwards

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

from abc import ABCMeta, abstractmethod
from typing import List, Callable


# TODO: metaclass=ABSMeta, @abstractmethod, @staticmethod
#  (with @abstractmethod) and ensure all subclasses implement
#  all @abstractmethod and and @abstractmethod @staticmethod.
class HalfBankAbstractInterface(metaclass=ABCMeta):
    """TODO: docstring"""

    # Most instructions are preceded by section masks.
    #
    # 5.1 WRITE LOGIC
    # ~~~~~~~~~~~~~~~
    #
    #   in shorter, BNF-style notation
    #
    #   <SRC>       = (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
    #                 NOTA BENE: <SRC> does NOT include SB!
    #
    #   SB[x]       = <SRC>
    #   SB[x, y]    = <SRC>
    #   SB[x, y, z] = <SRC>
    #
    #   where x, y, z are each one of RN_REG_0 .. RN_REG_15.
    #
    #   SB[x] is shorthand for SB[x, x, x],
    #   SB[x, y] is shorthand for SB[x, y, y]
    #
    #
    # 5.2 READ LOGIC
    # ~~~~~~~~~~~~~~
    #
    #   ------------------------------------------------------
    #         immediate instructions  op  arg1
    #   ------------------------------------------------------
    #     1.  RL = 0                  =   0
    #     2.  RL = 1                  =   1
    #   ------------------------------------------------------
    #         combining instructions  op  arg1   comb  arg2
    #   ------------------------------------------------------
    #     3.  RL =  <SB>              =   <SB>
    #     4.  RL =  <SRC>             =                <SRC>
    #     5.  RL =  <SB>  &  <SRC>    =   <SB>    &    <SRC>
    #
    #    10.  RL |=  <SB>             |=  <SB>
    #    11.  RL |=  <SRC>            |=               <SRC>
    #    12.  RL |=  <SB> &  <SRC>    |=  <SB>    &    <SRC>
    #
    #    13.  RL &=  <SB>             &=  <SB>
    #    14.  RL &=  <SRC>            &=               <SRC>
    #    15.  RL &=  <SB> &  <SRC>    &=  <SB>    &    <SRC>
    #
    #    18.  RL ^=  <SB>             ^=  <SB>
    #    19.  RL ^=  <SRC>            ^=               <SRC>
    #    20.  RL ^=  <SB> &  <SRC>    ^=  <SB>    &    <SRC>
    #   ------------------------------------------------------
    #         special cases           op  arg1   comb  arg2
    #   ------------------------------------------------------
    #     6.  RL =   <SB> |  <SRC>    =   <SB>    |    <SRC>
    #     7.  RL =   <SB> ^  <SRC>    =   <SB>    ^    <SRC>
    #
    #     8.  RL =  ~<SB> &  <SRC>    =  ~<SB>    &    <SRC>
    #     9.  RL =   <SB> & ~<SRC>    =   <SB>    &   ~<SRC>
    #
    #    16.  RL &= ~<SB>             &= ~<SB>
    #    17.  RL &= ~<SRC>            &= ~<SRC>
    #   ------------------------------------------------------
    #
    #   In addition the following instructions may be supported by HW but not
    #   supported by SW (no dedicated read-control register nor apl-preproc
    #   support):
    #
    #    18.  RL = ~RL & <SRC>
    #    19.  RL = ~RL & <SB>
    #    20.  RL = ~RL & (<SB> & <SRC>)
    #    21.  RL &= ~<SB> | ~<SRC>
    #
    # 5.3 R-SEL LOGIC
    # ~~~~~~~~~~~~~~~
    #
    #    GL = RL
    #    GGL = RL
    #    RSP16 = RL

    #  ___             _   _              _
    # | _ \___ __ _ __| | | |   ___  __ _(_)__
    # |   / -_) _` / _` | | |__/ _ \/ _` | / _|
    # |_|_\___\__,_\__,_| |____\___/\__, |_\__|
    #                               |___/

    # Read-logic Instruction 1 and 2
    @abstractmethod
    def set_rl(self, mask, bit):
        """Read-logic Instructions 1 and 2
           [APL  1] msk: RL = 1
           [APL  2] msk: RL = 0
        Set all columns of RL to a given bit through a mask"""
        raise NotImplementedError("HalfBank.set_rl")

    #   ____     ______    _______     ______
    #  / / /    <  <  /   <  / / /    <  / _ \
    # /_  _/    / // /    / /_  _/    / /\_, /
    #  /_/( )  /_//_( )  /_/ /_/( )  /_//___/
    #     |/        |/          |/

    # Read-logic instructions 4, 11, 14, 19
    #  4.  RL  =  <SRC>
    # 11.  RL |=  <SRC>
    # 14.  RL &=  <SRC>
    # 19.  RL ^=  <SRC>

    # Read-logic instruction 4
    @abstractmethod
    def rl_from_src(self, sections, src):
        """Read-logic Instruction 4
           [APL  4] msk: RL = <SRC>;"""
        raise NotImplementedError('HalfBank.rl_from_src')

    # Read-logic instruction 11
    @abstractmethod
    def rl_or_equals_src(self, sections, src):
        """Read-logic Instruction 11
           [APL 11] msk: RL |= <SRC>;"""
        raise NotImplementedError('HalfBank.rl_or_equals_src')

    # Read-logic instruction 14
    @abstractmethod
    def rl_and_equals_src(self, sections, src):
        """Read-logic Instruction 14
           [APL 14] msk: RL &= <SRC>;"""
        raise NotImplementedError('HalfBank.rl_and_equals_src')

    # Read-logic instruction 19
    @abstractmethod
    def rl_xor_equals_src(self, sections, src):
        """Read-logic Instruction 19
           [APL 19] msk: RL ^= <SRC>;"""
        raise NotImplementedError('HalfBank.rl_xor_equals_src')

    #    ____     ______      _______     ______
    #   |_  /    <  / _ \    <  /_  /    <  ( _ )
    #  _/_ <_    / / // /    / //_ <_    / / _  |
    # /____( )  /_/\___( )  /_/____( )  /_/\___/
    #      |/          |/          |/

    # Read-logic instructions 3, 10, 13, 18

    # Read-logic Instruction 3
    @abstractmethod
    def rl_from_sb(self, sections, sb):
        """Read-logic Instruction 3
           [APL  3] msk: RL = <SB>"""
        raise NotImplementedError("HalfBank.rl from sb")

    # Read-logic Instruction 10
    @abstractmethod
    def rl_or_equals_sb(self, sections, sb):
        """Read-logic Instruction 10
           [APL 10] msk: RL |= <SB>"""
        raise NotImplementedError("HalfBank.rl or-equals sb")

    # Read-logic Instruction 13
    @abstractmethod
    def rl_and_equals_sb(self, sections, sb):
        """Read-logic Instruction 13
        [APL 13] msk: RL &= <SB>"""
        raise NotImplementedError("HalfBank.rl and-equals sb")

    # Read-logic Instruction 18
    @abstractmethod
    def rl_xor_equals_sb(self, sections, sb):
        """Read-logic Instruction 13
        [APL 18] msk: RL ^= <SB>"""
        raise NotImplementedError("HalfBank.rl xor-equals sb")

    #    ____     ______      _______     ___  ___
    #   / __/    <  /_  |    <  / __/    |_  |/ _ \
    #  /__ \_    / / __/_    / /__ \_   / __// // /
    # /____( )  /_/____( )  /_/____( ) /____/\___/
    #      |/          |/          |/

    # Read-logic instruction 5, 12, 15, 20

    # Read-logic Instruction 5
    @abstractmethod
    def rl_from_sb_and_src(self, sections, sb, src):
        """Read-logic Instruction 5
           [APL  5] msk: RL = <SB> & <SRC>"""
        raise NotImplementedError("HalfBank.rl from sb and src")

    # Read-logic Instruction 12
    @abstractmethod
    def rl_or_equals_sb_and_src(self, sections, sb, src):
        """Read-logic Instruction 12
           [APL  5] msk: RL |= <SB> & <SRC>"""
        raise NotImplementedError("HalfBank.rl or-equals sb and src")

    # Read-logic Instruction 15
    @abstractmethod
    def rl_and_equals_sb_and_src(self, sections, sb, src):
        """Read-logic Instruction 15
           [APL 15] msk: RL &= <SB> & <SRC>"""
        raise NotImplementedError("HalfBank.rl and-equals sb and src")

    # Read-logic Instruction 20
    @abstractmethod
    def rl_xor_equals_sb_and_src(self, sections, sb, src):
        """Read-logic Instruction 20
           [APL 15] msk: RL ^= <SB> & <SRC>"""
        raise NotImplementedError("HalfBank.rl and-equals sb and src")

    #   ____     ____    ___      ___
    #  / __/    /_  /   ( _ )    / _ \
    # / _ \_     / /   / _  |    \_, /
    # \___( )   /_( )  \___( )  /___/
    #     |/      |/       |/

    # Read-logic instruction 6, 7, 8, 9

    # Read-logic Instruction 6
    @abstractmethod
    def rl_from_sb_or_src(self, sections, sb, src):
        """Read-logic Instruction 6
           [APL  6] msk: RL = <SB> | <SRC>"""
        raise NotImplementedError("HalfBank.rl from sb or src")

    # Read-logic Instruction 7
    @abstractmethod
    def rl_from_sb_xor_src(self, sections, sb, src):
        """Read-logic Instruction 7
           [APL  7] msk: RL = <SB> ^ <SRC>"""
        raise NotImplementedError("HalfBank.rl from sb xor src")

    # Read-logic Instruction 8
    @abstractmethod
    def rl_from_inv_sb_and_src(self, sections, sb, src):
        """Read-logic Instruction 8
           [APL  8] msk: RL = ~<SB> & <SRC>"""
        raise NotImplementedError("HalfBank.rl from inv sb and src")

    # Read-logic Instruction 9
    @abstractmethod
    def rl_from_sb_and_inv_src(self, sections, sb, src):
        """Read-logic Instruction 9
           [APL 9] msk: RL = <SB> & ~<SRC>"""
        raise NotImplementedError("HalfBank.rl from sb sb and inv src")

    # Read-logic Instruction 16
    @abstractmethod
    def rl_and_equals_inv_sb(self, sections, sb):
        """Read-logic Instruction 16
           [APL 16] msk: RL &= ~<SB>"""
        raise NotImplementedError('HalfBank.rl_and_equals_inv_sb')

    #  ___     ___      _   _              _
    # | _ \___/ __| ___| | | |   ___  __ _(_)__
    # |   /___\__ \/ -_) | | |__/ _ \/ _` | / _|
    # |_|_\   |___/\___|_| |____\___/\__, |_\__|
    #                                |___/

    @abstractmethod
    def gl_from_rl(self, sections):
        """[APL] msk: GL = RL;"""
        raise NotImplementedError("HalfBank.gl from rl")

    @abstractmethod
    def rsp16_from_rl(self, sections):
        """[APL] msk: RSP16 = RL;"""
        raise NotImplementedError("HalfBank.rsp 16 from rl")

    @abstractmethod
    def rsp256_from_rsp16(self):
        """[APL] RSP256 = RSP16;"""
        raise NotImplementedError("HalfBank.rsp256 from rsp16")

    @abstractmethod
    def rsp2k_from_rsp256(self):
        """[APL] RSP2K = RSP256;"""
        raise NotImplementedError("HalfBank.rsp2k from rsp256")

    @abstractmethod
    def rsp32k_from_rsp2k(self):
        """[APL] RSP32K = RSP2K;"""
        raise NotImplementedError("HalfBank.rsp32k from rsp2k")

    @abstractmethod
    def noop(self):
        """[APL] NOOP;"""
        raise NotImplementedError("HalfBank.noop")

    @abstractmethod
    def rsp_end(self):
        """[APL] RSP_END;"""
        raise NotImplementedError("HalfBank.rsp end")

    # __      __   _ _         _              _
    # \ \    / / _(_) |_ ___  | |   ___  __ _(_)__
    #  \ \/\/ / '_| |  _/ -_) | |__/ _ \/ _` | / _|
    #   \_/\_/|_| |_|\__\___| |____\___/\__, |_\__|
    #                                   |___/

    def NRL(self):
        raise NotImplementedError("HalfBank.NRL()")

    def ERL(self):
        raise NotImplementedError("HalfBank.ERL()")

    def WRL(self):
        raise NotImplementedError("HalfBank.WRL()")

    def SRL(self):
        raise NotImplementedError("HalfBank.SRL()")

    @abstractmethod
    def sb_from_src(self, sections, sb, src):
        """[APL] msk: <SB> = <SRC>"""
        raise NotImplementedError("HalfBank.sb from src")
