r"""
 BLECCI.py

 By Brian Beckman, Ilan Graidy, Eli Ehrman, Dylon Edwards, John D. Cook

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

from abc import ABC

from open_belex.bleir.types import (SB, SBParameter, and_eq, assign, cond_eq,
                                    conjoin, disjoin, invert, masked, or_eq,
                                    sb_parameter, statement, xor, xor_eq)
from open_belex.common.half_bank import HalfBankAbstractInterface as HalfBank


class BLECCI(HalfBank, ABC):
    r"""BLE -> C Compiler Interface

    --------------------------------------------------------------------
    Tartan Cheat Sheet:

    The original tartan paper, called "tartan.pdf," is in the
    docs/Tartan folder. It is written in mathematical notation. This
    cheat-sheet is in programmer notation, for programmers who are
    writing belops or other code-generators. It has references back to
    the original paper when appropriate.

    All assignment or copy operations have the form

        L' op= L ^ ( M & ( {L ^, nil}  {D, D {&, |} L} ) )

    where curly braces enclose options and with ^ bit-wise XOR, |
    bit-wise OR, and & bit-wise AND. In the original paper, simple
    assignment is equation 14, and L is called A:

        A' = A + (M (.) (A + D))

    where (.) is a circle-dot operator meaning AND or Hadamard product,
    and + is XOR.

    L' is the new value of L after the op.

    The "op" copies data from the donor matrix D into the marked plats
    and masked sections of L. The op can also mix original data from L
    via Boolean XOR, AND, and OR. The mask matrix, M, specifies the
    marked plats and masked sections of L to receive the data from the
    corresponding marked plats and masked sections of D. The unmarked
    plats and unmasked sections of L are undisturbed.

    L is an lvalue VR. Its section mask, ls, is accounted for in
    mask-matrix M as described below. D is a Data or Donor matrix,
    constructed from rvalue and rvalue2 as shown below.

    We speak of "copying or mixing data from D into the marked plats and
    masked sections of L." The matrix M specifies the marked plats and
    masked sections of L as an outer product of a section mask and a
    plat mask. A section mask, as always, is a 16-bit integer. A plat
    mask has the shape of a wordline: a bit string of length 2048. Plat
    masks are also called "markers" or marks in the APL jargon. The
    marks come from a section or wordline in a marker VR. The ON bits in
    the marker wordline indicate the masked-on (columns) in both L and
    D. The sections come from the section-mask part of an lvalue
    parameter of a belops call.

    Spelled out, the logical mixing operations follow this scheme:

        := :: L ^ ( M & ( L ^   D       ) )  # assignment
        ^= :: L ^ ( M & (       D       ) )  # assignment with XOR
        &= :: L ^ ( M & ( L ^ ( D & L ) ) )  # assignment with AND
        |= :: L ^ ( M & ( L ^ ( D | L ) ) )  # assignment with OR

    The meanings, in pseudo-belex, are

        L[ls, m[ms]] <= D()
        L[ls, m[ms]] ^= D()
        L[ls, m[ms]] &= D()
        L[ls, m[ms]] |= D()

    where L is a VR containing data to be partially overwritten; ls is a
    section mask; m is a VR containing a plat mask --- a row or section,
    ms, of marker bits; and D is a donor matrix containing data to be
    moved or mixed into L.

    In belex, :=, for simple assignment, is written <=.

    Additional commands are required to permute, shift, or duplicate
    sections of D before assignment, if desired. We do not address such
    permutations, shifts, or duplications in this cheat sheet. We also
    do not address permutations, shifts, or duplications in the plat
    dimensions.

    We clarify below with examples.

     __  __          _               __  __         _
    |  \/  |__ _ _ _| |_____ _ _ ___|  \/  |__ _ __| |__
    | |\/| / _` | '_| / / -_) '_|___| |\/| / _` (_-< / /
    |_|  |_\__,_|_| |_\_\___|_|     |_|  |_\__,_/__/_\_\
     __  __      _       _       __  __
    |  \/  |__ _| |_ _ _(_)_ __ |  \/  |
    | |\/| / _` |  _| '_| \ \ / | |\/| |
    |_|  |_\__,_|\__|_| |_/_\_\ |_|  |_|


    M is a plat-marker + section-mask matrix, also called a mask matrix
    or a marker matrix. It has ON bits where the plats of the Marker
    wordline cross the section Mask of the lvalue, namely lsections, and
    zeros elsewhere. The markers and section mask come from a call of a
    belops:

        def belop_1(self,
                lvalue: Union[WLs, ABs],  # <~~~~~~~ lsections from here
                op: BelOps,
                rvalue: Union[RValue, WLs, ABs, IVs],
                rvalue2: Optional[Union[RValue, WLs]] = None,
                rvalue2_op: BelOp = BelOp.AND,
                markers: Optional[WLs] = None)  # <~~~ markers from here

    M has the shape of a VR and is often stored in a temporary VR;
    though sometimes we don't need a temporary VR and can store M in RL.

    Every section in M that has any ON marker bits must have the same ON
    plat-marker bits. That is what the paper means in equation 10 by an
    "outer product" of a section mask \psi and a marker mask \mu. For
    example:

    M = [ ..11 .1.1 ...1 .1.. .1.1 ...1 .... .1.1,
          .... .... .... .... .... .... .... ....,
          ..11 .1.1 ...1 .1.. .1.1 ...1 .... .1.1,
          .... .... .... .... .... .... .... ....,
          ..11 .1.1 ...1 .1.. .1.1 ...1 .... .1.1,
          .... .... .... .... .... .... .... ....,
          etc. ]
          (dots mean 0's for visual clarity)

    has marker bits ON in the prime-numbered plats (up to 31), of the
    even-numbered sections, and zero in all other plats and sections. M
    is the outer product (using bit-wise AND as multiplication) of a
    section mask "5555" (all the even sections / wordlines) and a marker
    string 0011 0101 0001 0100 0101 0001 0000 0101 == 0x_3514_5105.

    The "tartan" pattern in the mas matrix, the pattern that gives the
    Tartan theory its name, is obvious in the sketch above.

    One way to copy markers into the lsections of the M matrix is with
    code like the following:

    Let the markers be stored in a single wordline in section ms of SB
    number mvr. Let the APL register parameters corresponding to mvr and
    ms be mvr_rp and ms_rp, respectively; that is, we want, for example

        mvr_rp = RN_REG_3
        ms_rp  = SM_REG_1

    in the final code that calls APL (the frag-caller).

    Let fs_rp be an APL SM-register parameter corresponding to a full
    section mask, "FFFF". Let tvr be a temporary VR and tvr_rp its
    corresponding APL RN-register parameter. Finally, let ls be the left
    section mask from the lvalue, and let ls_rp be its corresponding
    SM-register parameter.

        # SET UP M, THE MARKER MATRIX:

        # (1) Copy the marker wordline from mvr into section ms of RL
        #     through register-parameters ms_rp and mvr_rp.
        # low-level belex: RL[ms] <= mvr()
        self.rl_from_sb(ms_rp, mvr_rp)
        # (2) Copy the marker wordline from ms:RL into GL.
        # low-level belex: GL[ms] <= RL()
        self.gl_from_rl(ms_rp)
        # Empirical fact that on C-sim this noop is necessary.
        # (3) Give RL and GL time to sync up.
        # low-level belex: NOOP()
        self.noop()
        # (4) Clear all sections of RL through section-mask fs in
        #     register-parameter fs_rp
        # low-level belex: RL[:] <= 0
        self.set_rl(fs_rp, bit=0)
        # (5) Clear tvr (through register parameters):
        # low-level belex: tvr[:] <= RL()
        self.sb_from_src(fs_rp, tvr_rp, 'RL')
        # (6) Load marker wordlines into sections ls of tvr (through
        #     register-parameters).
        # low-level belex: tvr[ls] <= GL()
        self.sb_from_src(ls_rp, tvr_rp, 'GL')
        # POSTCONDITION: tvr contains M.

        # LOW-LEVEL BELEX (LLB)

        with apl_commands():
          RL[ms] <= mvr()   # pull marker 1s out of section ms of mvr
          GL[ms] <= RL()    # deposit those 1s in GL
        NOOP()
        RL[:] <= 0          # clear RL
        with apl_commands():
          tvr[~ls] <= RL()  # clear unmarked sections of tvr
          tvr[ls]  <= GL()  # deposit marker 1s in sections ls of tvr

    The C-sim revealed that a noop is necessary at step 3. The reason
    comes from obscure details of the hardware, but the C-sim is a
    reliable guide as to when that "wait-state" is needed. In a later
    example, we will see a similar sequence of instructions that,
    empirically, does not require this noop.

    In the case of all-plat operations, a common case called "full
    markers," we must create something like the following:

    M = [ 1111 1111 1111 1111 1111 1111 1111 1111,
          0000 0000 0000 0000 0000 0000 0000 0000,
          1111 1111 1111 1111 1111 1111 1111 1111,
          0000 0000 0000 0000 0000 0000 0000 0000,
          1111 1111 1111 1111 1111 1111 1111 1111,
          0000 0000 0000 0000 0000 0000 0000 0000,
          ... ]

    We can do that without an mvr, as follows

        # LOW-LEVEL BELEX (LLB)

        with apl_commands():
          RL[ms] <= 1       # put constant 1s in section ms of RL
          GL[ms] <= RL()    # deposit those 1s in GL
        NOOP()
        RL[:] <= 0          # clear RL
        with apl_commands():
          tvr[~ls] <= RL()  # clear unmarked sections of tvr
          tvr[ls]  <= GL()  # deposit marker 1s in sections ls of tvr

     ___                     __  __      _       _       ___
    |   \ ___ _ _  ___ _ _  |  \/  |__ _| |_ _ _(_)_ __ |   \
    | |) / _ \ ' \/ _ \ '_| | |\/| / _` |  _| '_| \ \ / | |) |
    |___/\___/_||_\___/_|   |_|  |_\__,_|\__|_| |_/_\_\ |___/


    For an example of setting up a donor matrix, D, consider copying
    immediates from rvalue into the marked sections of L, from
    _do_tartan_iv_k, implemented in this file. Let ds be a section
    mask corresponding to the 16-bit immediate value, which comes from
    the belop call here:

            def belop_1(self,
                lvalue: Union[WLs, ABs],  # <~~~~~~~ lsections from here
                op: BelOps,
                rvalue: Union[RValue, WLs, ABs, IVs],  # <~~~~ IV

    and, as usual, let ds_rp be a register-parameter (reg-param)
    corresponding to ds. Here are the APL mimics:

        # Set up D, in RL, donor matrix: the IV in all plats
        # (1) Set the masked-on bits of RL to 1.
        # low-level belex: RL[ds] <= 1
        self.set_rl(ds_rp, bit=1)
        # (2) Set the masked-off bits of RL to 0.
        # low-level belex: RL[~ds] <= 0
        self.set_rl(~ds_rp, bit=0)
        # POSTCONDITION: RL contains D.

        # LOW-LEVEL BELEX (LLB)

        # Set up D, in RL, donor matrix: the IV in all plats
        with apl_commands():
          RL[ds]  <= 1
          RL[~ds] <= 0

    Notice we do not need a second temporary VR, here, for D because the
    caller gave us an explicit section mask for the immediate value. We
    can do all the rest of our work in RL, now

     ___                _   _            _ _ _
    | __|__ _ _  _ __ _| |_(_)___ _ _   / | | |
    | _|/ _` | || / _` |  _| / _ \ ' \  | |_  _|
    |___\__, |\_,_\__,_|\__|_\___/_||_| |_| |_|
           |_|

    Equation 14, or our version of it, is easy to implement because the
    operations right-associate. Also notice that we use full section
    masks, "FFFF" == fs via reg-param fs_rp, from here on out. All the
    masking for the lvalue sections has been accounted for in M:

        # The rest of this is always the same for op = ASSIGN.
        # PRECONDITION: RL contains D, tvr contains M
        # Add L: RL <- L ^ D
        # low-level belex: RL[:] ^= lvr()
        self.rl_xor_equals_sb(fs_rp, lvr_rp)
        # Hadamard product with M: RL <- M & (L ^ D)
        # low-level belex: RL[:] &= tvr()
        self.rl_and_equals_sb(fs_rp, tvr_rp)
        # Add L again: RL <- L ^ (M & (L ^ D))
        # low-level belex: RL[:] ^= lvr()
        self.rl_xor_equals_sb(fs_rp, lvr_rp)
        # Put result into L's slots: L <- RL
        # low-level belex: lvr[:] <= RL()
        self.sb_from_src(fs_rp, lvr_rp, 'RL')

        # LOW-LEVEL BELEX
        # PRECONDITION: RL contains D, tvr contains M

        RL[:]  ^= lvr()  # RL <- L ^ D
        RL[:]  &= tvr()  # RL <- M & (L ^ D)
        RL[:]  ^= lvr()  # RL <- L ^ (M & (L ^ D))
        lvr[:] <= RL()   # L' <- L ^ (M & (L ^ D))

    What about the other Tartan operations? They differ only on the
    right-hand sides of M's & operator.

        := :: L ^ ( M & ( L ^   D       ) )  # assignment
        ^= :: L ^ ( M & (       D       ) )  # assignment with XOR
        &= :: L ^ ( M & ( L ^ ( D & L ) ) )  # assignment with AND
        |= :: L ^ ( M & ( L ^ ( D | L ) ) )  # assignment with OR

    The last one is Boolean, not in GF2, because GF2 doesn't have OR.

    The original Tartan expression for |= is

        |= :: L ^ ( M & ( D ^ ( D & L ) ) )  # assignment with OR

    Entirely in GF2, with ^ for plus and & for times.

    This is undesirable because, in a right-associative encoding, the
    innermost sub-expression on the right will overwrite RL == D, the
    precondition, with D & L, and we will need a temporary register to
    hold a copy of the original D.

    However D|L = (D^L)^(D&L) in GF2, and the original Tartan
    sub-expression

        D ^ (D & L) <=> L^D^L ^ (D & L) <=> L ^ (D^L)^(D&L) <=>
        L ^ (D | L)

    which is right-associative Boolean, supported by the machine, and
    uses D = RL just once, bypassing the need for a temporary register.
    Here are all four tartan assignment sequences, for reference

        # PRECONDITION for all: RL contains D, tvr contains M

        # LOW-LEVEL BELEX

         := :: L ^ ( M & ( L ^   D       ) )  # assignment

        RL[:]  ^= lvr()  # RL <- L ^ D
        RL[:]  &= tvr()  # RL <- M & (L ^ D)
        RL[:]  ^= lvr()  # RL <- L ^ (M & (L ^ D))
        lvr[:] <= RL()   # L' <- L ^ (M & (L ^ D))

         ^= :: L ^ ( M & (       D       ) )  # assignment with XOR

        RL[:]  &= tvr()  # RL <- M & D
        RL[:]  ^= lvr()  # RL <- L ^ (M & D)
        lvr[:] <= RL()   # L' <- L ^ (M & D)

         &= :: L ^ ( M & ( L ^ ( D & L ) ) )  # assignment with AND

        RL[:]  &= lvr()  # RL <- D & L
        RL[:]  ^= lvr()  # RL <- L ^ (D & L)
        RL[:]  &= tvr()  # RL <- M & (L ^ (D & L))
        RL[:]  ^= lvr()  # RL <- L ^ (M & (L ^ (D & L)))
        lvr[:] <= RL()   # L' <- L ^ (M & (L ^ (D & L)))

         |= :: L ^ ( M & ( L ^ ( D | L ) ) )  # assignment with OR

        RL[:]  |= lvr()  # RL <- D | L
        RL[:]  ^= lvr()  # RL <- L ^ (D | L)
        RL[:]  &= tvr()  # RL <- M & (L ^ (D | L))
        RL[:]  ^= lvr()  # RL <- L ^ (M & (L ^ (D | L)))
        lvr[:] <= RL()   # L' <- L ^ (M & (L ^ (D | L)))

    That covers the case of writing immediate values to marked plats
    specified in mvr and ms. If the user does not specify such marker
    plats, the belops assumes "all plats." We get that by a small
    difference in the set-up of M. Instead of

        # (1) Copy the marker wordline into section ms of RL.
        # low-level belex: RL[:] <= mvr()
        self.rl_from_sb(ms_rp, mvr_rp)

    we write

        # (1) Turn ON all bits in RL
        # low-level belex: RL[:] <= 1
        self.set_rl(fs_rp, '1'),

    before pulling the markers into GL. All the rest is the same.

       _            _   _              ___                     _
      /_\  _ _  ___| |_| |_  ___ _ _  | __|_ ____ _ _ __  _ __| |___
     / _ \| ' \/ _ \  _| ' \/ -_) '_| | _|\ \ / _` | '  \| '_ \ / -_)
    /_/ \_\_||_\___/\__|_||_\___|_|   |___/_\_\__,_|_|_|_| .__/_\___|
                                                     |_|

    Consider the following belop call:

        pyble.belop_1(
            lvalue=WLs(VRs=[lvr], sections=[5, 7, 11]),
            op=BelOps.ASSIGN,
            rvalue=RValue(val=WLs(VRs=[rvr], sections=[4, 5, 6]),
                          sweeping_op=SweepingBelOps.AND_ACROSS_SECTIONS,
                          inv=True),
            markers=WLs(VRs=[mvr], sections=Mask(msec)))

    The meaning is to take sections 4, 5, and 6 from rvalue VR rvr, AND
    them together --- sweep them --- into a single wordline of length
    NPLATS=2048, then to copy that swept wordline into sections 5, 7,
    and 11 of lvalue VR lvr, but only in the marked plats.

        def do_sequential_sweep(
            self,
            lvalue: Union[WLs, ABs],
            op: BelOps,
            rvalue: Optional[Union[RValue, WLs]],
            markers: Optional[WLs]):

    It is implemented as follows. Most of it will be familiar from the
    examples above. Here, the markers are in msec of mvr; a full mask,
    "FFFF", is in fsec; the lvalue sections are in lsec and the rvalue
    sections are in rsec; all with corresponding reg-params. The names
    of the reg-params uniformly have suffix "_rp". That naming
    convention is enforced in the new @basic_block decorator, which
    automates most of the boilerplate for new code-generators.

    The first bit is exactly the same, only with slightly longer names
    for the sections: fsec instead of fs, lsec instead of fs, etc.

        # Set up M, the Marker Matrix
        # (1) Copy the marker wordline into section ms of RL.
        # low-level belex: RL[msec] <= mvr()
        self.rl_from_sb(msec_rp, mvr_rp)
        # (2) Copy the marker wordline into GL.
        # low-level belex: GL[msec] <= RL()
        self.gl_from_rl(msec_rp)
        # Empirical fact that on C-sim this noop is necessary
        # (3) Give RL and GL time to sync up.
        # low-level belex: NOOP()
        self.noop()
        # (4) Clear RL (the rest is the same for all Tartan "assign"
        #     operations that adhere to equation 14).
        self.set_rl(fsec_rp, bit=0),
        # low-level belex: RL[:] <= 0
        # (5) Clear tvr:
        self.sb_from_src(fsec_rp, tvr_rp, 'RL'),
        # low-level belex: tvr[:] <= RL()
        # (6) Load marker wordlines into sections ls of tvr.
        # low-level belex: tvr[lsec] <= GL()
        self.sb_from_src(lsec_rp, tvr_rp, 'GL'),

    Only the donor matrix, D, differs between this example, sweeping,
    versus the prior example, setting immediates. If "inv" is true, we
    use a convenient APL instruction, rl_and_equals_inv_sb. Otherwise,
    we use the rl_and_equals_sb. The result is deposited in GL and then
    back into all sections of RL (for obscure reasons, the noop is not
    needed when constructing D in this example; the C-sim reliably warns
    when such a noop is necessary).

        # Set up D in RL: the sweep value S into every section
        # low-level belex: RL[:] <= 1
        self.set_rl(fsec_rp, bit=1)
        # low-level belex: RL[rsec] &= (rvr() if inv else ~rvr())
        self.rl_and_equals_inv_sb(rsec_rp, rvr_rp) \
            if inv else \
            self.rl_and_equals_sb(rsec_rp, rvr_rp),
        # low-level belex: GL[rsec] <= RL()
        self.gl_from_rl(rsec_rp)
        # low-level belex: RL[:] <= GL()
        self.rl_from_src(fsec_rp, 'GL')

    The final sequence is exactly the same as before.

        # The rest of this is always the same for op = ASSIGN.
        # PRECONDITION: RL contains D, tvr contains M
        # Add L: RL <- L ^ D
        # low-level belex: RL[:] ^= lvr()
        self.rl_xor_equals_sb(fs_rp, lvr_rp)
        # Hadamard product with M: RL <- M & (L ^ D)
        # low-level belex: RL[:] &= tvr()
        self.rl_and_equals_sb(fs_rp, tvr_rp)
        # Add L again: RL <- L ^ (M & (L ^ D))
        # low-level belex: RL[:] ^= lvr()
        self.rl_xor_equals_sb(fs_rp, lvr_rp)
        # Put result into L's slots: L <- RL
        # low-level belex: lvr[:] <= RL()
        self.sb_from_src(fs_rp, lvr_rp, 'RL')

    """

    # Begin the body of BLECCI.

    @staticmethod
    def _unpack_sbs_for_blecci(sbs):
        r"""SB expressions can contain up to 3 RNs (register numbers).
        SB[x], SB[x, y], SB[x, y, z] are allowed. SB[x, x], etc., i.e.,
        where some of the RN numbers are the same. Unpack such
        expressions for BLECCI.
        """
        if isinstance(sbs, str):
            sbs = [sb.strip() for sb in sbs.split(",")]
        if isinstance(sbs, SBParameter.__args__):
            sbs = [sbs]
        sbs = [sb_parameter(sb) for sb in sbs]
        return sbs

    #   _   _    _    _     _____     ____    _    _   _ _  __
    #  | | | |  / \  | |   |  ___|   | __ )  / \  | \ | | |/ /
    #  | |_| | / _ \ | |   | |_ _____|  _ \ / _ \ |  \| | ' /
    #  |  _  |/ ___ \| |___|  _|_____| |_) / ___ \| |\  | . \
    #  |_| |_/_/_  \_\_____|_|___ ___|____/_/ _ \_\_|_\_|_|\_\
    #      |_ _| \ | |_   _| ____|  _ \|  ___/ \  / ___| ____|
    #       | ||  \| | | | |  _| | |_) | |_ / _ \| |   |  _|
    #       | || |\  | | | | |___|  _ <|  _/ ___ \ |___| |___
    #      |___|_| \_| |_| |_____|_| \_\_|/_/   \_\____|_____|

    # There are stubs for every APL instruction in the half-bank
    # interface. We do not implement them until there is a test or an
    # application for them. They are implemented on an 'as-needed'
    # basis, like most of the rest of BLECCI.

    #  ___             _   _              _
    # | _ \___ __ _ __| | | |   ___  __ _(_)__
    # |   / -_) _` / _` | | |__/ _ \/ _` | / _|
    # |_|_\___\__,_\__,_| |____\___/\__, |_\__|
    #                               |___/

    #   ___    ___
    #  <  /   |_  |
    #  / /   / __/
    # /_( ) /____/
    #   |/

    # Read-logic Instructions 1 and 2

    @staticmethod
    @statement
    def set_rl(sections, bit):
        """Read-logic Instructions 1 and 2
           [APL  1] msk: RL = 1
           [APL  2] msk: RL = 0
        Set all plats of RL to a given bit through a section mask"""
        result = masked(sections, assign("RL", bit))
        return result

    #   ____     ______    _______     _______    ______
    #  / / /    <  <  /   <  / / /    <  /_  /   <  / _ \
    # /_  _/    / // /    / /_  _/    / / / /    / /\_, /
    #  /_/( )  /_//_( )  /_/ /_/( )  /_/ /_( )  /_//___/
    #     |/        |/          |/         |/

    # Read-logic instructions 4, 11, 14, 17, 19
    #   |  4.  RL  =  <SRC>           | :=               <SRC>  |
    #   | 11.  RL |=  <SRC>           | |=               <SRC>  |
    #   | 14.  RL &=  <SRC>           | &=               <SRC>  |
    #   | 17.  RL &= ~<SRC>           | &=              ~<SRC>  |
    #   | 19.  RL ^=  <SRC>           | ^=               <SRC>  |

    #   ____
    #  / / /
    # /_  _/
    #  /_/

    @staticmethod
    @statement
    def rl_from_src(sections, src):
        """Read-logic Instruction 4
           [APL  4] msk: RL = <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        result = masked(sections, assign("RL", src))
        return result

    @staticmethod
    @statement
    def rl_from_inv_src(sections, src):
        """Read-logic Instruction 4
           [APL  4] msk: RL = <SRC>;"""
        result = masked(sections, assign("RL", invert(src)))
        return result

    #   ______
    #  <  <  /
    #  / // /
    # /_//_/

    @staticmethod
    @statement
    def rl_or_equals_src(sections, src):
        """Read-logic Instruction 11
           [APL 11] msk: RL |= <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        result = masked(sections, or_eq("RL", src))
        return result

    @staticmethod
    @statement
    def rl_or_equals_inv_src(sections, src):
        """Read-logic Instruction 11
           [APL 11] msk: RL |= ~<SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        result = masked(sections, or_eq("RL", invert(src)))
        return result

    #   _______
    #  <  / / /
    #  / /_  _/
    # /_/ /_/

    @staticmethod
    @statement
    def rl_and_equals_src(sections, src):
        """Read-logic Instruction 14
           [APL 14] msk: RL &= <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        result = masked(sections, and_eq("RL", src))
        return result

    #   _______
    #  <  /_  /
    #  / / / /
    # /_/ /_/

    @staticmethod
    @statement
    def rl_and_equals_inv_src(sections, src):
        """Read-logic Instruction 17
           [APL 17] msk: RL &= ~<SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        result = masked(sections, and_eq("RL", invert(src)))
        return result

    #   ______
    #  <  / _ \
    #  / /\_, /
    # /_//___/

    @staticmethod
    @statement
    def rl_xor_equals_src(sections, src):
        """Read-logic Instruction 19
           [APL 19] msk: RL ^= <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        result = masked(sections, xor_eq("RL", src))
        return result

    @staticmethod
    @statement
    def rl_xor_equals_inv_src(sections, src):
        """Read-logic Instruction 19
           [APL 19] msk: RL ^= <SRC>;
           <SRC> is one of (INV_)?[GL, GGL, RSP16, RL [NEWS]RL]
                NOTA BENE: <SRC> does NOT include SB!"""
        result = masked(sections, xor_eq("RL", invert(src)))
        return result

    #    ____     ______      _______     _______     ______
    #   |_  /    <  / _ \    <  /_  /    <  / __/    <  ( _ )
    #  _/_ <_    / / // /    / //_ <_    / / _ \_    / / _  |
    # /____( )  /_/\___( )  /_/____( )  /_/\___( )  /_/\___/
    #      |/          |/          |/          |/

    # Read-logic instructions 3, 10, 13, 16, 18
    #   |  3.  RL  =  <SB>            | :=  <SB>                |
    #   | 10.  RL |=  <SB>            | |=  <SB>                |
    #   | 13.  RL &=  <SB>            | &=  <SB>                |
    #   | 16.  RL &= ~<SB>            | &= ~<SB>                |
    #   | 18.  RL ^=  <SB>            | ^=  <SB>                |

    #    ____
    #   |_  /
    #  _/_ <
    # /____/

    @classmethod
    @statement
    def rl_from_sb(cls, sections, sb):
        """Read-logic Instruction 3
           [APL  3] msk: RL = <SB>;
        In BLECCI, 'sections' must be an SM_REG_k, k in [0..15],
        and 'sb' must be an RN_REG_k, k in [0..15], or an iterable
        of such, as in SB[x], SB[x, y], SB[x, y, z]."""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, assign("RL", SB[sbublecci]))
        return result

    @classmethod
    @statement
    def rl_from_inv_sb(cls, sections, sb):
        """Read-logic Instruction 3
           [APL  3] msk: RL = ~<SB>;"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, assign("RL", ~SB[sbublecci]))
        return result

    #   ______
    #  <  / _ \
    #  / / // /
    # /_/\___/

    @classmethod
    @statement
    def rl_or_equals_sb(cls, sections, sb):
        """Read-logic Instruction 10
           [APL 10] msk: RL |= <SB>
        In BLECCI, 'sections' must be an SM_REG_k, k in [0..15],
        and 'sb' must be an RN_REG_k, k in [0..15]."""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, or_eq("RL", SB[sbublecci]))
        return result

    #   _______
    #  <  /_  /
    #  / //_ <
    # /_/____/

    @classmethod
    @statement
    def rl_and_equals_sb(cls, sections, sb):
        """Read-logic Instruction 13
        [APL 13] msk: RL &= <SB>
        In BLECCI, 'sections' must be an SM_REG_k, k in [0..15],
        and 'sb' must be an RN_REG_k, k in [0..15]."""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, and_eq("RL", SB[sbublecci]))
        return result

    #   _______
    #  <  / __/
    #  / / _ \
    # /_/\___/

    @classmethod
    @statement
    def rl_and_equals_inv_sb(cls, sections, sb):
        """Read-logic Instruction 16
           [APL 16] msk: RL &= ~<SB>
        In BLECCI, 'sections' must be an SM_REG_k, k in [0..15],
        and 'sb' must be an RN_REG_k, k in [0..15]."""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, and_eq("RL", ~SB[sbublecci]))
        return result

    #   ______
    #  <  ( _ )
    #  / / _  |
    # /_/\___/

    @classmethod
    @statement
    def rl_xor_equals_sb(cls, sections, sb):
        """Read-logic Instruction 13
        [APL 18] msk: RL ^= <SB>
        In BLECCI, 'sections' must be an SM_REG_k, k in [0..15],
        and 'sb' must be an RN_REG_k, k in [0..15]."""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, xor_eq("RL", SB[sbublecci]))
        return result

    #    ____     ______      _______     ___  ___
    #   / __/    <  /_  |    <  / __/    |_  |/ _ \
    #  /__ \_    / / __/_    / /__ \_   / __// // /
    # /____( )  /_/____( )  /_/____( ) /____/\___/
    #      |/          |/          |/

    # Read-logic instruction 5, 12, 15, 20

    #   |  5.  RL  =  <SB> &  <SRC>   | :=  <SB>    &    <SRC>  |
    #   | 12.  RL |=  <SB> &  <SRC>   | |=  <SB>    &    <SRC>  |
    #   | 15.  RL &=  <SB> &  <SRC>   | &=  <SB>    &    <SRC>  |
    #   | 20.  RL ^=  <SB> &  <SRC>   | ^=  <SB>    &    <SRC>  |

    #    ____
    #   / __/
    #  /__ \
    # /____/

    @classmethod
    @statement
    def rl_from_sb_and_src(cls, sections, sb, src):
        """Read-logic Instruction 5
           [APL  5] msk: RL = <SB> & <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, assign("RL", conjoin(SB[sbublecci], src)))
        return result

    #   ______
    #  <  /_  |
    #  / / __/
    # /_/____/

    @classmethod
    @statement
    def rl_or_equals_sb_and_src(cls, sections, sb, src):
        """Read-logic Instruction 12
           [APL  5] msk: RL |= <SB> & <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, or_eq("RL", conjoin(SB[sbublecci], src)))
        return result

    @classmethod
    @statement
    def rl_or_equals_sb_and_inv_src(cls, sections, sb, src):
        """Read-logic Instruction 12
           [APL  5] msk: RL |= <SB> & ~<SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections,
                        or_eq("RL",
                              conjoin(SB[sbublecci],
                                      invert(src))))
        return result

    #   _______
    #  <  / __/
    #  / /__ \
    # /_/____/

    @classmethod
    @statement
    def rl_and_equals_sb_and_src(cls, sections, sb, src):
        """Read-logic Instruction 15
           [APL 15] msk: RL &= <SB> & <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, and_eq("RL", conjoin(SB[sbublecci], src)))
        return result

    @classmethod
    @statement
    def rl_and_equals_sb_and_inv_src(cls, sections, sb, src):
        """Read-logic Instruction 15
           [APL 15] msk: RL &= <SB> & ~<SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections,
                        and_eq("RL",
                               conjoin(SB[sbublecci],
                                       invert(src))))
        return result

    #    ___  ___
    #   |_  |/ _ \
    #  / __// // /
    # /____/\___/

    @classmethod
    @statement
    def rl_xor_equals_sb_and_src(cls, sections, sb, src):
        """Read-logic Instruction 20
           [APL 15] msk: RL ^= <SB> & <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, xor_eq("RL", conjoin(SB[sbublecci], src)))
        return result

    @classmethod
    @statement
    def rl_xor_equals_sb_and_inv_src(cls, sections, sb, src):
        """Read-logic Instruction 20
           [APL 15] msk: RL ^= <SB> & ~<SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections,
                        xor_eq("RL",
                               conjoin(SB[sbublecci],
                                       invert(src))))
        return result

    #   ____     ____    ___      ___
    #  / __/    /_  /   ( _ )    / _ \
    # / _ \_     / /   / _  |    \_, /
    # \___( )   /_( )  \___( )  /___/
    #     |/      |/       |/

    # Read-logic instruction 6, 7, 8, 9
    #   |  6.  RL  =  <SB> |  <SRC>   | :=  <SB>    |    <SRC>  |
    #   |  7.  RL  =  <SB> ^  <SRC>   | :=  <SB>    ^    <SRC>  |
    #   |  8.  RL  = ~<SB> &  <SRC>   | := ~<SB>    &    <SRC>  |
    #   |  9.  RL  =  <SB> & ~<SRC>   | :=  <SB>    &   ~<SRC>  |

    #   ____
    #  / __/
    # / _ \
    # \___/

    @classmethod
    @statement
    def rl_from_sb_or_src(cls, sections, sb, src):
        """Read-logic Instruction 6
           [APL  6] msk: RL = <SB> | <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, assign("RL", disjoin(SB[sbublecci], src)))
        return result

    @classmethod
    @statement
    def rl_from_sb_or_inv_src(cls, sections, sb, src):
        """Read-logic Instruction 6
           [APL  6] msk: RL = <SB> | ~<SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections,
                        assign("RL",
                               disjoin(SB[sbublecci],
                                       invert(src))))
        return result

    #  ____
    # /_  /
    #  / /
    # /_/

    @classmethod
    @statement
    def rl_from_sb_xor_src(cls, sections, sb, src):
        """Read-logic Instruction 7
           [APL  7] msk: RL = <SB> ^ <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, assign("RL", xor(SB[sbublecci], src)))
        return result

    #   ___
    #  ( _ )
    # / _  |
    # \___/

    @classmethod
    @statement
    def rl_from_inv_sb_and_src(cls, sections, sb, src):
        """Read-logic Instruction 8
           [APL  8] msk: RL = ~<SB> & <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections,
                        assign("RL",
                             conjoin(~SB[sbublecci],
                                     src)))
        return result

    @classmethod
    @statement
    def rl_from_inv_sb_and_inv_src(cls, sections, sb, src):
        """Read-logic Instruction 8
           [APL  8] msk: RL = ~<SB> & ~<SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections,
                        assign("RL",
                             conjoin(~SB[sbublecci],
                                     invert(src))))
        return result

    #   ___
    #  / _ \
    #  \_, /
    # /___/

    @classmethod
    @statement
    def rl_from_sb_and_inv_src(cls, sections, sb, src):
        """Read-logic Instruction 9
           [APL  9] msk: RL = <SB> & ~<SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections,
                        assign("RL",
                               conjoin(SB[sbublecci],
                                       invert(src))))
        return result

    #  ___    __             _      _   _        __
    # / _ \  / /_ ____ _ _ _(_)__ _| |_(_)___ _ _\ \
    # \_, / | |\ V / _` | '_| / _` |  _| / _ \ ' \| |
    #  /_/  | | \_/\__,_|_| |_\__,_|\__|_\___/_||_| |
    #        \_\                                 /_/

    @classmethod
    @statement
    def rl_from_sb_xor_inv_src(cls, sections, sb, src):
        """undocumented variation of instruction 9
           [APL  9] msk: RL = <SB> ^ ~<SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections,
                        assign("RL",
                               xor(SB[sbublecci],
                                   invert(src))))
        return result

    #  ___     ___      _   _              _
    # | _ \___/ __| ___| | | |   ___  __ _(_)__
    # |   /___\__ \/ -_) | | |__/ _ \/ _` | / _|
    # |_|_\   |___/\___|_| |____\___/\__, |_\__|
    #                                |___/

    @staticmethod
    @statement
    def gl_from_rl(sections):
        """[APL] msk: GL = RL;
        In BLECCI, 'sections' must be an SM_REG_k, k in [0..15]."""
        result = masked(sections, assign("GL", "RL"))
        return result

    @staticmethod
    @statement
    def ggl_from_rl(sections):
        """[APL] msk: GGL = RL;
        In BLECCI, 'sections' must be an SM_REG_k, k in [0..15]."""
        result = masked(sections, assign("GGL", "RL"))
        return result

    @staticmethod
    @statement
    def ggl_from_rl_and_lx(sections, lx_addr):
        """[APL] msk: GGL = RL & <LX>;
        In BLECCI, 'sections' must be an SM_REG_k, k in [0..15]."""
        result = masked(sections, assign("GGL", conjoin("RL", lx_addr)))
        return result

    @staticmethod
    @statement
    def ggl_from_lx(lx_addr):
        """[APL] GGL = <LX>;"""
        result = assign("GGL", lx_addr)
        return result

    @staticmethod
    @statement
    def lgl_from_lx(lx_addr):
        """[APL] LGL = <LX>;"""
        result = assign("LGL", lx_addr)
        return result

    @staticmethod
    @statement
    def lx_from_ggl(lx_addr):
        result = assign(lx_addr, "GGL")
        return result

    @staticmethod
    @statement
    def lx_from_lgl(lx_addr):
        result = assign(lx_addr, "LGL")
        return result

    @staticmethod
    @statement
    def rsp16_from_rl(sections):
        """[APL] msk: RSP16 = RL;"""
        result = masked(sections, assign("RSP16", "RL"))
        return result

    @staticmethod
    @statement
    def rsp16_from_rsp256():
        """[APL] RSP16 = RSP256;"""
        result = assign("RSP16", "RSP256")
        return result

    @staticmethod
    @statement
    def rsp256_from_rsp16():
        """[APL] RSP256 = RSP16;"""
        result = assign("RSP256", "RSP16")
        return result

    @staticmethod
    @statement
    def rsp256_from_rsp2k():
        """[APL] RSP256 = RSP2K;"""
        result = assign("RSP256", "RSP2K")
        return result

    @staticmethod
    @statement
    def rsp2k_from_rsp256():
        """[APL] RSP2K = RSP256;"""
        result = assign("RSP2K", "RSP256")
        return result

    @staticmethod
    @statement
    def rsp2k_from_rsp32k():
        """[APL] RSP2K = RSP32K;"""
        result = assign("RSP2K", "RSP32K")
        return result

    @staticmethod
    @statement
    def rsp32k_from_rsp2k():
        """[APL] RSP32K = RSP2K;"""
        result = assign("RSP32K", "RSP2K")
        return result

    @staticmethod
    @statement
    def noop():
        """[APL] NOOP;  # needed for synch and timing."""
        result = f'NOOP'
        return result

    @staticmethod
    @statement
    def rsp_end():
        """[APL] RSP_END;"""
        result = f'RSP_END'
        return result

    # __      __   _ _         _              _
    # \ \    / / _(_) |_ ___  | |   ___  __ _(_)__
    #  \ \/\/ / '_| |  _/ -_) | |__/ _ \/ _` | / _|
    #   \_/\_/|_| |_|\__\___| |____\___/\__, |_\__|
    #                                   |___/

    @staticmethod
    def NRL():
        """[APL] NRL;  # <src> consisting of RL with its sections shifted 1 index toward max"""
        result = f"NRL"
        return result

    @staticmethod
    def ERL():
        """[APL] ERL;  # <src> consisting of RL with its plats shifted 1 index toward min"""
        result = f"ERL"
        return result

    @staticmethod
    def WRL():
        """[APL] WRL;  # <src> consisting of RL with its plats shifted 1 index toward max"""
        result = f"WRL"
        return result

    @staticmethod
    def SRL():
        """[APL] SRL;  # <src> consisting of RL with its sections shifted 1 index toward min"""
        result = f"SRL"
        return result

    @classmethod
    @statement
    def sb_from_src(cls, sections, sb, src):
        """[APL] msk: <SB> = <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, assign(SB[sbublecci], src))
        return result

    @classmethod
    @statement
    def sb_from_inv_src(cls, sections, sb, src):
        """[APL] msk: <SB> = ~<SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, assign(SB[sbublecci], invert(src)))
        return result

    @classmethod
    @statement
    def sb_cond_equals_src(cls, sections, sb, src):
        """[APL] msk: <SB> ?= <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, cond_eq(SB[sbublecci], src))
        return result

    @classmethod
    @statement
    def sb_cond_equals_inv_src(cls, sections, sb, src):
        """[APL] msk: <SB> ?= <SRC>"""
        sbublecci = cls._unpack_sbs_for_blecci(sb)
        result = masked(sections, cond_eq(SB[sbublecci], invert(src)))
        return result
