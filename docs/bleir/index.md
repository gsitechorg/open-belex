# BLEIR

An intermediate language for BLECCI. The purpose of this IR is to emit
code for each APL mimic through the `__str__` dunder of the `STATEMENT`
type. The ASTs (abstract syntax trees) can be passed downstream for
transformation and optimization.

For an updated EBNF grammar, please see docs/bleir/index.md

```
    APL MIMIC (see APL COMMANDS in README.md) BLEIR EXPRESSION
--- READ LOGIC
 1. set_rl(sects, bit)                        masked(sects, assign("RL", bit))
 2. rl_from_src(sects, src)                   masked(sects, assign("RL", src))
 3. rl_or_equals_src(sects, src)              masked(sects, or_eq("RL", src))
 4. rl_and_equals_src(sects, src)             masked(sects, and_eq("RL", src))
 5. rl_and_equals_inv_src(sects, src)         masked(sects, and_eq("RL", invert(src)))
 6. rl_xor_equals_src(sects, src)             masked(sects, xor_eq("RL", src))
 7. rl_from_sb(sects, sb)                     masked(sects, assign("RL", SB[sbub]))
 8. rl_or_equals_sb(sects, sb)                masked(sects, or_eq("RL", SB[sbub]))
 9. rl_and_equals_sb(sects, sb)               masked(sects, and_eq("RL", SB[sbub]))
10. rl_and_equals_inv_sb(sects, sb)           masked(sects, and_eq("RL", invert(SB[sbub])))
11. rl_xor_equals_sb(sects, sb)               masked(sects, xor_eq("RL", SB[sbub]))
12. rl_from_sb_and_src(sects, sb, src)        masked(sects, assign("RL", conjoin(SB[sbub], src)))
13. rl_or_equals_sb_and_src(sects, sb, src)   masked(sects, or_eq("RL", conjoin(SB[sbub], src)))
14. rl_and_equals_sb_and_src(sects, sb, src)  masked(sects, and_eq("RL", conjoin(SB[sbub], src)))
15. rl_xor_equals_sb_and_src(sects, sb, src)  masked(sects, xor_eq("RL", conjoin(SB[sbub], src)))
16. rl_from_sb_or_src(sects, sb, src)         masked(sects, assign("RL", disjoin(SB[sbub], src)))
17. rl_from_sb_xor_src(sects, sb, src)        masked(sects, assign("RL", xor(SB[sbub], src)))
18. rl_from_inv_sb_and_src(sects, sb, src)    masked(sects, assign("RL", conjoin(invert(SB[sbub]), src)))
19. rl_from_sb_and_inv_src(sects, sb, src)    masked(sects, assign("RL", conjoin(SB[sbub], invert(src))))
--- BROADCAST (R-SEL)
21. gl_from_rl(sects)                         masked(sects, assign("GL", "RL"))
22. rsp16_from_rl(sects)                      masked(sects, assign("RSP16", "RL"))
23. rsp256_from_rsp16(self)                   assign("RSP256", "RSP16")
24. rsp2k_from_rsp256(self)                   assign("RSP2K", "RSP256")
25. rsp32k_from_rsp2k(self)                   assign("RSP32K", "RSP2K")
26. noop(self)                                f'NOOP;'
27. rsp_end(self)                             f'RSP_END;'
--- WRITE_LOGIC
28. sb_from_src(sects, sb, src)               masked(sects, assign(SB[sbub], src))
```

Examples (approximate) ASTs:

```
set_rl: MASKED                          rl_from_inv_sb_and_src: MASKED
|-- mask: MASK                          |-- mask: MASK
|   |-- expression: SM_REG              |   |-- expression: SM_REG
|   |   `-- identifier: str             |   |   `-- identifier: str
|   |       `-- "fs"                    |   |       `-- "lvr"
|   `-- operator: Optional[UNARY_OP]    |   `-- operator: Optional[UNARY_OP]
|       `-- None                        |       `-- None
`-- assignment: ASSIGNMENT              `-- assignment: ASSIGNMENT
    `-- operation: READ                     `-- operation: READ
        |-- operator: ASSIGN_OP                 |-- operator: ASSIGN_OP
        |   `-- ASSIGN_OP.EQ                    |   `-- ASSIGN_OP.EQ
        `-- rvalue: UNARY_EXPR                  `-- rvalue: BINARY_EXPR
            `-- expression: BIT_EXPR                |-- operator: BINOP
                `-- BIT_EXPR.ONE                    |   `-- BINOP.AND
                                                    |-- left_operand: UNARY_SB
                                                    |   |-- expression: SB_EXPR
                                                    |   |   x: RN_REG
                                                    |   |   `-- identifier: str
                                                    |   |       `-- "rvr"
                                                    |   `-- operator: Optional[UNARY_OP]
                                                    |       `-- UNARY_OP.NEGATE
                                                    `-- right_operand: UNARY_SRC
                                                        |-- expression: SRC_EXPR
                                                        |   `-- SRC_EXPR.RL
                                                        `-- operator: Optional[UNARY_OP]
                                                            `-- None
```

TODO: is the following commentary concerning LIR and telemetry relevant any more?

Example of transformation of BLEIR into LIR: line 2907 of `test_BLECCI_1.py`, namely

```
    bellib_vr_and(pyble, a_and_cry, a, cry, msk_txt)
```

generates the following BLEIR

```
    [STATEMENT(expr=MASKED(
        mask=MASK(
            expression=SM_REG(
                identifier="r2sec")),
        assignment=ASSIGNMENT(
            operation=READ(
                operator=ASSIGN_OP.EQ,
                rvalue=UNARY_EXPR(
                    expression=UNARY_SB(
                        expression=SB_EXPR(
                            x=RN_REG(
                                identifier="r2vr")))))))),
     STATEMENT(expr=MASKED(
         mask=MASK(
             expression=SM_REG(
                 identifier="rsec)),
         assignment=ASSIGNMENT(
             operation=READ(
                 operator=ASSIGN_OP.EQ,
                 rvalue=BINARY_EXPR(
                     operator=BINOP.AND,
                     left_operand=UNARY_SB(
                         expression=SB_EXPR(
                             x=RN_REG(
                                 identifier="rvr))),
                     right_operand=UNARY_SRC(
                         expression=SRC_EXPR.RL)))))),
     STATEMENT(expr=MASKED(
         mask=MASK(
             expression=SM_REG(
                 identifier="lsec")),
         assignment=ASSIGNMENT(
             operation=WRITE(
                 lvalue=SB_EXPR(
                     x=RN_REG(
                         identifier="lvr")),
                 rvalue=UNARY_SRC(
                     expression=SRC_EXPR.RL)))))]
```

which we need to transform into LIR for the optimizer

```
    ut_aAc = irbcb1.AND(ir_a, ir_cry)
```

1 Jan 2021

it seems very easy to go from `ut_aAc = irbcb1.AND(mask, ir_a, ir_cry)` to
`bellib_vr_and(pyble, a_and_cry, a, cry, mask)`, then to belops, where the
user's variables are lost, and become `lsec_rp`, `rsec_rp`, etc.

Going the other way is the old Telemetry problem again. There is no easy
way to go from bellib/belop to LIR without tracking the user's variable
names and mapping them from the internal names in the bellib/belop
implementations.

Imagine a top-level expression language, a variation of LIR (adding
masks/indices), and our front-end work is to generate bellibs and belops
from it:

Here's what's happening now:

```
(LIR (IntermediateRepresentation) <-> optimizers) -> straight to APL
```

```
bellib / belop / blecci -> BLEIR -> straight to APL
```

Suppose `(LIR <-> optim) -> BLEIR -> straight to APL`,
or `(LIR <-> optim) -> bellib / belop / blecci -> BLEIR -> straight to APL`

We could have both of those, `LIR->BLEIR`, `LIR->bellibEtc->BLEIR`, mix and
match but optimizers only work at LIR level.

We had been thinking `bellibEtc -> LIR(crunch,crunch) -> BLEIR -> APL`.

As the  old "telemetry" problem again ... by the time control gets to
bellib/belop, the user's variables have been erased. Bellib/Belop only
knows `lsec`, `rsec`, `lvr`, `rvr`, etc. It doesn't know `a`, `b`,
 `a_and_cry`, etc.

If we don't make LIR higher than bellib/belop, then we must telemeter
the user's variables into the belops. That doesn't seem right.

## Grammar

EBNF grammar for BLEIR using ANTLRv4 syntax:

```antlr
// Top-level element of a BLEIR application that describes the application
// and its operations.
//
// Parameters:
//     name: identifies the application.
//     examples: initial states and expected final states of executions of the
//               application.
//     calls: FragmentCallerCalls that define the application.
//     doc_comment: optional description of the application's purpose.
//     metadata: meta-information about this Snippet such as the name of its
//               APL header.
//     library_callers: sequence of fragment callers to include in the
//                      generated APL that are not called explicitly.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_snippet(
//            self: BLEIRVisitor,
//            snippet: Snippet) -> Any
//     2. BLEIRTransformer.transform_snippet(
//            self: BLEIRTransformer,
//            snippet: Snippet) -> Snippet
//     3. BLEIRListener.enter_snippet(
//            self: BLEIRListener,
//            snippet: Snippet) -> None
//     4. BLEIRListener.exit_snippet(
//            self: BLEIRListener,
//            snippet: Snippet) -> None
Snippet:
    name=str
    ( examples+=Example )*
    ( calls+=( FragmentCallerCall | MultiLineComment | SingleLineComment ) )*
    ( doc_comment=MultiLineComment )?
    ( metadata=( SnippetMetadata | Any ) )?
    ( library_callers=FragmentCaller )?;


// str(object='') -> str
// str(bytes_or_buffer[, encoding[, errors]]) -> str
//
// Create a new string object from the given object. If encoding or
// errors is specified, then the object must expose a data buffer
// that will be decoded using the given encoding and error handler.
// Otherwise, returns the result of object.__str__() (if defined)
// or repr(object).
// encoding defaults to sys.getdefaultencoding().
// errors defaults to 'strict'.
str: /* see relevant docs */;


// Contains the initial and expected final states for a single execution of
// the application.
//
// Parameters:
//     expected_value: expected output of a single VR.
//     parameters: inital state of the application.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_example(
//            self: BLEIRVisitor,
//            example: Example) -> Any
//     2. BLEIRTransformer.transform_example(
//            self: BLEIRTransformer,
//            example: Example) -> Example
//     3. BLEIRListener.enter_example(
//            self: BLEIRListener,
//            example: Example) -> None
//     4. BLEIRListener.exit_example(
//            self: BLEIRListener,
//            example: Example) -> None
Example:
    expected_value=ValueParameter
    ( parameters+=ValueParameter )*;


// Represents the value of an `Example` parameter and references the vector
// register that should receive it.
//
// Parameters:
//     identifier: name of this parameter.
//     row_number: identifies the vector register that should hold the value.
//     value: numpy array of 2048, 16-bit unsigned integers.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_value_parameter(
//            self: BLEIRVisitor,
//            value_parameter: ValueParameter) -> Any
//     2. BLEIRTransformer.transform_value_parameter(
//            self: BLEIRTransformer,
//            value_parameter: ValueParameter) -> ValueParameter
//     3. BLEIRListener.enter_value_parameter(
//            self: BLEIRListener,
//            value_parameter: ValueParameter) -> None
//     4. BLEIRListener.exit_value_parameter(
//            self: BLEIRListener,
//            value_parameter: ValueParameter) -> None
ValueParameter:
    identifier=str
    row_number=int
    value=ndarray;


// int([x]) -> integer
// int(x, base=10) -> integer
//
// Convert a number or string to an integer, or return 0 if no arguments
// are given.  If x is a number, return x.__int__().  For floating point
// numbers, this truncates towards zero.
//
// If x is not a number or if base is given, then x must be a string,
// bytes, or bytearray instance representing an integer literal in the
// given base.  The literal can be preceded by '+' or '-' and be surrounded
// by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
// Base 0 means to interpret the base from the string as an integer literal.
// >>> int('0b100', base=0)
// 4
int: /* see relevant docs */;


// ndarray(shape, dtype=float, buffer=None, offset=0,
//         strides=None, order=None)
//
// An array object represents a multidimensional, homogeneous array
// of fixed-size items.  An associated data-type object describes the
// format of each element in the array (its byte-order, how many bytes it
// occupies in memory, whether it is an integer, a floating point number,
// or something else, etc.)
//
// Arrays should be constructed using `array`, `zeros` or `empty` (refer
// to the See Also section below).  The parameters given here refer to
// a low-level method (`ndarray(...)`) for instantiating an array.
//
// For more information, refer to the `numpy` module and examine the
// methods and attributes of an array.
//
// Parameters
// ----------
// (for the __new__ method; see Notes below)
//
// shape : tuple of ints
//     Shape of created array.
// dtype : data-type, optional
//     Any object that can be interpreted as a numpy data type.
// buffer : object exposing buffer interface, optional
//     Used to fill the array with data.
// offset : int, optional
//     Offset of array data in buffer.
// strides : tuple of ints, optional
//     Strides of data in memory.
// order : {'C', 'F'}, optional
//     Row-major (C-style) or column-major (Fortran-style) order.
//
// Attributes
// ----------
// T : ndarray
//     Transpose of the array.
// data : buffer
//     The array's elements, in memory.
// dtype : dtype object
//     Describes the format of the elements in the array.
// flags : dict
//     Dictionary containing information related to memory use, e.g.,
//     'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
// flat : numpy.flatiter object
//     Flattened version of the array as an iterator.  The iterator
//     allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
//     assignment examples; TODO).
// imag : ndarray
//     Imaginary part of the array.
// real : ndarray
//     Real part of the array.
// size : int
//     Number of elements in the array.
// itemsize : int
//     The memory use of each array element in bytes.
// nbytes : int
//     The total number of bytes required to store the array data,
//     i.e., ``itemsize * size``.
// ndim : int
//     The array's number of dimensions.
// shape : tuple of ints
//     Shape of the array.
// strides : tuple of ints
//     The step-size required to move from one element to the next in
//     memory. For example, a contiguous ``(3, 4)`` array of type
//     ``int16`` in C-order has strides ``(8, 2)``.  This implies that
//     to move from element to element in memory requires jumps of 2 bytes.
//     To move from row-to-row, one needs to jump 8 bytes at a time
//     (``2 * 4``).
// ctypes : ctypes object
//     Class containing properties of the array needed for interaction
//     with ctypes.
// base : ndarray
//     If the array is a view into another array, that array is its `base`
//     (unless that array is also a view).  The `base` array is where the
//     array data is actually stored.
//
// See Also
// --------
// array : Construct an array.
// zeros : Create an array, each element of which is zero.
// empty : Create an array, but leave its allocated memory unchanged (i.e.,
//         it contains "garbage").
// dtype : Create a data-type.
// numpy.typing.NDArray : An ndarray alias :term:`generic <generic type>`
//                        w.r.t. its `dtype.type <numpy.dtype.type>`.
//
// Notes
// -----
// There are two modes of creating an array using ``__new__``:
//
// 1. If `buffer` is None, then only `shape`, `dtype`, and `order`
//    are used.
// 2. If `buffer` is an object exposing the buffer interface, then
//    all keywords are interpreted.
//
// No ``__init__`` method is needed because the array is fully initialized
// after the ``__new__`` method.
//
// Examples
// --------
// These examples illustrate the low-level `ndarray` constructor.  Refer
// to the `See Also` section above for easier ways of constructing an
// ndarray.
//
// First mode, `buffer` is None:
//
// >>> np.ndarray(shape=(2,2), dtype=float, order='F')
// array([[0.0e+000, 0.0e+000], # random
//        [     nan, 2.5e-323]])
//
// Second mode:
//
// >>> np.ndarray((2,), buffer=np.array([1,2,3]),
// ...            offset=np.int_().itemsize,
// ...            dtype=int) # offset = 1*itemsize, i.e. skip first element
// array([2, 3])
ndarray: /* see relevant docs */;


// Represents a concrete call to a fragment through its fragment caller.
//
// For example, if you want to call the Fragment, `set_rl`, you would call it
// through `set_rl_caller` with actual parameters:
//
//     set_rl_caller(4, 0xBEEF);
//
// Parameters:
//     caller: FragmentCaller that ferries calls to the desired Fragment.
//     parameters: actual parameters used by the Fragment's operations.
//     metadata: bits of information about this FragmentCallerCall that might
//               affect such things as its code is generated.
//     comment: an optional comment associated with this FragmentCallerCall
//              that might include such things as debugger information.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_fragment_caller_call(
//            self: BLEIRVisitor,
//            fragment_caller_call: FragmentCallerCall) -> Any
//     2. BLEIRTransformer.transform_fragment_caller_call(
//            self: BLEIRTransformer,
//            fragment_caller_call: FragmentCallerCall) -> FragmentCallerCall
//     3. BLEIRListener.enter_fragment_caller_call(
//            self: BLEIRListener,
//            fragment_caller_call: FragmentCallerCall) -> None
//     4. BLEIRListener.exit_fragment_caller_call(
//            self: BLEIRListener,
//            fragment_caller_call: FragmentCallerCall) -> None
FragmentCallerCall:
    caller=FragmentCaller
    ( parameters+=( int | str ) )*
    ( metadata=( CallMetadata | Any ) )?
    ( comment=( MultiLineComment | SingleLineComment | TrailingComment ) )?;


// Represents a C function that calls a fragment. It is within the
// FragmentCaller that registers are allocated for parameters and it is the
// FragmentCaller that is called from the C main function.
//
// An example FragmentCaller might look like the following:
//
//     void set_rl_caller(
//             u16 lvr_vp,
//             u16 msk_vp)
//     {   apl_set_rn_reg(RN_REG_0, lvr_vp);
//         apl_set_sm_reg(SM_REG_0, msk_vp);
//         RUN_FRAG_ASYNC(
//             set_rl(
//                 lvr_rp=RN_REG_0,
//                 msk_rp=SM_REG_0));   }
//
// Parameters:
//     fragment: the Fragment to call.
//     registers: an optional sequence of AllocatedRegisters that specifies
//                which registers should receive the actual parameters.
//     metadata: mapping over meta-information about this fragment caller,
//               such as whether it was built by BELEX or BLECCI.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_fragment_caller(
//            self: BLEIRVisitor,
//            fragment_caller: FragmentCaller) -> Any
//     2. BLEIRTransformer.transform_fragment_caller(
//            self: BLEIRTransformer,
//            fragment_caller: FragmentCaller) -> FragmentCaller
//     3. BLEIRListener.enter_fragment_caller(
//            self: BLEIRListener,
//            fragment_caller: FragmentCaller) -> None
//     4. BLEIRListener.exit_fragment_caller(
//            self: BLEIRListener,
//            fragment_caller: FragmentCaller) -> None
FragmentCaller:
    fragment=Fragment
    ( registers=( AllocatedRegister | MultiLineComment | SingleLineComment | TrailingComment ) )?
    ( metadata=( CallerMetadata | Any ) )?;


// Represents a block of STATEMENTs and MultiStatements are executed with a
// given list of parameters.
//
// An example Fragment might look like the following:
//
//     APL_FRAG set_rl(
//             RN_REG rvr_rp,
//             SM_REG msk_rp)
//     {   {
//         msk_rp: RL = SB[rvr_rp];
//         }   };
//
// Parameters:
//     identifier: name to assign the Fragment.
//     parameters: sequence of RN_REGs and SM_REGs covering those used by the
//                 operations.
//     operations: sequence of instructions to execute in the provided order.
//     doc_comment: an optional comment describing the purpose of the
//                  Fragment.
//     metadata: a mapping over FragmentMetadata -> Value pairs containing
//               meta information about this Fragment.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_fragment(
//            self: BLEIRVisitor,
//            fragment: Fragment) -> Any
//     2. BLEIRTransformer.transform_fragment(
//            self: BLEIRTransformer,
//            fragment: Fragment) -> Fragment
//     3. BLEIRListener.enter_fragment(
//            self: BLEIRListener,
//            fragment: Fragment) -> None
//     4. BLEIRListener.exit_fragment(
//            self: BLEIRListener,
//            fragment: Fragment) -> None
Fragment:
    identifier=str
    ( parameters+=( RN_REG | RE_REG | EWE_REG | L1_REG | L2_REG | SM_REG ) )*
    ( operations+=( MultiStatement | STATEMENT | MultiLineComment | SingleLineComment ) )*
    ( doc_comment=MultiLineComment )?
    ( metadata=( FragmentMetadata | Any ) )?
    ( children=Fragment )?;


// Identifies a vector register (VR).
//
// Parameters:
//     identifier: name representing the VR, e.g. `lvr`.
//     comment: optional comment to associate with this RN_REG.
//     initial_value: (optional) initial value to write across all plats of
//                    the VR associated with this register (identified by its
//                    row number). RN_REGs with initial values do not have
//                    corresponding parameters. Instead, they will be
//                    initialized as local constant variables within the
//                    fragment caller body.
//     register: pre-defined register id to allocate for this RN_REG.
//     row_number: constant row number to assign the corresponding RN_REG (not
//                 its corresponding VR value).
//     is_lowered: whether this register is lowered, or referenced from a
//                 global context instead of from the parameter list or local
//                 variables. Lowering is the opposite of lambda lifting (see:
//                 https://en.wikipedia.org/wiki/Lambda_lifting).
//     is_literal: whether this instance represents an RN_REG literal and not
//                 a variable (e.g. RN_REG_0).
//     is_temporary Whether this register is part of the fragment but not its
//                  caller.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rn_reg(
//            self: BLEIRVisitor,
//            rn_reg: RN_REG) -> Any
//     2. BLEIRTransformer.transform_rn_reg(
//            self: BLEIRTransformer,
//            rn_reg: RN_REG) -> RN_REG
//     3. BLEIRListener.enter_rn_reg(
//            self: BLEIRListener,
//            rn_reg: RN_REG) -> None
//     4. BLEIRListener.exit_rn_reg(
//            self: BLEIRListener,
//            rn_reg: RN_REG) -> None
RN_REG:
    identifier=str
    ( comment=InlineComment )?
    ( initial_value=int )?
    ( register=int )?
    ( row_number=int )?
    is_lowered=bool
    is_literal=bool
    is_temporary=bool;


// Comments that appear, intermingled, within expressions, parameters,
// etc., like so:
//
//     APL_FRAG foo(RN_REG lvr_rp /* lvalue */,
//                  RN_REG rvr_rp /* rvalue */,
//                  SM_REG msk_rp /* section mask */)
//
// Parameters:
//     value: string content of the comment.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_inline_comment(
//            self: BLEIRVisitor,
//            inline_comment: InlineComment) -> Any
//     2. BLEIRTransformer.transform_inline_comment(
//            self: BLEIRTransformer,
//            inline_comment: InlineComment) -> InlineComment
//     3. BLEIRListener.enter_inline_comment(
//            self: BLEIRListener,
//            inline_comment: InlineComment) -> None
//     4. BLEIRListener.exit_inline_comment(
//            self: BLEIRListener,
//            inline_comment: InlineComment) -> None
InlineComment:
    value=str;


NoneType: /* see relevant docs */;


// bool(x) -> bool
//
// Returns True when the argument x is true, False otherwise.
// The builtins True and False are the only two instances of the class bool.
// The class bool is a subclass of the class int, and cannot be subclassed.
bool: /* see relevant docs */;


// Identifies a special type of register that represents combinations over
// all 24 VRs.
//
// Parameters:
//     identifier: name representing the register parameter.
//     comment: optional comment to associate with this RE_REG.
//     initial_value: (optional) initial value to write across all plats of
//                    the memory associated with this register.
//     register: pre-defined register id to allocate for this RE_REG.
//     row_mask: 24-bit mask representing the combination of VRs associated
//               with this register.
//     is_lowered: whether this register is lowered, or referenced from a
//                 global context instead of from the parameter list or local
//                 variables. Lowering is the opposite of lambda lifting (see:
//                 https://en.wikipedia.org/wiki/Lambda_lifting).
//     is_literal: whether this instance represents an RE_REG literal and not
//                 a variable (e.g. RE_REG_0).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_re_reg(
//            self: BLEIRVisitor,
//            re_reg: RE_REG) -> Any
//     2. BLEIRTransformer.transform_re_reg(
//            self: BLEIRTransformer,
//            re_reg: RE_REG) -> RE_REG
//     3. BLEIRListener.enter_re_reg(
//            self: BLEIRListener,
//            re_reg: RE_REG) -> None
//     4. BLEIRListener.exit_re_reg(
//            self: BLEIRListener,
//            re_reg: RE_REG) -> None
RE_REG:
    identifier=str
    ( comment=InlineComment )?
    ( initial_value=int )?
    ( register=int )?
    ( row_mask=int )?
    is_lowered=bool
    is_literal=bool;


// Identifies a special type of register that represents combinations over
// half word-lines (1024 of the 2048 plats).
//
// Parameters:
//     identifier: name representing the register parameter.
//     comment: optional comment to associate with this EWE_REG.
//     initial_value: (optional) initial value to write across all plats of
//                    the memory associated with this register.
//     register: pre-defined register id to allocate for this EWE_REG.
//     wordline_mask: 10-bit mask representing the combination of plats
//                    associated with this register.
//     is_lowered: whether this register is lowered, or referenced from a
//                 global context instead of from the parameter list or local
//                 variables. Lowering is the opposite of lambda lifting (see:
//                 https://en.wikipedia.org/wiki/Lambda_lifting).
//     is_literal: whether this instance represents an EWE_REG literal and not
//                 a variable (e.g. EWE_REG_0).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_ewe_reg(
//            self: BLEIRVisitor,
//            ewe_reg: EWE_REG) -> Any
//     2. BLEIRTransformer.transform_ewe_reg(
//            self: BLEIRTransformer,
//            ewe_reg: EWE_REG) -> EWE_REG
//     3. BLEIRListener.enter_ewe_reg(
//            self: BLEIRListener,
//            ewe_reg: EWE_REG) -> None
//     4. BLEIRListener.exit_ewe_reg(
//            self: BLEIRListener,
//            ewe_reg: EWE_REG) -> None
EWE_REG:
    identifier=str
    ( comment=InlineComment )?
    ( initial_value=int )?
    ( register=int )?
    ( wordline_mask=int )?
    is_lowered=bool
    is_literal=bool;


// Identifies a register associated with the level of memory just above
// MMB. This level is used for I/O and spilling and restoring MMB registers.
// It is slower to access this level than MMB but not as slow as accessing L2.
//
// Parameters:
//     identifier: name representing the register parameter.
//     comment: optional comment to associate with this register parameter.
//     register: pre-defined register id to allocate for this register
//               parameter.
//     bank_group_row: 13-bit scalar representing the encoded bank_id,
//                     group_id, and row_id used to identify the memory for
//                     this register parameter. A bank_id is comprised of 2
//                     bits, a group_id is comprised of 2 bits, and a row_id
//                     is comprised of 9 bits. The various ids are combined as
//                     follows:
//
//                         bank_group_row = (bank_id << 11) \
//                                        | (group_id << 9) \
//                                        | row_id.
//
//                     It is important to understand that although the maximum
//                     value of the bank_group_row is (1 << 13) = 8192, the
//                     maximum value supported by the hardware is 6144.
//     is_lowered: whether this register is lowered, or referenced from a
//                 global context instead of from the parameter list or local
//                 variables. Lowering is the opposite of lambda lifting (see:
//                 https://en.wikipedia.org/wiki/Lambda_lifting).
//     is_literal: whether this instance represents a register literal and not
//                 a variable (e.g. L1_ADDR_REG_0).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_l1_reg(
//            self: BLEIRVisitor,
//            l1_reg: L1_REG) -> Any
//     2. BLEIRTransformer.transform_l1_reg(
//            self: BLEIRTransformer,
//            l1_reg: L1_REG) -> L1_REG
//     3. BLEIRListener.enter_l1_reg(
//            self: BLEIRListener,
//            l1_reg: L1_REG) -> None
//     4. BLEIRListener.exit_l1_reg(
//            self: BLEIRListener,
//            l1_reg: L1_REG) -> None
L1_REG:
    identifier=str
    ( comment=InlineComment )?
    ( register=int )?
    ( bank_group_row=( int | int | int | int ) )?
    is_lowered=bool
    is_literal=bool;


// Identifies a register associated with the level of memory just above L1.
// This level is used for I/O as data must be written to L2 before L1 and
// vice-versa.
//
// Parameters:
//     identifier: name representing the register parameter.
//     comment: optional comment to associate with this register parameter.
//     register: pre-defined register id to allocate for this register
//               parameter.
//     value: register value to assign this parameter.
//     is_lowered: whether this register is lowered, or referenced from a
//                 global context instead of from the parameter list or local
//                 variables. Lowering is the opposite of lambda lifting (see:
//                 https://en.wikipedia.org/wiki/Lambda_lifting).
//     is_literal: whether this instance represents a register literal and not
//                 a variable (e.g. L2_ADDR_REG_0).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_l2_reg(
//            self: BLEIRVisitor,
//            l2_reg: L2_REG) -> Any
//     2. BLEIRTransformer.transform_l2_reg(
//            self: BLEIRTransformer,
//            l2_reg: L2_REG) -> L2_REG
//     3. BLEIRListener.enter_l2_reg(
//            self: BLEIRListener,
//            l2_reg: L2_REG) -> None
//     4. BLEIRListener.exit_l2_reg(
//            self: BLEIRListener,
//            l2_reg: L2_REG) -> None
L2_REG:
    identifier=str
    ( comment=InlineComment )?
    ( register=int )?
    ( value=int )?
    is_lowered=bool
    is_literal=bool;


// Identifies a section mask (SM).
//
// Parameters:
//     identifier: name representing the SM, e.g. `fs`.
//     comment: optional comment to associate with this SM_REG.
//     negated: helper attribute for parsing APL, specifies whether this
//              SM_REG should be negated (have its 1s flipped to 0s, and
//              vice-versa).
//     constant_value: optional constant value to assign this register.
//                     SM_REGs with constant values do not have corresponding
//                     parameters generated for them. Instead, they will be
//                     allocated within the fragment caller body.
//     register: pre-defined register to allocate for this SM_REG.
//     is_section: whether this SM_REG should be treated as a section value
//                 instead of a mask value. SM_REGs marked as sections will
//                 have their corresponding actual (section) values
//                 transformed into mask values as follows:
//                 sm_vp = (0x0001 << sm_vp).
//     is_lowered: whether this register is lowered, or referenced from a
//                 global context instead of from the parameter list or local
//                 variables. Lowering is the opposite of lambda lifting (see:
//                 https://en.wikipedia.org/wiki/Lambda_lifting).
//     is_literal: whether this instance represents a register literal and not
//                 a variable (e.g. SM_REG_0).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_sm_reg(
//            self: BLEIRVisitor,
//            sm_reg: SM_REG) -> Any
//     2. BLEIRTransformer.transform_sm_reg(
//            self: BLEIRTransformer,
//            sm_reg: SM_REG) -> SM_REG
//     3. BLEIRListener.enter_sm_reg(
//            self: BLEIRListener,
//            sm_reg: SM_REG) -> None
//     4. BLEIRListener.exit_sm_reg(
//            self: BLEIRListener,
//            sm_reg: SM_REG) -> None
SM_REG:
    identifier=str
    ( comment=InlineComment )?
    ( negated=bool )?
    ( constant_value=int )?
    ( register=int )?
    is_section=bool
    is_lowered=bool
    is_literal=bool;


// A collection of 1-to-4 STATEMENTs that are executed in parallel as a
// single STATEMENT. With two exceptions, the STATEMENTs should not depend on
// each and should yield the same half-bank state after executing in any
// order. The two exceptions are WRITE-before-READ and READ-before-BROADCAST,
// which operate in terms of half clocks and can be combined with dependencies
// in the same MultiStatement.
//
// A MultiStatement takes any of the following forms:
//     1. { STATEMENT; }
//     2. { STATEMENT; STATEMENT; }
//     3. { STATEMENT; STATEMENT; STATEMENT; }
//     4. { STATEMENT; STATEMENT; STATEMENT; STATEMENT; }
//
// Parameters:
//     statements: unordered collection of STATEMENTs to execute in parallel.
//     comment: an optional comment to associate with this MultiStatement.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_multi_statement(
//            self: BLEIRVisitor,
//            multi_statement: MultiStatement) -> Any
//     2. BLEIRTransformer.transform_multi_statement(
//            self: BLEIRTransformer,
//            multi_statement: MultiStatement) -> MultiStatement
//     3. BLEIRListener.enter_multi_statement(
//            self: BLEIRListener,
//            multi_statement: MultiStatement) -> None
//     4. BLEIRListener.exit_multi_statement(
//            self: BLEIRListener,
//            multi_statement: MultiStatement) -> None
MultiStatement:
    ( statements+=( STATEMENT | MultiLineComment | SingleLineComment ) )*
    ( comment=( MultiLineComment | SingleLineComment | TrailingComment ) )?;


// A semicolon-terminated operation that operates on the half-bank. A
// STATEMENT might be a MASKED operation, SPECIAL operation, or any of the RSP
// assignments.
//
// Parameters:
//     operation: manipulates the state of the half-bank in some fashion.
//     comment: an optional comment to associate with this STATEMENT.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_statement(
//            self: BLEIRVisitor,
//            statement: STATEMENT) -> Any
//     2. BLEIRTransformer.transform_statement(
//            self: BLEIRTransformer,
//            statement: STATEMENT) -> STATEMENT
//     3. BLEIRListener.enter_statement(
//            self: BLEIRListener,
//            statement: STATEMENT) -> None
//     4. BLEIRListener.exit_statement(
//            self: BLEIRListener,
//            statement: STATEMENT) -> None
STATEMENT:
    operation=( MASKED | SPECIAL | RSP16_ASSIGNMENT | RSP256_ASSIGNMENT | RSP2K_ASSIGNMENT | RSP32K_ASSIGNMENT | GGL_ASSIGNMENT | LGL_ASSIGNMENT | LX_ASSIGNMENT | GlassStatement )
    ( comment=( MultiLineComment | SingleLineComment | TrailingComment ) )?;


// Represents an operation that requires a section mask.
//
// Parameters:
//     mask: section mask to apply to the assignment operation.
//     assignment: assigns some rvalue to an lvalue under the given mask.
//     read_write_inhibit: whether the corresponding operation should be
//                         read/write inhibited, that is, whether its sections
//                         should be disabled to conserve energy.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_masked(
//            self: BLEIRVisitor,
//            masked: MASKED) -> Any
//     2. BLEIRTransformer.transform_masked(
//            self: BLEIRTransformer,
//            masked: MASKED) -> MASKED
//     3. BLEIRListener.enter_masked(
//            self: BLEIRListener,
//            masked: MASKED) -> None
//     4. BLEIRListener.exit_masked(
//            self: BLEIRListener,
//            masked: MASKED) -> None
MASKED:
    mask=MASK
    ( assignment=ASSIGNMENT )?
    ( read_write_inhibit=ReadWriteInhibit )?;


// Optionally transforms a section mask operand with some unary operator.
//
// A MASK takes either of the following forms:
//     1. msk
//     2. ~msk
//     3. msk<<num_bits
//     4. ~(msk<<num_bits)
//
// Parameters:
//     expression: section mask operand to transform.
//     operator: optional unary operator to transform the expression.
//     read_write_inhibit: whether this mask should be read/write inhibited,
//                         that is, whether its sections should be disabled to
//                         conserve energy.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_mask(
//            self: BLEIRVisitor,
//            mask: MASK) -> Any
//     2. BLEIRTransformer.transform_mask(
//            self: BLEIRTransformer,
//            mask: MASK) -> MASK
//     3. BLEIRListener.enter_mask(
//            self: BLEIRListener,
//            mask: MASK) -> None
//     4. BLEIRListener.exit_mask(
//            self: BLEIRListener,
//            mask: MASK) -> None
MASK:
    expression=( SM_REG | SHIFTED_SM_REG )
    ( operator=UNARY_OP )?
    ( read_write_inhibit=ReadWriteInhibit )?;


// Shifts a section mask to the left by the specified number of bits.
//
// A SHIFTED_SM_REG takes the following form:
//     1. msk<<num_bits
//
// where `num_bits` is between 0 and 16.
//
// Parameters:
//     register: section mask whose value should be shifted.
//     num_bits: number of bits to shift the section mask to the left.
//     negated: helper attribute that specifies whether the section mask bits
//              should be negated after they have been shifted.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_shifted_sm_reg(
//            self: BLEIRVisitor,
//            shifted_sm_reg: SHIFTED_SM_REG) -> Any
//     2. BLEIRTransformer.transform_shifted_sm_reg(
//            self: BLEIRTransformer,
//            shifted_sm_reg: SHIFTED_SM_REG) -> SHIFTED_SM_REG
//     3. BLEIRListener.enter_shifted_sm_reg(
//            self: BLEIRListener,
//            shifted_sm_reg: SHIFTED_SM_REG) -> None
//     4. BLEIRListener.exit_shifted_sm_reg(
//            self: BLEIRListener,
//            shifted_sm_reg: SHIFTED_SM_REG) -> None
SHIFTED_SM_REG:
    register=( SM_REG | SHIFTED_SM_REG )
    num_bits=int
    ( negated=bool )?;


// Internal wrapper to hold a forward reference.
SHIFTED_SM_REG: /* see relevant docs */;


// Negates the bits of the operand such that its 1s become 0s and its 0s
// become 1s, (e.g. `~0b10 == 0b01`).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_unary_op(
//            self: BLEIRVisitor,
//            unary_op: UNARY_OP) -> Any
//     2. BLEIRTransformer.transform_unary_op(
//            self: BLEIRTransformer,
//            unary_op: UNARY_OP) -> UNARY_OP
//     3. BLEIRListener.enter_unary_op(
//            self: BLEIRListener,
//            unary_op: UNARY_OP) -> None
//     4. BLEIRListener.exit_unary_op(
//            self: BLEIRListener,
//            unary_op: UNARY_OP) -> None
UNARY_OP:
    ( NEGATE="~" );


// A special token type used to represent read-write inhibited sections.
// When masked sections are read-write inhibited they are disabled to reduce
// power consumption. You should never enable read-write inhibit over sections
// you want to use.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_read_write_inhibit(
//            self: BLEIRVisitor,
//            read_write_inhibit: ReadWriteInhibit) -> Any
//     2. BLEIRTransformer.transform_read_write_inhibit(
//            self: BLEIRTransformer,
//            read_write_inhibit: ReadWriteInhibit) -> ReadWriteInhibit
//     3. BLEIRListener.enter_read_write_inhibit(
//            self: BLEIRListener,
//            read_write_inhibit: ReadWriteInhibit) -> None
//     4. BLEIRListener.exit_read_write_inhibit(
//            self: BLEIRListener,
//            read_write_inhibit: ReadWriteInhibit) -> None
ReadWriteInhibit:
    ( RWINH_SET="RWINH_SET"
    | RWINH_RST="RWINH_RST" );


// Performs an assignment to either RL, an SB set, or one of the RSPs.
//
// Parameters:
//     operation: assigns to either RL, an SB set, or one of the RSPs.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_assignment(
//            self: BLEIRVisitor,
//            assignment: ASSIGNMENT) -> Any
//     2. BLEIRTransformer.transform_assignment(
//            self: BLEIRTransformer,
//            assignment: ASSIGNMENT) -> ASSIGNMENT
//     3. BLEIRListener.enter_assignment(
//            self: BLEIRListener,
//            assignment: ASSIGNMENT) -> None
//     4. BLEIRListener.exit_assignment(
//            self: BLEIRListener,
//            assignment: ASSIGNMENT) -> None
ASSIGNMENT:
    operation=( READ | WRITE | BROADCAST | RSP16_ASSIGNMENT | RSP256_ASSIGNMENT | RSP2K_ASSIGNMENT | RSP32K_ASSIGNMENT );


// Assigns the rvalue to RL.
//
// Per the READ LOGIC table of the README.md, a READ may take any of the
// following forms:
//
//     +----------------------------------|-------------------------+
//     | APL                              | BEL                     |
//     +----------------------------------|-------------------------+
//     |      immediate APL commands      | op  arg1                |
//     +----------------------------------|-------------------------+
//     |  1.  msk: RL  = 0                | :=   0                  |
//     |  2.  msk: RL  = 1                | :=   1                  |
//     +----------------------------------|-------------------------+
//     |      combining APL commands      | op  arg1   comb  arg2   |
//     +----------------------------------|-------------------------+
//     |  3.  msk: RL  =  <SB>            | :=  <SB>                |
//     |  4.  msk: RL  =  <SRC>           | :=               <SRC>  |
//     |  5.  msk: RL  =  <SB> &  <SRC>   | :=  <SB>    &    <SRC>  |
//     |                                  |                         |
//     | 10.  msk: RL |=  <SB>            | |=  <SB>                |
//     | 11.  msk: RL |=  <SRC>           | |=               <SRC>  |
//     | 12.  msk: RL |=  <SB> &  <SRC>   | |=  <SB>    &    <SRC>  |
//     |                                  |                         |
//     | 13.  msk: RL &=  <SB>            | &=  <SB>                |
//     | 14.  msk: RL &=  <SRC>           | &=               <SRC>  |
//     | 15.  msk: RL &=  <SB> &  <SRC>   | &=  <SB>    &    <SRC>  |
//     |                                  |                         |
//     | 18.  msk: RL ^=  <SB>            | ^=  <SB>                |
//     | 19.  msk: RL ^=  <SRC>           | ^=               <SRC>  |
//     | 20.  msk: RL ^=  <SB> &  <SRC>   | ^=  <SB>    &    <SRC>  |
//     +----------------------------------|-------------------------+
//     |      special cases               | op  arg1   comb  arg2   |
//     +----------------------------------|-------------------------+
//     |  6.  msk: RL  =  <SB> |  <SRC>   | :=  <SB>    |    <SRC>  |
//     |  7.  msk: RL  =  <SB> ^  <SRC>   | :=  <SB>    ^    <SRC>  |
//     |                                  |                         |
//     |  8.  msk: RL  = ~<SB> &  <SRC>   | := ~<SB>    &    <SRC>  |
//     |  9.  msk: RL  =  <SB> & ~<SRC>   | :=  <SB>    &   ~<SRC>  |
//     |                                  |                         |
//     | 16.  msk: RL &= ~<SB>            | &= ~<SB>                |
//     | 17.  msk: RL &= ~<SRC>           | &= ~<SRC>               |
//     +----------------------------------|-------------------------+
//
// In addition, the following APL commands may be supported by HW but not
// supported by APL concrete syntax because they have no dedicated
// read-control register:
//
//     21.  msk: RL = ~RL & <SRC>
//     22.  msk: RL = ~RL & <SB>
//     23.  msk: RL = ~RL & (<SB> & <SRC>)
//     24.  msk: RL &= ~<SB> | ~<SRC>
//
// Parameters:
//     operator: any ASSIGN_OP except COND_EQ.
//     rvalue: either a unary or binary expression whose evaluation should be
//             assigned to RL in the method specified by the assignment
//             operator.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_read(
//            self: BLEIRVisitor,
//            read: READ) -> Any
//     2. BLEIRTransformer.transform_read(
//            self: BLEIRTransformer,
//            read: READ) -> READ
//     3. BLEIRListener.enter_read(
//            self: BLEIRListener,
//            read: READ) -> None
//     4. BLEIRListener.exit_read(
//            self: BLEIRListener,
//            read: READ) -> None
READ:
    operator=ASSIGN_OP
    rvalue=( UNARY_EXPR | BINARY_EXPR );


// Represents the various assignment operations.
//
// 1. EQ      := direct assignment from the rvalue to the lvalue
//               (e.g. `x = y`).
// 2. AND_EQ  := assigns the lvalue the conjunction of the lvalue and the
//               rvalue (e.g. `x &= y`).
// 3. OR_EQ   := assigns the lvalue the disjunction of the lvalue and the
//               rvalue (e.g. `x |= y`).
// 4. XOR_EQ  := assigns the lvalue the exclusive disjunction of the
//               lvalue and the rvalue (1s everywhere the lvalue is 1 or 0
//               and the rvalue is its complement, and 0s everywhere the
//               lvalue is equal to the rvalue) (e.g. `x ^= y`).
// 5. COND_EQ := equivalent to OR_EQ at the logical level but potentially
//               implemented differently at the hardware level
//               (e.g. `x ?= y`).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_assign_op(
//            self: BLEIRVisitor,
//            assign_op: ASSIGN_OP) -> Any
//     2. BLEIRTransformer.transform_assign_op(
//            self: BLEIRTransformer,
//            assign_op: ASSIGN_OP) -> ASSIGN_OP
//     3. BLEIRListener.enter_assign_op(
//            self: BLEIRListener,
//            assign_op: ASSIGN_OP) -> None
//     4. BLEIRListener.exit_assign_op(
//            self: BLEIRListener,
//            assign_op: ASSIGN_OP) -> None
ASSIGN_OP:
    ( EQ="="
    | AND_EQ="&="
    | OR_EQ="|="
    | XOR_EQ="^="
    | COND_EQ="?=" );


// Type alias surrounding each of the transformable unary operands.
//
// A UNARY_EXPR takes any of the following forms:
//     1. <SB>
//     2. ~<SB>
//     3. <SRC>
//     4. ~<SRC>
//     5. 1
//     6. 0
//
// Parameters:
//     expression: transformable unary expression.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_unary_expr(
//            self: BLEIRVisitor,
//            unary_expr: UNARY_EXPR) -> Any
//     2. BLEIRTransformer.transform_unary_expr(
//            self: BLEIRTransformer,
//            unary_expr: UNARY_EXPR) -> UNARY_EXPR
//     3. BLEIRListener.enter_unary_expr(
//            self: BLEIRListener,
//            unary_expr: UNARY_EXPR) -> None
//     4. BLEIRListener.exit_unary_expr(
//            self: BLEIRListener,
//            unary_expr: UNARY_EXPR) -> None
UNARY_EXPR:
    expression=( UNARY_SB | UNARY_SRC | BIT_EXPR );


// Optionally transforms an SB_EXPR operand with some unary operator.
//
// A UNARY_SB takes either of the following forms:
//     1. <SB>
//     2. ~<SB>
//
// Parameters:
//     expression: transformable SB_EXPR operand.
//     operator: optional unary operator to transform the expression.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_unary_sb(
//            self: BLEIRVisitor,
//            unary_sb: UNARY_SB) -> Any
//     2. BLEIRTransformer.transform_unary_sb(
//            self: BLEIRTransformer,
//            unary_sb: UNARY_SB) -> UNARY_SB
//     3. BLEIRListener.enter_unary_sb(
//            self: BLEIRListener,
//            unary_sb: UNARY_SB) -> None
//     4. BLEIRListener.exit_unary_sb(
//            self: BLEIRListener,
//            unary_sb: UNARY_SB) -> None
UNARY_SB:
    expression=SB_EXPR
    ( operator=UNARY_OP )?;


// Collection of 1-to-3 RN_REGs. When used as the lvalue of a WRITE, each
// register mapped to by the RN_REGs receives the same value. When used as the
// rvalue of a READ, the registers are conjoined togeter to form a single
// rvalue operand.
//
// An SB_EXPR takes any of the following forms:
//     1. SB[x]
//     2. SB[x,y]
//     3. SB[x,y,z]
//
// - The form SB[x] is equivalent to SB[x,x,x].
// - The form SB[x,y] is equivalent to SB[x,y,y].
// - The form SB[x,y,z] is equivalent to itself.
//
// Parameters:
//    parameters: A collection of a single RE_REG or 1-to-3 RN_REGs.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_sb_expr(
//            self: BLEIRVisitor,
//            sb_expr: SB_EXPR) -> Any
//     2. BLEIRTransformer.transform_sb_expr(
//            self: BLEIRTransformer,
//            sb_expr: SB_EXPR) -> SB_EXPR
//     3. BLEIRListener.enter_sb_expr(
//            self: BLEIRListener,
//            sb_expr: SB_EXPR) -> None
//     4. BLEIRListener.exit_sb_expr(
//            self: BLEIRListener,
//            sb_expr: SB_EXPR) -> None
SB_EXPR:
    parameters=( RN_REG | RN_REG | RN_REG | RN_REG | RN_REG | RN_REG | RE_REG | EWE_REG );


// Optionally transforms a <SRC> with some unary operator.
//
// A UNARY_SRC may take any of the following forms:
//     1. <SRC>
//     2. ~<SRC>
//
// Parameters:
//     expression: recipient (operand) of the provided operator.
//     operator: performs an optional transformation on the expression.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_unary_src(
//            self: BLEIRVisitor,
//            unary_src: UNARY_SRC) -> Any
//     2. BLEIRTransformer.transform_unary_src(
//            self: BLEIRTransformer,
//            unary_src: UNARY_SRC) -> UNARY_SRC
//     3. BLEIRListener.enter_unary_src(
//            self: BLEIRListener,
//            unary_src: UNARY_SRC) -> None
//     4. BLEIRListener.exit_unary_src(
//            self: BLEIRListener,
//            unary_src: UNARY_SRC) -> None
UNARY_SRC:
    expression=SRC_EXPR
    ( operator=UNARY_OP )?;


// Available <SRC> values for READ and WRITE.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_src_expr(
//            self: BLEIRVisitor,
//            src_expr: SRC_EXPR) -> Any
//     2. BLEIRTransformer.transform_src_expr(
//            self: BLEIRTransformer,
//            src_expr: SRC_EXPR) -> SRC_EXPR
//     3. BLEIRListener.enter_src_expr(
//            self: BLEIRListener,
//            src_expr: SRC_EXPR) -> None
//     4. BLEIRListener.exit_src_expr(
//            self: BLEIRListener,
//            src_expr: SRC_EXPR) -> None
SRC_EXPR:
    ( RL="RL"
    | NRL="NRL"
    | ERL="ERL"
    | WRL="WRL"
    | SRL="SRL"
    | GL="GL"
    | GGL="GGL"
    | RSP16="RSP16"
    | INV_RL="INV_RL"
    | INV_NRL="INV_NRL"
    | INV_ERL="INV_ERL"
    | INV_WRL="INV_WRL"
    | INV_SRL="INV_SRL"
    | INV_GL="INV_GL"
    | INV_GGL="INV_GGL"
    | INV_RSP16="INV_RSP16" );


// Possible bit values for the rvalue of READ.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_bit_expr(
//            self: BLEIRVisitor,
//            bit_expr: BIT_EXPR) -> Any
//     2. BLEIRTransformer.transform_bit_expr(
//            self: BLEIRTransformer,
//            bit_expr: BIT_EXPR) -> BIT_EXPR
//     3. BLEIRListener.enter_bit_expr(
//            self: BLEIRListener,
//            bit_expr: BIT_EXPR) -> None
//     4. BLEIRListener.exit_bit_expr(
//            self: BLEIRListener,
//            bit_expr: BIT_EXPR) -> None
BIT_EXPR:
    ( ZERO="0"
    | ONE="1" );


// Represents a binary operation performed over two unary operands. The
// left_operand is always an <SB> or ~<SB> and the right_operand is always a
// <SRC> or ~<SRC>.
//
// A BINARY_EXPR takes any of the following forms:
//     1. <SB> <op> <SRC>
//     2. ~<SB> <op> <SRC>
//     3. <SB> <op> ~<SRC>
//     4. ~<SB> <op> ~<SRC>
//
// Parameters:
//     operator: binary operator to apply to the operands.
//     left_operand: appears on the left side of the operation.
//     right_operand: appears on the right side of the operation.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_binary_expr(
//            self: BLEIRVisitor,
//            binary_expr: BINARY_EXPR) -> Any
//     2. BLEIRTransformer.transform_binary_expr(
//            self: BLEIRTransformer,
//            binary_expr: BINARY_EXPR) -> BINARY_EXPR
//     3. BLEIRListener.enter_binary_expr(
//            self: BLEIRListener,
//            binary_expr: BINARY_EXPR) -> None
//     4. BLEIRListener.exit_binary_expr(
//            self: BLEIRListener,
//            binary_expr: BINARY_EXPR) -> None
BINARY_EXPR:
    operator=BINOP
    left_operand=( UNARY_SB | RL_EXPR )
    right_operand=( UNARY_SRC | L1_REG | L2_REG | LXRegWithOffsets );


// Represents the various binary operations that may be performed on rvalue
// operands.
//
//     1. AND := performs a conjunction (e.g. `x & y`).
//     2. OR  := performs a disjunction (e.g. `x | y`).
//     3. XOR := performs an exclusive disjunction (e.g. `x ^ y`).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_binop(
//            self: BLEIRVisitor,
//            binop: BINOP) -> Any
//     2. BLEIRTransformer.transform_binop(
//            self: BLEIRTransformer,
//            binop: BINOP) -> BINOP
//     3. BLEIRListener.enter_binop(
//            self: BLEIRListener,
//            binop: BINOP) -> None
//     4. BLEIRListener.exit_binop(
//            self: BLEIRListener,
//            binop: BINOP) -> None
BINOP:
    ( AND="&"
    | OR="|"
    | XOR="^" );


// Possible lvalues for READ and rvalues for BROADCAST.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rl_expr(
//            self: BLEIRVisitor,
//            rl_expr: RL_EXPR) -> Any
//     2. BLEIRTransformer.transform_rl_expr(
//            self: BLEIRTransformer,
//            rl_expr: RL_EXPR) -> RL_EXPR
//     3. BLEIRListener.enter_rl_expr(
//            self: BLEIRListener,
//            rl_expr: RL_EXPR) -> None
//     4. BLEIRListener.exit_rl_expr(
//            self: BLEIRListener,
//            rl_expr: RL_EXPR) -> None
RL_EXPR:
    ( RL="RL" );


// Represents an offset L1_REG or L2_REG. The offset may contain a bank,
// group, and row only for L1_REG. If it is for an L2_REG it may contain only
// a row offset. Each L1_REG and L2_REG value represents a coordinate to which
// the offset values are added. Offsets avoid unnecessary parameters.
//
// Parameters:
//     parameter: the L1_REG or L2_REG to which the offset is applicable.
//     row_id: an offset added to the row coordinate of the LX_REG.
//     group_id: an offset added to the group coordinate of the LX_REG.
//     bank_id: an offset added to the bank coordinate of the LX_REG.
//     comment: an optional comment describing this offset.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_lx_reg_with_offsets(
//            self: BLEIRVisitor,
//            lx_reg_with_offsets: LXRegWithOffsets) -> Any
//     2. BLEIRTransformer.transform_lx_reg_with_offsets(
//            self: BLEIRTransformer,
//            lx_reg_with_offsets: LXRegWithOffsets) -> LXRegWithOffsets
//     3. BLEIRListener.enter_lx_reg_with_offsets(
//            self: BLEIRListener,
//            lx_reg_with_offsets: LXRegWithOffsets) -> None
//     4. BLEIRListener.exit_lx_reg_with_offsets(
//            self: BLEIRListener,
//            lx_reg_with_offsets: LXRegWithOffsets) -> None
LXRegWithOffsets:
    parameter=( L1_REG | L2_REG | LXRegWithOffsets )
    row_id=int
    group_id=int
    bank_id=int
    ( comment=( MultiLineComment | SingleLineComment | TrailingComment ) )?;


// Internal wrapper to hold a forward reference.
LXRegWithOffsets: /* see relevant docs */;


// Comments that appear by themselves over multiple lines, like so:
//
//     /**
//      * this is
//      * a
//      * multi-line
//      * comment
//      */
//
// Parameters:
//     lines: sequence of strings to print, one per line.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_multi_line_comment(
//            self: BLEIRVisitor,
//            multi_line_comment: MultiLineComment) -> Any
//     2. BLEIRTransformer.transform_multi_line_comment(
//            self: BLEIRTransformer,
//            multi_line_comment: MultiLineComment) -> MultiLineComment
//     3. BLEIRListener.enter_multi_line_comment(
//            self: BLEIRListener,
//            multi_line_comment: MultiLineComment) -> None
//     4. BLEIRListener.exit_multi_line_comment(
//            self: BLEIRListener,
//            multi_line_comment: MultiLineComment) -> None
MultiLineComment:
    ( lines+=str )*;


// Comments that appear by themselves on a single-line, like so:
//
//     /* this is a single-line comment */
//
// Parameters:
//     line: string content of the comment.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_single_line_comment(
//            self: BLEIRVisitor,
//            single_line_comment: SingleLineComment) -> Any
//     2. BLEIRTransformer.transform_single_line_comment(
//            self: BLEIRTransformer,
//            single_line_comment: SingleLineComment) -> SingleLineComment
//     3. BLEIRListener.enter_single_line_comment(
//            self: BLEIRListener,
//            single_line_comment: SingleLineComment) -> None
//     4. BLEIRListener.exit_single_line_comment(
//            self: BLEIRListener,
//            single_line_comment: SingleLineComment) -> None
SingleLineComment:
    line=str;


// Comments that appear after statements, like so:
//
//     msk_rp: RL = 1;  /* assign 1 to RL guided by msk_rp */
//
// Parameters:
//     value: string content of the comment.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_trailing_comment(
//            self: BLEIRVisitor,
//            trailing_comment: TrailingComment) -> Any
//     2. BLEIRTransformer.transform_trailing_comment(
//            self: BLEIRTransformer,
//            trailing_comment: TrailingComment) -> TrailingComment
//     3. BLEIRListener.enter_trailing_comment(
//            self: BLEIRListener,
//            trailing_comment: TrailingComment) -> None
//     4. BLEIRListener.exit_trailing_comment(
//            self: BLEIRListener,
//            trailing_comment: TrailingComment) -> None
TrailingComment:
    value=str;


// Assigns a <SRC> to the given collection of vector registers.
//
// A WRITE may take any of the following assignment forms:
//     1. msk: SB[x] = <SRC>;
//     2. msk: SB[x,y] = <SRC>;
//     3. msk: SB[x,y,z] = <SRC>;
//
// A WRITE may take any of the following conditional assignment forms:
//     1. msk: SB[x] ?= <SRC>;
//     2. msk: SB[x,y] ?= <SRC>;
//     3. msk: SB[x,y,z] ?= <SRC>;
//
// Parameters:
//     operator: specifies the method of assignment (either EQ or COND_EQ).
//     lvalue: recipient collection of vector registers.
//     rvalue: expression to assign the vector registers.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_write(
//            self: BLEIRVisitor,
//            write: WRITE) -> Any
//     2. BLEIRTransformer.transform_write(
//            self: BLEIRTransformer,
//            write: WRITE) -> WRITE
//     3. BLEIRListener.enter_write(
//            self: BLEIRListener,
//            write: WRITE) -> None
//     4. BLEIRListener.exit_write(
//            self: BLEIRListener,
//            write: WRITE) -> None
WRITE:
    operator=ASSIGN_OP
    lvalue=SB_EXPR
    rvalue=UNARY_SRC;


// Performs a contraction over either the sections or plats of RL and
// assigns it to the lvalue.
//
// A BROADCAST may take any of the following forms:
//     1. msk: GL = RL;
//     2. msk: GGL = RL;
//     3. msk: RSP16 = RL;
//
// Parameters:
//     lvalue: recipient of the contracted value of RL.
//     rvalue: what to broadcast to the lvalue; note that LX regs are only
//             applicable for GGL.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_broadcast(
//            self: BLEIRVisitor,
//            broadcast: BROADCAST) -> Any
//     2. BLEIRTransformer.transform_broadcast(
//            self: BLEIRTransformer,
//            broadcast: BROADCAST) -> BROADCAST
//     3. BLEIRListener.enter_broadcast(
//            self: BLEIRListener,
//            broadcast: BROADCAST) -> None
//     4. BLEIRListener.exit_broadcast(
//            self: BLEIRListener,
//            broadcast: BROADCAST) -> None
BROADCAST:
    lvalue=BROADCAST_EXPR
    rvalue=( RL_EXPR | L1_REG | L2_REG | LXRegWithOffsets | BINARY_EXPR );


// Possible lvalues for BROADCAST.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_broadcast_expr(
//            self: BLEIRVisitor,
//            broadcast_expr: BROADCAST_EXPR) -> Any
//     2. BLEIRTransformer.transform_broadcast_expr(
//            self: BLEIRTransformer,
//            broadcast_expr: BROADCAST_EXPR) -> BROADCAST_EXPR
//     3. BLEIRListener.enter_broadcast_expr(
//            self: BLEIRListener,
//            broadcast_expr: BROADCAST_EXPR) -> None
//     4. BLEIRListener.exit_broadcast_expr(
//            self: BLEIRListener,
//            broadcast_expr: BROADCAST_EXPR) -> None
BROADCAST_EXPR:
    ( GL="GL"
    | GGL="GGL"
    | RSP16="RSP16" );


// Assigns some value to RSP16.
//
// An RSP16_ASSIGNMENT may take the following form:
//     1. RSP16 = RSP256;
//
// RSP16 may take the value of either RSP256 or RL, but an assignment of RL is
// considered a broadcast and requires a section mask. Assigning to RSP16 from
// RSP256 does not require a section mask.
//
// Parameters:
//     rvalue: an allowed rvalue to assign to RSP16 (e.g. RSP256)
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp16_assignment(
//            self: BLEIRVisitor,
//            rsp16_assignment: RSP16_ASSIGNMENT) -> Any
//     2. BLEIRTransformer.transform_rsp16_assignment(
//            self: BLEIRTransformer,
//            rsp16_assignment: RSP16_ASSIGNMENT) -> RSP16_ASSIGNMENT
//     3. BLEIRListener.enter_rsp16_assignment(
//            self: BLEIRListener,
//            rsp16_assignment: RSP16_ASSIGNMENT) -> None
//     4. BLEIRListener.exit_rsp16_assignment(
//            self: BLEIRListener,
//            rsp16_assignment: RSP16_ASSIGNMENT) -> None
RSP16_ASSIGNMENT:
    rvalue=RSP16_RVALUE;


// Possible rvalues for RSP16_ASSIGNMENT.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp16_rvalue(
//            self: BLEIRVisitor,
//            rsp16_rvalue: RSP16_RVALUE) -> Any
//     2. BLEIRTransformer.transform_rsp16_rvalue(
//            self: BLEIRTransformer,
//            rsp16_rvalue: RSP16_RVALUE) -> RSP16_RVALUE
//     3. BLEIRListener.enter_rsp16_rvalue(
//            self: BLEIRListener,
//            rsp16_rvalue: RSP16_RVALUE) -> None
//     4. BLEIRListener.exit_rsp16_rvalue(
//            self: BLEIRListener,
//            rsp16_rvalue: RSP16_RVALUE) -> None
RSP16_RVALUE:
    ( RSP256="RSP256" );


// Assigns some value to RSP256.
//
// A RSP256_ASSIGNMENT may take either of the following forms:
//     1. RSP256 = RSP16;
//     2. RSP256 = RSP2K;
//
// Parameters:
//     rvalue: an allowed rvalue to assign to RSP256 (either RSP16 or
//             RSP2K).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp256_assignment(
//            self: BLEIRVisitor,
//            rsp256_assignment: RSP256_ASSIGNMENT) -> Any
//     2. BLEIRTransformer.transform_rsp256_assignment(
//            self: BLEIRTransformer,
//            rsp256_assignment: RSP256_ASSIGNMENT) -> RSP256_ASSIGNMENT
//     3. BLEIRListener.enter_rsp256_assignment(
//            self: BLEIRListener,
//            rsp256_assignment: RSP256_ASSIGNMENT) -> None
//     4. BLEIRListener.exit_rsp256_assignment(
//            self: BLEIRListener,
//            rsp256_assignment: RSP256_ASSIGNMENT) -> None
RSP256_ASSIGNMENT:
    rvalue=RSP256_RVALUE;


// Possible rvalues for RSP256_ASSIGNMENT.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp256_rvalue(
//            self: BLEIRVisitor,
//            rsp256_rvalue: RSP256_RVALUE) -> Any
//     2. BLEIRTransformer.transform_rsp256_rvalue(
//            self: BLEIRTransformer,
//            rsp256_rvalue: RSP256_RVALUE) -> RSP256_RVALUE
//     3. BLEIRListener.enter_rsp256_rvalue(
//            self: BLEIRListener,
//            rsp256_rvalue: RSP256_RVALUE) -> None
//     4. BLEIRListener.exit_rsp256_rvalue(
//            self: BLEIRListener,
//            rsp256_rvalue: RSP256_RVALUE) -> None
RSP256_RVALUE:
    ( RSP16="RSP16"
    | RSP2K="RSP2K" );


// Assigns some value to RSP2K.
//
// An RSP2K_ASSIGNMENT may take either of the following forms:
//     1. RSP2K = RSP256;
//     2. RSP2K = RSP32K;
//
// Parameters:
//     rvalue: an allowed rvalue to assign to RSP2K (either RSP256 or
//             RSP32K).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp2k_assignment(
//            self: BLEIRVisitor,
//            rsp2k_assignment: RSP2K_ASSIGNMENT) -> Any
//     2. BLEIRTransformer.transform_rsp2k_assignment(
//            self: BLEIRTransformer,
//            rsp2k_assignment: RSP2K_ASSIGNMENT) -> RSP2K_ASSIGNMENT
//     3. BLEIRListener.enter_rsp2k_assignment(
//            self: BLEIRListener,
//            rsp2k_assignment: RSP2K_ASSIGNMENT) -> None
//     4. BLEIRListener.exit_rsp2k_assignment(
//            self: BLEIRListener,
//            rsp2k_assignment: RSP2K_ASSIGNMENT) -> None
RSP2K_ASSIGNMENT:
    rvalue=RSP2K_RVALUE;


// Possible rvalues for RSP2K_ASSIGNMENT.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp2k_rvalue(
//            self: BLEIRVisitor,
//            rsp2k_rvalue: RSP2K_RVALUE) -> Any
//     2. BLEIRTransformer.transform_rsp2k_rvalue(
//            self: BLEIRTransformer,
//            rsp2k_rvalue: RSP2K_RVALUE) -> RSP2K_RVALUE
//     3. BLEIRListener.enter_rsp2k_rvalue(
//            self: BLEIRListener,
//            rsp2k_rvalue: RSP2K_RVALUE) -> None
//     4. BLEIRListener.exit_rsp2k_rvalue(
//            self: BLEIRListener,
//            rsp2k_rvalue: RSP2K_RVALUE) -> None
RSP2K_RVALUE:
    ( RSP256="RSP256"
    | RSP32K="RSP32K" );


// Assigns some value to RSP32K.
//
// An RSP32K_ASSIGNMENT may take the following form:
//     1. RSP32K = RSP2K;
//
// Parameters:
//     rvalue: an allowed rvalue to assign to RSP32K (e.g. RSP2K).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp32k_assignment(
//            self: BLEIRVisitor,
//            rsp32k_assignment: RSP32K_ASSIGNMENT) -> Any
//     2. BLEIRTransformer.transform_rsp32k_assignment(
//            self: BLEIRTransformer,
//            rsp32k_assignment: RSP32K_ASSIGNMENT) -> RSP32K_ASSIGNMENT
//     3. BLEIRListener.enter_rsp32k_assignment(
//            self: BLEIRListener,
//            rsp32k_assignment: RSP32K_ASSIGNMENT) -> None
//     4. BLEIRListener.exit_rsp32k_assignment(
//            self: BLEIRListener,
//            rsp32k_assignment: RSP32K_ASSIGNMENT) -> None
RSP32K_ASSIGNMENT:
    rvalue=RSP32K_RVALUE;


// Possible rvalues for RSP32K_ASSIGNMENT.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp32k_rvalue(
//            self: BLEIRVisitor,
//            rsp32k_rvalue: RSP32K_RVALUE) -> Any
//     2. BLEIRTransformer.transform_rsp32k_rvalue(
//            self: BLEIRTransformer,
//            rsp32k_rvalue: RSP32K_RVALUE) -> RSP32K_RVALUE
//     3. BLEIRListener.enter_rsp32k_rvalue(
//            self: BLEIRListener,
//            rsp32k_rvalue: RSP32K_RVALUE) -> None
//     4. BLEIRListener.exit_rsp32k_rvalue(
//            self: BLEIRListener,
//            rsp32k_rvalue: RSP32K_RVALUE) -> None
RSP32K_RVALUE:
    ( RSP2K="RSP2K" );


// Statements that don't belong anywhere else.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_special(
//            self: BLEIRVisitor,
//            special: SPECIAL) -> Any
//     2. BLEIRTransformer.transform_special(
//            self: BLEIRTransformer,
//            special: SPECIAL) -> SPECIAL
//     3. BLEIRListener.enter_special(
//            self: BLEIRListener,
//            special: SPECIAL) -> None
//     4. BLEIRListener.exit_special(
//            self: BLEIRListener,
//            special: SPECIAL) -> None
SPECIAL:
    ( NOOP="NOOP"
    | RSP_END="RSP_END"
    | RSP_START_RET="RSP_START_RET" );


// Represents an assignment to GGL from an LX register. It is important to
// note that only L1_REG is supported. Assigning to GGL from L2 is untested
// and may not work. The LX_ADDR type was extracted from the APL grammar and
// may not semantically apply to both L1 and L2.
//
// Parameters:
//     rvalue: The L1_REG or L2_REG (or an offset) to assign GGL.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_ggl_assignment(
//            self: BLEIRVisitor,
//            ggl_assignment: GGL_ASSIGNMENT) -> Any
//     2. BLEIRTransformer.transform_ggl_assignment(
//            self: BLEIRTransformer,
//            ggl_assignment: GGL_ASSIGNMENT) -> GGL_ASSIGNMENT
//     3. BLEIRListener.enter_ggl_assignment(
//            self: BLEIRListener,
//            ggl_assignment: GGL_ASSIGNMENT) -> None
//     4. BLEIRListener.exit_ggl_assignment(
//            self: BLEIRListener,
//            ggl_assignment: GGL_ASSIGNMENT) -> None
GGL_ASSIGNMENT:
    rvalue=( L1_REG | L2_REG | LXRegWithOffsets );


// Assigns to LGL the memory associated with the corresponding L1_REG,
// L2_REG, or an offset of either. LGL is used to shuttle data between L1 and
// L2 registers.
//
// Parameters:
//     rvalue: Either an L1_REG, L2_REG, or offset to assign LGL.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_lgl_assignment(
//            self: BLEIRVisitor,
//            lgl_assignment: LGL_ASSIGNMENT) -> Any
//     2. BLEIRTransformer.transform_lgl_assignment(
//            self: BLEIRTransformer,
//            lgl_assignment: LGL_ASSIGNMENT) -> LGL_ASSIGNMENT
//     3. BLEIRListener.enter_lgl_assignment(
//            self: BLEIRListener,
//            lgl_assignment: LGL_ASSIGNMENT) -> None
//     4. BLEIRListener.exit_lgl_assignment(
//            self: BLEIRListener,
//            lgl_assignment: LGL_ASSIGNMENT) -> None
LGL_ASSIGNMENT:
    rvalue=( L1_REG | L2_REG | LXRegWithOffsets );


// Assigns to either an L1_REG or L2_REG the value contained within GGL or
// LGL. It is important to note that a GGL rvalue is only applicable to L1,
// but LGL may apply to either L1 or L2.
//
// Parameters:
//     lvalue: the L1 or L2 register recipient.
//     rvalue: the storage device of the data, either GGL or LGL.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_lx_assignment(
//            self: BLEIRVisitor,
//            lx_assignment: LX_ASSIGNMENT) -> Any
//     2. BLEIRTransformer.transform_lx_assignment(
//            self: BLEIRTransformer,
//            lx_assignment: LX_ASSIGNMENT) -> LX_ASSIGNMENT
//     3. BLEIRListener.enter_lx_assignment(
//            self: BLEIRListener,
//            lx_assignment: LX_ASSIGNMENT) -> None
//     4. BLEIRListener.exit_lx_assignment(
//            self: BLEIRListener,
//            lx_assignment: LX_ASSIGNMENT) -> None
LX_ASSIGNMENT:
    lvalue=( L1_REG | L2_REG | LXRegWithOffsets )
    rvalue=( GGL_EXPR | LGL_EXPR );


// Special Token type for expressions that accept only GGL.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_ggl_expr(
//            self: BLEIRVisitor,
//            ggl_expr: GGL_EXPR) -> Any
//     2. BLEIRTransformer.transform_ggl_expr(
//            self: BLEIRTransformer,
//            ggl_expr: GGL_EXPR) -> GGL_EXPR
//     3. BLEIRListener.enter_ggl_expr(
//            self: BLEIRListener,
//            ggl_expr: GGL_EXPR) -> None
//     4. BLEIRListener.exit_ggl_expr(
//            self: BLEIRListener,
//            ggl_expr: GGL_EXPR) -> None
GGL_EXPR:
    ( GGL="GGL" );


// Special Token type for expressions that accept only LGL.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_lgl_expr(
//            self: BLEIRVisitor,
//            lgl_expr: LGL_EXPR) -> Any
//     2. BLEIRTransformer.transform_lgl_expr(
//            self: BLEIRTransformer,
//            lgl_expr: LGL_EXPR) -> LGL_EXPR
//     3. BLEIRListener.enter_lgl_expr(
//            self: BLEIRListener,
//            lgl_expr: LGL_EXPR) -> None
//     4. BLEIRListener.exit_lgl_expr(
//            self: BLEIRListener,
//            lgl_expr: LGL_EXPR) -> None
LGL_EXPR:
    ( LGL="LGL" );


// GlassStatement(subject, sections, plats, fmt, order)
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_glass_statement(
//            self: BLEIRVisitor,
//            glass_statement: GlassStatement) -> Any
//     2. BLEIRTransformer.transform_glass_statement(
//            self: BLEIRTransformer,
//            glass_statement: GlassStatement) -> GlassStatement
//     3. BLEIRListener.enter_glass_statement(
//            self: BLEIRListener,
//            glass_statement: GlassStatement) -> None
//     4. BLEIRListener.exit_glass_statement(
//            self: BLEIRListener,
//            glass_statement: GlassStatement) -> None
GlassStatement:
    subject=( EWE_REG | L1_REG | L2_REG | LGL_EXPR | LXRegWithOffsets | RE_REG | RN_REG | RSP256_EXPR | RSP2K_EXPR | RSP32K_EXPR | SRC_EXPR )
    ( sections+=int )*
    ( plats+=int )*
    fmt=GlassFormat
    order=GlassOrder;


// Possible lvalues for RSP256_ASSIGNMENT.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp256_expr(
//            self: BLEIRVisitor,
//            rsp256_expr: RSP256_EXPR) -> Any
//     2. BLEIRTransformer.transform_rsp256_expr(
//            self: BLEIRTransformer,
//            rsp256_expr: RSP256_EXPR) -> RSP256_EXPR
//     3. BLEIRListener.enter_rsp256_expr(
//            self: BLEIRListener,
//            rsp256_expr: RSP256_EXPR) -> None
//     4. BLEIRListener.exit_rsp256_expr(
//            self: BLEIRListener,
//            rsp256_expr: RSP256_EXPR) -> None
RSP256_EXPR:
    ( RSP256="RSP256" );


// Possible lvalues for RSP2K_ASSIGNMENT.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp2k_expr(
//            self: BLEIRVisitor,
//            rsp2k_expr: RSP2K_EXPR) -> Any
//     2. BLEIRTransformer.transform_rsp2k_expr(
//            self: BLEIRTransformer,
//            rsp2k_expr: RSP2K_EXPR) -> RSP2K_EXPR
//     3. BLEIRListener.enter_rsp2k_expr(
//            self: BLEIRListener,
//            rsp2k_expr: RSP2K_EXPR) -> None
//     4. BLEIRListener.exit_rsp2k_expr(
//            self: BLEIRListener,
//            rsp2k_expr: RSP2K_EXPR) -> None
RSP2K_EXPR:
    ( RSP2K="RSP2K" );


// Possible lvalues for RSP32K_ASSIGNMENT.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_rsp32k_expr(
//            self: BLEIRVisitor,
//            rsp32k_expr: RSP32K_EXPR) -> Any
//     2. BLEIRTransformer.transform_rsp32k_expr(
//            self: BLEIRTransformer,
//            rsp32k_expr: RSP32K_EXPR) -> RSP32K_EXPR
//     3. BLEIRListener.enter_rsp32k_expr(
//            self: BLEIRListener,
//            rsp32k_expr: RSP32K_EXPR) -> None
//     4. BLEIRListener.exit_rsp32k_expr(
//            self: BLEIRListener,
//            rsp32k_expr: RSP32K_EXPR) -> None
RSP32K_EXPR:
    ( RSP32K="RSP32K" );


// An enumeration.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_glass_format(
//            self: BLEIRVisitor,
//            glass_format: GlassFormat) -> Any
//     2. BLEIRTransformer.transform_glass_format(
//            self: BLEIRTransformer,
//            glass_format: GlassFormat) -> GlassFormat
//     3. BLEIRListener.enter_glass_format(
//            self: BLEIRListener,
//            glass_format: GlassFormat) -> None
//     4. BLEIRListener.exit_glass_format(
//            self: BLEIRListener,
//            glass_format: GlassFormat) -> None
GlassFormat:
    ( BIN="bin"
    | HEX="hex" );


// An enumeration.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_glass_order(
//            self: BLEIRVisitor,
//            glass_order: GlassOrder) -> Any
//     2. BLEIRTransformer.transform_glass_order(
//            self: BLEIRTransformer,
//            glass_order: GlassOrder) -> GlassOrder
//     3. BLEIRListener.enter_glass_order(
//            self: BLEIRListener,
//            glass_order: GlassOrder) -> None
//     4. BLEIRListener.exit_glass_order(
//            self: BLEIRListener,
//            glass_order: GlassOrder) -> None
GlassOrder:
    ( LEAST_SIGNIFICANT_BIT_FIRST="lsb"
    | MOST_SIGNIFICANT_BIT_FIRST="msb" );


// Metadata associated explicitly with Fragment instances.
// 1. ORIGINAL_IDENTIFIER := the name of the original fragment, such as that
//                           before it is partitioned or has its name
//                           obfuscated.
// 2. IS_LOW_LEVEL := whether the Fragment is defined by BLECCI instead of
//                 BELEX.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_fragment_metadata(
//            self: BLEIRVisitor,
//            fragment_metadata: FragmentMetadata) -> Any
//     2. BLEIRTransformer.transform_fragment_metadata(
//            self: BLEIRTransformer,
//            fragment_metadata: FragmentMetadata) -> FragmentMetadata
//     3. BLEIRListener.enter_fragment_metadata(
//            self: BLEIRListener,
//            fragment_metadata: FragmentMetadata) -> None
//     4. BLEIRListener.exit_fragment_metadata(
//            self: BLEIRListener,
//            fragment_metadata: FragmentMetadata) -> None
FragmentMetadata:
    ( ORIGINAL_IDENTIFIER="ORIGINAL_IDENTIFIER"
    | IS_LOW_LEVEL="IS_LOW_LEVEL" );


// Special type indicating an unconstrained type.
//
// - Any is compatible with every type.
// - Any assumed to have all methods.
// - All values assumed to be instances of Any.
//
// Note that all the above statements are true from the point of view of
// static type checkers. At runtime, Any should not be used with instance
// or class checks.
//
Any: /* see relevant docs */;


// Internal wrapper to hold a forward reference.
Fragment: /* see relevant docs */;


// Represents a register parameter that has been assigned a specific
// register, e.g. RN_REG_0 or SM_REG_11.
//
// Parameters:
//     parameter: RN_REG or SM_REG being assigned a register.
//     register: name of the specific register being assigned.
//     comment: optional comment to associate with this AllocatedRegister.
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_allocated_register(
//            self: BLEIRVisitor,
//            allocated_register: AllocatedRegister) -> Any
//     2. BLEIRTransformer.transform_allocated_register(
//            self: BLEIRTransformer,
//            allocated_register: AllocatedRegister) -> AllocatedRegister
//     3. BLEIRListener.enter_allocated_register(
//            self: BLEIRListener,
//            allocated_register: AllocatedRegister) -> None
//     4. BLEIRListener.exit_allocated_register(
//            self: BLEIRListener,
//            allocated_register: AllocatedRegister) -> None
AllocatedRegister:
    parameter=( RN_REG | RE_REG | EWE_REG | L1_REG | L2_REG | SM_REG )
    register=str
    ( comment=( MultiLineComment | SingleLineComment | TrailingComment ) )?;


// Metadata attributes associated with fragment callers.
// 1. BUILD_EXAMPLES := a function that returns a sequence of Examples
// 2. REGISTER_MAP := a mapping of parameter ids to register indices
// 3. ARGS_BY_REG_NYM := mapping over sequences of params having the same
//                       registers; if there is more than one param
//                       associated with a register an a copy operation is
//                       implied (high-level BELEX)
// 4. OUT_PARAM := identifier of the return parameter (high-level BELEX)
// 5. IS_HIGH_LEVEL := specifies that the fragment was built with
//                           BELEX
// 6. IS_LOW_LEVEL := specifies that the fragment was built with BLECCI
// 7. SHOULD_FAIL := indicates that the generated unit test should be a
//                   negative test (i.e. code failures imply test
//                   successes).
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_caller_metadata(
//            self: BLEIRVisitor,
//            caller_metadata: CallerMetadata) -> Any
//     2. BLEIRTransformer.transform_caller_metadata(
//            self: BLEIRTransformer,
//            caller_metadata: CallerMetadata) -> CallerMetadata
//     3. BLEIRListener.enter_caller_metadata(
//            self: BLEIRListener,
//            caller_metadata: CallerMetadata) -> None
//     4. BLEIRListener.exit_caller_metadata(
//            self: BLEIRListener,
//            caller_metadata: CallerMetadata) -> None
CallerMetadata:
    ( BUILD_EXAMPLES="BUILD_EXAMPLES"
    | REGISTER_MAP="REGISTER_MAP"
    | ARGS_BY_REG_NYM="ARGS_BY_REG_NYM"
    | OUT_PARAM="OUT_PARAM"
    | IS_HIGH_LEVEL="IS_HIGH_LEVEL"
    | IS_LOW_LEVEL="IS_LOW_LEVEL"
    | SHOULD_FAIL="SHOULD_FAIL" );


// Metadata attributes to attach to function calls.
//
// 1. IS_INITIALIZER := specifies whether the function call should be
//                   executed before anything else, even before the
//                   parameters are written to their corresponding vector
//                   registers.
// 2. IS_HIGH_LEVEL := specifies the fragment was defined by high-level BELEX
// 3. IS_LOW_LEVEL := specifies the fragment was defined by low-level BELEX
//                 or BLECCI
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_call_metadata(
//            self: BLEIRVisitor,
//            call_metadata: CallMetadata) -> Any
//     2. BLEIRTransformer.transform_call_metadata(
//            self: BLEIRTransformer,
//            call_metadata: CallMetadata) -> CallMetadata
//     3. BLEIRListener.enter_call_metadata(
//            self: BLEIRListener,
//            call_metadata: CallMetadata) -> None
//     4. BLEIRListener.exit_call_metadata(
//            self: BLEIRListener,
//            call_metadata: CallMetadata) -> None
CallMetadata:
    ( IS_INITIALIZER="IS_INITIALIZER"
    | IS_HIGH_LEVEL="IS_HIGH_LEVEL"
    | IS_LOW_LEVEL="IS_LOW_LEVEL" );


// Meta-information about Snippet instances, including:
// 1. APL_FUNCS_HEADER := name of the header of the APL file
// 2. APL_FUNCS_SOURCE := name of the source of the APL file
//
// Relevant bleir.walkables methods:
//     1. BLEIRVisitor.visit_snippet_metadata(
//            self: BLEIRVisitor,
//            snippet_metadata: SnippetMetadata) -> Any
//     2. BLEIRTransformer.transform_snippet_metadata(
//            self: BLEIRTransformer,
//            snippet_metadata: SnippetMetadata) -> SnippetMetadata
//     3. BLEIRListener.enter_snippet_metadata(
//            self: BLEIRListener,
//            snippet_metadata: SnippetMetadata) -> None
//     4. BLEIRListener.exit_snippet_metadata(
//            self: BLEIRListener,
//            snippet_metadata: SnippetMetadata) -> None
SnippetMetadata:
    ( APL_FUNCS_HEADER="APL_FUNCS_HEADER"
    | APL_FUNCS_SOURCE="APL_FUNCS_SOURCE" );
```
