# Belex Glass

Belex provides a utility, `Belex.glass`, for printing internal values of APU
types (subjects). Valid subjects include:

* `VR`
* `L1` (with and without an offset)
* `L2` (with and without an offset)
* `RL`
* `INV_RL`
* `NRL`
* `INV_NRL`
* `ERL`
* `INV_ERL`
* `WRL`
* `INV_WRL`
* `SRL`
* `INV_SRL`
* `GL`
* `INV_GL`
* `GGL`
* `INV_GGL`
* `RSP16`
* `INV_RSP16`
* `RSP256`
* `RSP2K`
* `RSP32K`
* `LGL`

`Belex.glass` accepts the following parameters:

* `subject: Glassible` is an instance of any subject described above.
* `comment: Optional[str]` is an optional string comment to print with the glass statement.
* `plats: Optional[Indices]` is an optional parameter that specifies which plats
  (e.g. columns) you want to print. If you do not specify this value, all the
  subject's plats will be printed. Any value of type
  [`Indices`](./types.md#Indices) is supported.
* `sections: Optional[Indices]` is an optional parameter that specifies which
  sections (e.g. rows) you want to print. If you do not specify this value, all
  the subject's sections will be printed. Any value of type
  [`Indices`](./types.md#Indices) is supported. Note that unmasked sections will
  be included as zero sections during formatting.
* `fmt: Optinoal[str]` specifies the output format of the subject. `fmt="hex"` specifies
  that the subject should be printed in hexadecimal form, where each nibble (4
  consecutive sections beginning at section 0) is combined to form a hex char.
  `fmt="bin"` specifies that the subject should be printed in binary form, where
  each section gets its own row. If you do not specify this parameter, the
  subject will be printed in hexadecimal form.
* `order: Optional[str]` specifies how the sections should be sorted for
  formatting. `order="msb"` specifies that the most-significant bit should
  appear at the top; this is the default order for `fmt="hex"`. `order="lsb"`
  specifies that the least-significant bit should appear at the top; this is the
  default order for `fmt="bin"`.
* `baloon: bool` specifies whether the subject should be "balooned" to be the
  same shape as a VR. Balooning consists of repeating sections and plats an
  equal number of times (per dimension) until the subject has the same shape as
  a VR. Balooning a subject makes it easier to understand how operations
  involving it and VRs relate. For example, `RL[::] <= x() & GL()` performs a
  conjunction between VR `x` and `GL` (even though `GL` consists of a single
  section) by taking the conjunction between each section of `x` and `GL` and
  reading it into the corresponding section of `RL`. When you baloon `GL`, it
  makes this operation easy to understand. The default behavior is to baloon
  subjects.

Subject views (calls to `Belex.glass`) may be captured for testing by passing a
`deque` or `list` in the parameter list to the `@belex_apl` fragment in the
optional `captured_glass` kwarg. For example:

```python
@belex_apl
def frag_w_glass(Belex, ...):
    ...
    view = Belex.glass(RL, plats=32)  # view all sections and the first 32 plats of RL
    Belex.assert_true(view == ...)  # non-capturing assertion of a glass statement
    # ^^^ If you use the non-capturing method, be aware that the glass statement
    # may change with respect to parameters. Also, you should wrap your assertion
    # within Belex.assert_true(...) because Belex.glass(...) will only generate
    # output within an interactive debugging session.  Belex.assert_true(...) is
    # context-aware and will only perform the assertion during an interactive
    # debugging session.
    ...
    
@parameterized_belex_test
def test_frag_w_glass(diri: DIRI):
    ...
    captured_glass = deque()
    frag_w_glass(..., captured_glass=captured_glass)
    assert captured_glass.popleft() == ...
    ...
```
