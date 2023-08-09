# Common Belex Types

## Integer

A Belex `Integer` includes Python's built-in `int` type as well as the Numpy
(`np`) integer types. This is to make variables compatible with both Python and
Numpy integer types, especially those sensitive to the variable type.

All integer types represented by `Integer` follow:

* `int` (Python built-in)
* `np.int8`
* `np.int16`
* `np.int32`
* `np.int64`
* `np.uint8`
* `np.uint16`
* `np.uint32`
* `np.uint64`

## Indices

A Belex `Indices` represents a sequence of index elements. Examples of indices
may include sections or plats of MMB data types (`RL`, `GL`, `RSP16`, etc.).

Supported `Indices` values follow:

* `str` is either a sequence of one or more single-char hex literals each
  representing a specific section in the range `0` through `F`, or a hex literal
  beginning with `0x` in the range `0x0000` through `0xFFFF` representing a
  section mask, where each bit is a 0-indexed section. If the string is not
  prefixed with `0x`, each digit is assumed to represent an independent section.
  For example, `"024CD"` will be parsed as `[0, 2, 4, 12, 14]`.
* `Integer` is any integer in the respective range (e.g. `0` through `15` for VR
  sections or `0` through `32767` for VR plats). If this describes a section, it
  is a specific section and not a section mask. Note that we have no special
  logic for Python integer literals, so `0x3` (int literal) is parsed with the
  Python parser and fed into Belex as the integer `3`, not the hex literal
  `0x3`. For this reason, we support hex literal passed as strings so we may
  perform the parsing.
* `Sequence[Integer]` is any (Python) sequence type of explicit `Integer`s. A
  sequence type may be, for instance, a `list`, `tuple`, Numpy array, or custom
  type (implements `__len__` and `__getitem__`). Duplicates and unordered
  sequences are supported, but are discouraged. An empty sequence will be
  expanded to all values in the respective range; this behavior was required for
  high-level Belex' accessors, of which `x()` implied all sections of variable
  `x`.
* `Sequence[bool]` is any (Python) sequence of `bool`s specifying explicitly
  which indices to select. The sequence must be the same length as the number of
  indices in the target, such as `16` for sections or `32k` for VR plats. If you
  choose to use a boolean map for plats, until we support ellipses I would
  recommend constructing a numpy array of True or False values and
  setting/unsetting specific plats as needed.
* `Sequence[str]` is any (Python) sequence of `str`s specifying hex values of
  section numbers (not section masks). Each element may be in the range `0`
  through `F`. This type is only really useful for sections, not plats. An
  example follows: `["0", "2", "4", "C", "E"]` will be parsed as
  `[0, 2, 4, 12, 14]`, which is equivalent to the string literal `"024CE"`.
* `Sequence[Union[Integer, str]]` is any (Python) sequence of mixed element
  types matching the above descriptions for `Sequence[Integer]` and
  `Sequence[str]`.
* `Set[Union[Integer, str]]` is any (Python) set of mixed index types
  serving the same purpose as `Sequence[Union[Integer, str]]`.
* `Set[bool]` is any (Python) `set` of bools types serving the same purpose
  as `Sequence[bool]`.
* `Iterator[Union[Integer, str]]` is any (Python) iterator or generator of
  mixed index types serving the same purpose as `Sequence[Union[Integer, str]]`.
* `Iterator[bool]` is any (Python) iterator or generator of
  mixed index types serving the same purpose as `Sequence[bool]`.
* `Dict[Union[Integer, str], bool]` is any (Python) dictionary that maps
  specific sections to whether to include them. Unlike a `Sequence[bool]`, the
  dictionary does not need to cover every index in the respective range, but may
  if desired. The keys must represent specific sections and not masks.
* `range` is any range of indices in the respective bounds, e.g. `range(0, 4)`
  for the first `4` sections of a VR.
* `slice` is any slice of indices in the respective range. A `slice` follows the
  same semantics as `slice` literals within square bracket accessors of lists.
  For example, `slice(0, 4)` is equivalent to `0:4`; `slice(None, 8)` is
  equivalent to `:8`; `slice(4)` is equivalent to `4:`; `slice(0, 16, 4)` is
  equivalent to `0:16:4`, which is the section mask `0x1111`; `slice(None, None, None)`
  is equivalent to `::`, which covers all values.
