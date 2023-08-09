# Belex Temporary and Immediate Types

## Belex.VR

A `Belex.VR(initial_value: Optional[int])` is a VR temporary that may be used to
store intermediate computations within a `@belex_apl` fragment. It accepts an
optional `initial_value` parameter that specifies a constant value in the range
`0x0000` through `0xFFFF`. The initial value is written to every plat in the VR
immediately prior to its first use in the fragment.

## Belex.Mask

A `Belex.Mask(constant_value: Indices)` is a section mask immediate that may be
used in place of a user parameter for section masks that do not change. It
requires a `constant_value` to be specified, which may be any value of type
[`Indices`](./types.md#Indices). Square bracket accessors for `VR`, `RL`, etc.
accept such immediates, as well as immediate literals which are any value of type
[`Indices`](./types.md#Indices) -- the same values that would be provided to the
`constant_value` parameter of the `Belex.Mask` constructor.

## Belex.Section

A `Belex.Section(constant_value: Integer)` is a section immediate, much like a
`Belex.Mask` except that it covers exactly one section. It has a required
`constant_value` parameter, which specifies which section it covers.
