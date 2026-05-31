### Security

- `Emily.Native.conv_general/8` now rejects a non-positive `groups`
  argument with `ArgumentError` instead of crashing the BEAM VM. MLX's
  convolution checks compute `in_channels % groups`, so `groups <= 0`
  (or a large value that narrows to zero through the `int64 → int`
  conversion) was an integer modulo-by-zero — a SIGFPE that bypassed the
  NIF's exception path and terminated the entire node. The guard
  validates the un-narrowed value at the NIF boundary.
