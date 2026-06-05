// Pure op cores: the `mx::array`-building expression for each Emily op,
// free of NIF / async / enif plumbing. One source of truth per op,
// called from BOTH the eager per-op NIF (c_src/ops/*.cpp) and the
// Expr-compiler program replay (c_src/program.cpp). Keeping the core
// here means the compiled (single-NIF) path can never numerically drift
// from the eager path — they invoke the same function.
//
// CM0 ships only `add` (the prototype op). CM1 fills in the full
// primitive set as each c_src/ops/*.cpp op is split into core + thin NIF.

#pragma once

#include <mlx/mlx.h>

#include <stdexcept>
#include <string>

namespace emily::ops {

namespace mx = mlx::core;

// --- Binary elementwise ---

inline mx::array add_core(const mx::array &a, const mx::array &b,
                          mx::Stream &s) {
  return mx::add(a, b, s);
}

// --- Shape ---

// Reverse `a` along `axis` via a negative-stride slice. `axis` may be
// negative (normalized against the rank); a scalar (ndim 0) is returned
// unchanged. The slice bounds are built as mx::Shape directly because the
// stop sentinel (`-dim-1`) is negative, which the shape-validation helper
// (to_mlx_shape) rejects by design.
inline mx::array flip_core(const mx::array &a, int64_t axis, mx::Stream &s) {
  const auto &shape = a.shape();
  const int ndim = static_cast<int>(shape.size());
  if (ndim == 0) {
    return a;
  }
  int ax = static_cast<int>(axis);
  if (ax < 0) {
    ax += ndim;
  }
  if (ax < 0 || ax >= ndim) {
    throw std::invalid_argument("[flip] axis " + std::to_string(axis) +
                                " out of range for ndim " +
                                std::to_string(ndim));
  }
  mx::Shape starts(ndim, 0);
  mx::Shape stops(shape.begin(), shape.end());
  mx::Shape strides(ndim, 1);
  starts[ax] = shape[ax] - 1;
  stops[ax] = -shape[ax] - 1;
  strides[ax] = -1;
  return mx::slice(a, std::move(starts), std::move(stops), std::move(strides),
                   s);
}

} // namespace emily::ops
