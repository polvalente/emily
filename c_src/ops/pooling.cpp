// Window / pooling reductions and scatters.
//
// MLX exposes no direct window_sum/max/min/product primitives. We
// compose each op as pad -> as_strided (sliding-window view) -> reduce,
// mirroring MLX's own nn/layers/pooling.py but generalised to N-D (Nx
// passes a per-axis window shape, strides, padding, and dilation).
//
// window_scatter_max/min are on the MaxPool backward path (Nx rewrites
// grad(window_max) into window_scatter_max), so we lift them here too
// rather than leave the backward pass on via_binary. The scatter path:
// pad -> as_strided -> flatten kernel axes -> argmax-with-tie-break
// (last-occurrence per Nx semantics) -> decompose flat arg into
// per-axis kernel indices -> build absolute padded-coord indices ->
// scatter_add source values into a full(init_value) tensor -> slice
// back to the unpadded shape.

#include "../emily/async.hpp"
#include "../emily/op_cores.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::wrap;
using emily::WorkerThread;

namespace {

// The window cores (do_pad / sliding_windows_view / window_reduce_core)
// live in emily/op_cores.hpp so the eager NIFs below and the Expr-compiler
// program replay (c_src/program.cpp) share one implementation and can't
// numerically drift. window_scatter_impl below still composes do_pad +
// sliding_windows_view from there.
using emily::ops::do_pad;
using emily::ops::sliding_windows_view;
using emily::ops::window_reduce_core;
using emily::ops::WindowReduceKind;

// -------------------- Reductions --------------------

#define EMILY_WINDOW_REDUCE(op_name, kind)                                    \
  fine::Term op_name##_nif(                                                    \
      ErlNifEnv *env,                                                          \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> t,                                             \
      std::vector<int64_t> window_shape,                                       \
      std::vector<int64_t> strides,                                            \
      std::vector<int64_t> pad_lo,                                             \
      std::vector<int64_t> pad_hi,                                             \
      std::vector<int64_t> dilations,                                          \
      fine::ResourcePtr<Tensor> init_value) {                                  \
    return async_encoded(env, w,                                               \
        [t = std::move(t), window_shape = std::move(window_shape),             \
         strides = std::move(strides), pad_lo = std::move(pad_lo),             \
         pad_hi = std::move(pad_hi), dilations = std::move(dilations),         \
         init_value = std::move(init_value)](mx::Stream &s) {                  \
          return wrap(window_reduce_core(                                      \
              t->array, window_shape, strides, pad_lo, pad_hi, dilations,      \
              init_value->array, kind, s));                                    \
        });                                                                    \
  }                                                                            \
  FINE_NIF(op_name##_nif, 0);

EMILY_WINDOW_REDUCE(window_sum,     WindowReduceKind::Sum)
EMILY_WINDOW_REDUCE(window_max,     WindowReduceKind::Max)
EMILY_WINDOW_REDUCE(window_min,     WindowReduceKind::Min)
EMILY_WINDOW_REDUCE(window_product, WindowReduceKind::Product)

#undef EMILY_WINDOW_REDUCE

// -------------------- Scatter variants --------------------

// Shared body: select-and-scatter. `is_max` picks between argmax
// (for window_scatter_max) and argmin (for window_scatter_min).
//
// Tie-break semantics: Nx uses `>=` / `<=` in its select_and_scatter,
// i.e. the LAST-occurrence winner. MLX's argmax/argmin return
// FIRST-occurrence. We build `mask_pos = (flat_view == selector) *
// arange(K)` and argmax that — for tied positions the pos multiplier
// makes the later index strictly larger, giving last-occurrence.
mx::array window_scatter_impl(
    const mx::array &tensor,
    const mx::array &source,
    const mx::array &init_value,
    const std::vector<int64_t> &window_shape,
    const std::vector<int64_t> &strides,
    const std::vector<int64_t> &pad_lo,
    const std::vector<int64_t> &pad_hi,
    bool is_max,
    mx::Stream &s) {
  int rank = static_cast<int>(window_shape.size());
  auto original_shape = tensor.shape();

  // 1. Pad input with init_value.
  auto padded = do_pad(tensor, pad_lo, pad_hi, init_value, s);
  auto padded_shape = padded.shape();

  // 2. Sliding-window view. Scatter variants don't take dilations in
  //    Nx's API, so dilation is implicitly 1 per axis.
  std::vector<int64_t> dilations(rank, 1);
  std::vector<int64_t> out_dims;
  auto view =
      sliding_windows_view(padded, window_shape, strides, dilations, out_dims, s);

  // 3. Flatten the kernel axes so we can argmax across the whole window
  //    in a single reduction.
  int64_t K = 1;
  for (int i = 0; i < rank; ++i)
    K *= window_shape[i];

  mx::Shape flat_view_shape;
  flat_view_shape.reserve(rank + 1);
  for (int i = 0; i < rank; ++i)
    flat_view_shape.push_back(static_cast<mx::ShapeElem>(out_dims[i]));
  flat_view_shape.push_back(static_cast<mx::ShapeElem>(K));

  auto flat_view = mx::reshape(view, flat_view_shape, s);
  int last_axis = rank;

  // 4. Argmax-with-tie-break. `selector` is the per-window max/min; mask
  //    is 1 where the kernel element equals the winner; mask*pos gives
  //    later-matching positions a higher value, so argmax picks the
  //    last-occurrence kernel index.
  auto selector = is_max
                      ? mx::max(flat_view, last_axis, /*keepdims=*/true, s)
                      : mx::min(flat_view, last_axis, /*keepdims=*/true, s);
  auto mask = mx::equal(flat_view, selector, s);

  auto pos_1d = mx::arange(0.0, static_cast<double>(K), 1.0, mx::int32, s);
  mx::Shape pos_shape(rank + 1, 1);
  pos_shape[rank] = static_cast<mx::ShapeElem>(K);
  auto pos = mx::reshape(pos_1d, pos_shape, s);

  auto mask_i = mx::astype(mask, mx::int32, s);
  auto mask_pos = mx::multiply(mask_i, pos, s);
  auto last_arg = mx::argmax(mask_pos, last_axis, /*keepdims=*/false, s);

  // 5. Decompose flat kernel index into per-axis kernel indices.
  //    k_idx[R-1] = last_arg % window[R-1];
  //    k_idx[R-2] = (last_arg / window[R-1]) % window[R-2]; ...
  std::vector<mx::array> k_idx;
  k_idx.reserve(rank);
  for (int i = 0; i < rank; ++i)
    k_idx.push_back(last_arg); // placeholder; overwritten below

  mx::array remaining = last_arg;
  for (int i = rank - 1; i >= 0; --i) {
    auto w_i = mx::array(static_cast<int32_t>(window_shape[i]), mx::int32);
    k_idx[i] = mx::remainder(remaining, w_i, s);
    if (i > 0) {
      remaining = mx::floor_divide(remaining, w_i, s);
    }
  }

  // 6. Per-axis absolute indices into the padded tensor:
  //    abs_idx[i][out_coord] = out_coord[i] * stride[i] + k_idx[i][out_coord]
  mx::Shape out_shape_s;
  out_shape_s.reserve(rank);
  for (int i = 0; i < rank; ++i)
    out_shape_s.push_back(static_cast<mx::ShapeElem>(out_dims[i]));

  std::vector<mx::array> abs_indices;
  abs_indices.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    auto base_i =
        mx::arange(0.0, static_cast<double>(out_dims[i]), 1.0, mx::int32, s);
    mx::Shape bcast(rank, 1);
    bcast[i] = static_cast<mx::ShapeElem>(out_dims[i]);
    base_i = mx::reshape(base_i, bcast, s);
    auto stride_i = mx::array(static_cast<int32_t>(strides[i]), mx::int32);
    auto base_times = mx::multiply(base_i, stride_i, s);
    auto bt = mx::broadcast_to(base_times, out_shape_s, s);
    abs_indices.push_back(mx::add(bt, k_idx[i], s));
  }

  // 7. Reshape source to IDX_SHAPE + [1]*rank so MLX scatter_add treats
  //    each index tuple as a single-point write.
  mx::Shape source_reshape;
  source_reshape.reserve(2 * rank);
  for (int i = 0; i < rank; ++i)
    source_reshape.push_back(static_cast<mx::ShapeElem>(out_dims[i]));
  for (int i = 0; i < rank; ++i)
    source_reshape.push_back(1);
  auto source_r = mx::reshape(source, source_reshape, s);
  source_r = mx::astype(source_r, tensor.dtype(), s);

  // 8. Output buffer starts filled with init_value. Matches Nx's
  //    select_and_scatter: unselected positions retain init_value;
  //    selected positions receive init_value + sum(source values).
  auto padded_out = mx::full(padded_shape, init_value, tensor.dtype(), s);

  // 9. Scatter-add all selected contributions in one dispatch.
  std::vector<int> axes(rank);
  std::iota(axes.begin(), axes.end(), 0);
  auto scattered = mx::scatter_add(padded_out, abs_indices, source_r, axes, s);

  // 10. Slice back to original (unpadded) shape.
  mx::Shape slice_start, slice_stop, slice_strides_v;
  slice_start.reserve(rank);
  slice_stop.reserve(rank);
  slice_strides_v.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    slice_start.push_back(static_cast<mx::ShapeElem>(pad_lo[i]));
    slice_stop.push_back(
        static_cast<mx::ShapeElem>(pad_lo[i] + original_shape[i]));
    slice_strides_v.push_back(1);
  }
  return mx::slice(scattered, slice_start, slice_stop, slice_strides_v, s);
}

fine::Term window_scatter_max_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> t,
    fine::ResourcePtr<Tensor> source,
    fine::ResourcePtr<Tensor> init_value,
    std::vector<int64_t> window_shape,
    std::vector<int64_t> strides,
    std::vector<int64_t> pad_lo,
    std::vector<int64_t> pad_hi) {
  return async_encoded(env, w,
      [t = std::move(t), source = std::move(source),
       init_value = std::move(init_value),
       window_shape = std::move(window_shape),
       strides = std::move(strides), pad_lo = std::move(pad_lo),
       pad_hi = std::move(pad_hi)](mx::Stream &s) {
        return wrap(window_scatter_impl(
            t->array, source->array, init_value->array, window_shape, strides,
            pad_lo, pad_hi, /*is_max=*/true, s));
      });
}
FINE_NIF(window_scatter_max_nif, 0);

fine::Term window_scatter_min_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> t,
    fine::ResourcePtr<Tensor> source,
    fine::ResourcePtr<Tensor> init_value,
    std::vector<int64_t> window_shape,
    std::vector<int64_t> strides,
    std::vector<int64_t> pad_lo,
    std::vector<int64_t> pad_hi) {
  return async_encoded(env, w,
      [t = std::move(t), source = std::move(source),
       init_value = std::move(init_value),
       window_shape = std::move(window_shape),
       strides = std::move(strides), pad_lo = std::move(pad_lo),
       pad_hi = std::move(pad_hi)](mx::Stream &s) {
        return wrap(window_scatter_impl(
            t->array, source->array, init_value->array, window_shape, strides,
            pad_lo, pad_hi, /*is_max=*/false, s));
      });
}
FINE_NIF(window_scatter_min_nif, 0);

} // namespace
