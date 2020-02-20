// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock, Igor Gudich
#include "scipp/core/apply.h"
#include "scipp/core/except.h"
#include "scipp/core/variable.h"
#include "scipp/units/except.h"

namespace scipp::core {

bool isBinEdge(const Dim dim, Dimensions edges, const Dimensions &toMatch) {
  edges.resize(dim, edges[dim] - 1);
  return edges[dim] == toMatch[dim];
}

bool is1D(Dimensions edges) { return edges.shape().size() == 1; }

template <class T> class VariableConceptT;
template <class DataType, class OldCoordType, class NewCoordType>
// Special rebin version for rebinning inner dimension to a joint new coord.
static void rebinInner(const Dim dim, const VariableConceptT<DataType> &oldT,
                       VariableConceptT<DataType> &newT,
                       const VariableConceptT<OldCoordType> &oldCoordT,
                       const VariableConceptT<NewCoordType> &newCoordT,
                       bool variances = false) {
  scipp::span<const DataType> oldData;
  scipp::span<DataType> newData;
  if (variances) {
    oldData = oldT.variances();
    newData = newT.variances();
  } else {
    oldData = oldT.values();
    newData = newT.values();
  }
  const auto oldSize = oldT.dims()[dim];
  const auto newSize = newT.dims()[dim];
  const auto count = oldT.dims().volume() / oldSize;
  const auto &xold = oldCoordT.values();
  const auto &xnew = newCoordT.values();
  // This function assumes that dimensions between coord and data either
  // match, or coord is 1D.
  const bool jointOld = oldCoordT.dims().shape().size() == 1;
  const bool jointNew = newCoordT.dims().shape().size() == 1;
  for (scipp::index c = 0; c < count; ++c) {
    scipp::index iold = 0;
    scipp::index inew = 0;
    const scipp::index oldEdgeOffset = jointOld ? 0 : c * (oldSize + 1);
    const scipp::index newEdgeOffset = jointNew ? 0 : c * (newSize + 1);
    const auto oldOffset = c * oldSize;
    const auto newOffset = c * newSize;
    while ((iold < oldSize) && (inew < newSize)) {
      auto xo_low = xold[oldEdgeOffset + iold];
      auto xo_high = xold[oldEdgeOffset + iold + 1];
      auto xn_low = xnew[newEdgeOffset + inew];
      auto xn_high = xnew[newEdgeOffset + inew + 1];

      if (xn_high <= xo_low)
        inew++; /* old and new bins do not overlap */
      else if (xo_high <= xn_low)
        iold++; /* old and new bins do not overlap */
      else {
        // delta is the overlap of the bins on the x axis
        auto delta = std::min<double>(xn_high, xo_high) -
                     std::max<double>(xn_low, xo_low);

        auto owidth = xo_high - xo_low;
        newData[newOffset + inew] += oldData[oldOffset + iold] * delta / owidth;

        if (xn_high > xo_high) {
          iold++;
        } else {
          inew++;
        }
      }
    }
  }
}

template <typename T>
void rebin_non_inner(const Dim dim, const VariableConstView &oldT,
                     Variable &newT, const VariableConstView &oldCoordT,
                     const VariableConstView &newCoordT) {
  const auto oldSize = oldT.dims()[dim];
  const auto newSize = newT.dims()[dim];

  const auto *xold = oldCoordT.values<T>().data();
  const auto *xnew = newCoordT.values<T>().data();
  // This function assumes that dimensions between coord and data
  // coord is 1D.
  int iold = 0;
  int inew = 0;
  while ((iold < oldSize) && (inew < newSize)) {
    auto xo_low = xold[iold];
    auto xo_high = xold[iold + 1];
    auto xn_low = xnew[inew];
    auto xn_high = xnew[inew + 1];

    if (xn_high <= xo_low)
      inew++; /* old and new bins do not overlap */
    else if (xo_high <= xn_low)
      iold++; /* old and new bins do not overlap */
    else {
      // delta is the overlap of the bins on the x axis
      auto delta = std::min(xn_high, xo_high) - std::max(xn_low, xo_low);

      auto owidth = xo_high - xo_low;
      newT.slice({dim, inew}) +=
          astype(oldT.slice({dim, iold}) * delta / owidth, newT.dtype());
      if (xn_high > xo_high) {
        iold++;
      } else {
        inew++;
      }
    }
  }
}

Variable rebin(const VariableConstView &var, const Dim dim,
               const VariableConstView &oldCoord,
               const VariableConstView &newCoord) {
  expect::notSparse(var);
  expect::notSparse(oldCoord);
  expect::notSparse(newCoord);

  // Rebin could also implemented for count-densities. However, it may be better
  // to avoid this since it increases complexity. Instead, densities could
  // always be computed on-the-fly for visualization, if required.
  expect::unit_any_of(var, {units::counts, units::Unit(units::dimensionless)});

  auto do_rebin = [dim](auto &&out, auto &&old, auto &&oldCoord_,
                        auto &&newCoord_) {
    // Dimensions of *this and old are guaranteed to be the same.
    const auto &oldT = *old;
    const auto &oldCoordT = *oldCoord_;
    const auto &newCoordT = *newCoord_;
    auto &outT = *out;
    const auto &out_dims = outT.dims();

    // dimension along which the data is being rebinned
    const bool rebin_dim_valid = out_dims.inner() == dim;

    const bool input_valid = isBinEdge(dim, oldCoordT.dims(), oldT.dims());

    const bool output_valid =
        is1D(newCoordT.dims()) && isBinEdge(dim, newCoordT.dims(), out_dims);

    if (rebin_dim_valid && input_valid && output_valid) {
      rebinInner(dim, oldT, outT, oldCoordT, newCoordT);
      if (oldT.hasVariances())
        rebinInner(dim, oldT, outT, oldCoordT, newCoordT, true);
    } else if (!rebin_dim_valid) {
      // TODO the new coord should be 1D or the same dim as newCoord.
      throw std::runtime_error(
          "The new coord should be the same dimensions as the output coord.");
    } else if (!input_valid) {
      throw std::runtime_error(
          "The input does not have coordinates with bin-edges.");
    } else if (!output_valid) {
      throw std::runtime_error(
          "The output is not 1D or does not have coordinates with bin-edges.");
    }
  };

  auto dims = var.dims();
  dims.resize(dim, newCoord.dims()[dim] - 1);
  Variable rebinned(var, dims);
  if (rebinned.dims().inner() == dim) {
    using mask_rebinning_t = std::tuple<bool, bool, double, double>;
    apply_in_place<double, float, std::tuple<float, float, double, double>,
                   std::tuple<float, float, float, double>, mask_rebinning_t>(
        do_rebin, rebinned, var, oldCoord, newCoord);
  } else {
    if (newCoord.dims().ndim() > 1)
      throw std::runtime_error(
          "Not inner rebin works only for 1d coordinates for now.");
    switch (oldCoord.dtype()) {
    case dtype<double>:
      rebin_non_inner<double>(dim, var, rebinned, oldCoord, newCoord);
      break;
    case dtype<float>:
      rebin_non_inner<float>(dim, var, rebinned, oldCoord, newCoord);
      break;
    default:
      throw std::runtime_error(
          "Rebinning is possible only for double and float types.");
    }
  }
  return rebinned;
}

} // namespace scipp::core
