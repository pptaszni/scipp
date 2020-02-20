// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <numeric>

#include "scipp/core/except.h"
#include "scipp/core/indexed_slice_view.h"
#include "scipp/core/sort.h"
#include "scipp/core/tag_util.h"

namespace scipp::core {

template <class T> struct MakePermutation {
  static auto apply(const VariableConstView &key) {
    if (key.dims().ndim() != 1)
      throw except::DimensionError("Sort key must be 1-dimensional");

    // Variances are ignored for sorting.
    const auto &values = key.values<T>();

    std::vector<scipp::index> permutation(values.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(
        permutation.begin(), permutation.end(),
        [&](scipp::index i, scipp::index j) { return values[i] < values[j]; });
    return permutation;
  }
};

static auto makePermutation(const VariableConstView &key) {
  return CallDType<double, float, int64_t, int32_t, bool,
                   std::string>::apply<MakePermutation>(key.dtype(), key);
}

/// Return a Variable sorted based on key.
Variable sort(const VariableConstView &var, const VariableConstView &key) {
  return concatenate(
      IndexedSliceView{var, key.dims().inner(), makePermutation(key)});
}

/// Return a DataArray sorted based on key.
DataArray sort(const DataArrayConstView &array, const VariableConstView &key) {
  return concatenate(
      IndexedSliceView{array, key.dims().inner(), makePermutation(key)});
}

/// Return a DataArray sorted based on coordinate.
DataArray sort(const DataArrayConstView &array, const Dim &key) {
  return sort(array, array.coords()[key]);
}

/// Return a DataArray sorted based on labels.
DataArray sort(const DataArrayConstView &array, const std::string &key) {
  return sort(array, array.labels()[key]);
}

/// Return a Dataset sorted based on key.
Dataset sort(const DatasetConstView &dataset, const VariableConstView &key) {
  return concatenate(
      IndexedSliceView{dataset, key.dims().inner(), makePermutation(key)});
}

/// Return a Dataset sorted based on coordinate.
Dataset sort(const DatasetConstView &dataset, const Dim &key) {
  return sort(dataset, dataset.coords()[key]);
}

/// Return a Dataset sorted based on labels.
Dataset sort(const DatasetConstView &dataset, const std::string &key) {
  return sort(dataset, dataset.labels()[key]);
}

} // namespace scipp::core
