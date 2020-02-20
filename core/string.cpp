// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <set>

#include "scipp/core/dataset.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/slice.h"
#include "scipp/core/string.h"
#include "scipp/core/tag_util.h"

namespace scipp::core {

std::ostream &operator<<(std::ostream &os, const Dim dim) {
  return os << to_string(dim);
}

std::ostream &operator<<(std::ostream &os, const VariableConstView &variable) {
  return os << to_string(variable);
}

std::ostream &operator<<(std::ostream &os, const VariableView &variable) {
  return os << VariableConstView(variable);
}

std::ostream &operator<<(std::ostream &os, const Variable &variable) {
  return os << VariableConstView(variable);
}

std::ostream &operator<<(std::ostream &os, const DataArrayConstView &data) {
  return os << to_string(data);
}

std::ostream &operator<<(std::ostream &os, const DataArrayView &data) {
  return os << DataArrayConstView(data);
}

std::ostream &operator<<(std::ostream &os, const DataArray &data) {
  return os << DataArrayConstView(data);
}

std::ostream &operator<<(std::ostream &os, const DatasetConstView &dataset) {
  return os << to_string(dataset);
}

std::ostream &operator<<(std::ostream &os, const DatasetView &dataset) {
  return os << DatasetConstView(dataset);
}

std::ostream &operator<<(std::ostream &os, const Dataset &dataset) {
  return os << DatasetConstView(dataset);
}

constexpr const char *tab = "    ";

std::string to_string(const Dimensions &dims) {
  if (dims.empty())
    return "{}";
  std::string s = "{{";
  for (int32_t i = 0; i < dims.shape().size(); ++i)
    s += to_string(dims.denseLabels()[i]) + ", " +
         std::to_string(dims.shape()[i]) + "}, {";
  if (dims.sparse())
    s += to_string(dims.sparseDim()) + ", [sparse]}, {";
  s.resize(s.size() - 3);
  s += "}";
  return s;
}

std::string to_string(const bool b) { return b ? "True" : "False"; }

std::string to_string(const DType dtype) {
  switch (dtype) {
  case DType::String:
    return "string";
  case DType::Bool:
    return "bool";
  case DType::DataArray:
    return "DataArray";
  case DType::Dataset:
    return "Dataset";
  case DType::Float:
    return "float32";
  case DType::Double:
    return "float64";
  case DType::Int32:
    return "int32";
  case DType::Int64:
    return "int64";
  case DType::SparseFloat:
    return "sparse_float32";
  case DType::SparseDouble:
    return "sparse_float64";
  case DType::SparseInt64:
    return "sparse_int64";
  case DType::SparseInt32:
    return "sparse_int32";
  case DType::EigenVector3d:
    return "vector_3_float64";
  case DType::PyObject:
    return "PyObject";
  case DType::Unknown:
    return "unknown";
  default:
    return "unregistered dtype";
  };
}

std::string to_string(const Slice &slice) {
  std::string end = slice.end() >= 0 ? ", " + std::to_string(slice.end()) : "";
  return "Slice(" + to_string(slice.dim()) + ", " +
         std::to_string(slice.begin()) + end + ")\n";
}

std::string make_dims_labels(const VariableConstView &variable,
                             const Dimensions &datasetDims) {
  const auto &dims = variable.dims();
  if (dims.empty())
    return "()";
  std::string diminfo = "(";
  for (const auto dim : dims.denseLabels()) {
    diminfo += to_string(dim);
    if (datasetDims.contains(dim) && (datasetDims[dim] + 1 == dims[dim]))
      diminfo += " [bin-edges]";
    diminfo += ", ";
  }
  if (variable.dims().sparse()) {
    diminfo += to_string(variable.dims().sparseDim());
    diminfo += " [sparse]";
    diminfo += ", ";
  }
  diminfo.resize(diminfo.size() - 2);
  diminfo += ")";
  return diminfo;
}

auto &to_string(const std::string &s) { return s; }
auto to_string(const std::string_view s) { return s; }
auto to_string(const char *s) { return std::string(s); }

template <class T> struct ValuesToString {
  static auto apply(const VariableConstView &var) {
    return array_to_string(var.template values<T>());
  }
};
template <class T> struct VariancesToString {
  static auto apply(const VariableConstView &var) {
    return array_to_string(var.template variances<T>());
  }
};

template <template <class> class Callable, class... Args>
auto apply(const DType dtype, Args &&... args) {
  return callDType<Callable>(
      std::tuple<double, float, int64_t, int32_t, std::string, bool,
                 sparse_container<double>, sparse_container<float>,
                 sparse_container<int64_t>, DataArray, Dataset,
                 Eigen::Vector3d>{},
      dtype, std::forward<Args>(args)...);
}

template <class Key, class Var>
auto format_variable(const Key &key, const Var &variable,
                     const Dimensions &datasetDims = Dimensions()) {
  if (!variable)
    return std::string(tab) + "invalid variable\n";
  std::stringstream s;
  const auto dtype = variable.dtype();
  const std::string colSep("  ");
  s << tab << std::left << std::setw(24) << to_string(key);
  s << colSep << std::setw(9) << to_string(variable.dtype());
  s << colSep << std::setw(15) << '[' + variable.unit().name() + ']';
  s << colSep << make_dims_labels(variable, datasetDims);
  s << colSep;
  if (dtype == DType::PyObject)
    s << "[PyObject]";
  else
    s << apply<ValuesToString>(variable.data().dtype(),
                               VariableConstView(variable));
  if (variable.hasVariances())
    s << colSep
      << apply<VariancesToString>(variable.data().dtype(),
                                  VariableConstView(variable));
  s << '\n';
  return s.str();
}

template <class Key>
auto format_data_view(const Key &name, const DataArrayConstView &data,
                      const Dimensions &datasetDims = Dimensions()) {
  std::stringstream s;
  if (data.hasData())
    s << format_variable(name, data.data(), datasetDims);
  else
    s << tab << name << '\n';
  for (const auto &[dim, coord] : data.coords())
    if (coord.dims().sparse()) {
      s << tab << tab << "Sparse coordinate:\n";
      s << format_variable(std::string(tab) + std::string(tab) + to_string(dim),
                           coord, datasetDims);
    }
  bool sparseLabels = false;
  for (const auto &[label_name, labels] : data.labels())
    if (labels.dims().sparse()) {
      if (!sparseLabels) {
        s << tab << tab << "Sparse labels:\n";
        sparseLabels = true;
      }
      s << format_variable(std::string(tab) + std::string(tab) +
                               std::string(label_name),
                           labels, datasetDims);
    }

  if (!data.attrs().empty()) {
    s << tab << "Attributes:\n";
    for (const auto &[attr_name, var] : data.attrs())
      s << tab << tab << format_variable(attr_name, var, datasetDims);
  }
  return s.str();
}

std::string to_string(const Variable &variable) {
  return format_variable("<scipp.Variable>", variable);
}

std::string to_string(const VariableConstView &variable) {
  return format_variable("<scipp.VariableView>", variable);
}

template <class D>
std::string do_to_string(const D &dataset, const std::string &id,
                         const Dimensions &dims) {
  std::stringstream s;
  s << id + '\n';
  s << "Dimensions: " << to_string(dims) << '\n';

  if (!dataset.coords().empty()) {
    s << "Coordinates:\n";
    for (const auto &[dim, var] : dataset.coords()) {
      /* Only print the dense coordinates here (sparse coordinates will appear
       * with their corresponding data item) */
      if (var.dims().sparseDim() == Dim::Invalid)
        s << format_variable(dim, var, dims);
    }
  }
  if (!dataset.labels().empty()) {
    s << "Labels:\n";
    for (const auto &[name, var] : dataset.labels())
      s << format_variable(name, var, dims);
  }
  if (!dataset.attrs().empty()) {
    s << "Attributes:\n";
    for (const auto &[name, var] : dataset.attrs())
      s << format_variable(name, var, dims);
  }
  if (!dataset.masks().empty()) {
    s << "Masks:\n";
    for (const auto &[name, var] : dataset.masks())
      s << format_variable(name, var, dims);
  }

  if constexpr (std::is_same_v<D, DataArray> ||
                std::is_same_v<D, DataArrayConstView>) {
    s << "Data:\n" << format_data_view(dataset.name(), dataset);
  } else {
    if (!dataset.empty())
      s << "Data:\n";
    std::set<std::string> sorted_items;
    for (const auto &item : dataset)
      sorted_items.insert(item.name());
    for (const auto &name : sorted_items)
      s << format_data_view(name, dataset[name], dims);
  }

  s << '\n';
  return s.str();
}

template <class T> Dimensions dimensions(const T &dataset) {
  Dimensions datasetDims;
  Dim sparse = Dim::Invalid;
  for (const auto &item : dataset) {
    const auto &dims = item.dims();
    for (const auto dim : dims.labels()) {
      if (!datasetDims.contains(dim)) {
        if (dim == dims.sparseDim())
          sparse = dim;
        else
          datasetDims.add(dim, dims[dim]);
      }
    }
  }
  for (const auto &coord : dataset.coords()) {
    const auto &dims = coord.second.dims();
    for (const auto dim : dims.labels())
      if (!datasetDims.contains(dim))
        datasetDims.add(dim, dims[dim]);
  }
  for (const auto &labels : dataset.labels()) {
    const auto &dims = labels.second.dims();
    for (const auto dim : dims.labels())
      if (!datasetDims.contains(dim))
        datasetDims.add(dim, dims[dim]);
  }
  if (sparse != Dim::Invalid && !datasetDims.contains(sparse))
    datasetDims.addInner(sparse, Dimensions::Sparse);
  return datasetDims;
}

std::string to_string(const DataArray &data) {
  return do_to_string(data, "<scipp.DataArray>", data.dims());
}

std::string to_string(const DataArrayConstView &data) {
  return do_to_string(data, "<scipp.DataArrayView>", data.dims());
}

std::string to_string(const Dataset &dataset) {
  return do_to_string(dataset, "<scipp.Dataset>", dimensions(dataset));
}

std::string to_string(const DatasetConstView &dataset) {
  return do_to_string(dataset, "<scipp.DatasetView>", dimensions(dataset));
}
} // namespace scipp::core
