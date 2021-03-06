// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock

#include "scipp/units/unit.h"

#include "scipp/core/dtype.h"
#include "scipp/core/except.h"
#include "scipp/core/tag_util.h"

#include "scipp/variable/comparison.h"
#include "scipp/variable/operations.h"
#include "scipp/variable/transform.h"
#include "scipp/variable/variable.h"

#include "scipp/dataset/dataset.h"
#include "scipp/dataset/sort.h"

#include "bind_data_access.h"
#include "bind_operators.h"
#include "bind_slice_methods.h"
#include "dtype.h"
#include "numpy.h"
#include "py_object.h"
#include "pybind11.h"
#include "rename.h"

using namespace scipp;
using namespace scipp::variable;

namespace py = pybind11;

template <class T> struct MakeVariable {
  static Variable apply(const std::vector<Dim> &labels, py::array values,
                        const std::optional<py::array> &variances,
                        const units::Unit unit) {
    // Pybind11 converts py::array to py::array_t for us, with all sorts of
    // automatic conversions such as integer to double, if required.
    py::array_t<T> valuesT(values);
    py::buffer_info info = valuesT.request();
    Dimensions dims(labels, {info.shape.begin(), info.shape.end()});
    auto var = variances
                   ? makeVariable<T>(Dimensions{dims}, Values{}, Variances{})
                   : makeVariable<T>(Dimensions(dims));
    copy_flattened<T>(valuesT, var.template values<T>());
    if (variances) {
      py::array_t<T> variancesT(*variances);
      info = variancesT.request();
      core::expect::equals(
          dims, Dimensions(labels, {info.shape.begin(), info.shape.end()}));
      copy_flattened<T>(variancesT, var.template variances<T>());
    }
    var.setUnit(unit);
    return var;
  }
};

template <class T> struct MakeVariableDefaultInit {
  static Variable apply(const std::vector<Dim> &labels,
                        const std::vector<scipp::index> &shape,
                        const units::Unit unit, const bool variances) {
    Dimensions dims(labels, shape);
    auto var = variances
                   ? makeVariable<T>(Dimensions{dims}, Values{}, Variances{})
                   : makeVariable<T>(Dimensions{dims});
    var.setUnit(unit);
    return var;
  }
};

template <class ST> struct MakeODFromNativePythonTypes {
  template <class T> struct Maker {
    static Variable apply(const units::Unit unit, const ST &value,
                          const std::optional<ST> &variance) {
      auto var = variance ? makeVariable<T>(Values{T(value)},
                                            Variances{T(variance.value())})
                          : makeVariable<T>(Values{T(value)});
      var.setUnit(unit);
      return var;
    }
  };

  static Variable make(const units::Unit unit, const ST &value,
                       const std::optional<ST> &variance,
                       const py::object &dtype) {
    return core::CallDType<double, float, int64_t, int32_t, bool>::apply<Maker>(
        scipp_dtype(dtype), unit, value, variance);
  }
};

template <class T>
Variable init_1D_no_variance(const std::vector<Dim> &labels,
                             const std::vector<scipp::index> &shape,
                             const std::vector<T> &values,
                             const units::Unit &unit) {
  Variable var;
  var = makeVariable<T>(Dims(labels), Shape(shape),
                        Values(values.begin(), values.end()));
  var.setUnit(unit);
  return var;
}

template <class T>
auto do_init_0D(const T &value, const std::optional<T> &variance,
                const units::Unit &unit) {
  using Elem = std::conditional_t<std::is_same_v<T, py::object>,
                                  scipp::python::PyObject, T>;
  Variable var;
  if (variance)
    var = makeVariable<Elem>(Values{value}, Variances{*variance});
  else
    var = makeVariable<Elem>(Values{value});
  var.setUnit(unit);
  return var;
}

Variable doMakeVariable(const std::vector<Dim> &labels, py::array &values,
                        std::optional<py::array> &variances,
                        const units::Unit unit, const py::object &dtype) {
  // Use custom dtype, otherwise dtype of data.
  const auto dtypeTag =
      dtype.is_none() ? scipp_dtype(values.dtype()) : scipp_dtype(dtype);

  if (labels.size() == 1 && !variances) {
    if (dtypeTag == core::dtype<std::string>) {
      std::vector<scipp::index> shape(values.shape(),
                                      values.shape() + values.ndim());
      return init_1D_no_variance(labels, shape,
                                 values.cast<std::vector<std::string>>(), unit);
    }

    if (dtypeTag == core::dtype<Eigen::Vector3d> ||
        dtypeTag == core::dtype<Eigen::Quaterniond>) {
      std::vector<scipp::index> shape(values.shape(),
                                      values.shape() + values.ndim() - 1);
      if (dtypeTag == core::dtype<Eigen::Vector3d>) {
        return init_1D_no_variance(
            labels, shape, values.cast<std::vector<Eigen::Vector3d>>(), unit);
      } else {
        const auto &arr = values.cast<std::vector<std::vector<double>>>();
        std::vector<Eigen::Quaterniond> qvec;
        qvec.reserve(arr.size());
        for (size_t i = 0; i < arr.size(); i++)
          qvec.emplace_back(arr[i].data());
        return init_1D_no_variance(labels, shape, qvec, unit);
      }
    }
  }

  return core::CallDType<double, float, int64_t, int32_t,
                         bool>::apply<MakeVariable>(dtypeTag, labels, values,
                                                    variances, unit);
}

Variable makeVariableDefaultInit(const std::vector<Dim> &labels,
                                 const std::vector<scipp::index> &shape,
                                 const units::Unit unit, py::object &dtype,
                                 const bool variances) {
  return core::CallDType<
      double, float, int64_t, int32_t, bool, event_list<double>,
      event_list<float>, event_list<int64_t>, event_list<int32_t>, DataArray,
      Dataset, Eigen::Vector3d,
      Eigen::Quaterniond>::apply<MakeVariableDefaultInit>(scipp_dtype(dtype),
                                                          labels, shape, unit,
                                                          variances);
}

template <class T> void bind_init_0D(py::class_<Variable> &c) {
  c.def(py::init([](const T &value, const std::optional<T> &variance,
                    const units::Unit &unit) {
          return do_init_0D(value, variance, unit);
        }),
        py::arg("value"), py::arg("variance") = std::nullopt,
        py::arg("unit") = units::Unit(units::dimensionless));
}

// This function is used only to bind native python types: pyInt -> int64_t;
// pyFloat -> double; pyBool->bool
template <class T>
void bind_init_0D_native_python_types(py::class_<Variable> &c) {
  c.def(py::init([](const T &value, const std::optional<T> &variance,
                    const units::Unit &unit, py::object &dtype) {
          static_assert(std::is_same_v<T, int64_t> ||
                        std::is_same_v<T, double> || std::is_same_v<T, bool>);
          if (dtype.is_none())
            return do_init_0D(value, variance, unit);
          else {
            return MakeODFromNativePythonTypes<T>::make(unit, value, variance,
                                                        dtype);
          }
        }),
        py::arg("value"), py::arg("variance") = std::nullopt,
        py::arg("unit") = units::Unit(units::dimensionless),
        py::arg("dtype") = py::none());
}

void bind_init_0D_numpy_types(py::class_<Variable> &c) {
  c.def(py::init([](py::buffer &b, const std::optional<py::buffer> &v,
                    const units::Unit &unit, py::object &dtype) {
          py::buffer_info info = b.request();
          if (info.ndim == 0) {
            auto arr = py::array(b);
            auto varr = v ? std::optional{py::array(*v)} : std::nullopt;
            return doMakeVariable({}, arr, varr, unit, dtype);
          } else if (info.ndim == 1 &&
                     scipp_dtype(dtype) == core::dtype<Eigen::Vector3d>) {
            return do_init_0D<Eigen::Vector3d>(
                b.cast<Eigen::Vector3d>(),
                v ? std::optional(v->cast<Eigen::Vector3d>()) : std::nullopt,
                unit);
          } else if (info.ndim == 1 &&
                     scipp_dtype(dtype) == core::dtype<Eigen::Quaterniond>) {
            return do_init_0D<Eigen::Quaterniond>(
                Eigen::Quaterniond(b.cast<std::vector<double>>().data()),
                v ? std::optional(Eigen::Quaterniond(
                        v->cast<std::vector<double>>().data()))
                  : std::nullopt,
                unit);
          } else {
            throw scipp::except::VariableError(
                "Wrong overload for making 0D variable.");
          }
        }),
        py::arg("value").noconvert(), py::arg("variance") = std::nullopt,
        py::arg("unit") = units::Unit(units::dimensionless),
        py::arg("dtype") = py::none());
}

void bind_init_list(py::class_<Variable> &c) {
  c.def(py::init([](const std::array<Dim, 1> &label, const py::list &values,
                    const std::optional<py::list> &variances,
                    const units::Unit &unit, py::object &dtype) {
          auto arr = py::array(values);
          auto varr =
              variances ? std::optional(py::array(*variances)) : std::nullopt;
          auto dims = std::vector<Dim>{label[0]};
          return doMakeVariable(dims, arr, varr, unit, dtype);
        }),
        py::arg("dims"), py::arg("values"), py::arg("variances") = std::nullopt,
        py::arg("unit") = units::Unit(units::dimensionless),
        py::arg("dtype") = py::none());
}

void bind_init_0D_list_eigen(py::class_<Variable> &c) {
  c.def(
      py::init([](const py::list &value,
                  const std::optional<py::list> &variance,
                  const units::Unit &unit, py::object &dtype) {
        if (scipp_dtype(dtype) == core::dtype<Eigen::Vector3d>) {
          return do_init_0D<Eigen::Vector3d>(
              Eigen::Vector3d(value.cast<std::vector<double>>().data()),
              variance ? std::optional(variance->cast<Eigen::Vector3d>())
                       : std::nullopt,
              unit);
        } else if (scipp_dtype(dtype) == core::dtype<Eigen::Quaterniond>) {
          return do_init_0D<Eigen::Quaterniond>(
              Eigen::Quaterniond(value.cast<std::vector<double>>().data()),
              variance ? std::optional(Eigen::Quaterniond(
                             variance->cast<std::vector<double>>().data()))
                       : std::nullopt,
              unit);
        } else {
          throw scipp::except::VariableError(
              "Cannot create 0D Variable from list of values with this dtype.");
        }
      }),
      py::arg("value"), py::arg("variance") = std::nullopt,
      py::arg("unit") = units::Unit(units::dimensionless),
      py::arg("dtype") = py::none());
}

template <class T, class... Ignored>
void bind_astype(py::class_<T, Ignored...> &c) {
  c.def(
      "astype",
      [](const T &self, const DType type) { return astype(self, type); },
      py::call_guard<py::gil_scoped_release>(),
      R"(
        Converts a Variable to a different type.

        :raises: If the variable cannot be converted to the requested dtype.
        :return: New Variable with specified dtype.
        :rtype: Variable)");
}

void init_variable(py::module &m) {
  py::class_<Variable> variable(m, "Variable",
                                R"(
    Array of values with dimension labels and a unit, optionally including an array of variances.)");
  bind_init_0D<DataArray>(variable);
  bind_init_0D<Dataset>(variable);
  bind_init_0D<std::string>(variable);
  bind_init_0D<Eigen::Vector3d>(variable);
  bind_init_0D<Eigen::Quaterniond>(variable);
  variable.def(py::init<const VariableView &>())
      .def(py::init(&makeVariableDefaultInit),
           py::arg("dims") = std::vector<Dim>{},
           py::arg("shape") = std::vector<scipp::index>{},
           py::arg("unit") = units::Unit(units::dimensionless),
           py::arg("dtype") = py::dtype::of<double>(),
           py::arg("variances").noconvert() = false)
      .def(py::init(&doMakeVariable), py::arg("dims"),
           py::arg("values"), // py::array
           py::arg("variances") = std::nullopt,
           py::arg("unit") = units::Unit(units::dimensionless),
           py::arg("dtype") = py::none())
      .def("rename_dims", &rename_dims<Variable>, py::arg("dims_dict"),
           "Rename dimensions.")
      .def(
          "copy", [](const Variable &self) { return self; },
          py::call_guard<py::gil_scoped_release>(), "Return a (deep) copy.")
      .def(
          "__copy__", [](Variable &self) { return Variable(self); },
          py::call_guard<py::gil_scoped_release>(), "Return a (deep) copy.")
      .def(
          "__deepcopy__",
          [](Variable &self, py::dict) { return Variable(self); },
          py::call_guard<py::gil_scoped_release>(), "Return a (deep) copy.")
      .def_property_readonly("dtype", &Variable::dtype)
      .def(
          "__radd__",
          [](Variable &a, double &b) {
            return a + b * units::Unit(units::dimensionless);
          },
          py::is_operator())
      .def(
          "__rsub__",
          [](Variable &a, double &b) {
            return b * units::Unit(units::dimensionless) - a;
          },
          py::is_operator())
      .def(
          "__rmul__",
          [](Variable &a, double &b) {
            return a * (b * units::Unit(units::dimensionless));
          },
          py::is_operator())
      .def("__repr__", [](const Variable &self) { return to_string(self); });

  bind_init_list(variable);
  // Order matters for pybind11's overload resolution. Do not change.
  bind_init_0D_numpy_types(variable);
  bind_init_0D_native_python_types<bool>(variable);
  bind_init_0D_native_python_types<int64_t>(variable);
  bind_init_0D_native_python_types<double>(variable);
  bind_init_0D<py::object>(variable);
  bind_init_0D_list_eigen(variable);
  //------------------------------------

  py::class_<VariableConstView>(m, "VariableConstView")
      .def(py::init<const Variable &>())
      .def(
          "copy", [](const VariableConstView &self) { return Variable(self); },
          "Return a (deep) copy.")
      .def("__copy__",
           [](const VariableConstView &self) { return Variable(self); })
      .def("__deepcopy__",
           [](VariableView &self, py::dict) { return Variable(self); })
      .def("__repr__",
           [](const VariableConstView &self) { return to_string(self); });

  py::class_<VariableView, VariableConstView> variableView(
      m, "VariableView", py::buffer_protocol(), R"(
        View for Variable, representing a sliced or transposed view onto a variable;
        Mostly equivalent to Variable, see there for details.)");
  variableView.def_buffer(&make_py_buffer_info);
  variableView.def(py::init<Variable &>())
      .def(
          "__radd__",
          [](VariableView &a, double &b) {
            return a + b * units::Unit(units::dimensionless);
          },
          py::is_operator())
      .def(
          "__rsub__",
          [](VariableView &a, double &b) {
            return b * units::Unit(units::dimensionless) - a;
          },
          py::is_operator())
      .def(
          "__rmul__",
          [](VariableView &a, double &b) {
            return a * (b * units::Unit(units::dimensionless));
          },
          py::is_operator());

  bind_astype(variable);
  bind_astype(variableView);

  bind_slice_methods(variable);
  bind_slice_methods(variableView);

  bind_comparison<Variable>(variable);
  bind_comparison<VariableConstView>(variable);
  bind_comparison<Variable>(variableView);
  bind_comparison<VariableConstView>(variableView);

  bind_in_place_binary<Variable>(variable);
  bind_in_place_binary<VariableConstView>(variable);
  bind_in_place_binary<Variable>(variableView);
  bind_in_place_binary<VariableConstView>(variableView);
  bind_in_place_binary_scalars(variable);
  bind_in_place_binary_scalars(variableView);

  bind_binary<Variable>(variable);
  bind_binary<VariableConstView>(variable);
  bind_binary<DataArrayView>(variable);
  bind_binary<Variable>(variableView);
  bind_binary<VariableConstView>(variableView);
  bind_binary<DataArrayView>(variableView);
  bind_binary_scalars(variable);
  bind_binary_scalars(variableView);

  bind_boolean_unary(variable);
  bind_boolean_unary(variableView);
  bind_boolean_operators<Variable>(variable);
  bind_boolean_operators<VariableConstView>(variable);
  bind_boolean_operators<Variable>(variableView);
  bind_boolean_operators<VariableConstView>(variableView);

  bind_data_properties(variable);
  bind_data_properties(variableView);

  py::implicitly_convertible<std::string, Dim>();
  py::implicitly_convertible<Variable, VariableConstView>();
  py::implicitly_convertible<Variable, VariableView>();

  m.def(
      "reshape",
      [](const VariableView &self, const std::vector<Dim> &labels,
         const py::tuple &shape) {
        Dimensions dims(labels, shape.cast<std::vector<scipp::index>>());
        return self.reshape(dims);
      },
      py::arg("x"), py::arg("dims"), py::arg("shape"), R"(
        Reshape a variable.

        :param x: Data to reshape.
        :param dims: List of new dimensions.
        :param shape: New extents in each dimension.
        :raises: If the volume of the old shape is not equal to the volume of the new shape.
        :return: New variable with requested dimension labels and shape.
        :rtype: Variable)");

  m.def(
      "abs", [](const VariableConstView &self) { return abs(self); },
      py::arg("x"), py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise absolute value.

        :raises: If the dtype has no absolute value, e.g., if it is a string
        :seealso: :py:class:`scipp.norm` for vector-like dtype
        :return: Copy of the input with values replaced by the absolute values
        :rtype: Variable)");

  m.def(
      "abs",
      [](const VariableConstView &self, const VariableView &out) {
        return abs(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise absolute value.

        :raises: If the dtype has no absolute value, e.g., if it is a string
        :seealso: :py:class:`scipp.norm` for vector-like dtype
        :return: Input with values replaced by the absolute values
        :rtype: Variable)");

  m.def("dot", py::overload_cast<const Variable &, const Variable &>(&dot),
        py::arg("x"), py::arg("y"), py::call_guard<py::gil_scoped_release>(),
        R"(
        Element-wise dot-product.

        :raises: If the dtype is not a vector such as :py:class:`scipp.dtype.vector_3_double`
        :return: New variable with scalar elements based on the two inputs.
        :rtype: Variable)");

  m.def("concatenate",
        py::overload_cast<const VariableConstView &, const VariableConstView &,
                          const Dim>(&concatenate),
        py::arg("x"), py::arg("y"), py::arg("dim"),
        py::call_guard<py::gil_scoped_release>(), R"(
        Concatenate input variables along the given dimension.

        Concatenation can happen in two ways:
        - Along an existing dimension, yielding a new dimension extent given by the sum of the input's extents.
        - Along a new dimension that is not contained in either of the inputs, yielding an output with one extra dimensions.

        :param x: First Variable.
        :param y: Second Variable.
        :param dim: Dimension along which to concatenate.
        :raises: If the dtype or unit does not match, or if the dimensions and shapes are incompatible.
        :return: New variable containing all elements of the input variables.
        :rtype: Variable)");

  m.def("filter",
        py::overload_cast<const Variable &, const Variable &>(&filter),
        py::arg("x"), py::arg("filter"),
        py::call_guard<py::gil_scoped_release>(), R"(
        Selects elements for a Variable using a filter (mask).

        The filter variable must be 1D and of bool type.
        A true value in the filter means the corresponding element in the input is selected and will be copied to the output.
        A false value in the filter discards the corresponding element in the input.

        :raises: If the filter variable is not 1 dimensional.
        :return: New variable containing the data selected by the filter
        :rtype: Variable)");

  m.def("mean", py::overload_cast<const VariableConstView &, const Dim>(&mean),
        py::arg("x"), py::arg("dim"), py::call_guard<py::gil_scoped_release>(),
        R"(
        Element-wise mean over the specified dimension, if variances are present, the new variance is computated as standard-deviation of the mean.

        If the input has variances, the variances stored in the ouput are based on the "standard deviation of the mean", i.e., :math:`\sigma_{mean} = \sigma / \sqrt{N}`.
        :math:`N` is the length of the input dimension.
        :math:`sigma` is estimated as the average of the standard deviations of the input elements along that dimension.
        This assumes that elements follow a normal distribution.

        :raises: If the dimension does not exist, or the dtype cannot be summed, e.g., if it is a string
        :seealso: :py:class:`scipp.sum`
        :return: New variable containing the mean.
        :rtype: Variable)");

  m.def(
      "mean",
      [](const VariableConstView &x, const Dim dim, const VariableView &out) {
        return mean(x, dim, out);
      },
      py::arg("x"), py::arg("dim"), py::arg("out"),
      py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise mean over the specified dimension, if variances are present, the new variance is computated as standard-deviation of the mean.

        If the input has variances, the variances stored in the ouput are based on the "standard deviation of the mean", i.e., :math:`\sigma_{mean} = \sigma / \sqrt{N}`.
        :math:`N` is the length of the input dimension.
        :math:`sigma` is estimated as the average of the standard deviations of the input elements along that dimension.
        This assumes that elements follow a normal distribution.

        :raises: If the dimension does not exist, or the dtype cannot be summed, e.g., if it is a string
        :seealso: :py:class:`scipp.sum`
        :return: Variable containing the mean.
        :rtype: Variable)");

  m.def("norm", py::overload_cast<const VariableConstView &>(&norm),
        py::arg("x"), py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise norm.

        :raises: If the dtype has no norm, i.e., if it is not a vector
        :seealso: :py:class:`scipp.abs` for scalar dtype
        :return: New variable with scalar elements computed as the norm values if the input elements.
        :rtype: Variable)");

  m.def("sort",
        py::overload_cast<const VariableConstView &, const VariableConstView &>(
            &sort),
        py::arg("data"), py::arg("key"),
        py::call_guard<py::gil_scoped_release>(),
        R"(Sort variable along a dimension by a sort key.

      :raises: If the key is invalid, e.g., if it has not exactly one dimension, or if its dtype is not sortable.
      :return: New sorted variable.
      :rtype: Variable)");

  m.def(
      "reciprocal",
      [](const VariableConstView &self) { return reciprocal(self); },
      py::arg("x"), py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise reciprocal.

        :return: Reciprocal of the input values.
        :rtype: Variable)");

  m.def(
      "reciprocal",
      [](const VariableConstView &self, const VariableView &out) {
        return reciprocal(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise reciprocal.

        :return: Reciprocal of the input values.
        :rtype: Variable)");

  m.def("split",
        py::overload_cast<const Variable &, const Dim,
                          const std::vector<scipp::index> &>(&split),
        py::call_guard<py::gil_scoped_release>(),
        "Split a Variable along a given Dimension.");

  m.def(
      "sqrt", [](const VariableConstView &self) { return sqrt(self); },
      py::arg("x"), py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise square-root.

        :raises: If the dtype has no square-root, e.g., if it is a string
        :return: Copy of the input with values replaced by the square-root.
        :rtype: Variable)");

  m.def(
      "sqrt",
      [](const VariableConstView &self, const VariableView &out) {
        return sqrt(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise square-root.

        :raises: If the dtype has no square-root, e.g., if it is a string
        :return: Copy of the input with values replaced by the square-root.
        :rtype: Variable)");

  m.def("sum", py::overload_cast<const VariableConstView &, const Dim>(&sum),
        py::arg("x"), py::arg("dim"), py::call_guard<py::gil_scoped_release>(),
        R"(
        Element-wise sum over the specified dimension.

        :param x: Data to sum.
        :param dim: Dimension over which to sum.
        :raises: If the dimension does not exist, or if the dtype cannot be summed, e.g., if it is a string
        :seealso: :py:class:`scipp.mean`
        :return: New variable containing the sum.
        :rtype: Variable)");

  m.def(
      "sum",
      [](const VariableConstView &self, const Dim dim,
         const VariableView &out) { return sum(self, dim, out); },
      py::arg("x"), py::arg("dim"), py::arg("out"),
      py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise sum over the specified dimension.

        :param x: Data to sum.
        :param dim: Dimension over which to sum.
        :raises: If the dimension does not exist, if the dtype cannot be summed, e.g., if it is a string or if the output variable contains the summing dimension.
        :seealso: :py:class:`scipp.mean`
        :return: Variable containing the sum.
        :rtype: Variable)");

  m.def(
      "sin", [](const VariableConstView &self) { return sin(self); },
      py::arg("x"), py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise sin.

        :raises: If the unit is not a plane-angle unit, or if the dtype has no sin, e.g., if it is an integer
        :return: Copy of the input with values replaced by the sin.
        :rtype: Variable)");

  m.def(
      "sin",
      [](const VariableConstView &self, const VariableView &out) {
        return sin(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise sin.

        :raises: If the unit is not a plane-angle unit, or if the dtype has no sin, e.g., if it is an integer
        :return: sin of input values.
        :rtype: Variable)");

  m.def(
      "cos", [](const VariableConstView &self) { return cos(self); },
      py::arg("x"), py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise cos.

        :raises: If the unit is not a plane-angle unit, or if the dtype has no cos, e.g., if it is an integer
        :return: Copy of the input with values replaced by the cos.
        :rtype: Variable)");

  m.def(
      "cos",
      [](const VariableConstView &self, const VariableView &out) {
        return cos(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise cos.

        :raises: If the unit is not a plane-angle unit, or if the dtype has no cos, e.g., if it is an integer
        :return: cos of input values.
        :rtype: Variable)");

  m.def(
      "tan", [](const Variable &self) { return tan(self); }, py::arg("x"),
      py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise tan.

        :raises: If the unit is not a plane-angle unit, or if the dtype has no tan, e.g., if it is an integer
        :return: Copy of the input with values replaced by the tan.
        :rtype: Variable)");

  m.def(
      "tan",
      [](const VariableConstView &self, const VariableView &out) {
        return tan(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise tan.

        :raises: If the unit is not a plane-angle unit, or if the dtype has no tan, e.g., if it is an integer
        :return: tan of input values.
        :rtype: Variable)");

  m.def(
      "asin", [](const Variable &self) { return asin(self); }, py::arg("x"),
      py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise asin.

        :raises: If the unit is dimensionless, or if the dtype has no asin, e.g., if it is an integer
        :return: Copy of the input with values replaced by the asin. Output unit is rad.
        :rtype: Variable)");

  m.def(
      "asin",
      [](const VariableConstView &self, const VariableView &out) {
        return asin(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise atan.

        :raises: If the unit is dimensionless, or if the dtype has no asin, e.g., if it is an integer
        :return: asin of input values. Output unit is rad.
        :rtype: Variable)");

  m.def(
      "acos", [](const Variable &self) { return acos(self); }, py::arg("x"),
      py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise acos.

        :raises: If the unit is dimensionless, or if the dtype has no acos, e.g., if it is an integer
        :return: Copy of the input with values replaced by the acos. Output unit is rad.
        :rtype: Variable)");

  m.def(
      "acos",
      [](const VariableConstView &self, const VariableView &out) {
        return acos(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise acos.

        :raises: If the unit is dimensionless, or if the dtype has no acos, e.g., if it is an integer
        :return: acos of input values. Output unit is rad.
        :rtype: Variable)");

  m.def(
      "atan", [](const Variable &self) { return atan(self); }, py::arg("x"),
      py::call_guard<py::gil_scoped_release>(), R"(
        Element-wise atan.

        :raises: If the unit is dimensionless, or if the dtype has no atan, e.g., if it is an integer
        :return: Copy of the input with values replaced by the atan. Output unit is rad.
        :rtype: Variable)");

  m.def(
      "atan",
      [](const VariableConstView &self, const VariableView &out) {
        return atan(self, out);
      },
      py::arg("x"), py::arg("out"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise atan.

        :raises: If the unit is dimensionless, or if the dtype has no atan, e.g., if it is an integer
        :return: atan of input values. Output unit is rad.
        :rtype: Variable)");

  m.def(
      "atan2", [](const Variable &y, const Variable &x) { return atan2(y, x); },
      py::arg("y"), py::arg("x"),
      R"(
        Element-wise atan2.

        :raises: If the units of inputs are different, or if the dtype has no atan2, e.g., if it is an integer
        :return: atan2 of input y and x. Output unit is rad.
        :rtype: Variable)");

  m.def(
      "atan2",
      [](const VariableConstView &y, const VariableConstView &x,
         const VariableView &out) { return atan2(y, x, out); },
      py::arg("y"), py::arg("x"), py::arg("out"),
      R"(
        Element-wise atan2 with out argument.

        :raises: If the units of inputs are different, or if the dtype has no atan2, e.g., if it is an integer
        :return: atan2 of input y and x, written to output. Output unit is rad.
        :rtype: VariableView)");

  m.def("all", py::overload_cast<const VariableConstView &, const Dim>(&all),
        py::arg("x"), py::arg("dim"), py::call_guard<py::gil_scoped_release>(),
        R"(
        Element-wise AND over the specified dimension.

        :param x: Data to reduce.
        :param dim: Dimension to reduce.
        :raises: If the dimension does not exist, or if the dtype is not bool
        :seealso: :py:class:`scipp.any`
        :return: New variable containing the reduced values.
        :rtype: Variable)");

  m.def("any", py::overload_cast<const VariableConstView &, const Dim>(&any),
        py::arg("x"), py::arg("dim"), py::call_guard<py::gil_scoped_release>(),
        R"(
        Element-wise OR over the specified dimension.

        :param x: Data to reduce.
        :param dim: Dimension to reduce.
        :raises: If the dimension does not exist, or if the dtype is not bool
        :seealso: :py:class:`scipp.all`
        :return: New variable containing the reduced values.
        :rtype: Variable)");

  m.def(
      "min",
      [](const VariableConstView &self, const Dim dim) {
        return min(self, dim);
      },
      py::arg("x"), py::arg("dim"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise min over the specified dimension.

        :param x: Data to reduce.
        :param dim: Dimension to reduce.
        :seealso: :py:class:`scipp.max`
        :return: New variable containing the min values.
        :rtype: Variable)");

  m.def(
      "max",
      [](const VariableConstView &self, const Dim dim) {
        return max(self, dim);
      },
      py::arg("x"), py::arg("dim"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise max over the specified dimension.

        :param x: Data to reduce.
        :param dim: Dimension to reduce.
        :seealso: :py:class:`scipp.min`
        :return: New variable containing the max values.
        :rtype: Variable)");

  m.def(
      "min", [](const VariableConstView &self) { return min(self); },
      py::arg("x"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise min over all of the input's dimensions.

        :param x: Data to reduce.
        :seealso: :py:class:`scipp.max`
        :return: New variable containing the min values.
        :rtype: Variable)");

  m.def(
      "max", [](const VariableConstView &self) { return max(self); },
      py::arg("x"), py::call_guard<py::gil_scoped_release>(),
      R"(
        Element-wise max over all of the input's dimensions.

        :param x: Data to reduce.
        :seealso: :py:class:`scipp.min`
        :return: New variable containing the max values.
        :rtype: Variable)");

  m.def(
      "nan_to_num",
      [](const VariableConstView &self,
         const std::optional<VariableConstView> &nan,
         const std::optional<VariableConstView> &posinf,
         const std::optional<VariableConstView> &neginf) {
        Variable out(self);
        if (nan)
          nan_to_num(out, *nan, out);
        if (posinf)
          positive_inf_to_num(out, *posinf, out);
        if (neginf)
          negative_inf_to_num(out, *neginf, out);
        return out;
      },
      py::call_guard<py::gil_scoped_release>(),
      R"(Element-wise special value replacement

       All elements in the output are identical to input except in the presence of a nan, inf or -inf.
       The function allows replacements to be separately specified for nan, inf or -inf values.
       You can choose to replace a subset of those special values by providing just the required key word arguments.
       If the replacement is value-only and the input has variances,
       the variance at the element(s) undergoing replacement are also replaced with the replacement value.
       If the replacement has a variance and the input has variances,
       the variance at the element(s) undergoing replacement are also replaced with the replacement variance.
       :raises: If the types of input and replacement do not match.
       :return: Input elements are replaced in output with specified subsitutions.
       :rtype: Variable)",
      py::arg("x"), py::arg("nan") = std::optional<VariableConstView>(),
      py::arg("posinf") = std::optional<VariableConstView>(),
      py::arg("neginf") = std::optional<VariableConstView>());

  m.def(
      "nan_to_num",
      [](const VariableConstView &self,
         const std::optional<VariableConstView> &nan,
         const std::optional<VariableConstView> &posinf,
         const std::optional<VariableConstView> &neginf, VariableView &out) {
        if (nan)
          nan_to_num(self, *nan, out);
        if (posinf)
          positive_inf_to_num(self, *posinf, out);
        if (neginf)
          negative_inf_to_num(self, *neginf, out);
        return out;
      },
      py::call_guard<py::gil_scoped_release>(),
      R"(Element-wise special value replacement

       All elements in the output are identical to input except in the presence of a nan, inf or -inf.
       The function allows replacements to be separately specified for nan, inf or -inf values.
       You can choose to replace a subset of those special values by providing just the required key word arguments.
       If the replacement is value-only and the input has variances,
       the variance at the element(s) undergoing replacement are also replaced with the replacement value.
       If the replacement has a variance and the input has variances,
       the variance at the element(s) undergoing replacement are also replaced with the replacement variance.
       :raises: If the types of input and replacement do not match.
       :return: Input elements are replaced in output with specified subsitutions.
       :rtype: Variable)",
      py::arg("x"), py::arg("nan") = std::optional<VariableConstView>(),
      py::arg("posinf") = std::optional<VariableConstView>(),
      py::arg("neginf") = std::optional<VariableConstView>(), py::arg("out"));

  m.def(
      "contains_events",
      [](const VariableConstView &self) { return contains_events(self); },
      R"(Return true if the variable contains event data.)");

  m.def(
      "contains_events",
      [](const DataArrayConstView &self) { return contains_events(self); },
      R"(Return true if the data array contains event data. Note that data may be stored as a scalar, but this returns true if any coord contains events.)");

  m.def(
      "less",
      [](const VariableConstView &x, const VariableConstView &y) {
        return less(x, y);
      },
      py::arg("x"), py::arg("y"),
      R"(
        Comparison returning the truth value of (x < y) element-wise.

        :raises: If the units of inputs are not the same, or if the dtypes of inputs are not double precision floats
        :return: Variable of booleans.
        :rtype: Variable)");

  m.def(
      "greater",
      [](const VariableConstView &x, const VariableConstView &y) {
        return greater(x, y);
      },
      py::arg("x"), py::arg("y"),
      R"(
        Comparison returning the truth value of (x > y) element-wise.

        :raises: If the units of inputs are not the same, or if the dtypes of inputs are not double precision floats
        :return: Variable of booleans.
        :rtype: Variable)");

  m.def(
      "greater_equal",
      [](const VariableConstView &x, const VariableConstView &y) {
        return greater_equal(x, y);
      },
      py::arg("x"), py::arg("y"),
      R"(
        Comparison returning the truth value of (x >= y) element-wise.

        :raises: If the units of inputs are not the same, or if the dtypes of inputs are not double precision floats
        :return: Variable of booleans.
        :rtype: Variable)");

  m.def(
      "less_equal",
      [](const VariableConstView &x, const VariableConstView &y) {
        return less_equal(x, y);
      },
      py::arg("x"), py::arg("y"),
      R"(
        Comparison returning the truth value of (x <= y) element-wise.

        :raises: If the units of inputs are not the same, or if the dtypes of inputs are not double precision floats
        :return: Variable of booleans.
        :rtype: Variable)");

  m.def(
      "equal",
      [](const VariableConstView &x, const VariableConstView &y) {
        return equal(x, y);
      },
      py::arg("x"), py::arg("y"),
      R"(
        Comparison returning the truth value of (x == y) element-wise.

        :raises: If the units of inputs are not the same, or if the dtypes of inputs are not double precision floats
        :return: Variable of booleans.
        :rtype: Variable)");

  m.def(
      "not_equal",
      [](const VariableConstView &x, const VariableConstView &y) {
        return not_equal(x, y);
      },
      py::arg("x"), py::arg("y"),
      R"(
        Comparison returning the truth value of (x != y) element-wise.

        :raises: If the units of inputs are not the same, or if the dtypes of inputs are not double precision floats
        :return: Variable of booleans.
        :rtype: Variable)");

  auto geom_m = m.def_submodule("geometry");
  geom_m.def(
      "position",
      [](const VariableConstView &x, const VariableConstView &y,
         const VariableConstView &z) { return geometry::position(x, y, z); },
      py::arg("x"), py::arg("y"), py::arg("z"),
      R"(
        Element-wise zip functionality to produce a vector_3_float64 from independent input variables.

        :raises: If the units of inputs are not all meters, or if the dtypes of inputs are not double precision floats
        :return: zip of input x, y and z. Output unit is meters.
        :rtype: Variable)");
  geom_m.def(
      "x", [](const VariableConstView &pos) { return geometry::x(pos); },
      py::arg("pos"),
      R"(
        un-zip functionality to produce a Variable of the x component of a vector_3_float64.

        :raises: If the units of inputs are not meters, or if the dtypes of inputs are not double precision floats
        :return: Extracted x component of input pos. Units in meters.
        :rtype: Variable)");
  geom_m.def(
      "y", [](const VariableConstView &pos) { return geometry::y(pos); },
      py::arg("pos"),
      R"(
        un-zip functionality to produce a Variable of the y component of a vector_3_float64.

        :raises: If the units of inputs are not meters, or if the dtypes of inputs are not double precision floats
        :return: Extracted y component of input pos. Units in meters.
        :rtype: Variable)");
  geom_m.def(
      "z", [](const VariableConstView &pos) { return geometry::z(pos); },
      py::arg("pos"),
      R"(
        un-zip functionality to produce a Variable of the z component of a vector_3_float64.

        :raises: If the units of inputs are not meters, or if the dtypes of inputs are not double precision floats
        :return: Extracted z component of input pos. Units in meters.
        :rtype: Variable)");
  geom_m.def(
      "rotate",
      [](const VariableConstView &pos, const VariableConstView &rot) {
        return geometry::rotate(pos, rot);
      },
      py::arg("position"), py::arg("rotation"),
      R"(
        Rotate a Variable of type vector_3_float64 using a Variable of type quaternion_float64.

        :raises: If the units of the inputs are not meter and dimensionless for the positions and the rotations, respectively.
        :return: a Variable containing the rotated position vectors.
        :rtype: Variable)");
  geom_m.def(
      "rotate",
      [](const VariableConstView &pos, const VariableConstView &rot,
         const VariableView &out) { return geometry::rotate(pos, rot, out); },
      py::arg("position"), py::arg("rotation"), py::arg("out"),
      R"(
        Rotate a Variable of type vector_3_float64 using a Variable of type quaternion_float64, and store in the result inside the provided out argument.

        :raises: If the units of the inputs are not meter and dimensionless for the positions and the rotations, respectively.
        :return: rotated vectors are written to out.
        :rtype: VariableView)");
}
