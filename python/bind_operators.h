// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef SCIPP_PYTHON_BIND_OPERATORS_H
#define SCIPP_PYTHON_BIND_OPERATORS_H

#include "pybind11.h"

namespace py = pybind11;

template <class Other, class T, class... Ignored>
void bind_comparison(pybind11::class_<T, Ignored...> &c) {
  c.def("__eq__", [](T &a, Other &b) { return a == b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__ne__", [](T &a, Other &b) { return a != b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
}

template <class Other, class T, class... Ignored>
void bind_in_place_binary(pybind11::class_<T, Ignored...> &c) {
  // In-place operators return py::object due to the way in-place operators work
  // in Python (assigning return value to this). This avoids extra copies, and
  // additionally ensures that all references to the object keep referencing the
  // same object after the operation.
  c.def("__iadd__",
        [](py::object &a, Other &b) {
          a.cast<T &>() += b;
          return a;
        },
        py::is_operator(), py::call_guard<py::gil_scoped_release>());
  c.def("__isub__",
        [](py::object &a, Other &b) {
          a.cast<T &>() -= b;
          return a;
        },
        py::is_operator(), py::call_guard<py::gil_scoped_release>());
  c.def("__imul__",
        [](py::object &a, Other &b) {
          a.cast<T &>() *= b;
          return a;
        },
        py::is_operator(), py::call_guard<py::gil_scoped_release>());
  c.def("__itruediv__",
        [](py::object &a, Other &b) {
          a.cast<T &>() /= b;
          return a;
        },
        py::is_operator(), py::call_guard<py::gil_scoped_release>());
}

template <class Other, class T, class... Ignored>
void bind_binary(pybind11::class_<T, Ignored...> &c) {
  c.def("__add__", [](T &a, Other &b) { return a + b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__sub__", [](T &a, Other &b) { return a - b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__mul__", [](T &a, Other &b) { return a * b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__truediv__", [](T &a, Other &b) { return a / b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
}

template <class T, class... Ignored>
void bind_in_place_binary_scalars(pybind11::class_<T, Ignored...> &c) {
  bind_in_place_binary<float>(c);
  bind_in_place_binary<double>(c);
  bind_in_place_binary<int32_t>(c);
  bind_in_place_binary<int64_t>(c);
}

template <class T, class... Ignored>
void bind_binary_scalars(pybind11::class_<T, Ignored...> &c) {
  bind_binary<float>(c);
  bind_binary<double>(c);
  bind_binary<int32_t>(c);
  bind_binary<int64_t>(c);
}

template <class T, class... Ignored>
void bind_boolean_unary(pybind11::class_<T, Ignored...> &c) {
  c.def("__invert__", [](T &a) { return ~a; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
}

template <class Other, class T, class... Ignored>
void bind_boolean_operators(pybind11::class_<T, Ignored...> &c) {
  c.def("__or__", [](T &a, Other &b) { return a | b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__xor__", [](T &a, Other &b) { return a ^ b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__and__", [](T &a, Other &b) { return a & b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__ior__",
        [](py::object &a, Other &b) {
          a.cast<T &>() |= b;
          return a;
        },
        py::is_operator(), py::call_guard<py::gil_scoped_release>());
  c.def("__ixor__",
        [](py::object &a, Other &b) {
          a.cast<T &>() ^= b;
          return a;
        },
        py::is_operator(), py::call_guard<py::gil_scoped_release>());
  c.def("__iand__",
        [](py::object &a, Other &b) {
          a.cast<T &>() &= b;
          return a;
        },
        py::is_operator(), py::call_guard<py::gil_scoped_release>());
}

#endif // SCIPP_PYTHON_BIND_OPERATORS_H
