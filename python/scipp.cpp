// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "pybind11.h"

namespace py = pybind11;

void init_dataset(py::module &);
void init_dimensions(py::module &);
void init_dtype(py::module &);
void init_neutron(py::module &);
void init_sparse_container(py::module &);
void init_units_neutron(py::module &);
void init_variable(py::module &);
void init_variable_view(py::module &);

PYBIND11_MODULE(_scipp, m) {
#ifdef SCIPP_VERSION
  m.attr("__version__") = py::str(SCIPP_VERSION);
#else
  m.attr("__version__") = py::str("unknown version");
#endif

  init_units_neutron(m);
  init_dataset(m);
  init_dimensions(m);
  init_dtype(m);
  init_neutron(m);
  init_sparse_container(m);
  init_variable(m);
  init_variable_view(m);
}
