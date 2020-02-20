// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "pybind11.h"

namespace py = pybind11;

void init_dataset(py::module &);
void init_detail(py::module &);
void init_dimensions(py::module &);
void init_dtype(py::module &);
void init_counts(py::module &);
void init_groupby(py::module &);
void init_neutron(py::module &);
void init_operations(py::module &);
void init_sparse_container(py::module &);
void init_units_neutron(py::module &);
void init_variable(py::module &);
void init_element_array_view(py::module &);

void init_core(py::module &m) {
  auto core = m.def_submodule("core");
  init_units_neutron(core);
  init_counts(core);
  init_dataset(core);
  init_dimensions(core);
  init_dtype(core);
  init_groupby(core);
  init_operations(core);
  init_sparse_container(core);
  init_variable(core);
  init_element_array_view(core);
}

PYBIND11_MODULE(_scipp, m) {
#ifdef SCIPP_VERSION
  m.attr("__version__") = py::str(SCIPP_VERSION);
#else
  m.attr("__version__") = py::str("unknown version");
#endif
  init_core(m);
  init_neutron(m);
  init_detail(m);
}
