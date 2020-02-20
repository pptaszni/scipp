// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/core/variable.h"

using namespace scipp;
using namespace scipp::core;

static auto make_sparse() {
  auto var = makeVariable<double>(Dims{Dim::Z, Dim::Y, Dim::X},
                                  Shape{3l, 2l, Dimensions::Sparse});
  scipp::index count = 0;
  for (auto &v : var.sparseValues<double>())
    v.resize(count++);
  return var;
}

static auto make_sparse_with_variances() {
  auto var = makeVariable<double>(
      Dimensions{{Dim::Z, 3}, {Dim::Y, 2}, {Dim::X, Dimensions::Sparse}},
      Values{}, Variances{});
  scipp::index count = 0;
  for (auto &v : var.sparseValues<double>())
    v.resize(count++);
  count = 0;
  for (auto &v : var.sparseVariances<double>())
    v.resize(count++);
  return var;
}

TEST(SparseCountsTest, fail_dense) {
  auto bad = makeVariable<double>(Values{1.0});
  EXPECT_ANY_THROW(sparse::counts(bad));
}

TEST(SparseCountsTest, no_variances) {
  const auto var = make_sparse();
  auto expected = makeVariable<scipp::index>(Dims{Dim::Z, Dim::Y}, Shape{3, 2},
                                             units::Unit(units::counts),
                                             Values{0, 1, 2, 3, 4, 5});

  EXPECT_EQ(sparse::counts(var), expected);
}

TEST(SparseCountsTest, variances) {
  const auto var = make_sparse_with_variances();
  auto expected = makeVariable<scipp::index>(Dims{Dim::Z, Dim::Y}, Shape{3, 2},
                                             units::Unit(units::counts),
                                             Values{0, 1, 2, 3, 4, 5});

  EXPECT_EQ(sparse::counts(var), expected);
}
