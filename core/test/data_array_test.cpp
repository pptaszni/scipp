// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/core/dataset.h"

#include "dataset_test_common.h"
#include "test_macros.h"

using namespace scipp;
using namespace scipp::core;

TEST(DataArrayTest, construct) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();

  DataArray array(dataset["data_xyz"]);
  EXPECT_EQ(array, dataset["data_xyz"]);
  // Comparison ignores the name, so this is tested separately.
  EXPECT_EQ(array.name(), "data_xyz");
}

TEST(DataArrayTest, sum_dataset_columns_via_DataArray) {
  DatasetFactory3D factory;
  auto dataset = factory.make();

  DataArray array(dataset["data_zyx"]);
  auto sum = array + dataset["data_xyz"];

  dataset["data_zyx"] += dataset["data_xyz"];

  // This would fail if the data items had attributes, since += preserves them
  // but + does not.
  EXPECT_EQ(sum, dataset["data_zyx"]);
}

TEST(DataArrayTest, reciprocal) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  DataArray array(dataset["data_zyx"]);
  EXPECT_EQ(reciprocal(array).data(), reciprocal(array.data()));
}

auto make_sparse() {
  auto var =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2l, Dimensions::Sparse});
  var.setUnit(units::us);
  auto vals = var.sparseValues<double>();
  vals[0] = {1.1, 2.2, 3.3};
  vals[1] = {1.1, 2.2, 3.3, 5.5};
  return DataArray(std::optional<Variable>(), {{Dim::X, var}});
}

auto make_histogram() {
  auto edges =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                           units::Unit(units::us), Values{0, 2, 4, 1, 3, 5});
  auto data = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0, 3.0},
                                   Variances{0.3, 0.4});

  return DataArray(data, {{Dim::X, edges}});
}

auto make_histogram_no_variance() {
  auto edges =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                           units::Unit(units::us), Values{0, 2, 4, 1, 3, 5});
  auto data = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0, 3.0});

  return DataArray(data, {{Dim::X, edges}});
}

TEST(DataArrayTest, astype) {
  DataArray a(
      makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}),
      {{Dim::X, makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{4, 5, 6})}});
  const auto x = astype(a, DType::Double);
  EXPECT_EQ(x.data(),
            makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1., 2., 3.}));
}

TEST(DataArraySparseArithmeticTest, fail_sparse_op_non_histogram) {
  const auto sparse = make_sparse();
  auto coord = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                    Values{0, 2, 1, 3});
  auto data = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0, 3.0},
                                   Variances{0.3, 0.4});
  DataArray not_hist(data, {{Dim::X, coord}});

  EXPECT_THROW(sparse * not_hist, except::SparseDataError);
  EXPECT_THROW(not_hist * sparse, except::SparseDataError);
  EXPECT_THROW(sparse / not_hist, except::SparseDataError);
}

TEST(DataArraySparseArithmeticTest, sparse_times_histogram) {
  const auto sparse = make_sparse();
  const auto hist = make_histogram();

  for (const auto result : {sparse * hist, hist * sparse}) {
    EXPECT_EQ(result.coords()[Dim::X], sparse.coords()[Dim::X]);
    EXPECT_TRUE(result.hasVariances());
    EXPECT_EQ(result.unit(), units::counts);

    const auto out_vals = result.data().sparseValues<double>();
    const auto out_vars = result.data().sparseVariances<double>();

    auto expected =
        makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 1, 1},
                             Variances{1, 1, 1}) *
        makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{2.0, 3.0, 3.0},
                             Variances{0.3, 0.4, 0.4});
    EXPECT_TRUE(equals(out_vals[0], expected.values<double>()));
    EXPECT_TRUE(equals(out_vars[0], expected.variances<double>()));
    // out of range of edges -> value set to 0, consistent with rebin behavior
    expected =
        makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 1, 1, 1},
                             Variances{1, 1, 1, 1}) *
        makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{2.0, 2.0, 3.0, 0.0},
                             Variances{0.3, 0.3, 0.4, 0.0});
    EXPECT_TRUE(equals(out_vals[1], expected.values<double>()));
    EXPECT_TRUE(equals(out_vars[1], expected.variances<double>()));
  }
  EXPECT_EQ(copy(sparse) *= hist, sparse * hist);
}

TEST(DataArraySparseArithmeticTest, sparse_times_histogram_without_variances) {
  const auto sparse = make_sparse();
  auto hist = make_histogram_no_variance();

  for (const auto result : {sparse * hist, hist * sparse}) {
    EXPECT_EQ(result.coords()[Dim::X], sparse.coords()[Dim::X]);
    EXPECT_TRUE(result.hasVariances());
    EXPECT_EQ(result.unit(), units::counts);

    const auto out_vals = result.data().sparseValues<double>();
    const auto out_vars = result.data().sparseVariances<double>();

    auto expected =
        makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 1, 1},
                             Variances{1, 1, 1}) *
        makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{2.0, 3.0, 3.0});
    EXPECT_TRUE(equals(out_vals[0], expected.values<double>()));
    EXPECT_TRUE(equals(out_vars[0], expected.variances<double>()));
    // out of range of edges -> value set to 0, consistent with rebin behavior
    expected = makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 1, 1, 1},
                                    Variances{1, 1, 1, 1}) *
               makeVariable<double>(Dims{Dim::X}, Shape{4},
                                    Values{2.0, 2.0, 3.0, 0.0});
    EXPECT_TRUE(equals(out_vals[1], expected.values<double>()));
    EXPECT_TRUE(equals(out_vars[1], expected.variances<double>()));
  }
  EXPECT_EQ(copy(sparse) *= hist, sparse * hist);
}

TEST(DataArraySparseArithmeticTest, sparse_with_values_times_histogram) {
  auto sparse = make_sparse();
  const auto hist = make_histogram();
  Variable data(sparse.coords()[Dim::X]);
  data.setUnit(units::counts);
  data *= 0.0;
  data += 2.0 * units::Unit(units::counts);
  sparse.setData(data);

  for (const auto result : {sparse * hist, hist * sparse}) {
    EXPECT_EQ(result.coords()[Dim::X], sparse.coords()[Dim::X]);
    EXPECT_TRUE(result.hasVariances());
    EXPECT_EQ(result.unit(), units::counts);
    const auto out_vals = result.data().sparseValues<double>();
    const auto out_vars = result.data().sparseVariances<double>();

    auto expected =
        makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{2, 2, 2},
                             Variances{0, 0, 0}) *
        makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{2.0, 3.0, 3.0},
                             Variances{0.3, 0.4, 0.4});
    EXPECT_TRUE(equals(out_vals[0], expected.values<double>()));
    EXPECT_TRUE(equals(out_vars[0], expected.variances<double>()));
    // out of range of edges -> value set to 0, consistent with rebin behavior
    expected =
        makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{2, 2, 2, 2},
                             Variances{0, 0, 0, 0}) *
        makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{2.0, 2.0, 3.0, 0.0},
                             Variances{0.3, 0.3, 0.4, 0.0});
    EXPECT_TRUE(equals(out_vals[1], expected.values<double>()));
    EXPECT_TRUE(equals(out_vars[1], expected.variances<double>()));
  }
}

TEST(DataArraySparseArithmeticTest, sparse_over_histogram) {
  const auto sparse = make_sparse();
  const auto hist = make_histogram();

  const auto result = sparse / hist;
  EXPECT_EQ(result.coords()[Dim::X], sparse.coords()[Dim::X]);
  EXPECT_TRUE(result.hasVariances());
  EXPECT_EQ(result.unit(), units::counts);
  const auto out_vals = result.data().sparseValues<double>();
  const auto out_vars = result.data().sparseVariances<double>();

  auto expected =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 1, 1},
                           Variances{1, 1, 1}) /
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{2.0, 3.0, 3.0},
                           Variances{0.3, 0.4, 0.4});
  EXPECT_TRUE(equals(out_vals[0], expected.values<double>()));
  EXPECT_TRUE(equals(out_vars[0], expected.variances<double>()));
  expected =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 1, 1, 1},
                           Variances{1, 1, 1, 1}) /
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{2.0, 2.0, 3.0, 0.0},
                           Variances{0.3, 0.3, 0.4, 0.0});
  EXPECT_TRUE(equals(out_vals[1], expected.values<double>()));
  EXPECT_TRUE(equals(span<const double>(out_vars[1]).subspan(0, 3),
                     expected.slice({Dim::X, 0, 3}).variances<double>()));
  EXPECT_TRUE(std::isnan(out_vars[1][3]));
}
