// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include <type_traits>

#include "scipp/core/dataset.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/except.h"

using namespace scipp;
using namespace scipp::core;

TEST(StringFormattingTest, to_string_Dataset) {
  Dataset a;
  a.setData("a", makeVariable<double>(Values{double{}}));
  a.setData("b", makeVariable<double>(Values{double{}}));
  // Create new dataset with same variables but different order
  Dataset b;
  b.setData("b", makeVariable<double>(Values{double{}}));
  b.setData("a", makeVariable<double>(Values{double{}}));
  // string representations should be the same
  EXPECT_EQ(to_string(a), to_string(b));
}

std::tuple<Dataset, Dataset> makeDatasets() {
  Dataset a;
  a.setCoord(Dim::X,
             makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  a.setCoord(Dim::Y,
             makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{1, 2, 3}));
  a.setCoord(Dim::Z,
             makeVariable<double>(Dims{Dim::Z}, Shape{3}, Values{1, 2, 3}));
  a.setLabels("label_1",
              makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{21, 22, 23}));
  a.setLabels("label_2",
              makeVariable<int>(Dims{Dim::Y}, Shape{3}, Values{21, 22, 23}));
  a.setLabels("label_3",
              makeVariable<int>(Dims{Dim::Z}, Shape{3}, Values{21, 22, 23}));
  a.setData("a", makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  a.setData("b", makeVariable<int>(Dims{Dim::Y}, Shape{3}, Values{1, 2, 3}));
  a.setData("c", makeVariable<int>(Dims{Dim::Z}, Shape{3}, Values{1, 2, 3}));

  Dataset b;
  b.setCoord(Dim::X,
             makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  b.setCoord(Dim::Y,
             makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{1, 2, 3}));
  b.setCoord(Dim::Z,
             makeVariable<double>(Dims{Dim::Z}, Shape{3}, Values{1, 2, 3}));
  b.setLabels("label_1",
              makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{21, 22, 23}));
  b.setLabels("label_2",
              makeVariable<int>(Dims{Dim::Y}, Shape{3}, Values{21, 22, 23}));
  b.setLabels("label_3",
              makeVariable<int>(Dims{Dim::Z}, Shape{3}, Values{21, 22, 23}));
  b.setData("a", makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  b.setData("b", makeVariable<int>(Dims{Dim::Y}, Shape{3}, Values{1, 2, 3}));
  b.setData("c", makeVariable<int>(Dims{Dim::Z}, Shape{3}, Values{1, 2, 3}));

  return std::make_tuple(a, b);
}

TEST(StringFormattingTest, to_string_MutableView) {
  auto [a, b] = makeDatasets();

  EXPECT_EQ(to_string(a.coords()), to_string(b.coords()));
  EXPECT_EQ(to_string(a.labels()), to_string(b.labels()));
  EXPECT_EQ(to_string(a.attrs()), to_string(b.attrs()));
}

TEST(StringFormattingTest, to_string_ConstView) {
  const auto [a, b] = makeDatasets();

  EXPECT_EQ(to_string(a.coords()), to_string(b.coords()));
  EXPECT_EQ(to_string(a.labels()), to_string(b.labels()));
  EXPECT_EQ(to_string(a.attrs()), to_string(b.attrs()));
}

TEST(StringFormattingTest, to_string_sparse_Dataset) {
  Dataset a;
  a.setSparseCoord("a", makeVariable<double>(Dims{Dim::Y, Dim::X},
                                             Shape{4l, Dimensions::Sparse}));
  ASSERT_NO_THROW(to_string(a));
}

TEST(ValidSliceTest, test_slice_range) {
  Dimensions dims{Dim::X, 3};
  EXPECT_NO_THROW(expect::validSlice(dims, Slice(Dim::X, 0)));
  EXPECT_NO_THROW(expect::validSlice(dims, Slice(Dim::X, 2)));
  EXPECT_NO_THROW(expect::validSlice(dims, Slice(Dim::X, 0, 3)));
  EXPECT_THROW(expect::validSlice(dims, Slice(Dim::X, 3)), except::SliceError);
  EXPECT_THROW(expect::validSlice(dims, Slice(Dim::X, -1)), except::SliceError);
  EXPECT_THROW(expect::validSlice(dims, Slice(Dim::X, 0, 4)),
               except::SliceError);
}

TEST(ValidSliceTest, test_dimension_contained) {
  Dimensions dims{{Dim::X, 3}, {Dim::Z, 3}};
  EXPECT_NO_THROW(expect::validSlice(dims, Slice(Dim::X, 0)));
  EXPECT_THROW(expect::validSlice(dims, Slice(Dim::Y, 0)), except::SliceError);
}
