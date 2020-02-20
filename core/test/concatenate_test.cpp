// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include "test_macros.h"
#include <gtest/gtest.h>

#include "scipp/core/dataset.h"

using namespace scipp;
using namespace scipp::core;

TEST(ConcatenateTest, simple_1d) {
  Dataset a;
  a.setCoord(Dim::X,
             makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  a.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{11, 12, 13}));
  a.setLabels("label_1",
              makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{21, 22, 23}));
  a.setMask("mask_1", makeVariable<bool>(Dims{Dim::X}, Shape{3},
                                         Values{false, true, false}));

  Dataset b;
  b.setCoord(Dim::X,
             makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{4, 5, 6}));
  b.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{14, 15, 16}));
  b.setLabels("label_1",
              makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{24, 25, 26}));
  b.setMask("mask_1", makeVariable<bool>(Dims{Dim::X}, Shape{3},
                                         Values{false, true, false}));

  const auto d = concatenate(a, b, Dim::X);

  EXPECT_EQ(d.coords()[Dim::X], makeVariable<int>(Dims{Dim::X}, Shape{6},
                                                  Values{1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(d["data_1"].data(),
            makeVariable<int>(Dims{Dim::X}, Shape{6},
                              Values{11, 12, 13, 14, 15, 16}));
  EXPECT_EQ(d.labels()["label_1"],
            makeVariable<int>(Dims{Dim::X}, Shape{6},
                              Values{21, 22, 23, 24, 25, 26}));
  EXPECT_EQ(d.masks()["mask_1"],
            makeVariable<bool>(Dims{Dim::X}, Shape{6},
                               Values{false, true, false, false, true, false}));
}

TEST(ConcatenateTest, simple_1d_histogram) {
  Dataset a;
  a.setCoord(Dim::X,
             makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  a.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{2}, Values{11, 12}));
  a.setLabels("edge_labels",
              makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{21, 22, 23}));
  a.setLabels("labels",
              makeVariable<int>(Dims{Dim::X}, Shape{2}, Values{21, 22}));
  a.setMask("masks",
            makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{false, true}));

  Dataset b;
  b.setCoord(Dim::X,
             makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{3, 4, 5}));
  b.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{2}, Values{13, 14}));
  b.setLabels("edge_labels",
              makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{23, 24, 25}));
  b.setLabels("labels",
              makeVariable<int>(Dims{Dim::X}, Shape{2}, Values{24, 25}));
  b.setMask("masks",
            makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{false, true}));

  Dataset expected;
  expected.setCoord(
      Dim::X, makeVariable<int>(Dims{Dim::X}, Shape{5}, Values{1, 2, 3, 4, 5}));
  expected.setData("data_1", makeVariable<int>(Dims{Dim::X}, Shape{4},
                                               Values{11, 12, 13, 14}));
  expected.setLabels(
      "edge_labels",
      makeVariable<int>(Dims{Dim::X}, Shape{5}, Values{21, 22, 23, 24, 25}));
  expected.setLabels("labels", makeVariable<int>(Dims{Dim::X}, Shape{4},
                                                 Values{21, 22, 24, 25}));
  expected.setMask("masks",
                   makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                      Values{false, true, false, true}));

  EXPECT_EQ(concatenate(a, b, Dim::X), expected);
}

TEST(ConcatenateTest, fail_when_histograms_have_non_overlapping_bins) {
  Dataset a;
  a.setCoord(Dim::X,
             makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  a.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{2}, Values{11, 12}));

  Dataset b;
  b.setCoord(Dim::X,
             makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{4, 5, 6}));
  b.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{2}, Values{13, 14}));

  EXPECT_THROW(concatenate(a, b, Dim::X), except::MismatchError<Variable>);
}

TEST(ConcatenateTest, fail_mixing_point_data_and_histogram) {
  Dataset pointData;
  pointData.setCoord(Dim::X, makeVariable<int>(Dims{Dim::X}, Shape{3}));
  pointData.setData("data_1", makeVariable<int>(Dims{Dim::X}, Shape{3}));

  Dataset histogram;
  histogram.setCoord(Dim::X, makeVariable<int>(Dims{Dim::X}, Shape{3}));
  histogram.setData("data_1", makeVariable<int>(Dims{Dim::X}, Shape{2}));

  EXPECT_THROW(concatenate(pointData, histogram, Dim::X), except::BinEdgeError);
}

TEST(ConcatenateTest, identical_non_dependant_data_is_copied) {
  const auto axis = makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});
  const auto data =
      makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{11, 12, 13});

  Dataset a;
  a.setCoord(Dim::X, axis);
  a.setData("data_1", data);

  Dataset b;
  b.setCoord(Dim::X, axis);
  b.setData("data_1", data);

  const auto d = concatenate(a, b, Dim::Y);

  EXPECT_EQ(d.coords()[Dim::X], axis);
  EXPECT_EQ(d["data_1"].data(), data);
}

TEST(ConcatenateTest, non_dependant_data_is_stacked) {
  const auto axis = makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});

  Dataset a;
  a.setCoord(Dim::X, axis);
  a.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{11, 12, 13}));

  Dataset b;
  b.setCoord(Dim::X, axis);
  b.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{14, 15, 16}));

  const auto d = concatenate(a, b, Dim::Y);

  EXPECT_EQ(d["data_1"].data(),
            makeVariable<int>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                              Values{11, 12, 13, 14, 15, 16}));
}

TEST(ConcatenateTest, concat_2d_coord) {
  Dataset a;
  a.setCoord(Dim::X,
             makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  a.setData("data_1",
            makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{11, 12, 13}));
  a.setLabels("label_1",
              makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{21, 22, 23}));
  a.setMask("mask_1", makeVariable<bool>(Dims{Dim::X}, Shape{3},
                                         Values{false, true, false}));

  Dataset b(a);
  b.coords()[Dim::X] += 3;
  b["data_1"].data() += 100;

  Dataset expected;
  expected.setCoord(
      Dim::X, makeVariable<int>(Dims{Dim::Y, Dim::X}, Shape{4, 3},
                                Values{1, 2, 3, 4, 5, 6, 4, 5, 6, 1, 2, 3}));
  expected.setData("data_1",
                   makeVariable<int>(Dims{Dim::Y, Dim::X}, Shape{4, 3},
                                     Values{11, 12, 13, 111, 112, 113, 111, 112,
                                            113, 11, 12, 13}));
  expected.setLabels(
      "label_1", makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{21, 22, 23}));
  expected.setMask("mask_1", makeVariable<bool>(Dims{Dim::X}, Shape{3},
                                                Values{false, true, false}));

  const auto ab = concatenate(a, b, Dim::Y);
  const auto ba = concatenate(b, a, Dim::Y);
  const auto abba = concatenate(ab, ba, Dim::Y);

  EXPECT_EQ(abba, expected);
}

TEST(ConcatenateTest, concatenate_sparse_no_data) {
  auto var1 =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, Dimensions::Sparse});
  auto var1_ = var1.sparseValues<double>();
  var1_[0] = {1, 2, 3};
  var1_[1] = {1, 2};
  const auto var2 = concatenate(var1, var1, Dim::X);

  const auto a =
      DataArray(std::optional<Variable>(), {{Dim::X, var1}}, {{"labs", var1}});
  const auto b =
      DataArray(std::optional<Variable>(), {{Dim::X, var2}}, {{"labs", var2}});

  const auto x = concatenate(a, b, Dim::X);
  const auto y = concatenate(a, b, Dim::Y);

  EXPECT_FALSE(x.hasData());
  EXPECT_FALSE(y.hasData());
  EXPECT_EQ(x.coords()[Dim::X], concatenate(var1, var2, Dim::X));
  EXPECT_EQ(x.labels()["labs"], concatenate(var1, var2, Dim::X));
  EXPECT_EQ(y.coords()[Dim::X], concatenate(var1, var2, Dim::Y));
  EXPECT_EQ(y.labels()["labs"], concatenate(var1, var2, Dim::Y));
}

TEST(ConcatenateTest, dataset_with_no_data_items) {
  Dataset a, b;
  a.setCoord(Dim::X,
             makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2}));
  a.setLabels("points",
              makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{.1, .2}));
  b.setCoord(Dim::X,
             makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{3, 4}));
  b.setLabels("points",
              makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{.3, .4}));

  const auto res = concatenate(a, b, Dim::X);

  EXPECT_EQ(res.coords()[Dim::X],
            makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4}));
  EXPECT_EQ(
      res.labels()["points"],
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{.1, .2, .3, .4}));
}

TEST(ConcatenateTest, dataset_with_no_data_items_histogram) {
  Dataset a, b;
  a.setCoord(Dim::X,
             makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  a.setLabels("histogram",
              makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{.1, .2}));
  b.setCoord(Dim::X,
             makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{3, 4, 5}));
  b.setLabels("histogram",
              makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{.3, .4}));

  const auto res = concatenate(a, b, Dim::X);

  EXPECT_EQ(res.coords()[Dim::X], makeVariable<double>(Dims{Dim::X}, Shape{5},
                                                       Values{1, 2, 3, 4, 5}));
  EXPECT_EQ(
      res.labels()["histogram"],
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{.1, .2, .3, .4}));
}
