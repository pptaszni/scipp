// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
#include <benchmark/benchmark.h>

#include "scipp/core/dataset.h"
#include "scipp/neutron/convert.h"

using namespace scipp;
using namespace scipp::core;

auto make_beamline(const scipp::index size) {
  Dataset beamline;

  beamline.setLabels(
      "source_position",
      makeVariable<Eigen::Vector3d>(units::Unit(units::m),
                                    Values{Eigen::Vector3d{0.0, 0.0, -10.0}}));
  beamline.setLabels(
      "sample_position",
      makeVariable<Eigen::Vector3d>(units::Unit(units::m),
                                    Values{Eigen::Vector3d{0.0, 0.0, 0.0}}));

  beamline.setLabels(
      "position", makeVariable<Eigen::Vector3d>(
                      Dims{Dim::Spectrum}, Shape{size}, units::Unit(units::m)));
  return beamline;
}

auto make_dense_coord_only(const scipp::index size, const scipp::index count,
                           bool transpose) {
  auto out = make_beamline(size);
  auto var = transpose ? makeVariable<double>(Dims{Dim::Tof, Dim::Spectrum},
                                              Shape{count, size})
                       : makeVariable<double>(Dims{Dim::Spectrum, Dim::Tof},
                                              Shape{size, count});
  out.setCoord(Dim::Tof, std::move(var));
  return out;
}

static void BM_neutron_convert(benchmark::State &state, const Dim targetDim) {
  const scipp::index nBin = state.range(0);
  const scipp::index nHist = 1e8 / nBin;
  const bool transpose = state.range(1);
  const auto dense = make_dense_coord_only(nHist, nBin, transpose);
  for (auto _ : state) {
    state.PauseTiming();
    Dataset data = dense;
    state.ResumeTiming();
    data = neutron::convert(std::move(data), Dim::Tof, targetDim);
    state.PauseTiming();
    data = Dataset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * nHist * nBin);
  state.SetBytesProcessed(state.iterations() * nHist * nBin * 2 *
                          sizeof(double));
  state.counters["positions"] = benchmark::Counter(nHist);
  state.counters["transpose"] = transpose;
}

auto make_sparse_coord_only(const scipp::index size, const scipp::index count) {
  auto out = make_beamline(size);
  auto var = makeVariable<double>(Dims{Dim::Spectrum, Dim::Tof},
                                  Shape{size, Dimensions::Sparse});
  auto vals = var.sparseValues<double>();
  for (scipp::index i = 0; i < size; ++i)
    vals[i].resize(count, 5000.0);
  out.setSparseCoord("", std::move(var));
  return out;
}

static void BM_neutron_convert_sparse(benchmark::State &state,
                                      const Dim targetDim) {
  const scipp::index nEvent = state.range(0);
  const scipp::index nHist = 1e8 / nEvent;
  const auto sparse = make_sparse_coord_only(nHist, nEvent);
  for (auto _ : state) {
    state.PauseTiming();
    Dataset data = sparse;
    state.ResumeTiming();
    data = neutron::convert(std::move(data), Dim::Tof, targetDim);
    state.PauseTiming();
    data = Dataset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * nHist * nEvent);
  state.SetBytesProcessed(state.iterations() * nHist * nEvent * 2 *
                          sizeof(double));
  state.counters["positions"] = benchmark::Counter(nHist);
}

// Params are:
// - nBin
// - transpose
BENCHMARK_CAPTURE(BM_neutron_convert, Dim::DSpacing, Dim::DSpacing)
    ->RangeMultiplier(2)
    ->Ranges({{8, 2 << 14}, {false, true}});
BENCHMARK_CAPTURE(BM_neutron_convert, Dim::Wavelength, Dim::Wavelength)
    ->RangeMultiplier(2)
    ->Ranges({{8, 2 << 14}, {false, true}});
BENCHMARK_CAPTURE(BM_neutron_convert, Dim::Energy, Dim::Energy)
    ->RangeMultiplier(2)
    ->Ranges({{8, 2 << 14}, {false, true}});

// Params are:
// - nEvent
BENCHMARK_CAPTURE(BM_neutron_convert_sparse, Dim::DSpacing, Dim::DSpacing)
    ->RangeMultiplier(2)
    ->Ranges({{8, 2 << 14}});
BENCHMARK_CAPTURE(BM_neutron_convert_sparse, Dim::Wavelength, Dim::Wavelength)
    ->RangeMultiplier(2)
    ->Ranges({{8, 2 << 14}});
BENCHMARK_CAPTURE(BM_neutron_convert_sparse, Dim::Energy, Dim::Energy)
    ->RangeMultiplier(2)
    ->Ranges({{8, 2 << 14}});

BENCHMARK_MAIN();
