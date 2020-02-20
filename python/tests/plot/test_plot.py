# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @file
# @author Neil Vaytet

from itertools import product

import numpy as np
import pytest

import scipp as sc
from scipp import Dim
from scipp.plot import plot

# TODO: For now we are just checking that the plot does not throw any errors.
# In the future it would be nice to check the output by either comparing
# checksums or by using tools like squish.


def make_dense_dataset(ndim=1,
                       variances=False,
                       binedges=False,
                       labels=False,
                       masks=False):

    dim_list = [Dim.Tof, Dim.X, Dim.Y, Dim.Z, Dim.Qx]

    N = 50
    M = 10

    d = sc.Dataset()
    shapes = []
    dims = []
    for i in range(ndim):
        n = N - (i * M)
        d.coords[dim_list[i]] = sc.Variable([dim_list[i]],
                                            np.arange(n + binedges).astype(
                                                np.float64))
        dims.append(dim_list[i])
        shapes.append(n)
    a = np.sin(np.arange(np.prod(shapes)).reshape(*shapes).astype(np.float64))
    d["Sample"] = sc.Variable(dims, values=a, unit=sc.units.counts)
    if variances:
        d["Sample"].variances = np.abs(np.random.normal(a * 0.1, 0.05))
    if labels:
        d.labels["somelabels"] = sc.Variable([dim_list[0]],
                                             values=np.linspace(
                                                 101., 105., shapes[0]),
                                             unit=sc.units.s)
    if masks:
        d.masks["mask"] = sc.Variable(dims,
                                      values=np.where(a > 0, True, False))
    return d


def make_sparse_dataset(ndim=1, data=False):

    dim_list = [Dim.Tof, Dim.X, Dim.Y, Dim.Z, Dim.Qx]

    N = 50
    M = 10

    dims = []
    shapes = []
    for i in range(1, ndim):
        n = N - (i * M)
        dims.append(dim_list[i])
        shapes.append(n)
    dims.append(dim_list[0])
    shapes.append(sc.Dimensions.Sparse)

    var = sc.Variable(dims=dims, shape=shapes, unit=sc.units.us)
    if data:
        dat = sc.Variable(dims=dims, shape=shapes, unit=sc.units.us)

    if ndim > 1:
        indices = tuple()
        for i in range(ndim - 1):
            indices += range(shapes[i]),
    else:
        indices = [0],
    # Now construct all indices combinations using itertools
    for ind in product(*indices):
        # And for each indices combination, slice the original
        # data down to the sparse dimension
        vslice = var
        if data:
            dslice = dat
        if ndim > 1:
            for i in range(ndim - 1):
                vslice = vslice[dims[i], ind[i]]
                if data:
                    dslice = dslice[dims[i], ind[i]]
        v = np.random.normal(float(N),
                             scale=2.0 * M,
                             size=int(np.random.rand() * N))
        vslice.values = v
        if data:
            dslice.values = v * 0.5

    d = sc.Dataset()
    for i in range(1, ndim):
        d.coords[dim_list[i]] = sc.Variable([dim_list[i]],
                                            values=np.arange(N - (i * M),
                                                             dtype=np.float),
                                            unit=sc.units.m)
    params = {"coords": {dim_list[0]: var}}
    if data:
        params["data"] = dat
    d["a"] = sc.DataArray(**params)
    return d


def test_plot_1d():
    d = make_dense_dataset(ndim=1)
    plot(d)


def test_plot_1d_with_variances():
    d = make_dense_dataset(ndim=1, variances=True)
    plot(d)


def test_plot_1d_bin_edges():
    d = make_dense_dataset(ndim=1, binedges=True)
    plot(d)


def test_plot_1d_with_labels():
    d = make_dense_dataset(ndim=1, labels=True)
    plot(d, axes=["somelabels"])


def test_plot_1d_log_axes():
    d = make_dense_dataset(ndim=1)
    plot(d, logx=True)
    plot(d, logy=True)
    plot(d, logxy=True)


def test_plot_1d_bin_edges_with_variances():
    d = make_dense_dataset(ndim=1, variances=True, binedges=True)
    plot(d)


def test_plot_1d_two_separate_entries():
    d = make_dense_dataset(ndim=1)
    d["Background"] = sc.Variable([Dim.Tof],
                                  values=2.0 * np.random.rand(50),
                                  unit=sc.units.kg)
    plot(d)


def test_plot_1d_two_entries_on_same_plot():
    d = make_dense_dataset(ndim=1)
    d["Background"] = sc.Variable([Dim.Tof],
                                  values=2.0 * np.random.rand(50),
                                  unit=sc.units.counts)
    plot(d)


def test_plot_1d_two_entries_hide_variances():
    d = make_dense_dataset(ndim=1, variances=True)
    d["Background"] = sc.Variable([Dim.Tof],
                                  values=2.0 * np.random.rand(50),
                                  unit=sc.units.counts)
    plot(d, variances=False)
    # When variances are not present, the plot does not fail, is silently does
    # not show variances
    print(d)
    plot(d, variances={"Sample": False, "Background": True})


def test_plot_1d_three_entries_with_labels():
    N = 50
    d = make_dense_dataset(ndim=1, labels=True)
    d["Background"] = sc.Variable([Dim.Tof],
                                  values=2.0 * np.random.rand(N),
                                  unit=sc.units.counts)
    d.coords[Dim.X] = sc.Variable([Dim.X],
                                  values=np.arange(N).astype(np.float64),
                                  unit=sc.units.m)
    d["Sample2"] = sc.Variable([Dim.X],
                               values=10.0 * np.random.rand(N),
                               unit=sc.units.counts)
    d.labels["Xlabels"] = sc.Variable([Dim.X],
                                      values=np.linspace(151., 155., N),
                                      unit=sc.units.s)
    plot(d, axes={Dim.X: "Xlabels", Dim.Tof: "somelabels"})


def test_plot_1d_with_masks():
    d = make_dense_dataset(ndim=1, masks=True)
    plot(d)


def test_plot_2d_image():
    d = make_dense_dataset(ndim=2)
    plot(d)


def test_plot_2d_image_with_axes():
    d = make_dense_dataset(ndim=2)
    plot(d, axes=[Dim.Tof, Dim.X])


def test_plot_2d_image_with_labels():
    d = make_dense_dataset(ndim=2, labels=True)
    plot(d, axes=[Dim.X, "somelabels"])


def test_plot_2d_image_with_variances():
    d = make_dense_dataset(ndim=2, variances=True)
    plot(d, variances=True)


def test_plot_2d_image_with_filename():
    d = make_dense_dataset(ndim=2)
    plot(d, filename="image.pdf")


def test_plot_2d_image_with_variances_with_filename():
    d = make_dense_dataset(ndim=2, variances=True)
    plot(d, variances=True, filename="val_and_var.pdf")


def test_plot_2d_image_with_bin_edges():
    d = make_dense_dataset(ndim=2, binedges=True)
    plot(d)


def test_plot_2d_with_masks():
    d = make_dense_dataset(ndim=2, masks=True)
    plot(d)


def test_plot_collapse():
    d = make_dense_dataset(ndim=2)
    plot(d, collapse=Dim.Tof)


def test_plot_sliceviewer():
    d = make_dense_dataset(ndim=3)
    plot(d)


def test_plot_sliceviewer_with_variances():
    d = make_dense_dataset(ndim=3, variances=True)
    plot(d, variances=True)


def test_plot_sliceviewer_with_two_sliders():
    d = make_dense_dataset(ndim=4)
    plot(d)


def test_plot_sliceviewer_with_axes():
    d = make_dense_dataset(ndim=3)
    plot(d, axes=[Dim.X, Dim.Tof, Dim.Y])


def test_plot_sliceviewer_with_labels():
    d = make_dense_dataset(ndim=3, labels=True)
    plot(d, axes=[Dim.X, Dim.Y, "somelabels"])


def test_plot_sliceviewer_with_labels_bad_dimension():
    d = make_dense_dataset(ndim=3, labels=True)
    with pytest.raises(Exception) as e:
        plot(d, axes=[Dim.Tof, Dim.Y, "somelabels"])
    assert str(e.value) == ("The dimension of the labels cannot also be "
                            "specified as another axis.")


def test_plot_sliceviewer_with_3d_projection():
    d = make_dense_dataset(ndim=3)
    plot(d, projection="3d")


def test_plot_sliceviewer_with_3d_projection_with_variances():
    d = make_dense_dataset(ndim=3, variances=True)
    plot(d, projection="3d", variances=True)


def test_plot_sliceviewer_with_3d_projection_with_labels():
    d = make_dense_dataset(ndim=3, labels=True)
    plot(d, projection="3d", axes=[Dim.X, Dim.Y, "somelabels"])


def test_plot_3d_with_filename():
    d = make_dense_dataset(ndim=3)
    plot(d, projection="3d", filename="a3dplot.html")


def test_plot_sliceviewer_with_1d_projection():
    d = make_dense_dataset(ndim=3)
    plot(d, projection="1d")


@pytest.mark.skip(reason="Volume rendering is not yet supported.")
def test_plot_volume():
    d = make_dense_dataset(ndim=3)
    plot(d, projection="volume")


def test_plot_convenience_methods():
    d = make_dense_dataset(ndim=3)
    sc.plot.image(d)
    sc.plot.threeslice(d)
    # sc.plot.volume(d)
    sc.plot.superplot(d)


def test_plot_1d_sparse_data():
    d = make_sparse_dataset(ndim=1)
    plot(d)


def test_plot_1d_sparse_data_with_weights():
    d = make_sparse_dataset(ndim=1, data=True)
    plot(d)


def test_plot_1d_sparse_data_with_int_bins():
    d = make_sparse_dataset(ndim=1)
    plot(d, bins=50)


def test_plot_1d_sparse_data_with_nparray_bins():
    d = make_sparse_dataset(ndim=1)
    plot(d, bins=np.linspace(0.0, 105.0, 50))


def test_plot_1d_sparse_data_with_Variable_bins():
    d = make_sparse_dataset(ndim=1)
    bins = sc.Variable([Dim.Tof],
                       values=np.linspace(0.0, 105.0, 50),
                       unit=sc.units.us)
    plot(d, bins=bins)


def test_plot_2d_sparse_data():
    d = make_sparse_dataset(ndim=2)
    plot(d)


def test_plot_2d_sparse_data_with_weights():
    d = make_sparse_dataset(ndim=2, data=True)
    plot(d)


def test_plot_2d_sparse_data_with_int_bins():
    d = make_sparse_dataset(ndim=2)
    plot(d, bins=50)


def test_plot_2d_sparse_data_with_nparray_bins():
    d = make_sparse_dataset(ndim=2)
    plot(d, bins=np.linspace(0.0, 105.0, 50))


def test_plot_2d_sparse_data_with_Variable_bins():
    d = make_sparse_dataset(ndim=2)
    bins = sc.Variable([Dim.Tof],
                       values=np.linspace(0.0, 105.0, 50),
                       unit=sc.units.us)
    plot(d, bins=bins)


def test_plot_3d_sparse_data():
    d = make_sparse_dataset(ndim=3)
    plot(d)


def test_plot_3d_sparse_data_with_weights():
    d = make_sparse_dataset(ndim=3, data=True)
    plot(d)


@pytest.mark.skip(reason="RuntimeError: Only the simple case histograms may "
                  "be constructed for now: 2 dims including sparse.")
def test_plot_3d_sparse_data_with_int_bins():
    d = make_sparse_dataset(ndim=3)
    plot(d, bins=50)


@pytest.mark.skip(reason="RuntimeError: Only the simple case histograms may "
                  "be constructed for now: 2 dims including sparse.")
def test_plot_3d_sparse_data_with_nparray_bins():
    d = make_sparse_dataset(ndim=3)
    plot(d, bins=np.linspace(0.0, 105.0, 50))


@pytest.mark.skip(reason="RuntimeError: Only the simple case histograms may "
                  "be constructed for now: 2 dims including sparse.")
def test_plot_3d_sparse_data_with_Variable_bins():
    d = make_sparse_dataset(ndim=3)
    bins = sc.Variable([Dim.Tof],
                       values=np.linspace(0.0, 105.0, 50),
                       unit=sc.units.us)
    plot(d, bins=bins)


def test_plot_variable():
    N = 50
    v1d = sc.Variable([Dim.Tof],
                      values=np.random.rand(N),
                      unit=sc.units.counts)
    v2d = sc.Variable([Dim.Tof, Dim.X],
                      values=np.random.rand(N, N),
                      unit=sc.units.K)
    v3d = sc.Variable([Dim.Tof, Dim.X, Dim.Y],
                      values=np.random.rand(N, N, N),
                      unit=sc.units.m)
    plot(v1d)
    plot(v2d)
    plot(v3d)


def test_plot_dataset_view():
    d = make_dense_dataset(ndim=2)
    plot(d[Dim.X, 0])


def test_plot_data_array():
    d = make_dense_dataset(ndim=1)
    plot(d["Sample"])


def test_plot_vector_axis_labels_1d():
    d = sc.Dataset()
    N = 10
    vecs = []
    for i in range(N):
        vecs.append(np.random.random(3))
    d.coords[Dim.X] = sc.Variable([Dim.X],
                                  values=vecs,
                                  unit=sc.units.m,
                                  dtype=sc.dtype.vector_3_float64)
    d["Sample"] = sc.Variable([Dim.X],
                              values=np.random.rand(N),
                              unit=sc.units.counts)
    plot(d)


def test_plot_string_axis_labels_1d():
    d = sc.Dataset()
    N = 10
    d.coords[Dim.X] = sc.Variable(
        dims=[Dim.X],
        values=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        unit=sc.units.m)
    d["Sample"] = sc.Variable([Dim.X],
                              values=np.random.rand(N),
                              unit=sc.units.counts)
    plot(d)


def test_plot_string_axis_labels_1d_short():
    d = sc.Dataset()
    N = 5
    d.coords[Dim.X] = sc.Variable(dims=[Dim.X],
                                  values=["a", "b", "c", "d", "e"],
                                  unit=sc.units.m)
    d["Sample"] = sc.Variable([Dim.X],
                              values=np.random.rand(N),
                              unit=sc.units.counts)
    plot(d)


def test_plot_string_and_vector_axis_labels_2d():
    N = 10
    M = 5
    vecs = []
    for i in range(N):
        vecs.append(np.random.random(3))
    d = sc.Dataset()
    d.coords[Dim.X] = sc.Variable([Dim.X],
                                  values=vecs,
                                  unit=sc.units.m,
                                  dtype=sc.dtype.vector_3_float64)
    d.coords[Dim.Y] = sc.Variable([Dim.Y],
                                  values=["a", "b", "c", "d", "e"],
                                  unit=sc.units.m)
    d["Signal"] = sc.Variable([Dim.Y, Dim.X],
                              values=np.random.random([M, N]),
                              unit=sc.units.counts)
    plot(d)


def test_plot_2d_with_dimension_of_size_1():
    N = 10
    M = 1
    x = np.arange(N, dtype=np.float64)
    y = np.arange(M, dtype=np.float64)
    z = np.arange(M + 1, dtype=np.float64)
    d = sc.Dataset()
    d.coords[Dim.X] = sc.Variable([Dim.X], values=x, unit=sc.units.m)
    d.coords[Dim.Y] = sc.Variable([Dim.Y], values=y, unit=sc.units.m)
    d.coords[Dim.Z] = sc.Variable([Dim.Z], values=z, unit=sc.units.m)
    d["a"] = sc.Variable([Dim.Y, Dim.X],
                         values=np.random.random([M, N]),
                         unit=sc.units.counts)
    d["b"] = sc.Variable([Dim.Z, Dim.X],
                         values=np.random.random([M, N]),
                         unit=sc.units.counts)
    plot(d["a"])
    plot(d["b"])


def test_sparse_data_slice_with_on_the_fly_histogram():
    N = 50
    M = 10
    var = sc.Variable(dims=[Dim.X, Dim.Tof],
                      shape=[M, sc.Dimensions.Sparse],
                      unit=sc.units.us)
    for i in range(M):
        v = np.random.normal(50.0, scale=20.0, size=int(np.random.rand() * N))
        var[Dim.X, i].values = v

    d = sc.Dataset()
    d.coords[Dim.X] = sc.Variable([Dim.X],
                                  values=np.arange(M),
                                  unit=sc.units.m)
    d['a'] = sc.DataArray(coords={Dim.Tof: var})
    d['b'] = sc.DataArray(coords={Dim.Tof: var * 1.1})
    plot(d[Dim.X, 4], bins=100)
