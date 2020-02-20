# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock
import math

import numpy as np
import pytest

import scipp as sc
from scipp import Dim


def make_variables():
    data = np.arange(1, 4, dtype=float)
    a = sc.Variable([sc.Dim.X], data)
    b = sc.Variable([sc.Dim.X], data)
    a_slice = a[sc.Dim.X, :]
    b_slice = b[sc.Dim.X, :]
    return a, b, a_slice, b_slice, data


def test_create_default():
    var = sc.Variable()
    assert var.dims == []
    assert var.sparse_dim is None
    assert var.dtype == sc.dtype.float64
    assert var.unit == sc.units.dimensionless
    assert var.value == 0.0


def test_create_default_dtype():
    var = sc.Variable([sc.Dim.X], [4])
    assert var.dtype == sc.dtype.float64


def test_create_with_dtype():
    var = sc.Variable(dims=[Dim.X], shape=[2], dtype=sc.dtype.float32)
    assert var.dtype == sc.dtype.float32


def test_create_with_numpy_dtype():
    var = sc.Variable(dims=[Dim.X], shape=[2], dtype=np.dtype(np.float32))
    assert var.dtype == sc.dtype.float32


def test_create_with_variances():
    assert sc.Variable(dims=[Dim.X], shape=[2]).variances is None
    assert sc.Variable(dims=[Dim.X], shape=[2],
                       variances=False).variances is None
    assert sc.Variable(dims=[Dim.X], shape=[2],
                       variances=True).variances is not None


def test_create_with_shape_and_variances():
    # If no values are given, variances must be Bool, cannot pass array.
    with pytest.raises(TypeError):
        sc.Variable(dims=[Dim.X], shape=[2], variances=np.arange(2))


def test_create_sparse():
    var = sc.Variable([sc.Dim.X, sc.Dim.Y], [4, sc.Dimensions.Sparse])
    assert var.dtype == sc.dtype.float64
    assert var.sparse_dim == sc.Dim.Y
    assert len(var.values) == 4
    for vals in var.values:
        assert len(vals) == 0


def test_create_from_numpy_1d():
    var = sc.Variable([sc.Dim.X], np.arange(4.0))
    assert var.dtype == sc.dtype.float64
    np.testing.assert_array_equal(var.values, np.arange(4))


def test_create_from_numpy_1d_bool():
    var = sc.Variable(dims=[sc.Dim.X], values=np.array([True, False, True]))
    assert var.dtype == sc.dtype.bool
    np.testing.assert_array_equal(var.values, np.array([True, False, True]))


def test_create_with_variances_from_numpy_1d():
    var = sc.Variable([sc.Dim.X],
                      values=np.arange(4.0),
                      variances=np.arange(4.0, 8.0))
    assert var.dtype == sc.dtype.float64
    np.testing.assert_array_equal(var.values, np.arange(4))
    np.testing.assert_array_equal(var.variances, np.arange(4, 8))


def test_create_scalar():
    var = sc.Variable(1.2)
    assert var.value == 1.2
    assert var.dims == []
    assert var.dtype == sc.dtype.float64
    assert var.unit == sc.units.dimensionless


def test_create_scalar_Dataset():
    dataset = sc.Dataset({'a': sc.Variable([sc.Dim.X], np.arange(4.0))})
    var = sc.Variable(dataset)
    assert var.value == dataset
    assert var.dims == []
    assert var.dtype == sc.dtype.Dataset
    assert var.unit == sc.units.dimensionless


def test_create_scalar_quantity():
    var = sc.Variable(1.2, unit=sc.units.m)
    assert var.value == 1.2
    assert var.dims == []
    assert var.dtype == sc.dtype.float64
    assert var.unit == sc.units.m


def test_create_via_unit():
    expected = sc.Variable(1.2, unit=sc.units.m)
    var = 1.2 * sc.units.m
    assert var == expected


def test_create_1D_string():
    var = sc.Variable(dims=[Dim.Row], values=['a', 'bb'], unit=sc.units.m)
    assert len(var.values) == 2
    assert var.values[0] == 'a'
    assert var.values[1] == 'bb'
    assert var.dims == [Dim.Row]
    assert var.dtype == sc.dtype.string
    assert var.unit == sc.units.m


def test_create_1D_vector_3_float64():
    var = sc.Variable(dims=[Dim.X],
                      values=[[1, 2, 3], [4, 5, 6]],
                      unit=sc.units.m,
                      dtype=sc.dtype.vector_3_float64)
    assert len(var.values) == 2
    np.testing.assert_array_equal(var.values[0], [1, 2, 3])
    np.testing.assert_array_equal(var.values[1], [4, 5, 6])
    assert var.dims == [Dim.X]
    assert var.dtype == sc.dtype.vector_3_float64
    assert var.unit == sc.units.m


def test_create_2D_inner_size_3():
    var = sc.Variable(dims=[Dim.X, Dim.Y],
                      values=np.arange(6.0).reshape(2, 3),
                      unit=sc.units.m)
    assert var.shape == [2, 3]
    np.testing.assert_array_equal(var.values[0], [0, 1, 2])
    np.testing.assert_array_equal(var.values[1], [3, 4, 5])
    assert var.dims == [Dim.X, Dim.Y]
    assert var.dtype == sc.dtype.float64
    assert var.unit == sc.units.m


def test_astype():
    var = sc.Variable([sc.Dim.X],
                      values=np.array([1, 2, 3, 4], dtype=np.int64))
    assert var.dtype == sc.dtype.int64

    var_as_float = var.astype(sc.dtype.float32)
    assert var_as_float.dtype == sc.dtype.float32


def test_astype_bad_conversion():
    var = sc.Variable([sc.Dim.X],
                      values=np.array([1, 2, 3, 4], dtype=np.int64))
    assert var.dtype == sc.dtype.int64

    with pytest.raises(RuntimeError):
        var.astype(sc.dtype.string)


def test_operation_with_scalar_quantity():
    reference = sc.Variable([sc.Dim.X], np.arange(4.0) * 1.5)
    reference.unit = sc.units.kg

    var = sc.Variable([sc.Dim.X], np.arange(4.0))
    var *= sc.Variable(1.5, unit=sc.units.kg)
    assert var == reference


def test_0D_scalar_access():
    var = sc.Variable()
    assert var.value == 0.0
    var.value = 1.2
    assert var.value == 1.2
    assert var.values.shape == ()
    assert var.values == 1.2


def test_0D_scalar_string():
    var = sc.Variable(value='a')
    assert var.value == 'a'
    var.value = 'b'
    assert var == sc.Variable(value='b')


def test_1D_scalar_access_fail():
    var = sc.Variable([Dim.X], (1, ))
    with pytest.raises(RuntimeError):
        assert var.value == 0.0
    with pytest.raises(RuntimeError):
        var.value = 1.2


def test_1D_access():
    var = sc.Variable([Dim.X], (2, ))
    assert len(var.values) == 2
    assert var.values.shape == (2, )
    var.values[1] = 1.2
    assert var.values[1] == 1.2


def test_1D_set_from_list():
    var = sc.Variable([Dim.X], (2, ))
    var.values = [1.0, 2.0]
    assert var == sc.Variable([Dim.X], values=[1.0, 2.0])


def test_1D_string():
    var = sc.Variable([Dim.X], values=['a', 'b'])
    assert len(var.values) == 2
    assert var.values[0] == 'a'
    assert var.values[1] == 'b'
    var.values = ['c', 'd']
    assert var == sc.Variable([Dim.X], values=['c', 'd'])


def test_1D_converting():
    var = sc.Variable([Dim.X], values=[1, 2])
    var.values = [3.3, 4.6]
    # floats get truncated
    assert var == sc.Variable([Dim.X], values=[3, 4])


def test_1D_dataset():
    var = sc.Variable([Dim.X], shape=(2, ), dtype=sc.dtype.Dataset)
    d1 = sc.Dataset({'a': 1.5 * sc.units.m})
    d2 = sc.Dataset({'a': 2.5 * sc.units.m})
    var.values = [d1, d2]
    assert var.values[0] == d1
    assert var.values[1] == d2


def test_1D_access_bad_shape_fail():
    var = sc.Variable([Dim.X], (2, ))
    with pytest.raises(RuntimeError):
        var.values = np.arange(3)


def test_2D_access():
    var = sc.Variable([Dim.X, Dim.Y], (2, 3))
    assert var.values.shape == (2, 3)
    assert len(var.values) == 2
    assert len(var.values[0]) == 3
    var.values[1] = 1.2  # numpy assigns to all elements in "slice"
    var.values[1][2] = 2.2
    assert var.values[1][0] == 1.2
    assert var.values[1][1] == 1.2
    assert var.values[1][2] == 2.2


def test_2D_access_bad_shape_fail():
    var = sc.Variable([Dim.X, Dim.Y], (2, 3))
    with pytest.raises(RuntimeError):
        var.values = np.ones(shape=(3, 2))


def test_2D_access_variances():
    var = sc.Variable([Dim.X, Dim.Y], (2, 3), variances=True)
    assert var.values.shape == (2, 3)
    assert var.variances.shape == (2, 3)
    var.values[1] = 1.2
    assert np.array_equal(var.variances, np.zeros(shape=(2, 3)))
    var.variances = np.ones(shape=(2, 3))
    assert np.array_equal(var.variances, np.ones(shape=(2, 3)))


def test_sparse_slice():
    var = sc.Variable([sc.Dim.X, sc.Dim.Y], [4, sc.Dimensions.Sparse])
    vals0 = var[Dim.X, 0].values
    assert len(vals0) == 0
    vals0.append(1.2)
    assert len(var[Dim.X, 0].values) == 1


def test_sparse_setitem():
    var = sc.Variable([sc.Dim.X, sc.Dim.Y], [4, sc.Dimensions.Sparse])
    # __setitem__ of vector
    var[Dim.X, 0].values = np.arange(4)
    assert len(var[Dim.X, 0].values) == 4
    # __setitem__ of span
    var.values[1] = np.arange(3)
    assert len(var[Dim.X, 1].values) == 3
    # __setitem__ of ElementArrayView
    var[Dim.X, :].values[2] = np.arange(2)
    assert len(var[Dim.X, 2].values) == 2


def test_sparse_setitem_sparse_fail():
    var = sc.Variable([sc.Dim.X, sc.Dim.Y], [4, sc.Dimensions.Sparse])
    with pytest.raises(RuntimeError):
        var.values = np.arange(3)


def test_sparse_setitem_shape_fail():
    var = sc.Variable([sc.Dim.X, sc.Dim.Y], [4, sc.Dimensions.Sparse])
    with pytest.raises(RuntimeError):
        var[Dim.X, 0].values = np.ones(shape=(3, 2))


def test_sparse_setitem_float():
    var = sc.Variable([sc.Dim.X, sc.Dim.Y], [4, sc.Dimensions.Sparse],
                      dtype=sc.dtype.float32)
    var[Dim.X, 0].values = np.arange(4)
    assert len(var[Dim.X, 0].values) == 4


def test_sparse_setitem_int64_t():
    var = sc.Variable([sc.Dim.X, sc.Dim.Y], [4, sc.Dimensions.Sparse],
                      dtype=sc.dtype.int64)
    var[Dim.X, 0].values = np.arange(4)
    assert len(var[Dim.X, 0].values) == 4


def test_create_dtype():
    var = sc.Variable([Dim.X], values=np.arange(4).astype(np.int64))
    assert var.dtype == sc.dtype.int64
    var = sc.Variable([Dim.X], values=np.arange(4).astype(np.int32))
    assert var.dtype == sc.dtype.int32
    var = sc.Variable([Dim.X], values=np.arange(4).astype(np.float64))
    assert var.dtype == sc.dtype.float64
    var = sc.Variable([Dim.X], values=np.arange(4).astype(np.float32))
    assert var.dtype == sc.dtype.float32
    var = sc.Variable([Dim.X], (4, ), dtype=np.dtype(np.float64))
    assert var.dtype == sc.dtype.float64
    var = sc.Variable([Dim.X], (4, ), dtype=np.dtype(np.float32))
    assert var.dtype == sc.dtype.float32
    var = sc.Variable([Dim.X], (4, ), dtype=np.dtype(np.int64))
    assert var.dtype == sc.dtype.int64
    var = sc.Variable([Dim.X], (4, ), dtype=np.dtype(np.int32))
    assert var.dtype == sc.dtype.int32


def test_get_slice():
    var = sc.Variable([Dim.X, Dim.Y], values=np.arange(0, 8).reshape(2, 4))
    var_slice = var[Dim.X, 1:2]
    assert var_slice == sc.Variable([Dim.X, Dim.Y],
                                    values=np.arange(4, 8).reshape(1, 4))


def test_slicing():
    var = sc.Variable([sc.Dim.X], values=np.arange(0, 3))
    var_slice = var[(sc.Dim.X, slice(0, 2))]
    assert isinstance(var_slice, sc.VariableView)
    assert len(var_slice.values) == 2
    assert np.array_equal(var_slice.values, np.array([0, 1]))


def test_iadd():
    expected = sc.Variable(2.2)
    a = sc.Variable(1.2)
    b = a
    a += 1.0
    assert a == expected
    assert b == expected
    # This extra check is important: It can happen that an implementation of,
    # e.g., __iadd__ does an in-place modification, updating `b`, but then the
    # return value is assigned to `a`, which could break the connection unless
    # the correct Python object is returned.
    a += 1.0
    assert a == b


def test_isub():
    expected = sc.Variable(2.2 - 1.0)
    a = sc.Variable(2.2)
    b = a
    a -= 1.0
    assert a == expected
    assert b == expected
    a -= 1.0
    assert a == b


def test_imul():
    expected = sc.Variable(2.4)
    a = sc.Variable(1.2)
    b = a
    a *= 2.0
    assert a == expected
    assert b == expected
    a *= 2.0
    assert a == b


def test_idiv():
    expected = sc.Variable(1.2)
    a = sc.Variable(2.4)
    b = a
    a /= 2.0
    assert a == expected
    assert b == expected
    a /= 2.0
    assert a == b


def test_iand():
    expected = sc.Variable(False)
    a = sc.Variable(True)
    b = a
    a &= sc.Variable(False)
    assert a == expected
    assert b == expected
    a |= sc.Variable(True)
    assert a == b


def test_ior():
    expected = sc.Variable(True)
    a = sc.Variable(False)
    b = a
    a |= sc.Variable(True)
    assert a == expected
    assert b == expected
    a &= sc.Variable(False)
    assert a == b


def test_ixor():
    expected = sc.Variable(True)
    a = sc.Variable(False)
    b = a
    a ^= sc.Variable(True)
    assert a == expected
    assert b == expected
    a ^= sc.Variable(True)
    assert a == b


def test_binary_plus():
    a, b, a_slice, b_slice, data = make_variables()
    c = a + b
    assert np.array_equal(c.values, data + data)
    c = a + 2.0
    assert np.array_equal(c.values, data + 2.0)
    c = a + b_slice
    assert np.array_equal(c.values, data + data)
    c += b
    assert np.array_equal(c.values, data + data + data)
    c += b_slice
    assert np.array_equal(c.values, data + data + data + data)
    c = 3.5 + c
    assert np.array_equal(c.values, data + data + data + data + 3.5)


def test_binary_minus():
    a, b, a_slice, b_slice, data = make_variables()
    c = a - b
    assert np.array_equal(c.values, data - data)
    c = a - 2.0
    assert np.array_equal(c.values, data - 2.0)
    c = a - b_slice
    assert np.array_equal(c.values, data - data)
    c -= b
    assert np.array_equal(c.values, data - data - data)
    c -= b_slice
    assert np.array_equal(c.values, data - data - data - data)
    c = 3.5 - c
    assert np.array_equal(c.values, 3.5 - data + data + data + data)


def test_binary_multiply():
    a, b, a_slice, b_slice, data = make_variables()
    c = a * b
    assert np.array_equal(c.values, data * data)
    c = a * 2.0
    assert np.array_equal(c.values, data * 2.0)
    c = a * b_slice
    assert np.array_equal(c.values, data * data)
    c *= b
    assert np.array_equal(c.values, data * data * data)
    c *= b_slice
    assert np.array_equal(c.values, data * data * data * data)
    c = 3.5 * c
    assert np.array_equal(c.values, data * data * data * data * 3.5)


def test_binary_divide():
    a, b, a_slice, b_slice, data = make_variables()
    c = a / b
    assert np.array_equal(c.values, data / data)
    c = a / 2.0
    assert np.array_equal(c.values, data / 2.0)
    c = a / b_slice
    assert np.array_equal(c.values, data / data)
    c /= b
    assert np.array_equal(c.values, data / data / data)
    c /= b_slice
    assert np.array_equal(c.values, data / data / data / data)


def test_in_place_binary_or():
    a = sc.Variable(False)
    b = sc.Variable(True)
    a |= b
    assert a == sc.Variable(True)

    a = sc.Variable([Dim.X], values=np.array([False, True, False, True]))
    b = sc.Variable([Dim.X], values=np.array([False, False, True, True]))
    a |= b
    assert a == sc.Variable([Dim.X],
                            values=np.array([False, True, True, True]))


def test_binary_or():
    a = sc.Variable(False)
    b = sc.Variable(True)
    assert (a | b) == sc.Variable(True)

    a = sc.Variable([Dim.X], values=np.array([False, True, False, True]))
    b = sc.Variable([Dim.X], values=np.array([False, False, True, True]))
    assert (a | b) == sc.Variable([Dim.X],
                                  values=np.array([False, True, True, True]))


def test_in_place_binary_and():
    a = sc.Variable(False)
    b = sc.Variable(True)
    a &= b
    assert a == sc.Variable(False)

    a = sc.Variable([Dim.X], values=np.array([False, True, False, True]))
    b = sc.Variable([Dim.X], values=np.array([False, False, True, True]))
    a &= b
    assert a == sc.Variable([Dim.X],
                            values=np.array([False, False, False, True]))


def test_binary_and():
    a = sc.Variable(False)
    b = sc.Variable(True)
    assert (a & b) == sc.Variable(False)

    a = sc.Variable([Dim.X], values=np.array([False, True, False, True]))
    b = sc.Variable([Dim.X], values=np.array([False, False, True, True]))
    assert (a & b) == sc.Variable([Dim.X],
                                  values=np.array([False, False, False, True]))


def test_in_place_binary_xor():
    a = sc.Variable(False)
    b = sc.Variable(True)
    a ^= b
    assert a == sc.Variable(True)

    a = sc.Variable([Dim.X], values=np.array([False, True, False, True]))
    b = sc.Variable([Dim.X], values=np.array([False, False, True, True]))
    a ^= b
    assert a == sc.Variable([Dim.X],
                            values=np.array([False, True, True, False]))


def test_binary_xor():
    a = sc.Variable(False)
    b = sc.Variable(True)
    assert (a ^ b) == sc.Variable(True)

    a = sc.Variable([Dim.X], values=np.array([False, True, False, True]))
    b = sc.Variable([Dim.X], values=np.array([False, False, True, True]))
    assert (a ^ b) == sc.Variable([Dim.X],
                                  values=np.array([False, True, True, False]))


def test_in_place_binary_with_scalar():
    v = sc.Variable([Dim.X], values=[10])
    copy = v.copy()

    v += 2
    v *= 2
    v -= 4
    v /= 2
    assert v == copy


def test_binary_equal():
    a, b, a_slice, b_slice, data = make_variables()
    assert a == b
    assert a == a_slice
    assert a_slice == b_slice
    assert b == a
    assert b_slice == a
    assert b_slice == a_slice


def test_binary_not_equal():
    a, b, a_slice, b_slice, data = make_variables()
    c = a + b
    assert a != c
    assert a_slice != c
    assert c != a
    assert c != a_slice


def test_abs():
    var = sc.Variable([Dim.X], values=np.array([0.1, -0.2]), unit=sc.units.m)
    expected = sc.Variable([Dim.X],
                           values=np.array([0.1, 0.2]),
                           unit=sc.units.m)
    assert sc.abs(var) == expected


def test_abs_out():
    var = sc.Variable([Dim.X], values=np.array([0.1, -0.2]), unit=sc.units.m)
    expected = sc.Variable([Dim.X],
                           values=np.array([0.1, 0.2]),
                           unit=sc.units.m)
    out = sc.abs(x=var, out=var)
    assert var == expected
    assert out == expected


def test_dot():
    a = sc.Variable(dims=[Dim.X],
                    values=[[1, 0, 0], [0, 1, 0]],
                    unit=sc.units.m,
                    dtype=sc.dtype.vector_3_float64)
    b = sc.Variable(dims=[Dim.X],
                    values=[[1, 0, 0], [1, 0, 0]],
                    unit=sc.units.m,
                    dtype=sc.dtype.vector_3_float64)
    expected = sc.Variable([Dim.X],
                           values=np.array([1.0, 0.0]),
                           unit=sc.units.m**2)
    assert sc.dot(a, b) == expected


def test_concatenate():
    var = sc.Variable([Dim.X], values=np.array([0.1, 0.2]), unit=sc.units.m)
    expected = sc.Variable([sc.Dim.X],
                           values=np.array([0.1, 0.2, 0.1, 0.2]),
                           unit=sc.units.m)
    assert sc.concatenate(var, var, Dim.X) == expected


def test_mean():
    var = sc.Variable([Dim.X, Dim.Y],
                      values=np.array([[0.1, 0.3], [0.2, 0.6]]),
                      unit=sc.units.m)
    expected = sc.Variable([Dim.X],
                           values=np.array([0.2, 0.4]),
                           unit=sc.units.m)
    assert sc.mean(var, Dim.Y) == expected


def test_mean_in_place():
    var = sc.Variable([Dim.X, Dim.Y],
                      values=np.array([[0.1, 0.3], [0.2, 0.6]]),
                      unit=sc.units.m)
    out = sc.Variable([Dim.X], values=np.array([0.0, 0.0]), unit=sc.units.m)
    expected = sc.Variable([Dim.X],
                           values=np.array([0.2, 0.4]),
                           unit=sc.units.m)
    view = sc.mean(var, Dim.Y, out)
    assert out == expected
    assert view == out


def test_norm():
    var = sc.Variable(dims=[Dim.X],
                      values=[[1, 0, 0], [3, 4, 0]],
                      unit=sc.units.m,
                      dtype=sc.dtype.vector_3_float64)
    expected = sc.Variable([Dim.X],
                           values=np.array([1.0, 5.0]),
                           unit=sc.units.m)
    assert sc.norm(var) == expected


def test_sqrt():
    var = sc.Variable([Dim.X], values=np.array([4.0, 9.0]), unit=sc.units.m**2)
    expected = sc.Variable([Dim.X],
                           values=np.array([2.0, 3.0]),
                           unit=sc.units.m)
    assert sc.sqrt(var) == expected


def test_sqrt_out():
    var = sc.Variable([Dim.X], values=np.array([4.0, 9.0]), unit=sc.units.m**2)
    expected = sc.Variable([Dim.X],
                           values=np.array([2.0, 3.0]),
                           unit=sc.units.m)
    out = sc.sqrt(x=var, out=var)
    assert var == expected
    assert out == expected


def test_sum():
    var = sc.Variable([Dim.X, Dim.Y],
                      values=np.array([[0.1, 0.3], [0.2, 0.6]]),
                      unit=sc.units.m)
    expected = sc.Variable([Dim.X],
                           values=np.array([0.4, 0.8]),
                           unit=sc.units.m)
    assert sc.sum(var, Dim.Y) == expected


def test_sum_in_place():
    var = sc.Variable([Dim.X, Dim.Y],
                      values=np.array([[0.1, 0.3], [0.2, 0.6]]),
                      unit=sc.units.m)
    out_var = sc.Variable([Dim.X],
                          values=np.array([0.0, 0.0]),
                          unit=sc.units.m)
    expected = sc.Variable([Dim.X],
                           values=np.array([0.4, 0.8]),
                           unit=sc.units.m)
    out_view = sc.sum(var, Dim.Y, out=out_var)
    assert out_var == expected
    assert out_view == expected


def test_variance_acess():
    v = sc.Variable()
    assert v.variance is None
    assert v.variances is None


def test_set_variance():
    values = np.random.rand(2, 3)
    variances = np.random.rand(2, 3)
    var = sc.Variable(dims=[Dim.X, Dim.Y], values=values)
    expected = sc.Variable(dims=[Dim.X, Dim.Y],
                           values=values,
                           variances=variances)

    assert var.variances is None
    assert var != expected

    var.variances = variances

    assert var.variances is not None
    assert var == expected


def test_copy_variance():
    values = np.random.rand(2, 3)
    variances = np.random.rand(2, 3)
    var = sc.Variable(dims=[Dim.X, Dim.Y], values=values)
    expected = sc.Variable(dims=[Dim.X, Dim.Y],
                           values=values,
                           variances=variances)

    assert var.variances is None
    assert var != expected

    var.variances = expected.variances

    assert var.variances is not None
    assert var == expected


def test_remove_variance():
    values = np.random.rand(2, 3)
    variances = np.random.rand(2, 3)
    var = sc.Variable(dims=[Dim.X, Dim.Y], values=values, variances=variances)
    expected = sc.Variable(dims=[Dim.X, Dim.Y], values=values)
    assert var.variances is not None
    var.variances = None
    assert var.variances is None
    assert var == expected


def test_set_variance_convert_dtype():
    values = np.random.rand(2, 3)
    variances = np.arange(6).reshape(2, 3)
    assert variances.dtype == np.int
    var = sc.Variable(dims=[Dim.X, Dim.Y], values=values)
    expected = sc.Variable(dims=[Dim.X, Dim.Y],
                           values=values,
                           variances=variances)

    assert var.variances is None
    assert var != expected

    var.variances = variances

    assert var.variances is not None
    assert var == expected


def test_sum_mean():
    var = sc.Variable([Dim.X], values=np.arange(5, dtype=np.int64))
    assert sc.sum(var, Dim.X) == sc.Variable(10)
    var = sc.Variable([Dim.X], values=np.arange(6, dtype=np.int64))
    assert sc.mean(var, Dim.X) == sc.Variable(2.5)


def test_make_variable_from_unit_scalar_mult_div():
    var = sc.Variable()
    var.unit = sc.units.m
    assert var == 0.0 * sc.units.m
    var.unit = sc.units.m**(-1)
    assert var == 0.0 / sc.units.m

    var = sc.Variable(np.float32())
    var.unit = sc.units.m
    assert var == np.float32(0.0) * sc.units.m
    var.unit = sc.units.m**(-1)
    assert var == np.float32(0.0) / sc.units.m


def test_construct_0d_numpy():
    v = sc.Variable([sc.Dim.X], values=np.array([0]), dtype=np.float32)
    var = sc.Variable(v[sc.Dim.X, 0])
    assert var == sc.Variable(np.float32())

    v = sc.Variable([sc.Dim.X], values=np.array([0]), dtype=np.float32)
    var = sc.Variable(v[sc.Dim.X, 0])
    var.unit = sc.units.m
    assert var == np.float32(0.0) * sc.units.m
    var.unit = sc.units.m**(-1)
    assert var == np.float32(0.0) / sc.units.m


def test_construct_0d_native_python_types():
    assert sc.Variable(2).dtype == sc.dtype.int64
    assert sc.Variable(2.0).dtype == sc.dtype.float64
    assert sc.Variable(True).dtype == sc.dtype.bool


def test_construct_0d_dtype():
    assert sc.Variable(2, dtype=np.int32).dtype == sc.dtype.int32
    assert sc.Variable(np.float64(2),
                       dtype=np.float32).dtype == sc.dtype.float32
    assert sc.Variable(1, dtype=np.bool).dtype == sc.dtype.bool


def test_rename_dims():
    values = np.arange(6).reshape(2, 3)
    xy = sc.Variable(dims=[Dim.X, Dim.Y], values=values)
    zy = sc.Variable(dims=[Dim.Z, Dim.Y], values=values)
    xy.rename_dims({Dim.X: Dim.Z})
    assert xy == zy


def test_create_1d_with_strings():
    v = sc.Variable([Dim.X], values=["aaa", "ff", "bb"])
    assert np.all(v.values == np.array(["aaa", "ff", "bb"]))


def test_bool_variable_repr():
    a = sc.Variable([Dim.X], values=np.array([False, True, True, False, True]))
    assert [expected in repr(a) for expected in ["True", "False", "..."]]


def test_reciprocal():
    var = sc.Variable([Dim.X], values=np.array([1.0, 2.0]))
    expected = sc.Variable([Dim.X], values=np.array([1.0 / 1.0, 1.0 / 2.0]))
    assert sc.reciprocal(var) == expected


def test_reciprocal_out():
    var = sc.Variable([Dim.X], values=np.array([1.0, 2.0]))
    expected = sc.Variable([Dim.X], values=np.array([1.0 / 1.0, 1.0 / 2.0]))
    out = sc.reciprocal(x=var, out=var)
    assert var == expected
    assert out == expected


def test_sin():
    var = sc.Variable([Dim.X],
                      values=np.array([0.0, math.pi]),
                      unit=sc.units.rad)
    expected = sc.Variable([Dim.X],
                           values=np.array([math.sin(0.0),
                                            math.sin(math.pi)]),
                           unit=sc.units.dimensionless)
    assert sc.sin(var) == expected


def test_sin_out():
    var = sc.Variable([Dim.X],
                      values=np.array([0.0, math.pi]),
                      unit=sc.units.rad)
    expected = sc.Variable([Dim.X],
                           values=np.array([math.sin(0.0),
                                            math.sin(math.pi)]),
                           unit=sc.units.dimensionless)
    out = sc.sin(x=var, out=var)
    assert var == expected
    assert out == expected


def test_cos():
    var = sc.Variable([Dim.X],
                      values=np.array([0.0, math.pi]),
                      unit=sc.units.rad)
    expected = sc.Variable([Dim.X],
                           values=np.array([math.cos(0.0),
                                            math.cos(math.pi)]),
                           unit=sc.units.dimensionless)
    assert sc.cos(var) == expected


def test_cos_out():
    var = sc.Variable([Dim.X],
                      values=np.array([0.0, math.pi]),
                      unit=sc.units.rad)
    expected = sc.Variable([Dim.X],
                           values=np.array([math.cos(0.0),
                                            math.cos(math.pi)]),
                           unit=sc.units.dimensionless)
    out = sc.cos(x=var, out=var)
    assert var == expected
    assert out == expected


def test_tan():
    var = sc.Variable([Dim.X],
                      values=np.array([0.0, math.pi / 2.]),
                      unit=sc.units.rad)
    expected = sc.Variable([Dim.X],
                           values=np.array(
                               [math.tan(0.0),
                                math.tan(math.pi / 2.)]),
                           unit=sc.units.dimensionless)
    assert sc.tan(var) == expected


def test_tan_out():
    var = sc.Variable([Dim.X],
                      values=np.array([0.0, math.pi / 2.]),
                      unit=sc.units.rad)
    expected = sc.Variable([Dim.X],
                           values=np.array(
                               [math.tan(0.0),
                                math.tan(math.pi / 2.)]),
                           unit=sc.units.dimensionless)
    out = sc.tan(x=var, out=var)
    assert var == expected
    assert out == expected


def test_asin():
    var = sc.Variable([Dim.X], values=np.array([0.0, 0.5]))
    expected = sc.Variable([Dim.X],
                           values=np.array([math.asin(0.0),
                                            math.asin(0.5)]),
                           unit=sc.units.rad)
    assert sc.asin(var) == expected


def test_asin_out():
    var = sc.Variable([Dim.X], values=np.array([0.0, 0.5]))
    expected = sc.Variable([Dim.X],
                           values=np.array([math.asin(0.0),
                                            math.asin(0.5)]),
                           unit=sc.units.rad)
    out = sc.asin(x=var, out=var)
    assert var == expected
    assert out == expected


def test_acos():
    var = sc.Variable([Dim.X], values=np.array([0.0, 0.5]))
    expected = sc.Variable([Dim.X],
                           values=np.array([math.acos(0.0),
                                            math.acos(0.5)]),
                           unit=sc.units.rad)
    assert sc.acos(var) == expected


def test_acos_out():
    var = sc.Variable([Dim.X], values=np.array([0.0, 0.5]))
    expected = sc.Variable([Dim.X],
                           values=np.array([math.acos(0.0),
                                            math.acos(0.5)]),
                           unit=sc.units.rad)
    out = sc.acos(x=var, out=var)
    assert var == expected
    assert out == expected


def test_atan():
    var = sc.Variable([Dim.X], values=np.array([0.0, 0.5]))
    expected = sc.Variable([Dim.X],
                           values=np.array([math.atan(0.0),
                                            math.atan(0.5)]),
                           unit=sc.units.rad)
    assert sc.atan(var) == expected


def test_atan_out():
    var = sc.Variable([Dim.X], values=np.array([0.0, 0.5]))
    expected = sc.Variable([Dim.X],
                           values=np.array([math.atan(0.0),
                                            math.atan(0.5)]),
                           unit=sc.units.rad)
    out = sc.atan(x=var, out=var)
    assert var == expected
    assert out == expected


@pytest.mark.parametrize("dims, lengths",
                         (([Dim.X], (sc.Dimensions.Sparse, )),
                          ([Dim.X, Dim.Y], (10, sc.Dimensions.Sparse)),
                          ([Dim.X, Dim.Y, Dim.Z],
                           (10, 10, sc.Dimensions.Sparse)),
                          ([Dim.X, Dim.Y, Dim.Z, Dim.Spectrum],
                           (10, 10, 10, sc.Dimensions.Sparse))))
def test_sparse_dim_has_none_shape(dims, lengths):
    data = sc.Variable(dims, shape=lengths)

    assert data.shape[-1] is None


def test_variable_data_array_binary_ops():
    a = sc.DataArray(1.0 * sc.units.m)
    var = 1.0 * sc.units.m
    assert a / var == var / a


def test_num_to_nan():
    a = sc.Variable(dims=[Dim.X], values=np.array([1, np.nan]))
    replace = sc.Variable(value=0.0)
    b = sc.nan_to_num(a, replace)
    expected = sc.Variable(dims=[Dim.X], values=np.array([1, replace.value]))
    assert b == expected


def test_num_to_nan_out():
    a = sc.Variable(dims=[Dim.X], values=np.array([1, np.nan]))
    out = sc.Variable(dims=[Dim.X], values=np.zeros(2))
    replace = sc.Variable(value=0.0)
    sc.nan_to_num(a, replace, out)
    expected = sc.Variable(dims=[Dim.X], values=np.array([1, replace.value]))
    assert out == expected
