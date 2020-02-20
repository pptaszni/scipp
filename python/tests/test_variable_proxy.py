# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock
import scipp as sc
from scipp import Dim
import numpy as np
import operator


def test_type():
    variable_slice = sc.Variable([Dim.X], np.arange(1, 10,
                                                    dtype=float))[Dim.X, :]
    assert type(variable_slice) == sc.VariableView


def test_astype():
    variable_slice = sc.Variable([Dim.X],
                                 values=np.arange(1, 10,
                                                  dtype=np.int64))[Dim.X, :]
    assert variable_slice.dtype == sc.dtype.int64

    var_as_float = variable_slice.astype(sc.dtype.float32)
    assert var_as_float.dtype == sc.dtype.float32


def apply_test_op(op, a, b, data):
    op(a, b)
    # Assume numpy operations are correct as comparitor
    op(data, b.values)
    assert np.array_equal(a.values, data)


def test_binary_operations():
    _a = sc.Variable([Dim.X], np.arange(1, 10, dtype=float))
    _b = sc.Variable([Dim.X], np.arange(1, 10, dtype=float))
    a = _a[Dim.X, :]
    b = _b[Dim.X, :]

    data = np.copy(a.values)
    c = a + b
    assert type(c) == sc.Variable
    assert np.array_equal(c.values, data + data)
    c = a - b
    assert np.array_equal(c.values, data - data)
    c = a * b
    assert np.array_equal(c.values, data * data)
    c = a / b
    assert np.array_equal(c.values, data / data)

    apply_test_op(operator.iadd, a, b, data)
    apply_test_op(operator.isub, a, b, data)
    apply_test_op(operator.imul, a, b, data)
    apply_test_op(operator.itruediv, a, b, data)


def test_binary_float_operations():
    _a = sc.Variable([Dim.X], np.arange(1, 10, dtype=float))
    a = _a[Dim.X, :]
    data = np.copy(a.values)
    c = a + 2.0
    assert np.array_equal(c.values, data + 2.0)
    c = a - 2.0
    assert np.array_equal(c.values, data - 2.0)
    c = a * 2.0
    assert np.array_equal(c.values, data * 2.0)
    c = a / 2.0
    assert np.array_equal(c.values, data / 2.0)
    c = 2.0 + a
    assert np.array_equal(c.values, data + 2.0)
    c = 2.0 - a
    assert np.array_equal(c.values, 2.0 - data)
    c = 2.0 * a
    assert np.array_equal(c.values, data * 2.0)


def test_equal_not_equal():
    _a = sc.Variable([Dim.X], np.arange(1, 10, dtype=float))
    _b = sc.Variable([Dim.X], np.arange(1, 10, dtype=float))
    a = _a[Dim.X, :]
    b = _b[Dim.X, :]
    c = a + 2.0
    assert a == b
    assert b == a
    assert a != c
    assert c != a


def test_correct_temporaries():
    v = sc.Variable([Dim.X], values=np.arange(100.0))
    b = sc.sqrt(v)[Dim.X, 0:10]
    assert len(b.values) == 10
    b = b[Dim.X, 2:5]
    assert len(b.values) == 3
