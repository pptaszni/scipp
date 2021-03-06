import numpy as np
import scipp as sc


def test_slice_dataset_with_data_only():
    d = sc.Dataset()
    d['data'] = sc.Variable(['y'], values=np.arange(10))
    sliced = d['y', :]
    assert d == sliced
    sliced = d['y', 2:6]
    assert sc.Variable(['y'], values=np.arange(2, 6)) == sliced['data'].data


def test_slice_dataset_with_coords_only():
    d = sc.Dataset()
    d.coords['y-coord'] = sc.Variable(['y'], values=np.arange(10))
    sliced = d['y', :]
    assert d == sliced
    sliced = d['y', 2:6]
    assert sc.Variable(['y'], values=np.arange(2,
                                               6)) == sliced.coords['y-coord']


def test_slice_dataset_with_attrs_only():
    d = sc.Dataset()
    d.attrs['y-attr'] = sc.Variable(['y'], values=np.arange(10))
    sliced = d['y', :]
    assert d == sliced
    sliced = d['y', 2:6]
    assert sc.Variable(['y'], values=np.arange(2, 6)) == sliced.attrs['y-attr']


def test_slice_dataset_with_masks_only():
    d = sc.Dataset()
    d.masks['y-mask'] = sc.Variable(['y'], values=np.arange(10))
    sliced = d['y', :]
    assert d == sliced
    sliced = d['y', 2:6]
    assert sc.Variable(['y'], values=np.arange(2, 6)) == sliced.masks['y-mask']
