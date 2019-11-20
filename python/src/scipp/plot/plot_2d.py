# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

# Scipp imports
from ..config import plot as config
from .render import render_plot
from .slicer import Slicer
from .tools import axis_label, parse_colorbar
from .._scipp.core import combine_masks


# Other imports
import numpy as np
import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image, ImageOps
from matplotlib import cm
from matplotlib.colors import Normalize
import hashlib


def plot_2d(input_data, axes=None, contours=False, cb=None, filename=None,
            name=None, figsize=None, show_variances=False, ndim=0,
            rasterize="auto", backend=None, mask_color=None, show_masks=True):
    """
    Plot a 2D slice through a N dimensional dataset. For every dimension above
    2, a slider is created to adjust the position of the slice in that
    particular dimension.
    """

    var = input_data[name]
    masks = input_data.masks
    if axes is None:
        axes = var.dims

    # Parse colorbar
    cbar = parse_colorbar(cb, plotly=True)

    # Make title
    title = axis_label(var=var, name=name, log=cbar["log"])

    if figsize is None:
        figsize = [config.width, config.height]

    layout = {"height": figsize[1], "width": figsize[0], 'hovermode': 'x'}
    if var.variances is not None and show_variances:
        layout["height"] = 0.7 * layout["height"]
        layout["xaxis2"] = {"matches": "x"}
        layout["yaxis2"] = {"matches": "y"}

    cbdict = {"title": title,
              "titleside": "right",
              "lenmode": 'fraction',
              "len": 1.05,
              "thicknessmode": 'fraction',
              "thickness": 0.03}

    # Automatically switch to rasterization if image is large
    if rasterize == "auto":
        imsize = 1
        # Find the two largest dimensions
        shapes = np.sort(var.shape)[::-1]
        for i in range(2):
            imsize *= shapes[i]
        rasterize = imsize > config.rasterize_threshold

    plot_type = 'heatmap'

    if rasterize:
        layout["xaxis"] = dict(showgrid=False, zeroline=False, autorange=True)
        layout["yaxis"] = dict(showgrid=False, zeroline=False, autorange=True)
        hoverinfo = 'skip'
    else:
        if contours:
            plot_type = 'contour'
        hoverinfo = "x+y+z"

    data = dict(x=[0.0, 1.0],
                y=[0.0, 1.0],
                z=[[0.0]],
                type=plot_type,
                colorscale=cbar["name"],
                colorbar=cbdict,
                opacity=int(not rasterize),
                hoverinfo=hoverinfo,
                meta="data",
                name="values"
                )

    sv = Slicer2d(data=data, layout=layout, input_data=var, axes=axes,
                  value_name=title, cb=cbar, show_variances=show_variances,
                  rasterize=rasterize, show_masks=show_masks, masks=masks)

    render_plot(static_fig=sv.fig, interactive_fig=sv.vbox, backend=backend,
                filename=filename)

    return


class Slicer2d(Slicer):

    def __init__(self, data, layout, input_data, axes,
                 value_name, cb, show_variances, rasterize, show_masks,
                 masks, surface3d=False):

        super().__init__(input_data, axes, value_name, cb, show_variances,
                         masks, button_options=['X', 'Y'])

        self.surface3d = surface3d
        self.rasterize = rasterize
        self.mask = None
        if self.show_masks:
            self.mask = combine_masks(masks)

        # Initialise Figure and VBox objects
        self.fig = None
        params = {"values": {"cbmin": "min", "cbmax": "max"},
                  "variances": None}
        if self.show_variances:
            params["variances"] = {"cbmin": "min_var", "cbmax": "max_var"}
            if self.surface3d:
                self.fig = go.FigureWidget(
                    make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                                  specs=[[{"type": "scene"},
                                          {"type": "scene"}]]))
            else:
                self.fig = go.FigureWidget(
                    make_subplots(rows=1, cols=2, horizontal_spacing=0.16))
            data["colorbar"]["x"] = 0.42
            data["colorbar"]["thickness"] = 0.02
            self.fig.add_trace(data, row=1, col=1)
            data["colorbar"]["title"] = "variances"
            data["colorbar"]["x"] = 1.0
            data["name"] = "variances"
            self.fig.add_trace(data, row=1, col=2)
            if self.show_masks  and not self.rasterize:
                data["colorscale"] = "Gray"
                data["hoverinfo"] = "none"
                data["showscale"] = False
                data["meta"] = "mask"
                data["name"] = "values"
                data["visible"] = show_masks
                self.fig.add_trace(data, row=1, col=1)
                data["name"] = "variances"
                self.fig.add_trace(data, row=1, col=2)

            self.fig.update_layout(**layout)
            if self.rasterize:
                self.fig.update_xaxes(row=1, col=1, **layout["xaxis"])
                self.fig.update_xaxes(row=1, col=2, **layout["xaxis"])
                self.fig.update_yaxes(row=1, col=1, **layout["yaxis"])
                self.fig.update_yaxes(row=1, col=2, **layout["yaxis"])
        else:
            self.fig = go.FigureWidget(data=[data], layout=layout)
            if self.show_masks and not self.rasterize:
                data["colorscale"] = "Gray"
                data["hoverinfo"] = "none"
                data["showscale"] = False
                data["meta"] = "mask"
                data["visible"] = show_masks
                self.fig.add_trace(data)

        # Set colorbar limits once to keep them constant for slicer
        # TODO: should there be auto scaling as slider value is changed?
        if self.surface3d:
            attr_names = ["cmin", "cmax"]
        else:
            attr_names = ["zmin", "zmax"]
        self.scalarMap = dict()
        for i, (key, val) in enumerate(sorted(params.items())):
            if val is not None:
                arr = getattr(self.input_data, key)
                if self.cb[val["cbmin"]] is not None:
                    vmin = self.cb[val["cbmin"]]
                else:
                    vmin = np.amin(arr[np.where(np.isfinite(arr))])
                if self.cb[val["cbmax"]] is not None:
                    vmax = self.cb[val["cbmax"]]
                else:
                    vmax = np.amax(arr[np.where(np.isfinite(arr))])

                if self.rasterize:
                    self.scalarMap[key] = cm.ScalarMappable(
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        cmap=self.cb["name"].lower())
                    self.scalarMap[key+"_mask"] = cm.ScalarMappable(
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        cmap="gray")

                args = {attr_names[0]: vmin, attr_names[1]: vmax, "selector": dict(name=key)}
                self.fig.update_traces(**args)

        if self.surface3d:
            self.fig.layout.scene1.zaxis.title = self.value_name
            if self.show_variances:
                self.fig.layout.scene2.zaxis.title = "variances"

        if self.rasterize:
            # Add background image
            im_params = {"opacity": 1.0, "layer": "below", "sizing": "stretch",
                         "source": None}
            im_list = [go.layout.Image(xref="x", yref="y", name="values", **im_params)]
            if self.show_masks:
                im_list.append(go.layout.Image(xref="x", yref="y", name="values_mask",
                                               **im_params))
            if self.show_variances:
                im_list.append(go.layout.Image(xref="x2", yref="y2", name="variances",
                                               **im_params))
                if self.show_masks:
                    im_list.append(go.layout.Image(xref="x2", yref="y2", name="variances_mask",
                                               **im_params))
            self.fig.update_layout(images=im_list)

        # Call update_slice once to make the initial image
        self.update_axes()
        self.update_slice(None)
        self.vbox = [self.fig] + self.vbox
        if self.show_masks:
            masks_button = widgets.ToggleButton(
                          value=show_masks,
                          description="Hide masks" if show_masks else "Show masks",
                          disabled=False,
                          button_style=""
                          )
            masks_button.observe(self.hide_show_masks, names="value")
            self.vbox += [masks_button]
        self.vbox = widgets.VBox(self.vbox)
        self.vbox.layout.align_items = 'center'

        return

    def update_buttons(self, owner, event, dummy):
        toggle_slider = False
        if not self.slider[owner.dim_str].disabled:
            toggle_slider = True
            self.slider[owner.dim_str].disabled = True
        for key, button in self.buttons.items():
            if (button.value == owner.value) and (key != owner.dim_str):
                if self.slider[key].disabled:
                    button.value = owner.old_value
                else:
                    button.value = None
                button.old_value = button.value
                if toggle_slider:
                    self.slider[key].disabled = False
        owner.old_value = owner.value
        self.update_axes()
        self.update_slice(None)

        return

    def update_axes(self):
        # Go through the buttons and select the right coordinates for the axes
        for key, button in self.buttons.items():
            if self.slider[key].disabled:
                but_val = button.value.lower()
                if self.rasterize:
                    xlims = [self.slider_x[key].values[0], self.slider_x[key].values[-1]]
                    args = {but_val: xlims}
                    for im in self.fig.layout["images"]:
                        im[but_val] = xlims[but_val == "y"]
                        im["size{}".format(but_val)] = xlims[1] - xlims[0]
                else:
                    args = {but_val: self.slider_x[key].values}
                self.fig.update_traces(**args)
                if self.surface3d:
                    self.fig.layout.scene1["{}axis_title".format(
                        but_val)] = axis_label(self.slider_x[key],
                                               name=self.slider_labels[key])
                    if self.show_variances:
                        self.fig.layout.scene2["{}axis_title".format(
                            but_val)] = axis_label(
                                self.slider_x[key],
                                name=self.slider_labels[key])
                else:
                    if self.show_variances:
                        func = getattr(self.fig, 'update_{}axes'.format(
                            but_val))
                        for i in range(2):
                            func(title_text=axis_label(
                                self.slider_x[key],
                                name=self.slider_labels[key]),
                                row=1, col=i+1)
                    else:
                        axis_str = "{}axis".format(but_val)
                        self.fig.layout[axis_str]["title"] = axis_label(
                            self.slider_x[key], name=self.slider_labels[key])

        return

    # Define function to update slices
    def update_slice(self, change):
        # The dimensions to be sliced have been saved in slider_dims
        vslice = self.input_data
        # Do we also need to slice the masks?
        if self.show_masks:
            mslice = self.mask

        # Slice along dimensions with active sliders
        button_dims = [None, None]
        for key, val in self.slider.items():
            if not val.disabled:
                self.lab[key].value = self.make_slider_label(
                    self.slider_x[key], val.value)
                vslice = vslice[val.dim, val.value]
                if self.show_masks:
                    # for i, (key, var) in enumerate(sorted(self.masks.items())):
                    mslice = mslice[val.dim, val.value]
            else:
                button_dims[self.buttons[key].value.lower() == "y"] = val.dim

        # Check if dimensions of arrays agree, if not, plot the transpose
        slice_dims = vslice.dims
        transp = slice_dims == button_dims
        mask_alpha = None
        if self.show_masks:
            mask_alpha = self.transpose_log(np.where(mslice.values, 1, 0),
                                            transp, False)
        if self.rasterize:
            im_sources = dict()
            val_array = self.transpose_log(vslice.values, transp, self.cb["log"])
            im_sources["values"] = self.to_image(val_array,
                                                 self.scalarMap["values"])

            if self.show_masks:
                im_sources["values_mask"] = self.to_image(
                    val_array, self.scalarMap["values_mask"], alpha=mask_alpha)
            if self.show_variances:
                var_array = self.transpose_log(vslice.variances, transp, self.cb["log"])
                im_sources["variances"] = self.to_image(
                    var_array, self.scalarMap["variances"])
                if self.show_masks:
                    im_sources["variances_mask"] = self.to_image(
                    var_array, self.scalarMap["variances_mask"], alpha=mask_alpha)

            for im in self.fig.layout["images"]:
                if im.name in im_sources.keys():
                    im["source"] = im_sources[im.name]


        else:
            self.update_heatmaps(vslice.values, transp,
                                 dict(meta="data", name="values"))
            if self.show_masks:
                self.update_heatmaps(
                    np.where(mslice.values, vslice.values, None), transp,
                    dict(meta="mask", name="values"))
            if self.show_variances:
                self.update_heatmaps(vslice.variances, transp,
                                 dict(meta="data", name="variances"))
                if self.show_masks:
                    self.update_heatmaps(
                        np.where(mslice.values, vslice.variances, None),
                        transp, dict(meta="mask", name="variances"))

        return

    def to_image(self, array, scal_map, alpha=None):
        data_colors = scal_map.to_rgba(array)
        if alpha is not None:
            data_colors[:,:,3] = alpha
        # Image is upside down by default and needs to be flipped
        return ImageOps.flip(Image.fromarray(np.uint8(data_colors*255)))

    def transpose_log(self, values, transp, log):
        if transp:
            values = values.T
        if log:
            with np.errstate(invalid="ignore", divide="ignore"):
                values = np.log10(values)
        return values

    def update_heatmaps(self, array, transp, selector):
        self.fig.update_traces(
            z=self.transpose_log(array, transp, self.cb["log"]),
            selector=selector)
        return

    def hide_show_masks(self, change):
        if self.rasterize:
            for im in self.fig.layout["images"]:
                if im.name.count("mask") > 0:
                    im["visible"] = change["new"]
        self.fig.update_traces(visible=change["new"],
                               selector=dict(meta="mask"))
        change["owner"].description = "Hide masks" if change["new"] else "Show masks"
        return
