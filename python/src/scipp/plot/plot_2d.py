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

    # # masks = dict()
    # show_masks = ((len(masks.keys()) > 0) and show_masks)
    # if show_masks:


    #     for i, (key, msk) in enumerate(sorted(input_data.masks)):
    #         if msk.dims == var.dims:
    #             if mask_color is None:
    #                 col = "#{}".format(hashlib.sha1(key.encode()).hexdigest()[:6])
    #             elif isinstance(mask_color, list):
    #                 col = mask_color[i]
    #             else:
    #                 col = mask_color
    #             # print(col)
    #             data.append(
    #                 dict(x=[0.0, 1.0],
    #                     y=[0.0, 1.0],
    #                     z=[[None]],
    #                     type=plot_type,
    #                     # colorscale=[[0, 'white'], [1, col]],
    #                     colorscale="Gray",
    #                     colorbar=cbdict,
    #                     opacity=1,
    #                     hovertext=[key],
    #                     hoverinfo="text",
    #                     showscale=False,
    #                     # showlegend=True
    #                     ))
    #             masks[key] = msk
    # show_masks = ((len(masks.keys()) > 0) and show_masks)

    # show_masks = ((len(input_data.masks) > 0) and show_masks)
    # for i, (key, msk) in enumerate(sorted(input_data.masks)):
    # # if show_masks:
    #     if mask_color is None:
    #         col = "#{}".format(hashlib.sha1(key.encode()).hexdigest()[:6])
    #     elif isinstance(mask_color, list):
    #         col = mask_color[i]
    #     else:
    #         col = mask_color
    #     print(col)
    #     data.append(
    #         dict(x=[0.0, 1.0],
    #             y=[0.0, 1.0],
    #             z=[[None]],
    #             type=plot_type,
    #             colorscale=[[0, 'white'], [1, 'black']],
    #             colorbar=cbdict,
    #             opacity=0.8,
    #             hovertext=[key],
    #             hoverinfo="text",
    #             showscale=False
    #             ))


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
                         show_masks, masks, button_options=['X', 'Y'])

        self.surface3d = surface3d
        self.rasterize = rasterize
        # self.show_masks = show_masks

        # self.masks = dict()

        # show_masks = ((len(input_data.masks) > 0) and show_masks)
        self.mask = None
        if self.show_masks:
            self.mask = combine_masks(masks)
        #     for i, (key, msk) in enumerate(sorted(masks)):
        #         if msk.dims == self.input_data.dims:
        #             if mask_color is None:
        #                 col = "#{}".format(hashlib.sha1(key.encode()).hexdigest()[:6])
        #             elif isinstance(mask_color, list):
        #                 col = mask_color[i]
        #             else:
        #                 col = mask_color
        #             # print(col)
        #             data.append(
        #                 dict(x=[0.0, 1.0],
        #                     y=[0.0, 1.0],
        #                     z=[[None]],
        #                     type=plot_type,
        #                     colorscale=[[0, 'white'], [1, col]],
        #                     colorbar=cbdict,
        #                     opacity=0.6,
        #                     hovertext=[key],
        #                     hoverinfo="text",
        #                     showscale=False
        #                     ))
        #             self.masks[key] = msk
        # self.show_masks = ((len(self.masks.keys()) > 0) and show_masks)

        # self.masks = masks

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
            # Make a backup of the original trace dict
            # trace = data.copy()
            data["colorbar"]["x"] = 0.42
            data["colorbar"]["thickness"] = 0.02
            self.fig.add_trace(data, row=1, col=1)
            # trace = data.copy()
            data["colorbar"]["title"] = "variances"
            data["colorbar"]["x"] = 1.0
            data["name"] = "variances"
            self.fig.add_trace(data, row=1, col=2)
            if self.show_masks:
                data["colorscale"] = "Gray"
                data["hoverinfo"] = "none"
                data["showscale"] = False
                data["meta"] = "mask"
                data["name"] = "values"
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
            if self.show_masks:
                data["colorscale"] = "Gray"
                data["hoverinfo"] = "none"
                data["showscale"] = False
                data["meta"] = "mask"
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

                if rasterize:
                    self.scalarMap[key] = cm.ScalarMappable(
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        cmap=self.cb["name"].lower())
                    self.scalarMap[key+"_mask"] = cm.ScalarMappable(
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        cmap="gray")

                # self.fig.data[i][attr_names[0]] = vmin
                # self.fig.data[i][attr_names[1]] = vmax
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
            im_list = [go.layout.Image(xref="x", yref="y", **im_params)]
            if self.show_variances:
                im_list.append(go.layout.Image(xref="x2", yref="y2",
                                               **im_params))
            self.fig.update_layout(images=im_list)

        # Call update_slice once to make the initial image
        self.update_axes()
        self.update_slice(None)
        self.vbox = [self.fig] + self.vbox
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
                # for i in range((1 + self.show_variances) * self.show_masks*(len(self.masks.keys()))):
                #     if self.rasterize:
                #         self.fig.data[i][but_val] = \
                #             self.slider_x[key].values[[0, -1]]
                #     else:
                #         self.fig.data[i][but_val] = \
                #             self.slider_x[key].values
                if self.rasterize:
                    # self.fig.data[i][but_val] = \
                    #     self.slider_x[key].values[[0, -1]]
                    [x0, x1] = self.slider_x[key].values[[0, -1]]
                    args = {but_val: [x0, x1]}
                    # self.fig.update_traces(marker=dict(color="red"), selector=dict(meta="data", name="values"))
                    for i in range(1 + self.show_variances):
                        self.fig.layout["images"][i][but_val] = x0
                        self.fig.layout["images"][i]["size{}".format(but_val)] = \
                            x1 - x0
                        # self.fig.layout["images"][indx]["y"] = self.fig.data[indx]["y"][-1]
                        # self.fig.layout["images"][indx]["sizey"] = \
                        #     self.fig.data[indx]["y"][-1] - self.fig.data[indx]["y"][0]
                else:
                    args = {but_val: self.slider_x[key].values}
                    # self.fig.data[i][but_val] = \
                    #     self.slider_x[key].values
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
            # mslice = [var for key, var in sorted(self.masks.items())]
            mslice = self.mask

        # Slice along dimensions with active sliders
        button_dims = [None, None]
        for key, val in self.slider.items():
            if not val.disabled:
                self.lab[key].value = str(
                    self.slider_x[key].values[val.value])
                vslice = vslice[val.dim, val.value]
                if self.show_masks:
                    # for i, (key, var) in enumerate(sorted(self.masks.items())):
                    mslice = mslice[val.dim, val.value]
            else:
                button_dims[self.buttons[key].value.lower() == "y"] = val.dim

        # Check if dimensions of arrays agree, if not, plot the transpose
        slice_dims = vslice.dims
        transp = slice_dims == button_dims
        # self.update_z2d(vslice.values, transp, self.cb["log"], 0, selector=dict(meta="data", name="values"))
        if self.rasterize:
            data_colors = self.scalarMap["values"].to_rgba(self.transpose_log(vslice.values, transp,
                                                        self.cb["log"]))
            # Image is upside down by default and needs to be flipped
            img = ImageOps.flip(Image.fromarray(np.uint8(data_colors*255)))
            if self.show_masks:
                mask_colors = self.scalarMap["values_mask"].to_rgba(self.transpose_log(
                    np.where(mslice.values, vslice.values, None), transp, self.cb["log"]))
                mask_img = ImageOps.flip(Image.fromarray(np.uint8(mask_colors*255)))
                img.paste(mask_img, (0, 0), mask_img)
                # Image.alpha_composite(background, foreground)
                # background.show()
            self.fig.layout["images"][0]["source"] = img
            if self.show_variances:
                data_colors = self.scalarMap["variances"].to_rgba(self.transpose_log(vslice.variances, transp,
                                                        self.cb["log"]))
                # Image is upside down by default and needs to be flipped
                img = ImageOps.flip(Image.fromarray(np.uint8(data_colors*255)))
                if self.show_masks:
                    mask_colors = self.scalarMap["variances_mask"].to_rgba(self.transpose_log(
                        np.where(mslice.values, vslice.variances, None), transp, self.cb["log"]))
                    mask_img = ImageOps.flip(Image.fromarray(np.uint8(mask_colors*255)))
                    img.paste(mask_img, (0, 0), mask_img)
                    # background.show()
                self.fig.layout["images"][1]["source"] = img


        else:
            self.fig.update_traces(z=self.transpose_log(vslice.values, transp,
                                                        self.cb["log"]),
                                   selector=dict(meta="data", name="values"))
            if self.show_masks:
                # for i, m in enumerate(mslice):
                    # shp = np.array(np.shape(m.values))
                    # # tile = np.zeros_like(m.values, dtype=np.int)
                    # # tile[::2, 1::2] = 1
                    # tile = np.tile([[0, 1], [1, 0]], (shp//2 + 1))[:shp[0], :shp[1]]
                    # print(np.shape(tile))
                # self.update_z2d(np.where(mslice.values, vslice.values, None),
                #                 transp, False, 0, selector=dict(meta="mask", name="values"))
                self.fig.update_traces(z=self.transpose_log(
                    np.where(mslice.values, vslice.values, None), transp, self.cb["log"]),
                           selector=dict(meta="mask", name="values"))

            if self.show_variances:
                self.fig.update_traces(z=self.transpose_log(vslice.variances, transp,
                                                        self.cb["log"]),
                                   selector=dict(meta="data", name="variances"))
                # self.update_z2d(vslice.variances, transp, self.cb["log"], 1, selector=dict(meta="data", name="variances"))
                if self.show_masks:
                    self.fig.update_traces(z=self.transpose_log(
                        np.where(mslice.values, vslice.variances, None), transp, self.cb["log"]),
                               selector=dict(meta="mask", name="variances"))
                    # self.update_z2d(np.where(mslice.values, vslice.variances, None),
                    #             transp, False, 1, selector=dict(meta="mask", name="variances"))

        return

    def transpose_log(self, values, transp, log):
        if transp:
            values = values.T
        if log:
            with np.errstate(invalid="ignore", divide="ignore"):
                values = np.log10(values)
        return values

    # def update_z2d(self, values, transp, log, indx, selector):
    #     if transp:
    #         values = values.T
    #     if log:
    #         with np.errstate(invalid="ignore", divide="ignore"):
    #             values = np.log10(values)
    #     if self.rasterize:
    #         seg_colors = self.scalarMap[indx].to_rgba(values)
    #         # Image is upside down by default and needs to be flipped
    #         img = ImageOps.flip(Image.fromarray(np.uint8(seg_colors*255)))

    #         # self.fig.layout["images"][indx]["x"] = self.fig.data[indx]["x"][0]
    #         # self.fig.layout["images"][indx]["sizex"] = \
    #         #     self.fig.data[indx]["x"][-1] - self.fig.data[indx]["x"][0]
    #         # self.fig.layout["images"][indx]["y"] = self.fig.data[indx]["y"][-1]
    #         # self.fig.layout["images"][indx]["sizey"] = \
    #         #     self.fig.data[indx]["y"][-1] - self.fig.data[indx]["y"][0]
    #         self.fig.layout["images"][indx]["source"] = img
    #     else:
    #         # self.fig.data[indx].z = values
    #         self.fig.update_traces(z=values, selector=selector)
    #     return
