# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

# Scipp imports
from .. import config
from .render import render_plot
from .slicer import Slicer
from ..utils import name_with_unit
from .._scipp.core import Variable

# Other imports
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import warnings


def plot_2d(scipp_obj_dict=None,
            axes=None,
            values=None,
            variances=None,
            masks=None,
            filename=None,
            figsize=None,
            mpl_axes=None,
            aspect=None,
            cmap=None,
            log=False,
            vmin=None,
            vmax=None,
            color=None,
            logx=False,
            logy=False,
            logxy=False):
    """
    Plot a 2D slice through a N dimensional dataset. For every dimension above
    2, a slider is created to adjust the position of the slice in that
    particular dimension.
    """

    sv = Slicer2d(scipp_obj_dict=scipp_obj_dict,
                  axes=axes,
                  values=values,
                  variances=variances,
                  masks=masks,
                  mpl_axes=mpl_axes,
                  aspect=aspect,
                  cmap=cmap,
                  log=log,
                  vmin=vmin,
                  vmax=vmax,
                  color=color,
                  logx=logx or logxy,
                  logy=logy or logxy)

    if mpl_axes is None:
        render_plot(figure=sv.fig, widgets=sv.vbox, filename=filename)

    return sv.members


class Slicer2d(Slicer):
    def __init__(self,
                 scipp_obj_dict=None,
                 axes=None,
                 values=None,
                 variances=None,
                 masks=None,
                 mpl_axes=None,
                 aspect=None,
                 cmap=None,
                 log=None,
                 vmin=None,
                 vmax=None,
                 color=None,
                 logx=False,
                 logy=False):

        super().__init__(scipp_obj_dict=scipp_obj_dict,
                         axes=axes,
                         values=values,
                         variances=variances,
                         masks=masks,
                         cmap=cmap,
                         log=log,
                         vmin=vmin,
                         vmax=vmax,
                         color=color,
                         aspect=aspect,
                         button_options=['X', 'Y'])

        self.members.update({"images": {}, "colorbars": {}})
        self.extent = {"x": [0, 1], "y": [0, 1]}

        # Get or create matplotlib axes
        self.fig = None
        cax = [None] * (1 + self.params["variances"][self.name]["show"])
        if mpl_axes is not None:
            if isinstance(mpl_axes, dict):
                ax = [None, None]
                for key, val in mpl_axes.items():
                    if key == "ax" or key == "ax_values":
                        ax[0] = val
                    if key == "cax" or key == "cax_values":
                        cax[0] = val
                    if key == "ax_variances":
                        ax[1] = val
                    if key == "cax_variances":
                        cax[1] = val
            else:
                # Case where only a single axis is given
                ax = [mpl_axes]
        else:
            self.fig, ax = plt.subplots(
                1,
                1 + self.params["variances"][self.name]["show"],
                figsize=(config.plot.width / config.plot.dpi,
                         config.plot.height /
                         (1.0 + self.params["variances"][self.name]["show"]) /
                         config.plot.dpi),
                dpi=config.plot.dpi,
                sharex=True,
                sharey=True)
            if not self.params["variances"][self.name]["show"]:
                ax = [ax]

        self.ax = dict()
        self.cax = dict()
        self.im = dict()
        self.cbar = dict()

        self.ax["values"] = ax[0]
        self.cax["values"] = cax[0]
        panels = ["values"]
        if self.params["variances"][self.name]["show"]:
            self.ax["variances"] = ax[1]
            self.cax["variances"] = cax[1]
            panels.append("variances")

        extent_array = np.array(list(self.extent.values())).flatten()
        for key in panels:
            if self.params[key][self.name]["show"]:
                self.im[key] = self.ax[key].imshow(
                    [[1.0, 1.0], [1.0, 1.0]],
                    norm=self.params[key][self.name]["norm"],
                    extent=extent_array,
                    origin="lower",
                    aspect=self.aspect,
                    interpolation="nearest",
                    cmap=self.params[key][self.name]["cmap"])
                self.ax[key].set_title(self.name if key ==
                                       "values" else "std dev.")
                if self.params[key][self.name]["cbar"]:
                    self.cbar[key] = plt.colorbar(self.im[key],
                                                  ax=self.ax[key],
                                                  cax=self.cax[key])
                    self.cbar[key].ax.set_ylabel(
                        name_with_unit(var=self.data_array, name=""))
                if self.cax[key] is None:
                    self.cbar[key].ax.yaxis.set_label_coords(-1.1, 0.5)
                self.members["images"][key] = self.im[key]
                self.members["colorbars"][key] = self.cbar[key]
                if self.params["masks"][self.name]["show"]:
                    self.im[self.get_mask_key(key)] = self.ax[key].imshow(
                        [[1.0, 1.0], [1.0, 1.0]],
                        extent=extent_array,
                        norm=self.params[key][self.name]["norm"],
                        origin="lower",
                        interpolation="nearest",
                        aspect=self.aspect,
                        cmap=self.params["masks"][self.name]["cmap"])
                if logx:
                    self.ax[key].set_xscale("log")
                if logy:
                    self.ax[key].set_yscale("log")

        # Call update_slice once to make the initial image
        self.update_axes()
        self.update_slice(None)
        self.vbox = widgets.VBox(self.vbox)
        self.vbox.layout.align_items = 'center'
        self.members["fig"] = self.fig
        self.members["ax"] = self.ax

        return

    def update_buttons(self, owner, event, dummy):
        toggle_slider = False
        if not self.slider[owner.dim].disabled:
            toggle_slider = True
            self.slider[owner.dim].disabled = True
        for dim, button in self.buttons.items():
            if (button.value == owner.value) and (dim != owner.dim):
                if self.slider[dim].disabled:
                    button.value = owner.old_value
                else:
                    button.value = None
                button.old_value = button.value
                if toggle_slider:
                    self.slider[dim].disabled = False
        owner.old_value = owner.value
        self.update_axes()
        self.update_slice(None)

        return

    def update_axes(self):
        # Go through the buttons and select the right coordinates for the axes
        axparams = {"x": {}, "y": {}}
        for dim, button in self.buttons.items():
            if self.slider[dim].disabled:
                but_val = button.value.lower()
                if not self.histograms[self.name][dim]:
                    xc = self.slider_x[self.name][dim].values
                    if self.slider_nx[self.name][dim] < 2:
                        xmin = xc[0] - 0.5
                        xmax = xc[0] + 0.5
                    else:
                        xmin = 1.5 * xc[0] - 0.5 * xc[1]
                        xmax = 1.5 * xc[-1] - 0.5 * xc[-2]
                    self.extent[but_val] = [xmin, xmax]
                else:
                    self.extent[but_val] = self.slider_x[
                        self.name][dim].values[[0, -1]]

                axparams[but_val]["labels"] = name_with_unit(
                    self.slider_x[self.name][dim],
                    name=self.slider_labels[self.name][dim])
                axparams[but_val]["dim"] = dim

        extent_array = np.array(list(self.extent.values())).flatten()
        for key in self.ax.keys():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.im[key].set_extent(extent_array)
                if self.params["masks"][self.name]["show"]:
                    self.im[self.get_mask_key(key)].set_extent(extent_array)
                self.ax[key].set_xlim(self.extent["x"])
                self.ax[key].set_ylim(self.extent["y"])
            self.ax[key].set_xlabel(axparams["x"]["labels"])
            self.ax[key].set_ylabel(axparams["y"]["labels"])
            for xy, param in axparams.items():
                if self.slider_ticks[self.name][param["dim"]] is not None:
                    getattr(self.ax[key], "set_{}ticklabels".format(xy))(
                        self.get_custom_ticks(ax=self.ax[key],
                                              dim=param["dim"],
                                              xy=xy))
        return

    def update_slice(self, change):
        """
        Slice data according to new slider value.
        """
        vslice = self.data_array
        if self.params["masks"][self.name]["show"]:
            mslice = self.masks
        # Slice along dimensions with active sliders
        button_dims = [None, None]
        for dim, val in self.slider.items():
            if not val.disabled:
                self.lab[dim].value = self.make_slider_label(
                    self.slider_x[self.name][dim], val.value)
                vslice = vslice[val.dim, val.value]
                # At this point, after masks were combined, all their
                # dimensions should be contained in the data_array.dims.
                if self.params["masks"][self.name]["show"]:
                    mslice = mslice[val.dim, val.value]
            else:
                button_dims[self.buttons[dim].value.lower() == "x"] = val.dim

        # Check if dimensions of arrays agree, if not, plot the transpose
        slice_dims = vslice.dims
        transp = slice_dims != button_dims

        if self.params["masks"][self.name]["show"]:
            shape_list = [self.shapes[self.name][bdim] for bdim in button_dims]
            # Use scipp's automatic broadcast functionality to broadcast
            # lower dimension masks to higher dimensions.
            # TODO: creating a Variable here could become expensive when
            # sliders are being used. We could consider performing the
            # automatic broadcasting once and store it in the Slicer class,
            # but this could create a large memory overhead if the data is
            # large.
            # Here, the data is at most 2D, so having the Variable creation
            # and broadcasting should remain cheap.
            msk = Variable(button_dims,
                           values=np.ones(shape_list, dtype=np.int32))
            msk *= Variable(mslice.dims, values=mslice.values.astype(np.int32))

        for key in self.ax.keys():
            arr = getattr(vslice, key)
            if key == "variances":
                arr = np.sqrt(arr)
            if transp:
                arr = arr.T
            self.im[key].set_data(arr)
            if self.params["masks"][self.name]["show"]:
                self.im[self.get_mask_key(key)].set_data(
                    self.mask_to_float(msk.values, arr))

        return

    def toggle_masks(self, change):
        for key in self.ax.keys():
            self.im[key + "_masks"].set_visible(change["new"])
        change["owner"].description = "Hide masks" if change["new"] else \
            "Show masks"
        return

    def get_mask_key(self, key):
        return key + "_masks"
