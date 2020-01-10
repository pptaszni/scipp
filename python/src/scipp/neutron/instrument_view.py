# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

# Scipp imports
from ..config import plot as config
from ..plot.render import render_plot
from ..plot.sciplot import SciPlot
from ..plot.sparse import histogram_sparse_data, make_bins
from ..plot.tools import parse_params
from ..utils import name_with_unit, value_to_string
from .._scipp import core as sc, neutron as sn

# Other imports
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib import cm
try:
    import ipyvolume as ipv
except ImportError:
    ipv = None
import IPython.display as disp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms


def instrument_view(scipp_obj=None, bins=None, masks=None, filename=None,
                    figsize=None, aspect="equal", cmap=None, log=False,
                    vmin=None, vmax=None, size=0.1, projection="3D",
                    nan_color="#d3d3d3", continuous_update=True,
                    dim=sc.Dim.Tof):
    """
    Plot a 2D or 3D view of the instrument.
    A slider is also generated to navigate the dimension (dim) given as an
    input. Since this is designed for neutron science, the default dimension
    is neutron time-of-flight (sc.Dim.Tof), but this could be anything
    (wavelength, temperature...)

    Example:

    import scipp.neutron as sn
    sample = sn.load(filename="PG3_4844_event.nxs")
    sn.instrument_view(sample)
    """

    iv = InstrumentView(scipp_obj=scipp_obj, bins=bins, masks=masks,
                        cmap=cmap, log=log, vmin=vmin, vmax=vmax,
                        aspect=aspect, size=size, projection=projection,
                        nan_color=nan_color, filename=filename,
                        continuous_update=continuous_update, dim=dim)

    render_plot(widgets=iv.box, filename=filename, ipv=ipv)

    return SciPlot(iv.members)


class UpdatablePatchCollection(PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


class InstrumentView:

    def __init__(self, scipp_obj=None, bins=None, masks=None, cmap=None,
                 log=None, vmin=None, vmax=None, aspect=None, size=1,
                 projection=None, nan_color=None, filename=None,
                 continuous_update=None, dim=None):

        self.fig2d = None
        self.fig = None
        self.scatter2d = None
        self.scatter3d = None
        self.outline = None
        self.size = size
        self.aspect = aspect
        self.do_update = None
        self.figurewidget = widgets.Output()
        self.figure2d = False
        self.figure3d = False
        self.image = None
        self.nan_color = nan_color
        self.log = log
        self.current_projection = None
        self.dim = dim
        self.patches = None

        self.data_arrays = {}
        tp = type(scipp_obj)
        if tp is sc.Dataset or tp is sc.DatasetProxy:
            for key, var in sorted(scipp_obj):
                if self.dim in var.dims:
                    self.data_arrays[key] = var
        elif tp is sc.DataArray or tp is sc.DataProxy:
            self.data_arrays[scipp_obj.name] = scipp_obj
        else:
            raise RuntimeError("Unknown input type: {}. Allowed inputs "
                               "are a Dataset or a DataArray (and their "
                               "respective proxies).".format(tp))

        self.globs = {"cmap": cmap, "log": log, "vmin": vmin, "vmax": vmax}
        self.params = {}
        self.hist_data_array = {}
        self.scalar_map = {}
        self.minmax = {}

        # Find the min/max time-of-flight limits and store them
        self.minmax["tof"] = [np.Inf, np.NINF, 1]
        for key, data_array in self.data_arrays.items():
            bins_here = bins
            if data_array.sparse_dim is not None and bins_here is None:
                bins_here = True
            if bins_here is not None:
                dim = None if data_array.sparse_dim is not None else self.dim
                spdim = None if data_array.sparse_dim is None else self.dim
                var = make_bins(
                    data_array=data_array, sparse_dim=spdim, dim=dim,
                    bins=bins_here,
                    padding=(data_array.sparse_dim is not None))
            else:
                var = data_array.coords[self.dim]
            self.minmax["tof"][0] = min(self.minmax["tof"][0], var.values[0])
            self.minmax["tof"][1] = max(self.minmax["tof"][1], var.values[-1])
            self.minmax["tof"][2] = var.shape[0]

        # Rebin all DataArrays to common Tof axis
        self.rebin_data(np.linspace(*self.minmax["tof"]))

        # Create dropdown menu to select the DataArray
        keys = list(self.hist_data_array.keys())
        self.dropdown = widgets.Dropdown(
            options=keys, description="Select entry:",
            layout={"width": "initial"})
        self.dropdown.observe(self.change_data_array, names="value")

        # Store current active data entry (DataArray)
        self.key = keys[0]

        # Create a Tof slider and its label
        self.tof_dim_indx = self.hist_data_array[self.key].dims.index(
            self.dim)
        self.slider = widgets.IntSlider(
            value=0, min=0, step=1,
            description=str(self.dim).replace("Dim.", ""),
            max=self.hist_data_array[self.key].shape[self.tof_dim_indx] - 1,
            continuous_update=continuous_update, readout=False)
        self.slider.observe(self.update_colors, names="value")
        self.label = widgets.Label()

        # Add text boxes to change number of bins/bin size
        # Here we use the generic Text widget as opposed to IntText and
        # FloatTextbecause it provides the 'on_submit' method which allows us
        # to modify the value of the bin width according to the number of bins
        # and vice versa without executing the rebinning twice, which would
        # happen with 'observe' since any change of value triggers the rebin.
        self.nbins = widgets.Text(
            value=str(self.hist_data_array[self.key].shape[self.tof_dim_indx]),
            description="Number of bins:",
            style={"description_width": "initial"},
            layout={"width": "170px"})
        self.nbins.on_submit(self.update_nbins)

        tof_values = self.hist_data_array[self.key].coords[self.dim].values
        self.bin_size = widgets.Text(
            value=str(tof_values[1] - tof_values[0]),
            description="Bin size:",
            layout={"width": "250px"})
        self.bin_size.on_submit(self.update_bin_size)

        self.detector_size = widgets.BoundedFloatText(
            value=self.size, min=0, description="Detector size",
            style={"description_width": "initial"},
            layout={"width": "170px"})
        self.detector_size.observe(self.change_detector_size, names="value")

        projections = ["3D", "Cylindrical X", "Cylindrical Y", "Cylindrical Z",
                       "Spherical X", "Spherical Y", "Spherical Z"]

        # Create toggle buttons to change projection
        self.buttons = {}
        for p in projections:
            self.buttons[p] = widgets.Button(
                description=p, disabled=False,
                button_style=("info" if (p == projection) else ""))
            self.buttons[p].on_click(self.change_projection)
        items = [self.buttons["3D"]]
        for x in "XYZ":
            items.append(self.buttons["Cylindrical {}".format(x)])
            items.append(self.buttons["Spherical {}".format(x)])
            if x != "Z":
                items.append(widgets.Label())

        self.togglebuttons = widgets.GridBox(
            items,
            layout=widgets.Layout(grid_template_columns="repeat(3, 150px)"))






        # Get detector positions
        self.det_pos = np.array(
            sn.position(self.hist_data_array[self.key]).values)
        # Find extents of the detectors
        for i, x in enumerate("xyz"):
            self.minmax[x] = [np.amin(self.det_pos[:, i]),
                              np.amax(self.det_pos[:, i])]
        self.box_size = {"x": 0, "y": 0, "z": 0}
        for ax in self.box_size.keys():
            self.box_size[ax] = np.ediff1d(self.minmax[ax])[0]
        self.box_size["max3d"] = np.amax(list(self.box_size.values()))




        self.fig = ipv.figure(width=config.width, height=config.height,
                                    animation=0, lighting=False)
        # if self.aspect == "equal":
        #     arrays = {}
        #     for ax in "xyz":
        #         diff = self.box_size["max"] - self.box_size[ax]
        #         arrays[ax] = [self.minmax[ax][0] - 0.5 * diff,
        #                       self.minmax[ax][1] + 0.5 * diff]

        #     outl_x, outl_y, outl_z = np.meshgrid(arrays["x"], arrays["y"],
        #                                          arrays["z"],
        #                                          indexing="ij")
        #     # self.outline = ipv.plot_wireframe(outl_x, outl_y, outl_z,
        #     #                                   color="black")

        perc_size = 100.0 * self.size / self.box_size["max3d"]
        self.scatter3d = ipv.scatter(x=np.zeros([1]),
                                     y=np.zeros([1]),
                                     z=np.zeros([1]),
                                     marker="square_2d", size=perc_size)

        # Place widgets in boxes
        self.vbox = widgets.VBox(
            [ipv.gcc(), widgets.HBox([self.dropdown, self.slider, self.label]),
             widgets.HBox([self.nbins, self.bin_size, self.detector_size]),
             self.togglebuttons])
        self.box = widgets.VBox([self.figurewidget, self.vbox])
        self.box.layout.align_items = "center"

        # # Protect against uninstalled ipyvolume
        # if ipv is None and projection == "3D":
        #     print("Warning: 3D projection requires ipyvolume to be "
        #           "installed. Use conda/pip install ipyvolume. Reverting to "
        #           "2D projection.")
        #     self.buttons[projections[1]].button_style = "info"
        #     self.buttons["3D"].button_style = ""
        #     self.buttons["3D"].disabled = True

        # # Render the plot here instead of at the top level because to capture
        # # the matplotlib output (if a 2D projection is requested to begin with,
        # # the output widget needs to be displayed first, before any mpl figure
        # # is displayed.
        # render_plot(widgets=self.box, filename=filename, ipv=ipv)

        # # Get detector positions
        # self.det_pos = np.array(
        #     sn.position(self.hist_data_array[self.key]).values)
        # # Find extents of the detectors
        # for i, x in enumerate("xyz"):
        #     self.minmax[x] = [np.amin(self.det_pos[:, i]),
        #                       np.amax(self.det_pos[:, i])]
        # self.box_size = {"x": 0, "y": 0, "z": 0}
        # for ax in self.box_size.keys():
        #     self.box_size[ax] = np.ediff1d(self.minmax[ax])
        # self.box_size["max"] = np.amax(list(self.box_size.values()))
        # # max_size = 0.0
        # #     dx = {"x": 0, "y": 0, "z": 0}
        # #     for ax in dx.keys():
        # #         dx[ax] = np.ediff1d(self.minmax[ax])
        # #     max_size = np.amax(list(dx.values()))

        # Update the figure
        self.change_projection(self.buttons[projection])

        # Create members object
        self.members = {"widgets": {"sliders": self.slider,
                                    "buttons": self.buttons,
                                    "text": {"nbins": self.nbins,
                                             "bin_size": self.bin_size},
                                    "dropdown": self.dropdown
                                    },
                        "fig2d": self.fig2d, "fig": self.fig,
                        "scatter2d": self.scatter2d,
                        "scatter3d": self.scatter3d,
                        "outline": self.outline}

        return

    def rebin_data(self, bins):
        """
        Rebin the original data to Tof given some bins. This is executed both
        on first instrument display and when either the number of bins or the
        bin width is changed.
        """
        for key, data_array in self.data_arrays.items():

            # Histogram the data in the Tof dimension
            if data_array.sparse_dim is not None:
                self.hist_data_array[key] = histogram_sparse_data(
                    data_array, data_array.sparse_dim, bins)
            else:
                self.hist_data_array[key] = sc.rebin(
                    data_array, self.dim,
                    make_bins(data_array=data_array, dim=self.dim,
                              bins=bins, padding=False))

            # Parse input parameters for colorbar
            self.params[key] = parse_params(
                globs=self.globs, array=self.hist_data_array[key].values)
            cmap = cm.get_cmap(self.params[key]["cmap"])
            cmap.set_bad(color=self.nan_color)
            self.scalar_map[key] = cm.ScalarMappable(
                cmap=cmap, norm=self.params[key]["norm"])
        return

    def update_colors(self, change):
        # self.do_update(change)

        arr = self.hist_data_array[self.key][self.dim, change["new"]].values
        if self.log:
            arr = np.ma.masked_where(arr <= 0, arr)
        self.scatter3d.color = self.scalar_map[self.key].to_rgba(arr)

        self.label.value = name_with_unit(
            var=self.hist_data_array[self.key].coords[self.dim],
            name=value_to_string(
                self.hist_data_array[self.key].coords[
                    self.dim].values[change["new"]]))
        return

    def change_projection(self, owner):

        if owner.description == self.current_projection:
            owner.button_style = "info"
            return
        if self.current_projection is not None:
            self.buttons[self.current_projection].button_style = ""

        # # Temporarily disable automatic plotting in notebook
        # if plt.isinteractive():
        #     plt.ioff()
        #     re_enable_interactive = True
        # else:
        #     re_enable_interactive = False


        update_children = False

        if owner.description == "3D":
            self.projection_3d()
            # self.do_update = self.update_colors_3d
        else:
            # if self.current_projection == "3D" or \
            #    self.current_projection is None:
            #     update_children = True
            self.projection_2d(owner.description)#, update_children)
            # self.do_update = self.update_colors_2d

        self.update_colors({"new": self.slider.value})

        self.current_projection = owner.description
        self.buttons[owner.description].button_style = "info"

        # # Re-enable automatic plotting in notebook
        # if re_enable_interactive:
        #     plt.ion()

        return

    def projection_3d(self):
        # # Initialise Figure
        # if not self.figure3d:
        #     self.fig3d = ipv.figure(width=config.width, height=config.height,
        #                             animation=0, lighting=False)
        #     # max_size = 0.0
        #     # dx = {"x": 0, "y": 0, "z": 0}
        #     # for ax in dx.keys():
        #     #     dx[ax] = np.ediff1d(self.minmax[ax])
        #     # max_size = np.amax(list(dx.values()))
        #     # Make plot outline if aspect ratio is to be conserved
        if self.aspect == "equal":
            arrays = {}
            for ax in "xyz":
                diff = self.box_size["max3d"] - self.box_size[ax]
                arrays[ax] = [self.minmax[ax][0] - 0.5 * diff,
                              self.minmax[ax][1] + 0.5 * diff]

        self.box_size["max"] = self.box_size["max3d"]

            # outl_x, outl_y, outl_z = np.meshgrid(arrays["x"], arrays["y"],
            #                                      arrays["z"],
            #                                      indexing="ij")
        #         self.outline = ipv.plot_wireframe(outl_x, outl_y, outl_z,
        #                                           color="black")
        #     # Try to guess marker size
        #     perc_size = 100.0 * self.size / self.box_size["max"]
        #     self.scatter3d = ipv.scatter(x=self.det_pos[:, 0],
        #                                  y=self.det_pos[:, 1],
        #                                  z=self.det_pos[:, 2],
        #                                  marker="square_2d", size=perc_size)
        #     self.figure3d = True

        # # self.figurewidget.clear_output()
        # # self.box.children = tuple([self.figurewidget, ipv.gcc(), self.vbox])

        self.scatter3d.x = self.det_pos[:, 0]
        self.scatter3d.y = self.det_pos[:, 1]
        self.scatter3d.z = self.det_pos[:, 2]

        

        self.fig.xlim = arrays["x"]
        self.fig.ylim = arrays["y"]
        self.fig.zlim = arrays["z"]
        ipv.xyzlabel("X", "Y", "Z")
        self.change_detector_size({"new": self.size})

        return

    # def update_colors_3d(self, change):
    #     arr = self.hist_data_array[self.key][self.dim, change["new"]].values
    #     if self.log:
    #         arr = np.ma.masked_where(arr <= 0, arr)
    #     self.scatter3d.color = self.scalar_map[self.key].to_rgba(arr)
    #     return

    def projection_2d(self, projection):
        # # Initialise figure if we switched from 3D view, if not re-use current
        # # figure.
        # if update_children:
        #     self.box.children = tuple([self.figurewidget, self.vbox])
        # if not self.figure2d:
        #     self.fig2d, self.ax = plt.subplots(
        #         1, 1, figsize=(config.width / config.dpi,
        #                        config.height / config.dpi))

        # if update_children:
        #     with self.figurewidget:
        #         disp.display(self.fig2d)

        # Compute cylindrical or spherical projections
        permutations = {"X": [0, 2, 1], "Y": [1, 0, 2], "Z": [2, 1, 0]}
        axis = projection[-1]

        theta = np.arctan2(self.det_pos[:, permutations[axis][2]],
                                self.det_pos[:, permutations[axis][1]])
        if projection.startswith("Cylindrical"):
            z_or_phi = self.det_pos[:, permutations[axis][0]]
            ytitle = "Z"
        elif projection.startswith("Spherical"):
            z_or_phi = np.arcsin(self.det_pos[:, permutations[axis][0]] /
                                 np.sqrt(self.det_pos[:, 0]**2 +
                                         self.det_pos[:, 1]**2 +
                                         self.det_pos[:, 2]**2))
            ytitle = "Phi"

        # self.theta = theta
        # self.z_or_phi = z_or_phi

        self.scatter3d.x = theta
        self.scatter3d.y = z_or_phi
        self.scatter3d.z = np.zeros_like(theta)

        self.minmax["theta"] = [np.amin(theta), np.amax(theta)]
        self.minmax["z_or_phi"] = [np.amin(z_or_phi), np.amax(z_or_phi)]
        for ax in ["theta", "z_or_phi"]:
            self.box_size[ax] = np.ediff1d(self.minmax[ax])[0]
        self.box_size["max2d"] = max(self.box_size["theta"], self.box_size["z_or_phi"])


        # if self.aspect == "equal":
        arrays = {}
        for ax in ["theta", "z_or_phi"]:
            diff = self.box_size["max2d"] - self.box_size[ax]
            arrays[ax] = [self.minmax[ax][0] - 0.5 * diff,
                          self.minmax[ax][1] + 0.5 * diff]
        print(self.box_size)
        print(arrays["theta"])
        print(arrays["z_or_phi"])

        # ipv.xyzlim(arrays["theta"], arrays["z_or_phi"], [-0.5*self.box_size["max2d"], 0.5*self.box_size["max2d"]])
        self.fig.xlim = arrays["theta"]
        self.fig.ylim = arrays["z_or_phi"]
        self.fig.zlim = [-0.5*self.box_size["max2d"], 0.5*self.box_size["max2d"]]
        
        ipv.xyzlabel("Theta", ytitle, " ")

        self.box_size["max"] = self.box_size["max2d"]

        self.change_detector_size({"new": self.size})

        # self.fig.xlim = arrays["theta"]
        # self.fig.ylim = arrays["z_or_phi"]
        # self.fig.zlim = [-0.5*self.box_size["max2d"], 0.5*self.box_size["max2d"]]

            # outl_x, outl_y, outl_z = np.meshgrid(arrays["x"], arrays["y"],
            #                                      arrays["z"],
            #                                      indexing="ij")

        # # Create the scatter
        # if not self.figure2d:
        #     # self.patches = []
        #     # for x, y in zip(theta, z_or_phi):
        #     #     self.patches.append(Rectangle((x-0.5*self.size, y-0.5*self.size),
        #     #                              self.size, self.size))

        #     self.scatter2d = self.generate_patch_collection()
        #     #     cmap=self.params[self.key]["cmap"],
        #     #     norm=self.params[self.key]["norm"],
        #     #     array=self.hist_data_array[
        #     #         self.key][self.dim, self.slider.value].values)
        #     self.ax.add_collection(self.scatter2d)
        #     self.save_xy = np.array([theta, z_or_phi]).T
        #     if self.params[self.key]["cbar"]:
        #         self.cbar = plt.colorbar(self.scatter2d, ax=self.ax)
        #         self.cbar.ax.set_ylabel(
        #             name_with_unit(var=self.hist_data_array[self.key],
        #                            name=""))
        #         self.cbar.ax.yaxis.set_label_coords(-1.1, 0.5)
        #     self.figure2d = True
        # else:
        #     self.scatter2d.set_offset_position("data")
        #     self.scatter2d.set_offsets(np.array([self.theta, self.z_or_phi]).T -
        #                                self.save_xy)
        # self.set_2d_axis_limits()
        return

    # def generate_patch_collection(self):
    #     patches = []
    #     for x, y in zip(self.theta, self.z_or_phi):
    #         patches.append(Rectangle((x-0.5*self.size, y-0.5*self.size),
    #                                  self.size, self.size))
    #     return PatchCollection(
    #             patches, cmap=self.params[self.key]["cmap"],
    #             norm=self.params[self.key]["norm"],
    #             array=self.hist_data_array[
    #                 self.key][self.dim, self.slider.value].values)

    # def set_2d_axis_limits(self):
    #     self.ax.set_xlim([np.amin(self.theta) - self.size,
    #                       np.amax(self.theta) + self.size])
    #     self.ax.set_ylim([np.amin(self.z_or_phi) - self.size,
    #                       np.amax(self.z_or_phi) + self.size])

    # def update_colors_2d(self, change):
    #     self.scatter2d.set_array(
    #         self.hist_data_array[self.key][self.dim, change["new"]].values)
    #     self.fig2d.canvas.draw_idle()
    #     return

    def update_nbins(self, owner):
        try:
            nbins = int(owner.value)
        except ValueError:
            print("Warning: could not convert value: {} to an "
                  "integer.".format(owner.value))
            return
        # self.rebin_data(nbins, from_nbins_text)
        self.rebin_data(np.linspace(
            self.minmax["tof"][0], self.minmax["tof"][1], nbins + 1))
        x = self.hist_data_array[self.key].coords[self.dim].values
        self.bin_size.value = str(x[1] - x[0])
        self.update_slider()
        return

    def update_bin_size(self, owner):
        try:
            binw = float(owner.value)
        except ValueError:
            print("Warning: could not convert value: {} to a "
                  "float.".format(owner.value))
            return
        self.rebin_data(np.arange(
            self.minmax["tof"][0], self.minmax["tof"][1], binw))
        self.nbins.value = str(self.hist_data_array[self.key].shape[
            self.tof_dim_indx])
        self.update_slider()
        return

    def update_slider(self):
        """
        Try to replace the slider around the same position
        """

        # Compute percentage position
        perc_pos = self.slider.value / self.slider.max
        # Compute new position
        nbins = int(self.nbins.value)
        new_pos = int(perc_pos * nbins)
        # Either place new upper boundary first, or change slider value first
        if new_pos > self.slider.max:
            self.slider.max = nbins
            self.slider.value = new_pos
        else:
            self.slider.value = new_pos
            self.slider.max = nbins
        return

    def change_data_array(self, change):
        self.key = change["new"]
        # if self.scatter2d is not None:
        #     # Apparently, you have to set norm, clim on PatchCollection and
        #     # clim on the colorbar to get this working. Only setting norm
        #     # seems to work only on the first change.
        #     self.scatter2d.set_norm(self.params[self.key]["norm"])
        #     self.scatter2d.set_clim(vmin=self.params[self.key]["vmin"],
        #                             vmax=self.params[self.key]["vmax"])
        #     self.cbar.set_clim(vmin=self.params[self.key]["vmin"],
        #                        vmax=self.params[self.key]["vmax"])
        self.update_colors({"new": self.slider.value})

    def change_detector_size(self, change):
        # self.scatter3d.size = 100.0 * self.size / self.box_size["max"]

        self.size = change["new"]
        self.scatter3d.size = 100.0 * self.size / self.box_size["max"]
        # if self.current_projection == "3D":
        #     self.scatter3d.size = 100.0 * self.size / self.box_size["max"]
        # else:
        #     self.ax.collections = []
        #     self.scatter2d = self.generate_patch_collection()
        #     print("Finished generating")
        #     self.ax.add_collection(self.scatter2d)
        #     print("fininshed drawing")
        #     self.set_2d_axis_limits()
        #     # # trans = transforms.Affine2D().scale(change["new"]/change["old"]) + self.ax.transData
        #     # scale = change["new"]/change["old"]
        #     # for patch in self.patches:
        #     #     patch.set_width(change["new"])
        #     #     patch.set_height(change["new"])
                
        #     #     # patch.set_xy(trans)
        #     #     # self.scatter2d.set_transform(trans)
        #     # self.fig2d.canvas.draw_idle()
        return
