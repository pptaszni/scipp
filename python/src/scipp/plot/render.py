# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet


def render_plot(figure=None, widgets=None, filename=None, ipv=None):
    """
    Render the plot using either file export or interactive display.
    """

    # Delay imports
    import IPython.display as disp
    import matplotlib

    if filename is not None:
        if ipv is not None:
            if filename.endswith(".html"):
                ipv.save(filename)
            else:
                raise RuntimeError("Only html export is supported for now "
                                   "when using ipyvolume.")
        else:
            figure.savefig(filename, bbox_inches="tight")
    else:
        if widgets is not None and (ipv is not None
                                    or matplotlib.get_backend() == "nbAgg"):
            disp.display(widgets)
    return
