"""
core/sample_preview.py

The Analysis tab's picture of one recorded sample, drawn the way the
Visualize tab draws the radar:

  Doppler   the reference script's LiveDopplerProcessor run over every frame
            of the sample, drawn by its LiveSpectrogramPlot with the
            reference velocity axis and the app's colour floor, exactly as
            the Visualize tab's main view draws live frames.
  range     range_map_v1.py's compute_range_map and plot_range_map, the
            script's own whole-recording figure, with the shared MTI setting
            and only the distance shown (core/range_map.py), as the Visualize
            tab's range map draws it.

compute() is plain numpy and can run on a worker thread; the render_*
functions make matplotlib figures and must run on the GUI thread. Both
renders come back as PNG bytes at a fixed figure size, so the picture is the
same whatever space the panel gives it.
"""

import io

import numpy as np

from core import range_map as RM
from core.reference_image import (
    REF_ANTENNA, REF_MAX_SPEED_M_S, current_jet_vmin, import_reference,
)

DOPPLER_FIGSIZE = (6.4, 4.8)      # inches; the Visualize tab's "Wide 6.4 x 4.8 in"
# range_map_v1.py's figure is 10 x 5 in with a constrained layout, which lays
# itself out again at any size; drawn at the spectrogram's width so the text
# stays readable in the panel.
RANGE_FIGSIZE = (6.4, 3.6)
DPI = 100


def compute(cube, mode=None) -> dict:
    """
    cube: (n_frame, n_ant, n_chirp, n_sample). Returns dict(doppler (n_frame,
    bins) dB oldest first, range_map (n_frame, n_bins) dB, range_m, n_frame,
    mode).
    """
    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError(f"expected (n_frame, n_ant, n_chirp, n_sample), got {cube.shape}")
    n_frame, _, n_chirp, n_sample = cube.shape
    ref = import_reference()
    proc = ref.LiveDopplerProcessor(n_sample=n_sample, n_chirp=n_chirp,
                                    history_length=n_frame)
    history = None
    for i in range(n_frame):
        history, _ = proc.process_frame(np.asarray(cube[i, REF_ANTENNA]))
    mode = mode or RM.mode()
    v1, _ = RM.import_range_map()
    range_map_db, range_m = v1.compute_range_map(cube, antenna=RM.ANTENNA, mode=mode)
    return {"doppler": np.array(history, copy=True), "range_map": range_map_db,
            "range_m": range_m, "n_frame": n_frame, "mode": mode}


def _png(fig, figsize, dpi):
    import matplotlib.pyplot as plt
    try:
        fig.set_size_inches(*figsize)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        return buf.getvalue()
    finally:
        plt.close(fig)


def _hide_window(fig):
    # The scripts open their figures with pyplot, which gives each a window
    # of its own; the picture is wanted, not the window.
    win = getattr(getattr(fig.canvas, "manager", None), "window", None)
    if win is not None:
        win.hide()


def render_doppler(pre: dict, scale: float = 1.0) -> bytes:
    """The Visualize tab's spectrogram figure for the whole sample, as PNG."""
    ref = import_reference()
    n = pre["n_frame"]
    plot = ref.LiveSpectrogramPlot(
        history_length=n,
        max_speed_m_s=REF_MAX_SPEED_M_S,
        jet_vmin=current_jet_vmin(),
        title="Doppler Spectrogram",
        orientation=ref.ORIENT_FRAME_X,
    )
    _hide_window(plot.fig)
    plot.draw(pre["doppler"], frame_end=n)
    return _png(plot.fig, DOPPLER_FIGSIZE, DPI * scale)


def render_range(pre: dict, scale: float = 1.0) -> bytes:
    """range_map_v1.py's figure for the whole sample, over the distance shown."""
    import matplotlib.pyplot as plt
    v1, _ = RM.import_range_map()
    s = RM.get_settings()
    lo, hi = RM.shown_bins(pre["range_m"], s["min"], s["max"])
    title = "Range map (moving)" if pre["mode"] == "mti" else "Range map (raw)"
    # Held off interactive mode so no window flashes up for this figure.
    was = plt.isinteractive()
    plt.ioff()
    try:
        fig = v1.plot_range_map(
            pre["range_map"][:, lo:hi + 1], pre["range_m"][lo:hi + 1],
            duration_s=pre["n_frame"] * RM.FRAME_TIME_S,
            jet_vmin=current_jet_vmin(), title=title,
        )
    finally:
        if was:
            plt.ion()
    return _png(fig, RANGE_FIGSIZE, DPI * scale)
