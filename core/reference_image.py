"""
core/reference_image.py

The Infineon SDK training image, drawn the reference script's way.

Students should see exactly the picture the model learns from. So the image
the Collect tab saves and the image the Test and VEX AIM tabs hand the model
live are both made here, by one function, from the reference script's own
pieces: its compute_recorded() for the spectrogram, its _jet_clim() for the
color scaling, matplotlib's jet colormap as its imshow applies it, and its
orientation, with the first Doppler bin (the most negative velocity) on the top
row, as imshow's default origin="upper" puts it.

The reference plot on screen differs only by the dashed grid it draws over the
picture, which is not part of the image.

compute_recorded() is used instead of the app's own vectorised port because the
two disagree. On real captures the port settles on a different range bin for
about a third of frames, so those columns show a different slice entirely. His
function is no slower.

The one adjustable is the color floor, jet_vmin. The reference default is
-20 dB. When nothing in a capture reaches that, _jet_clim falls back to a floor
40 dB below the brightest point, which lifts the receiver noise into mid-scale,
so every idle capture comes out as full-frame noise. Reduce noise lowers the
floor to -50 dB: below every idle peak measured (-39 to -28 dB) and above the
noise (about -58 dB). Idle stays dark, and gestures show more of their faint
returns because their color window widens.
"""

import numpy as np

# The reference script's own defaults, passed to it unchanged.
REF_MAX_SPEED_M_S = 6.19405905
REF_ANTENNA = 0
REF_JET_VMIN = -20.0

# The Reduce noise floor. See the module docstring for where it comes from.
REDUCED_NOISE_JET_VMIN = -50.0

# Saved training images are 400 x 300, as they always have been, and the
# model's image processor resizes them from there. Live inference makes the
# same 400 x 300 image and hands it to the same processor, so training and
# inference go through identical resizing.
TRAINING_SIZE = (400, 300)

_reduce_noise = False


def set_reduce_noise(on: bool):
    """Switch the color floor app-wide: display, saved images and inference."""
    global _reduce_noise
    _reduce_noise = bool(on)


def get_reduce_noise() -> bool:
    return _reduce_noise


def current_jet_vmin() -> float:
    return REDUCED_NOISE_JET_VMIN if _reduce_noise else REF_JET_VMIN


def import_reference():
    """
    The vendored reference module, with matplotlib pointed at Qt first.

    Imported lazily so the app can start without matplotlib, and so the first
    import happens on whichever thread asks. The Visualize and Collect tabs
    build their reference views on the GUI thread at startup, so by the time a
    capture or inference worker calls in, the module is already loaded.
    """
    import matplotlib
    matplotlib.use("QtAgg", force=False)
    from core import doppler_spectrogram_live as ref
    return ref


def reference_spectrogram(frames) -> np.ndarray:
    """
    The reference script's spectrogram for a capture.

    frames: (n_frame, n_ant, n_chirp, n_sample)
    returns (n_frame, doppler_bins) in dB, exactly as compute_recorded gives it
    """
    ref = import_reference()
    spectrogram, _ = ref.compute_recorded(np.asarray(frames), antenna=REF_ANTENNA)
    return spectrogram


def reference_rgb(spectrogram, jet_vmin=None) -> np.ndarray:
    """
    The picture the reference plot draws, as (doppler_bins, n_frame, 3) uint8.

    Same steps as plot_recorded_spectrogram(): transpose so rows are velocity,
    take the color limits from _jet_clim, map through jet with clipping, and
    keep row 0 on top.
    """
    from matplotlib import colormaps, colors
    ref = import_reference()
    floor = current_jet_vmin() if jet_vmin is None else float(jet_vmin)
    plot_data = np.asarray(spectrogram, dtype=np.float64).T
    vmin, vmax = ref._jet_clim(plot_data, floor)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return colormaps["jet"](norm(plot_data), bytes=True)[..., :3]


def training_image(spectrogram, jet_vmin=None):
    """
    The saved training image and the live inference image: one PIL RGB image,
    TRAINING_SIZE wide by high.
    """
    from PIL import Image
    rgb = np.ascontiguousarray(reference_rgb(spectrogram, jet_vmin))
    return Image.fromarray(rgb, "RGB").resize(TRAINING_SIZE, Image.BILINEAR)
