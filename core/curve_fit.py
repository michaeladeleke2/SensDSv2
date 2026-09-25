"""
core/curve_fit.py

Fit a mathematical model to a curve traced by hand over a spectrogram.

A student freezes the live display, traces the bright ridge a moving target
leaves, and fits a function to it. A Newton's cradle swinging toward and away
from the radar draws a sinusoid, so the fitted frequency can be checked against
a pendulum the student can measure with a ruler.

Freehand tracing is messy in two ways handled here: the same instant can be
traced several times over (clean_trace), and the hand wanders off the ridge
(snap_to_peak).
"""

from dataclasses import dataclass, field

import numpy as np

G = 9.81

MODEL_SINUSOID = "sinusoid"
MODEL_DAMPED = "damped_sinusoid"
MODEL_POLYNOMIAL = "polynomial"
MODEL_LINEAR = "linear"

MODEL_LABELS = {
    MODEL_SINUSOID: "Sinusoid",
    MODEL_DAMPED: "Damped sinusoid",
    MODEL_POLYNOMIAL: "Polynomial",
    MODEL_LINEAR: "Linear",
}


# ── models ───────────────────────────────────────────────────────────────────

def sinusoid(t, A, f, phi, c):
    return A * np.sin(2 * np.pi * f * t + phi) + c


def damped_sinusoid(t, A, tau, f, phi, c):
    return A * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + phi) + c


# ── trace cleanup ────────────────────────────────────────────────────────────

def clean_trace(points):
    """
    Sort a traced path by time and collapse repeated times.

    Tracing backtracks, so one instant can carry several velocities; each
    cluster of near equal times becomes its mean. Returns (t, v) arrays.
    """
    arr = np.asarray(list(points), dtype=float)
    if arr.size == 0:
        return np.empty(0), np.empty(0)
    arr = arr.reshape(-1, 2)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    t, v = arr[:, 0], arr[:, 1]
    if t.size == 1:
        return t.copy(), v.copy()

    # Scaled to the traced span so it means the same in any time window.
    span = float(t[-1] - t[0])
    tol = span * 1e-3 if span > 0 else 1e-9
    groups = np.concatenate(([0], np.cumsum(np.diff(t) > tol)))
    counts = np.bincount(groups)
    return (np.bincount(groups, weights=t) / counts,
            np.bincount(groups, weights=v) / counts)


def snap_to_peak(points, spectrogram, time_scale, vel_scale, window_bins=20):
    """
    Pull each traced point onto the strongest bin near it.

    points:      iterable of (time_s, velocity_m_s)
    spectrogram: (freq_bins, n_cols), row 0 the most negative velocity
    time_scale:  seconds per column
    vel_scale:   m/s per bin

    So the fit follows the signal rather than the steadiness of the hand.
    """
    spec = np.asarray(spectrogram)
    pts = [(float(t), float(v)) for t, v in points]
    if spec.ndim != 2 or spec.size == 0 or not time_scale or not vel_scale:
        return pts

    n_bins, n_cols = spec.shape
    half = n_bins / 2.0
    out = []
    for t, v in pts:
        col = min(n_cols - 1, max(0, int(round(t / time_scale))))
        row = int(round(v / vel_scale + half))
        lo = max(0, row - int(window_bins))
        hi = min(n_bins, row + int(window_bins) + 1)
        if hi <= lo:
            out.append((t, v))
            continue
        window = spec[lo:hi, col]
        # On a tie, which is what a window with no signal in it looks like,
        # keep the bin nearest the traced point rather than the lowest one.
        peaks = np.flatnonzero(window == window.max()) + lo
        best = int(peaks[np.argmin(np.abs(peaks - row))])
        out.append((t, (best - half) * vel_scale))
    return out


# ── result ───────────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    ok: bool
    model: str = ""
    params: tuple = ()
    param_text: str = ""
    equation: str = ""
    r_squared: float = float("nan")
    physics: str = ""
    message: str = ""
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    v: np.ndarray = field(default_factory=lambda: np.empty(0))
    predict: object = None          # callable: t -> fitted velocity


def _g(x) -> str:
    return f"{float(x):.3g}"


def _signed(x) -> str:
    """A term's sign folded into the operator, so no equation reads '+ -0.2'."""
    x = float(x)
    return f"- {_g(abs(x))}" if x < 0 else f"+ {_g(x)}"


def _r_squared(v, predicted) -> float:
    ss_res = float(np.sum((v - predicted) ** 2))
    ss_tot = float(np.sum((v - np.mean(v)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _poly_equation(coeffs) -> str:
    degree = len(coeffs) - 1
    parts = []
    for i, co in enumerate(coeffs):
        power = degree - i
        mag = _g(abs(co))
        term = mag if power == 0 else (f"{mag} t" if power == 1
                                       else f"{mag} t^{power}")
        if not parts:
            parts.append(f"-{term}" if co < 0 else term)
        else:
            parts.append(f"{'-' if co < 0 else '+'} {term}")
    return "v(t) = " + " ".join(parts)


def pendulum_readout(freq_hz) -> str:
    """Frequency, period, and the simple pendulum that would swing at it."""
    f = abs(float(freq_hz))
    if not np.isfinite(f) or f <= 0:
        return ""
    length = G / (4 * np.pi ** 2 * f ** 2)
    return (f"Frequency  {_g(f)} Hz\n"
            f"Period     {_g(1.0 / f)} s\n"
            f"Predicted length for a simple pendulum  {_g(length)} m")


# ── fitting ──────────────────────────────────────────────────────────────────

def _sinusoid_guess(t, v):
    """Amplitude from peak to peak, frequency from zero crossings, offset from
    the mean. curve_fit's defaults are all ones, which never converges here."""
    amplitude = (float(np.max(v)) - float(np.min(v))) / 2.0
    if amplitude <= 0:
        amplitude = 1.0
    offset = float(np.mean(v))
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    crossings = int(np.count_nonzero(np.diff(np.signbit(v - offset))))
    if span > 0 and crossings:
        freq = crossings / (2.0 * span)
    elif span > 0:
        freq = 1.0 / span
    else:
        freq = 1.0
    return amplitude, freq, offset, span


def _n_params(model, degree) -> int:
    if model == MODEL_SINUSOID:
        return 4
    if model == MODEL_DAMPED:
        return 5
    if model == MODEL_LINEAR:
        return 2
    return int(degree) + 1


def fit_curve(points, model=MODEL_SINUSOID, degree=2) -> FitResult:
    """
    Fit `model` to a traced path. Never raises: a fit that will not converge
    comes back as ok=False with something readable in `message`.
    """
    t, v = clean_trace(points)
    needed = _n_params(model, degree)
    if t.size < needed:
        return FitResult(
            ok=False,
            message=f"{MODEL_LABELS.get(model, model)} needs at least "
                    f"{needed} points; the trace has {t.size}.",
        )

    try:
        if model in (MODEL_SINUSOID, MODEL_DAMPED):
            from scipy.optimize import curve_fit

            amplitude, freq, offset, span = _sinusoid_guess(t, v)
            if model == MODEL_SINUSOID:
                fn = sinusoid
                p0 = [amplitude, freq, 0.0, offset]
                lower = [-np.inf, 0.0, -np.inf, -np.inf]
                upper = [np.inf, np.inf, np.inf, np.inf]
                names = ("A", "f", "phi", "c")
                units = ("m/s", "Hz", "rad", "m/s")
            else:
                fn = damped_sinusoid
                p0 = [amplitude, span if span > 0 else 1.0, freq, 0.0, offset]
                # tau must stay positive or the exponential runs away.
                lower = [-np.inf, 1e-6, 0.0, -np.inf, -np.inf]
                upper = [np.inf, np.inf, np.inf, np.inf, np.inf]
                names = ("A", "tau", "f", "phi", "c")
                units = ("m/s", "s", "Hz", "rad", "m/s")

            params, _ = curve_fit(fn, t, v, p0=p0, bounds=(lower, upper))
            params = tuple(float(p) for p in params)

            def predict(x, _fn=fn, _p=params):
                return _fn(np.asarray(x, dtype=float), *_p)

            if model == MODEL_SINUSOID:
                A, f, phi, c = params
                equation = (f"v(t) = {_g(A)} sin(2π {_g(f)} t "
                            f"{_signed(phi)}) {_signed(c)}")
            else:
                A, tau, f, phi, c = params
                equation = (f"v(t) = {_g(A)} e^(-t/{_g(tau)}) "
                            f"sin(2π {_g(f)} t {_signed(phi)}) {_signed(c)}")
            physics = pendulum_readout(f)
        else:
            deg = 1 if model == MODEL_LINEAR else int(degree)
            coeffs = np.polyfit(t, v, deg)
            params = tuple(float(co) for co in coeffs)

            def predict(x, _p=coeffs):
                return np.polyval(_p, np.asarray(x, dtype=float))

            equation = _poly_equation(coeffs)
            names = tuple(f"a{deg - i}" for i in range(len(coeffs)))
            units = ("",) * len(coeffs)
            physics = ""
    except Exception as e:
        return FitResult(
            ok=False,
            message=f"The fit did not converge ({type(e).__name__}). Trace "
                    f"more of the curve, or try a different model.",
        )

    param_text = "   ".join(
        f"{n} = {_g(p)}{(' ' + u) if u else ''}"
        for n, p, u in zip(names, params, units)
    )
    return FitResult(
        ok=True,
        model=model,
        params=params,
        param_text=param_text,
        equation=equation,
        r_squared=_r_squared(v, predict(t)),
        physics=physics,
        t=t,
        v=v,
        predict=predict,
    )
