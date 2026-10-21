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

Team feedback, as in the web version:
  - an Exponential model, y = A e^(k t) + c,
  - the curve can come from the range map (distance over time) as well as the
    Doppler spectrogram; quantity="range" writes r(t) in metres,
  - auto_trace finds the curve in the picture itself, and the positive and
    negative peaks of what it finds (find_peaks, peak_start) give the fit a
    starting point of its own; the start that ends with the smaller error wins.
"""

from dataclasses import dataclass, field

import numpy as np

G = 9.81

MODEL_SINUSOID = "sinusoid"
MODEL_DAMPED = "damped_sinusoid"
MODEL_POLYNOMIAL = "polynomial"
MODEL_LINEAR = "linear"
MODEL_EXP = "exponential"

MODEL_LABELS = {
    MODEL_SINUSOID: "Sinusoid",
    MODEL_DAMPED: "Damped sinusoid",
    MODEL_EXP: "Exponential",
    MODEL_POLYNOMIAL: "Polynomial",
    MODEL_LINEAR: "Linear",
}

# What the traced curve measures: speed (spectrogram) or distance (range map)
QUANTITIES = {"velocity": ("v", "m/s"), "range": ("r", "m")}


# ── models ───────────────────────────────────────────────────────────────────

def sinusoid(t, A, f, phi, c):
    return A * np.sin(2 * np.pi * f * t + phi) + c


def damped_sinusoid(t, A, tau, f, phi, c):
    return A * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + phi) + c


def exponential(t, A, k, c):
    return A * np.exp(k * t) + c


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


def snap_to_peak(points, spectrogram, time_scale, vel_scale, window_bins=20,
                 row_offset=None):
    """
    Pull each traced point onto the strongest bin near it.

    points:      iterable of (time_s, velocity_m_s)
    spectrogram: (freq_bins, n_cols), row 0 the most negative velocity
    time_scale:  seconds per column
    vel_scale:   m/s per bin (m per bin on a range map)
    row_offset:  the row of zero on the axis; the middle for velocity, 0 for
                 a range map

    So the fit follows the signal rather than the steadiness of the hand.
    """
    spec = np.asarray(spectrogram)
    pts = [(float(t), float(v)) for t, v in points]
    if spec.ndim != 2 or spec.size == 0 or not time_scale or not vel_scale:
        return pts

    n_bins, n_cols = spec.shape
    half = n_bins / 2.0 if row_offset is None else float(row_offset)
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
    quantity: str = "velocity"      # or "range"


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


def _poly_equation(coeffs, y="v") -> str:
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
    return f"{y}(t) = " + " ".join(parts)


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
    if model == MODEL_EXP:
        return 3
    if model == MODEL_LINEAR:
        return 2
    return int(degree) + 1


def exponential_guess(t, v):
    """
    Exponential start: for each rate on a grid, A and c follow from a straight
    line fit (y is linear in them once k is fixed); the rate with the smallest
    error wins. t is measured from the first point. Returns (A, k, c, cost).
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    best = (0.0, 0.0, float(np.mean(v)) if v.size else 0.0, np.inf)
    if not span > 0:
        return best
    ts = t - t[0]
    for r in 0.05 * 600.0 ** (np.arange(80) / 79.0):      # k * span from 0.05 to 30
        for sgn in (-1.0, 1.0):
            k = sgn * r / span
            x = np.exp(k * ts)
            M = np.column_stack([x, np.ones_like(x)])
            (A, c), *_ = np.linalg.lstsq(M, v, rcond=None)
            cost = float(np.sum((A * x + c - v) ** 2))
            if cost < best[3]:
                best = (float(A), float(k), float(c), cost)
    return best


# ── automatic tracing ────────────────────────────────────────────────────────

def auto_trace(spectrogram, time_scale, scale, row_offset=None,
               y_range=(-np.inf, np.inf), dead_zone=0.0):
    """
    Find the curve in the picture without a hand trace: in each column, the
    strongest bin. On the Doppler spectrogram the strongest bin moving toward
    the radar and the strongest moving away are compared and the louder one
    wins, skipping the still band around 0 m/s (dead_zone).

    spectrogram: (rows, cols), row 0 the bottom of the axis
    scale:       m/s (or m) per row; row_offset: the row of zero on the axis
    y_range:     the part of the axis to search (what is on screen)

    Columns whose peak is not clearly above the picture's background are
    skipped, and single-column jumps are pulled back to their neighbours.
    Returns [(t, value), ...].
    """
    spec = np.asarray(spectrogram, dtype=float)
    if spec.ndim != 2 or spec.size == 0:
        return []
    rows, cols = spec.shape
    off = rows / 2.0 if row_offset is None else float(row_offset)
    values = (np.arange(rows) - off) * scale
    allowed = np.flatnonzero((values >= y_range[0]) & (values <= y_range[1])
                             & (np.abs(values) >= dead_zone))
    if allowed.size == 0:
        return []
    sub = spec[allowed]
    floor = float(np.median(sub[::4]))
    best = np.argmax(sub, axis=0)
    peak_db = sub[best, np.arange(cols)]
    top = float(np.max(peak_db))
    if not (np.isfinite(top) and np.isfinite(floor)):
        return []
    threshold = floor + max(6.0, 0.35 * (top - floor))
    kept = np.flatnonzero(peak_db >= threshold)
    vals = values[allowed[best[kept]]]
    if vals.size == 0:
        return []
    # A point far from the median of its neighbours (two each side) is a jump
    # to some other reflection; it takes that median. The rest are kept as
    # found, so the peaks are not flattened.
    steps = np.abs(np.diff(vals))
    jump = max(4 * scale, 0.15 * float(np.ptp(vals)), 3 * float(np.median(steps)) if steps.size else 0.0)
    out = []
    for i, c in enumerate(kept):
        v = float(vals[i])
        if 2 <= i <= vals.size - 3:
            m = float(np.median(vals[i - 2:i + 3]))
            if abs(v - m) > jump:
                v = m
        out.append(((c + 0.5) * time_scale, v))
    return out


def find_peaks(t, v):
    """
    The positive and negative peaks of a traced curve. The curve is split
    where it crosses its middle (with a margin, so jitter near the middle does
    not count as a crossing); the highest point of each part above the middle
    and the lowest of each part below are its peaks. A part cut off by the
    edge of the trace may not reach its peak and is left out.

    Returns dict(positive=[(t, v)], negative=[(t, v)], all=[(t, v, sign)], middle).
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    n = t.size
    if n < 3:
        return {"positive": [], "negative": [], "all": [], "middle": float(v[0]) if n else 0.0}
    middle = float(np.mean(v))
    margin = 0.2 * (float(v.max()) - float(v.min())) / 2.0
    peaks, side, ext = [], 0, None
    for ti, vi in zip(t, v):
        s = 1 if vi > middle + margin else -1 if vi < middle - margin else 0
        if s and s != side:
            if ext:
                peaks.append(ext)
            side, ext = s, None
        if side and (ext is None or (vi > ext[1] if side > 0 else vi < ext[1])):
            ext = (float(ti), float(vi), side)
    if ext:
        peaks.append(ext)
    inside = [p for i, p in enumerate(peaks)
              if not ((i == 0 and p[0] == t[0]) or (i == len(peaks) - 1 and p[0] == t[-1]))]
    return {"positive": [(a, b) for a, b, s in inside if s > 0],
            "negative": [(a, b) for a, b, s in inside if s < 0],
            "all": inside, "middle": middle}


def peak_start(model, peaks):
    """
    A starting point read off the peaks: the time from a positive peak to the
    next negative one is half a swing, the height between them twice the
    amplitude, the first peak fixes the phase, and for a damped sinusoid the
    shrinking peaks give the decay time. None when the peaks cannot say.
    """
    pos, neg, allp = peaks["positive"], peaks["negative"], peaks["all"]
    if model not in (MODEL_SINUSOID, MODEL_DAMPED) or not pos or not neg or len(allp) < 2:
        return None
    halves = [b[0] - a[0] for a, b in zip(allp, allp[1:]) if a[2] != b[2]]
    if not halves:
        return None
    f = 1.0 / (2.0 * float(np.median(halves)))
    up = float(np.mean([p[1] for p in pos]))
    down = float(np.mean([p[1] for p in neg]))
    c = (up + down) / 2.0
    A = (up - down) / 2.0
    first = allp[0]
    phi = (np.pi / 2 if first[2] > 0 else -np.pi / 2) - 2 * np.pi * f * first[0]
    phi = float(np.arctan2(np.sin(phi), np.cos(phi)))
    if model == MODEL_SINUSOID:
        return [A, f, phi, c]
    # ln |peak - middle| falls on a line of slope -1/tau
    xs = np.array([p[0] for p in allp])
    ys = np.log(np.maximum(1e-9, np.abs(np.array([p[1] for p in allp]) - c)))
    tau = float(xs[-1] - xs[0]) or 1.0
    if xs.size > 1 and np.ptp(xs) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        if slope < 0:
            tau = -1.0 / slope
            A = float(np.exp(intercept))
    return [A, max(1e-3, tau), f, phi, c]


def fit_curve(points, model=MODEL_SINUSOID, degree=2, quantity="velocity",
              extra_starts=()) -> FitResult:
    """
    Fit `model` to a traced path. Never raises: a fit that will not converge
    comes back as ok=False with something readable in `message`.

    quantity:     "velocity" (v(t), m/s) or "range" (r(t), m)
    extra_starts: more starting points for the sinusoids (e.g. peak_start's);
                  the one that ends with the smallest error is kept
    """
    y, unit = QUANTITIES.get(quantity, QUANTITIES["velocity"])
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
                units = (unit, "Hz", "rad", unit)
            else:
                fn = damped_sinusoid
                p0 = [amplitude, span if span > 0 else 1.0, freq, 0.0, offset]
                # tau must stay positive or the exponential runs away.
                lower = [-np.inf, 1e-6, 0.0, -np.inf, -np.inf]
                upper = [np.inf, np.inf, np.inf, np.inf, np.inf]
                names = ("A", "tau", "f", "phi", "c")
                units = (unit, "s", "Hz", "rad", unit)

            # The desktop's start first; any extra start that ends with a
            # smaller error replaces it.
            params, best_cost, first_error = None, np.inf, None
            for start in [p0, *[s for s in extra_starts if s is not None and len(s) == len(p0)]]:
                start = [min(max(x, lo + 1e-9 if np.isfinite(lo) else x), hi) for x, lo, hi in zip(start, lower, upper)]
                try:
                    found, _ = curve_fit(fn, t, v, p0=start, bounds=(lower, upper))
                except Exception as e:
                    first_error = first_error or e
                    continue
                cost = float(np.sum((fn(t, *found) - v) ** 2))
                if cost < best_cost * (1 - 1e-9):
                    params, best_cost = found, cost
            if params is None:
                raise first_error
            params = tuple(float(p) for p in params)

            def predict(x, _fn=fn, _p=params):
                return _fn(np.asarray(x, dtype=float), *_p)

            if model == MODEL_SINUSOID:
                A, f, phi, c = params
                equation = (f"{y}(t) = {_g(A)} sin(2π {_g(f)} t "
                            f"{_signed(phi)}) {_signed(c)}")
            else:
                A, tau, f, phi, c = params
                equation = (f"{y}(t) = {_g(A)} e^(-t/{_g(tau)}) "
                            f"sin(2π {_g(f)} t {_signed(phi)}) {_signed(c)}")
            physics = pendulum_readout(f)
        elif model == MODEL_EXP:
            from scipy.optimize import curve_fit

            # Fitted with t from the first point, so e^(k t) stays in range
            # over a long window; A is moved back to t = 0 afterwards.
            A0, k0, c0, cost0 = exponential_guess(t, v)
            ts = t - t[0]
            try:
                (A0, k, c), _ = curve_fit(exponential, ts, v, p0=[A0, k0, c0], maxfev=5000)
            except Exception:
                if not np.isfinite(cost0):
                    raise
                k = k0
                c = c0
            A = float(A0) * float(np.exp(-k * t[0]))
            params = (A, float(k), float(c))

            def predict(x, _p=params):
                return exponential(np.asarray(x, dtype=float), *_p)

            equation = f"{y}(t) = {_g(A)} e^({_g(k)} t) {_signed(c)}"
            names = ("A", "k", "c")
            units = (unit, "1/s", unit)
            if k == 0:
                physics = ""
            else:
                tc = 1.0 / abs(k)
                physics = (f"{'Decays' if k < 0 else 'Grows'} with time constant {_g(tc)} s\n"
                           f"{'Half-life' if k < 0 else 'Doubling time'}  {_g(np.log(2) * tc)} s")
        else:
            deg = 1 if model == MODEL_LINEAR else int(degree)
            coeffs = np.polyfit(t, v, deg)
            params = tuple(float(co) for co in coeffs)

            def predict(x, _p=coeffs):
                return np.polyval(_p, np.asarray(x, dtype=float))

            equation = _poly_equation(coeffs, y)
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
        quantity=quantity if quantity in QUANTITIES else "velocity",
    )
