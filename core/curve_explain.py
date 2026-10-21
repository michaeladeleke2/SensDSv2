"""
core/curve_explain.py

The Curve Fit result in words a student can read: the model's general form,
the same equation with the fitted numbers in it, one card per variable saying
what it means, and what the fit predicts (period, pendulum length,
half-life). core/curve_fit.py keeps its plain text; this is the tab's version
of it. Equations are Qt rich text (<i>, <sup>, <sub>).
"""

import numpy as np

from core.curve_fit import (
    G, MODEL_DAMPED, MODEL_EXP, MODEL_LINEAR, MODEL_POLYNOMIAL, MODEL_SINUSOID,
    QUANTITIES,
)

MINUS = "−"


def g(x) -> str:
    """Three significant figures, written for a student: 32,400 rather than
    3.24e+04, and 1.23 × 10⁻⁵ for very small numbers."""
    x = float(x)
    a = abs(x)
    if np.isfinite(x) and 1e4 <= a < 1e12:
        return f"{float(f'{x:.3g}'):,.0f}".replace("-", MINUS)
    if np.isfinite(x) and 0 < a < 1e-3:
        mant, exp = f"{x:.2e}".split("e")
        return f"{float(mant):.3g} × 10<sup>{int(exp)}</sup>".replace("-", MINUS)
    return f"{x:.3g}".replace("-", MINUS)


def fit_quality(r2: float):
    """(word, level) for an R²: level is 'ok', 'warn' or 'bad'."""
    if r2 >= 0.95:
        return "Excellent fit", "ok"
    if r2 >= 0.8:
        return "Good fit", "ok"
    if r2 >= 0.5:
        return "Fair fit", "warn"
    return "Poor fit", "bad"


_WHAT = {
    "velocity": dict(noun="speed", thing="speed toward or away from the radar"),
    "range": dict(noun="distance", thing="distance from the radar"),
}


def _sign(x) -> str:
    return MINUS if x < 0 else "+"


def explain_fit(r) -> dict:
    """
    r: a successful FitResult. Returns dict(general, numbers, params, results,
    quality, level, what, unit, y); params and results are lists of
    dict(sym, name, value, unit, meaning).
    """
    q = _WHAT.get(r.quantity, _WHAT["velocity"])
    ysym, unit = QUANTITIES.get(r.quantity, QUANTITIES["velocity"])
    Y = f"<i>{ysym}</i>(<i>t</i>)"
    p = r.params
    params, results = [], []
    speed = q["noun"] == "speed"

    if r.model in (MODEL_SINUSOID, MODEL_DAMPED):
        damped = r.model == MODEL_DAMPED
        if damped:
            A, tau, f, phi, c = p
        else:
            A, f, phi, c = p
            tau = None
        general = (f"{Y} = <i>A</i> · " + ("e<sup>−<i>t</i>/<i>τ</i></sup> · " if damped else "")
                   + "sin(2π · <i>f</i> · <i>t</i> + <i>φ</i>) + <i>c</i>")
        numbers = (f"{Y} = {MINUS if A < 0 else ''}{g(abs(A))} · "
                   + (f"e<sup>−<i>t</i>/{g(tau)}</sup> · " if damped else "")
                   + f"sin(2π · {g(f)} · <i>t</i> {_sign(phi)} {g(abs(phi))}) {_sign(c)} {g(abs(c))}")
        meaning_a = ("How far the speed swings above and below its middle value. A bigger swing gives a bigger A."
                     if speed else
                     "How far the object moves toward and away from the radar, measured from the middle of its swing.")
        if damped:
            meaning_a += " This is its size at t = 0, before it dies down."
        params.append(dict(sym="<i>A</i>", name="Amplitude", value=g(A), unit=unit, meaning=meaning_a))
        if damped:
            span = float(r.t[-1] - r.t[0]) if len(r.t) > 1 else 0.0
            note = (" Here τ is far longer than your trace: the swing hardly dies out at all."
                    if span > 0 and tau > 10 * span else "")
            params.append(dict(sym="<i>τ</i>", name="Decay time", value=g(tau), unit="s",
                               meaning="How long the swing lasts. After τ seconds it has shrunk to about 37 % "
                                       "of its size; a bigger τ dies out more slowly." + note))
        params.append(dict(sym="<i>f</i>", name="Frequency", value=g(f), unit="Hz",
                           meaning="How many complete back-and-forth swings happen each second."))
        params.append(dict(sym="<i>φ</i>", name="Phase", value=g(phi), unit="rad",
                           meaning="Where in its swing the motion was at t = 0. One whole swing is 2π ≈ 6.28 rad."))
        params.append(dict(sym="<i>c</i>", name="Offset", value=g(c), unit=unit,
                           meaning=("The average speed. Near 0 means the object moves toward and away from the radar equally."
                                    if speed else "The middle of the swing: the average distance from the radar.")))
        fa = abs(f)
        if fa > 0:
            results.append(dict(sym="<i>T</i> = 1/<i>f</i>", name="Period", value=g(1 / fa), unit="s",
                                meaning="Seconds for one complete swing, there and back."))
            results.append(dict(sym="<i>L</i> = <i>g</i>/(2π<i>f</i>)²", name="Pendulum length",
                                value=g(G / (4 * np.pi ** 2 * fa ** 2)), unit="m",
                                meaning="The string length of a simple pendulum that swings this fast. Measure yours to check!"))

    elif r.model == MODEL_EXP:
        A, k, c = p
        general = f"{Y} = <i>A</i> · e<sup><i>k</i>·<i>t</i></sup> + <i>c</i>"
        numbers = f"{Y} = {g(A)} · e<sup>{g(k)}·<i>t</i></sup> {_sign(c)} {g(abs(c))}"
        params += [
            dict(sym="<i>A</i>", name="Starting gap", value=g(A), unit=unit,
                 meaning=f"How far the {q['noun']} starts from its final value c, at t = 0."),
            dict(sym="<i>k</i>", name="Rate", value=g(k), unit="1/s",
                 meaning=f"How quickly the {q['noun']} changes. Negative: it shrinks toward c. "
                         "Positive: it grows. Bigger numbers mean faster change."),
            dict(sym="<i>c</i>", name="Final value", value=g(c), unit=unit,
                 meaning=f"The {q['noun']} the curve settles toward (when k is negative)."),
        ]
        if k:
            tc = 1 / abs(k)
            results.append(dict(sym="1/|<i>k</i>|", name="Time constant", value=g(tc), unit="s",
                                meaning=("After this long the gap to c has shrunk to about 37 %." if k < 0
                                         else "After this long the gap to c has grown about 2.7 times.")))
            results.append(dict(sym="ln 2/|<i>k</i>|", name="Half-life" if k < 0 else "Doubling time",
                                value=g(np.log(2) * tc), unit="s",
                                meaning=("After this long the gap to c is half as big." if k < 0
                                         else "After this long the gap to c is twice as big.")))

    else:
        # Linear and polynomial: v(t) = a_n t^n + ... + a_1 t + a_0
        deg = len(p) - 1
        terms_g, terms_n = [], []
        for i, co in enumerate(p):
            power = deg - i
            tpart = "" if power == 0 else (" · <i>t</i>" if power == 1 else f" · <i>t</i><sup>{power}</sup>")
            terms_g.append(f"<i>a</i><sub>{power}</sub>{tpart}")
            mag = g(abs(co))
            terms_n.append((MINUS if co < 0 else "") + mag + tpart if i == 0 else f"{_sign(co)} {mag}{tpart}")
        general = f"{Y} = " + " + ".join(terms_g)
        numbers = f"{Y} = " + " ".join(terms_n)
        for i, co in enumerate(p):
            power = deg - i
            u = unit if power == 0 else f"{unit}/s" if power == 1 else f"{unit}/s<sup>{power}</sup>"
            if power == 0:
                name, meaning = "Starting value", f"The {q['noun']} at t = 0."
            elif power == 1 and deg == 1:
                name = "Slope"
                meaning = ("How fast the speed changes each second: the acceleration. Positive means it speeds up "
                           "away from the radar." if speed else
                           "How fast the distance changes each second: the speed. Positive means moving away from the radar.")
            elif power == 1:
                name, meaning = "Slope at t = 0", f"How fast the {q['noun']} is changing at t = 0."
            elif power == 2:
                name, meaning = "Bend", "How strongly the curve bends: positive bends up like a smile, negative down like a frown."
            else:
                name, meaning = f"t{_sup(power)} term", "Adds extra wiggle; the higher the power, the more it matters late in the trace."
            params.append(dict(sym=f"<i>a</i><sub>{power}</sub>", name=name, value=g(co), unit=u, meaning=meaning))
        if deg == 1 and speed:
            results.append(dict(sym="<i>a</i><sub>1</sub>", name="Acceleration", value=g(p[0]), unit="m/s²",
                                meaning="The slope of a speed-time line is the acceleration."))

    word, level = fit_quality(r.r_squared)
    return dict(general=general, numbers=numbers, params=params, results=results,
                quality=word, level=level, what=q["thing"], unit=unit, y=Y)


def _sup(n):
    return str(n).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def explain_peaks(peaks, unit) -> str:
    """What the peaks say, as Qt rich text; '' when there are none."""
    if not peaks or not peaks["all"]:
        return ""
    pos, neg, allp = peaks["positive"], peaks["negative"], peaks["all"]
    halves = [b[0] - a[0] for a, b in zip(allp, allp[1:]) if a[2] != b[2]]
    s = lambda n: "" if n == 1 else "s"
    parts = [f"Found <b>{len(pos)}</b> high peak{s(len(pos))} ▲ and <b>{len(neg)}</b> low peak{s(len(neg))} ▼."]
    if halves:
        m = float(np.mean(halves))
        parts.append(f"From a high peak to the next low one takes about <b>{g(m)} s</b>, half a swing, "
                     f"so one full swing takes about <b>{g(2 * m)} s</b>.")
    if pos and neg:
        up = float(np.mean([q[1] for q in pos]))
        down = float(np.mean([q[1] for q in neg]))
        parts.append(f"The peaks are about <b>{g(up - down)} {unit}</b> apart from top to bottom.")
    return " ".join(parts)
