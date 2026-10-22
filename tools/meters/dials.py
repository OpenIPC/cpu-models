# Red pointer dials: where the needle points, read without a network.
#
# A dial is annotated once, on the meter's face in cell widths
# (common.face_homography): its centre, roughly its radius, its multiplier.
# In each frame the face is drawn frontal around that centre, so the dial is
# a circle whatever the camera's angle, and the dial's own printed ring is
# found by a circle fit, which absorbs a slightly wrong annotation or cell
# fit.
#
# The needle is the darkest thing in the annulus between the hub and the
# numerals: a solid wedge from the hub out to about two thirds of the
# radius, where the numerals start. Its counterweight on the other side is
# shorter and stops before the annulus does. In luma alone (what the camera
# reads, and all an IR-lit camera has) the red needle is darker than the
# white face, and the numerals are left outside the annulus. The angle is
# taken clockwise from the face's up: these dials print 0 at the top,
# upright to the drum row, so the face's up is the dial's 0.
import math

import cv2
import numpy as np

from common import face_homography

K = 48          # face render: pixels per cell width
# Annulus, in dial radii, centred on the hub: chosen by the chain residual on
# a day of wall-camera frames (0.160 for 0.30-0.58 about the ring's centre,
# 0.111 for this about the hub's).
INNER, OUTER = 0.45, 0.75
MIN_CLARITY = 1.5   # a needle hidden by glare peaks no higher than this


def face_patch(y, cells, centre, half):
    """The face around `centre` (cell units), frontal, `half` units each way."""
    h, _, _ = face_homography(cells)
    cx, cy = centre
    S = np.array([[K, 0, -(cx - half) * K], [0, K, -(cy - half) * K], [0, 0, 1]])
    n = int(2 * half * K)
    return cv2.warpPerspective(y, S @ h, (n, n), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def find_ring(p, r_guess):
    """The dial's printed ring in a patch: (cx, cy, r) in patch pixels."""
    g = cv2.GaussianBlur(p, (0, 0), 1.2)
    rp = r_guess * K
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=K,
                          param1=60, param2=18, minRadius=int(rp * 0.75), maxRadius=int(rp * 1.25))
    c0 = np.array([p.shape[1] / 2, p.shape[0] / 2])
    if cs is None:
        return c0[0], c0[1], rp, False
    cs = cs[0]
    k = int(np.argmin(np.linalg.norm(cs[:, :2] - c0, axis=1)))
    cx, cy, r = cs[k]
    if np.hypot(cx - c0[0], cy - c0[1]) > 0.5 * rp:
        return c0[0], c0[1], rp, False
    return float(cx), float(cy), float(r), True


def hub(p, cx, cy, r):
    """The pivot: the needles here grow out of a dark disc about 0.28 of the
    dial's radius, and the printed ring's centre can be a few percent off it,
    which at these radii is degrees."""
    g = cv2.GaussianBlur(p.astype(np.float32), (0, 0), 1.0)
    rr = max(int(0.28 * r), 2)
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disk = (xx ** 2 + yy ** 2) <= rr * rr
    best = (np.inf, cx, cy)
    span = int(0.15 * r)
    for dy in range(-span, span + 1):
        for dx in range(-span, span + 1):
            X, Y = int(round(cx + dx)), int(round(cy + dy))
            win = g[Y - rr:Y + rr + 1, X - rr:X + rr + 1]
            if win.shape != disk.shape:
                continue
            v = float(win[disk].mean())
            if v < best[0]:
                best = (v, X, Y)
    return float(best[1]), float(best[2])


def needle(p, cx, cy, r):
    """The needle's angle (degrees clockwise from up) and how clear it is."""
    ang = np.arange(0, 360, 1.0)
    rad = np.linspace(INNER * r, OUTER * r, 12)
    a = np.radians(ang)[:, None]
    xs = (cx + rad[None, :] * np.sin(a)).astype(np.float32)
    ys = (cy - rad[None, :] * np.cos(a)).astype(np.float32)
    v = cv2.remap(p.astype(np.float32), xs, ys, cv2.INTER_LINEAR)
    dark = -v.mean(1)
    # the needle is ~25 degrees wide at these radii: smooth on that scale
    k = np.exp(-0.5 * (np.arange(-20, 21) / 6.0) ** 2)
    sm = np.convolve(np.concatenate([dark[-20:], dark, dark[:20]]), k / k.sum(), "same")[20:-20]
    i = int(np.argmax(sm))
    # sub-degree peak, circular
    y0, y1, y2 = sm[i - 1], sm[i], sm[(i + 1) % 360]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den else 0.0
    clarity = float((sm[i] - np.median(sm)) / (sm.std() + 1e-6))
    return (ang[i] + off) % 360, clarity, sm


def read_dial(y, cells, dial):
    """(value 0..10, clarity, ring found, patch, (cx, cy, r))."""
    r = dial["radius"]
    p = face_patch(y, cells, dial["centre"], 1.6 * r)
    cx, cy, rr, ok = find_ring(p, r)
    cx, cy = hub(p, cx, cy, rr)
    a, clarity, _ = needle(p, cx, cy, rr)
    return a / 36.0, clarity, ok, p, (cx, cy, rr)


def join(dials, last_unit):
    """The fraction of the last drum's unit the dials say has turned, or None
    when the dial worth a tenth of that unit cannot be read (the finer ones
    alone cannot say which tenth). `dials`: (value 0..10, clarity,
    multiplier) each, any order.

    Each dial's digit is its own needle's; its place within the digit is the
    next dial down's (a needle a hair short of 3 is 2.99 only if the dial
    below is near 9). A dial whose needle is not clear is left out; the one
    above it then keeps its own needle's fraction. Gears have backlash, so a
    needle's own fraction is otherwise used only to decide which digit the
    dial below puts it in."""
    ds = sorted((d for d in dials if d[1] >= MIN_CLARITY), key=lambda d: d[2])
    if not ds or not math.isclose(ds[-1][2], last_unit / 10, rel_tol=0.01):
        return None
    lower, lower_mult = ds[0][0], ds[0][2]
    for v, _, mult in ds[1:]:
        if math.isclose(mult, lower_mult * 10, rel_tol=0.01):
            f = lower / 10           # the dial below, as a tenth of this one's digit
            d = math.floor(v)
            if (v - d) - f > 0.5:
                d += 1
            elif f - (v - d) > 0.5:
                d -= 1
            lower = (d % 10) + f
        else:
            lower = v                # a gap in the chain: this needle alone
        lower_mult = mult
    return lower / 10


def with_drum(drum_pos, frac):
    """The last drum's reading with the dials' fraction: the drum says which
    digit, roughly (it rolls while the fraction runs from about .8 to 1), the
    dials say exactly how far."""
    return round(drum_pos - frac) + frac
