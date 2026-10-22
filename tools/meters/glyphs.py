# Drums: digits from a font, on a cylinder, seen through a window.
#
# Everything here is ink coverage in 0..1 at an oversampled scale; levels,
# blur, noise and the camera come later. A drum is the ten digits stacked at
# a fixed pitch around a cylinder, the next digit below the current one (the
# count rises as the drum turns up). Position p shows digit floor(p) shifted
# up by frac(p) of a pitch, the next one entering from below, and the window
# sees the strip through the cylinder's curvature: arc length s maps to the
# screen at R sin(s/R), and the strip darkens as it turns away by cos(s/R).
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_cache = {}


def font(path, size, weight=None, width=None):
    key = (path, size, weight, width)
    f = _cache.get(key)
    if f is None:
        f = ImageFont.truetype(path, size)
        try:
            axes = {a["name"]: a for a in f.get_variation_axes()}
        except (OSError, AttributeError):
            axes = {}
        if axes:
            vals = []
            for name, a in axes.items():
                v = a.get("default", a["minimum"])
                if name in (b"Weight", "Weight") and weight is not None:
                    v = weight
                if name in (b"Width", "Width") and width is not None:
                    v = width
                vals.append(min(max(v, a["minimum"]), a["maximum"]))
            f.set_variation_by_axes(vals)
        _cache[key] = f
    return f


def weight_range(path):
    try:
        for a in ImageFont.truetype(path, 10).get_variation_axes():
            if a["name"] in (b"Weight", "Weight"):
                return a["minimum"], a["maximum"]
    except (OSError, AttributeError):
        pass
    return None


def glyph(path, ch, height, weight=None, width=None, squeeze=1.0):
    """Ink of one character, its digit height scaled to `height` px, cropped
    to its ink box and squeezed horizontally by `squeeze`."""
    size = int(height * 1.6)
    f = font(path, size, weight, width)
    l, t, r, b = f.getbbox("0")
    scale = height / max(b - t, 1)
    if abs(scale - 1) > 0.02:
        f = font(path, max(int(size * scale), 4), weight, width)
    l, t, r, b = f.getbbox(ch)
    l0, t0, r0, b0 = f.getbbox("0")
    im = Image.new("L", (r - l + 4, b0 - t0 + 4))
    # Every digit sits on the same baseline and cap height as "0".
    ImageDraw.Draw(im).text((2 - l, 2 - t0), ch, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    if squeeze != 1.0:
        w = max(int(round(a.shape[1] * squeeze)), 1)
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (w, a.shape[0]), Image.LANCZOS), np.float32) / 255
    return a


def drum_window(path, pos, w, h, *, glyph_h=0.84, pitch=1.45, radius=0.95,
                weight=None, width=None, squeeze=1.0, dx=0.0, digits="0123456789"):
    """The ink a window of w x h px shows of a drum at position `pos`, and the
    drum's shading (1 facing the camera, less as it turns away).

    glyph_h: digit height / window height; pitch: digit spacing / digit
    height; radius: cylinder radius / window height; dx: horizontal offset of
    the digits / window width."""
    gh = glyph_h * h
    p = pitch * gh
    R = radius * h
    ink = np.zeros((h, w), np.float32)
    base = int(math.floor(pos))
    frac = pos - base
    # Screen row y in the window (centre 0) sees arc length s = R asin(y/R);
    # digit k (k = 0 the current one) is centred at s = (k - frac) * p.
    ys = np.arange(h, dtype=np.float32) - (h - 1) / 2
    s = R * np.arcsin(np.clip(ys / R, -1, 1))
    shade = np.cos(np.clip(s / R, -1.5, 1.5)).astype(np.float32)
    for k in (-1, 0, 1, 2):
        ch = digits[(base + k) % len(digits)]
        g = glyph(path, ch, int(round(gh * 1.0)), weight, width, squeeze)
        gh_px, gw = g.shape
        # rows of the glyph each screen row samples
        gy = (s - (k - frac) * p) + gh_px / 2
        valid = (gy >= 0) & (gy <= gh_px - 1)
        if not valid.any():
            continue
        x0 = int(round((w - gw) / 2 + dx * w))
        rows = np.where(valid)[0]
        gyv = gy[rows]
        i0 = np.floor(gyv).astype(int)
        i1 = np.minimum(i0 + 1, gh_px - 1)
        t = (gyv - i0)[:, None]
        band = g[i0] * (1 - t) + g[i1] * t
        xa, xb = max(x0, 0), min(x0 + gw, w)
        if xb > xa:
            ink[rows, xa:xb] = np.maximum(ink[rows, xa:xb], band[:, xa - x0:xb - x0])
    return ink, shade
