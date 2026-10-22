# fontfit.py <bank-dir> <fonts-dir> <out.json>
#
# Which fonts, at which weight, size and squeeze, draw digits like the ones a
# real meter shows. Each real digit of the bank is averaged over its frames
# (one physical glyph, many noise draws: the mean is that glyph), and every
# font is rendered as a drum at that digit, blurred to the real crop's
# softness, and compared by normalised cross-correlation over the window's
# interior; the frame of the printed box is not the font's business. The
# parameters are shared across a font's digits, as they are on a meter.
#
# The ranking is a prior for sampling, not a choice: the synthetic set draws
# every font, the close ones more often, because other meters use other faces
# and the network has to read them too.
import glob
import itertools
import json
import os
import sys

import cv2
import numpy as np

from common import pinned_fonts
from glyphs import drum_window, weight_range

MARGIN = 0.12  # of the crop, on each side: the printed frame


def interior(a):
    h, w = a.shape
    my, mx = int(h * MARGIN), int(w * MARGIN)
    a = a[my:h - my, mx:w - mx].astype(np.float32)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def real_means(bank):
    # Per (digit, meter): the mean of its crops. A meter's own offsets differ,
    # so meters are not averaged together.
    out = {}
    for f in glob.glob(f"{bank}/*/*.png"):
        d = os.path.basename(os.path.dirname(f))
        meter = os.path.basename(f).split("_")[2]
        out.setdefault((d, meter), []).append(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32))
    return {k: np.mean(v, 0) for k, v in out.items()}


def best_shift(ref, img):
    # The real crops sit a little off their boxes; search a small shift.
    best = -1.0
    h, w = img.shape
    for dy in (-4, -2, 0, 2, 4):
        for dx in (-4, -2, 0, 2, 4):
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            s = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
            best = max(best, float((interior(s) * ref).sum()))
    return best


def main(bank, fonts_dir, out):
    reals = real_means(bank)
    h, w = next(iter(reals.values())).shape
    refs = {k: interior(v) for k, v in reals.items()}
    results = []
    for path in (os.path.join(fonts_dir, f) for f in pinned_fonts(fonts_dir)):
        wr = weight_range(path)
        weights = [None] if wr is None else [x for x in (400, 500, 600, 700) if wr[0] <= x <= wr[1]]
        best = None
        for gh, wt, sq, sig in itertools.product((0.74, 0.8, 0.86), weights,
                                                 (0.85, 1.0, 1.15), (1.5, 2.5, 3.5)):
            scores = []
            for (d, meter), ref in refs.items():
                ink, shade = drum_window(path, float(d), w, h, glyph_h=gh, weight=wt, squeeze=sq)
                img = 0.9 * shade[:, None] * (1 - ink) + 0.05 * ink
                img = cv2.GaussianBlur(img, (0, 0), sig)
                scores.append(best_shift(ref, img))
            s = float(np.mean(scores))
            if best is None or s > best["score"]:
                best = {"font": os.path.basename(path), "score": round(s, 4), "glyph_h": gh,
                        "weight": wt, "squeeze": sq, "blur": sig,
                        "per_digit": {f"{d}/{m}": round(x, 3) for (d, m), x in zip(refs, scores)}}
        print(best["font"], best["score"], flush=True)
        results.append(best)
    results.sort(key=lambda r: -r["score"])
    json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
