# synth.py <out-dir> <n-patches> [seed] [--cameras a.json,...] [--wall <dir>] [--fontfit f.json]
#
# Drum cells as a camera sees them, with the digit each one shows.
#
# A sample starts as a frontal face: either a real meter out of a saved
# daylight frame, its drums painted over with rendered ones in the real
# drum's own levels, or a procedural plate (dark-on-light, light-on-dark and
# red decimal drums, one window or a box per drum, any number of drums). The
# face is posed in 3D (any rotation in the picture, tilted up to 50 degrees),
# put through a fisheye, blurred, lit with glare and IR reflections, given
# shot and read noise, an ISP's denoise and sharpening, quantised in limited
# or full range, and through the codec a camera uses (H.264, H.265 or JPEG,
# tiled into full frames so the blocks fall where a sensor's do). Each drum
# is then cut by the rectifier the camera runs (common.rectify), from its
# corners as a person would click them: off by a little.
#
# The real frames lend their faces only, never their digits, and only the
# cameras named with --cameras: every other camera stays a clean test.
#
# Out: <out>/crops/*.png (CELL_W x CELL_H) and <out>/labels.jsonl, one line a
# crop: its drum position (2.0 shows a 2; 2.5 is half way to 3), its class,
# and every choice that made it, so a score can be split by any of them;
# <out>/sheet/ keeps some whole patches with the cut quads drawn.
import argparse
import functools
import json
import math
import os
import subprocess
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np

from common import cells_at, face_homography, frame_luma, harvested, load_camera, pos_class, rectify
from glyphs import drum_window

PATCH = 192          # sensor pixels a side
FRAME_W, FRAME_H = 1920, 1080
COLS, ROWS = FRAME_W // PATCH, FRAME_H // PATCH
O = 6                # face oversampling: face pixels per cell-width unit / 10
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.cache/fonts")


# ---------------------------------------------------------------- positions

def drum_positions(rng, n):
    """Positions of n drums, most significant first."""
    if rng.random() < 0.5:
        # An odometer: the last drum turns continuously, each one above it
        # turns only while the one below goes from 9 to 0.
        digits = rng.integers(0, 10, n)
        pos = np.zeros(n)
        pos[-1] = digits[-1] + rng.random()
        for i in range(n - 2, -1, -1):
            lower = pos[i + 1] % 10
            pos[i] = digits[i] + max(0.0, lower - 9.0)
    else:
        # Balanced: every drum anywhere, half of them on a whole digit.
        pos = np.where(rng.random(n) < 0.5, rng.integers(0, 10, n).astype(float), rng.random(n) * 10)
    # Mechanical play: a drum on a whole digit is rarely exactly centred.
    pos = pos + rng.normal(0, 0.04, n)
    return np.mod(pos, 10)


# ---------------------------------------------------------------- styles

@functools.lru_cache(maxsize=None)
def font_prior(fontfit):
    fonts = sorted(f for f in os.listdir(FONTS_DIR) if f.endswith(".ttf"))
    fit = {}
    if fontfit and os.path.exists(fontfit):
        fit = {r["font"]: r for r in json.load(open(fontfit))}
    scores = np.array([fit.get(f, {}).get("score", 0.4) for f in fonts])
    w = np.exp((scores - scores.max()) * 20)  # the closest fonts often, every font sometimes
    return fonts, w / w.sum(), fit


def pick_font(rng, fontfit):
    fonts, w, fit = font_prior(fontfit)
    f = fonts[rng.choice(len(fonts), p=w)]
    r = fit.get(f, {})
    weight = r.get("weight")
    if weight is not None:
        weight = float(np.clip(weight + rng.normal(0, 60), 300, 800))
    return {"font": f, "weight": weight,
            # Wide spread on purpose: an Energomera's digits are thin and
            # short in a wide drum, a water meter's fill theirs.
            "glyph_h": float(np.clip(r.get("glyph_h", 0.8) + rng.normal(-0.04, 0.1), 0.5, 0.92)),
            "squeeze": float(np.clip(r.get("squeeze", 1.0) * rng.uniform(0.6, 1.2), 0.5, 1.3)),
            "pitch": float(rng.uniform(1.2, 1.65)), "radius": float(rng.uniform(0.7, 1.4)),
            "dx": float(rng.normal(0, 0.03))}


def render_drum(st, pos, w, h, paper, ink):
    a, shade = drum_window(os.path.join(FONTS_DIR, st["font"]), pos, w, h,
                           glyph_h=st["glyph_h"], pitch=st["pitch"], radius=st["radius"],
                           weight=st["weight"], squeeze=st["squeeze"], dx=st["dx"])
    # the drum darkens as it turns away from the window's centre
    paper_s = paper * (0.75 + 0.25 * shade[:, None])
    return paper_s * (1 - a) + ink * a


# ---------------------------------------------------------------- faces
#
# A face is (image 0..1, cell quads in face pixels TL TR BR BL, metadata).
# Face units: a cell is CW x CH with CW = 10 * O pixels.

CW = 10 * O


def proc_face(rng, n, fontfit):
    ch_ratio = rng.uniform(1.35, 1.9)            # cell height / width
    CH = int(CW * ch_ratio)
    gap = int(CW * rng.uniform(0.08, 0.6))
    one_window = rng.random() < 0.35              # one long window with dividers (Energomera)
    mx, my = int(CW * 2.5), int(CH * 2.0)
    W = 2 * mx + n * CW + (n - 1) * gap
    H = 2 * my + CH
    face_l = rng.uniform(0.55, 0.97)
    img = np.full((H, W), face_l, np.float32)
    # a little printing around: lines of text, a unit, a logo-ish block
    for _ in range(rng.integers(1, 5)):
        y = int(rng.uniform(0.05, 0.95) * H)
        if my - CH * 0.2 < y < my + CH * 1.2:
            continue
        x = int(rng.uniform(0, W * 0.6))
        m = np.zeros((H, W), np.uint8)
        cv2.putText(m, "".join(rng.choice(list("0123456789ABCMKWh-./")) for _ in range(rng.integers(3, 12))),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.6, 2.2) * O / 6, 255,
                    int(rng.integers(1, 4)), cv2.LINE_AA)
        m = m.astype(np.float32) / 255
        img = img * (1 - m) + rng.uniform(0.0, 0.35) * m
    st = pick_font(rng, fontfit)
    kind = rng.choice(["dark_on_light", "light_on_dark", "mixed"], p=[0.45, 0.3, 0.25])
    n_red = int(rng.integers(1, 4)) if kind == "mixed" else 0
    pos = drum_positions(rng, n)
    frame_ink = rng.uniform(0.0, 0.3)
    t = max(1, int(CW * rng.uniform(0.03, 0.09)))
    if one_window:
        x0, x1 = mx - gap // 2, mx + n * CW + (n - 1) * gap + gap // 2
        cv2.rectangle(img, (x0 - t, my - t), (x1 + t, my + CH + t), frame_ink, -1)
        cv2.rectangle(img, (x0, my), (x1, my + CH), rng.uniform(0.0, 0.25), -1)
    quads, styles = [], []
    for i in range(n):
        x = mx + i * (CW + gap)
        red = kind == "mixed" and i >= n - n_red
        if kind == "light_on_dark":
            paper, ink = rng.uniform(0.02, 0.25), rng.uniform(0.6, 0.98)
        elif red:
            # red drums in luma: a mid grey ground, the digit dark or light
            paper = rng.uniform(0.3, 0.7)
            ink = rng.uniform(0.0, 0.15) if rng.random() < 0.6 else rng.uniform(0.8, 1.0)
        else:
            paper, ink = rng.uniform(0.55, 0.98), rng.uniform(0.0, 0.25)
        if not one_window:
            cv2.rectangle(img, (x - t, my - t), (x + CW + t, my + CH + t), frame_ink, -1)
        img[my:my + CH, x:x + CW] = render_drum(st, pos[i], CW, CH, paper, ink)
        if kind == "light_on_dark" and rng.random() < 0.4:
            # An IR lamp beside the lens draws a bright line along each
            # cylinder, high in the window (seen on an Energomera at night).
            yb, wb = CH * rng.uniform(0.08, 0.4), CH * rng.uniform(0.04, 0.12)
            band = np.exp(-((np.arange(CH) - yb) / wb) ** 2)[:, None]
            img[my:my + CH, x:x + CW] += rng.uniform(0.3, 1.0) * band
        quads.append([[x, my], [x + CW, my], [x + CW, my + CH], [x, my + CH]])
        styles.append("red" if red else kind)
    if one_window and rng.random() < 0.5:
        # How a person marks a one-window meter: the window cut in n equal
        # parts, dividers and all, not each drum's own edges.
        x0, x1 = mx - gap / 2, mx + n * CW + (n - 1) * gap + gap / 2
        step = (x1 - x0) / n
        quads = [[[x0 + i * step, my], [x0 + (i + 1) * step, my], [x0 + (i + 1) * step, my + CH],
                  [x0 + i * step, my + CH]] for i in range(n)]
    if rng.random() < 0.5:  # a decimal comma or point between two drums
        k = int(rng.integers(1, n))
        cx = mx + k * (CW + gap) - gap // 2
        cv2.circle(img, (cx, my + CH + int(CH * 0.15)), max(2, int(CW * 0.08)), frame_ink, -1)
    return img, np.float32(quads), pos, {"face": "procedural", "drums": styles, **st}


def homography_frontal(quads_img, CH):
    """Map frame pixels to a frontal face in which the cells sit on a row,
    CW x CH each at their measured spacing."""
    h, pitch, _ = face_homography(quads_img)
    return np.diag([CW, CW, 1.0]) @ h, CW * pitch


def real_face(rng, frames, fontfit):
    y, meter = frames[rng.integers(len(frames))]
    q = np.float32(meter["cells"])
    n = len(q)
    wbox = np.mean([np.linalg.norm(c[1] - c[0]) for c in q])
    hbox = np.mean([np.linalg.norm(c[3] - c[0]) for c in q])
    CH = int(CW * hbox / wbox)
    hm, pitch = homography_frontal(q, CH)
    mx, my = int(CW * 2.5), int(CH * 2.0)
    W = int(2 * mx + (n - 1) * pitch + CW)
    H = 2 * my + CH
    shift = np.float64([[1, 0, mx], [0, 1, my], [0, 0, 1]])
    face = cv2.warpPerspective(y.astype(np.float32) / 255, shift @ hm, (W, H),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    # The frame's own light is the frame's; light is the camera model's job.
    # Bring the drums' paper to a level of its own choosing first.
    row = face[my:my + CH, mx:int(mx + (n - 1) * pitch + CW)]
    face = face * (rng.uniform(0.55, 0.95) / max(float(np.percentile(row, 90)), 0.05))
    st = pick_font(rng, fontfit)
    pos = drum_positions(rng, n)
    quads = []
    for i in range(n):
        x0 = int(mx + i * pitch)
        cell = face[my:my + CH, x0:x0 + CW]
        # The drum's own levels: its paper and its ink, from the real cell.
        inner = cell[int(CH * .1):int(CH * .9), int(CW * .15):int(CW * .85)]
        paper, ink = float(np.percentile(inner, 90)), float(np.percentile(inner, 4))
        # The faintest real drum here is barely legible in its own frame;
        # keep its paper, but give the digit enough ink to be a digit.
        ink = min(ink, paper * rng.uniform(0.3, 0.6))
        ix0, ix1 = int(CW * 0.1), int(CW * 0.9)
        iy0, iy1 = int(CH * 0.05), int(CH * 0.95)
        d = render_drum(st, pos[i], ix1 - ix0, iy1 - iy0, paper, ink)
        # Soften to the real face's own softness (it is a blurred
        # photograph; the render is sharp) and feather it in.
        d = cv2.GaussianBlur(d, (0, 0), rng.uniform(0.6, 1.4) * O / 2)
        m = np.zeros_like(d)
        m[2:-2, 2:-2] = 1
        m = cv2.GaussianBlur(m, (0, 0), O / 2)
        reg = face[my + iy0:my + iy1, x0 + ix0:x0 + ix1]
        face[my + iy0:my + iy1, x0 + ix0:x0 + ix1] = reg * (1 - m) + d * m
        quads.append([[x0, my], [x0 + CW, my], [x0 + CW, my + CH], [x0, my + CH]])
    return face, np.float32(quads), pos, {"face": "real:" + meter["_camera"] + "/" + meter["name"],
                                          "drums": ["real"] * n, **st}


# ---------------------------------------------------------------- the camera

def pose(rng, face_shape, quads, glyph_px):
    """A homography from face pixels to an undistorted sensor patch: the
    cell row centred, cell height `glyph_px`, any rotation, tilted."""
    H, W = face_shape
    c = quads.reshape(-1, 2).mean(0)
    ch = np.mean(quads[:, 3, 1] - quads[:, 0, 1])
    s = glyph_px / ch
    th = rng.uniform(0, 2 * math.pi)
    tilt = math.radians(min(abs(rng.normal(0, 25)), 50))
    phi = rng.uniform(0, 2 * math.pi)                 # direction of the tilt
    # face plane in metric units centred on the row; rotate by tilt about
    # the axis at phi, view with focal f at distance D
    f = PATCH * rng.uniform(0.8, 2.5)
    D = f / s
    a = np.array([math.cos(phi), math.sin(phi), 0.0])
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(tilt) * K + (1 - math.cos(tilt)) * K @ K
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    R = R @ Rz
    # plane point (u, v) -> R @ (u, v, 0) + (0, 0, D)
    P = np.array([[R[0, 0], R[0, 1], 0], [R[1, 0], R[1, 1], 0], [R[2, 0], R[2, 1], D]])
    Kc = np.array([[f, 0, PATCH / 2], [0, f, PATCH / 2], [0, 0, 1]])
    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
    h = Kc @ P @ T
    return h / h[2, 2], {"theta": round(math.degrees(th), 1), "tilt": round(math.degrees(tilt), 1)}


def fisheye(rng):
    """Where in a 1920x1080 frame the patch sits, and the lens's k1, k2:
    an undistorted normalised radius r maps to r (1 + k1 r^2 + k2 r^4),
    normalised by half the frame's width. k1 is kept where that stays
    monotonic out to the frame's corner, as a real lens's image does."""
    if rng.random() < 0.3:
        return None
    ox = rng.uniform(-FRAME_W / 2 + PATCH, FRAME_W / 2 - PATCH)
    oy = rng.uniform(-FRAME_H / 2 + PATCH, FRAME_H / 2 - PATCH)
    return (ox, oy, rng.uniform(-0.2, 0.0), rng.uniform(-0.02, 0.02))


def _radial(fe):
    _, _, k1, k2 = fe
    ru = np.linspace(0, 1.6, 4001)
    rd = ru * (1 + k1 * ru ** 2 + k2 * ru ** 4)
    ok = np.concatenate([[True], np.diff(rd) > 0])
    stop = np.argmin(ok) if not ok.all() else len(ok)
    return ru[:stop], rd[:stop]


def distort_pts(pts, fe):
    if fe is None:
        return pts
    ox, oy, k1, k2 = fe
    nrm = FRAME_W / 2
    p = (pts - PATCH / 2 + [ox, oy]) / nrm
    r2 = (p ** 2).sum(-1, keepdims=True)
    p = p * (1 + k1 * r2 + k2 * r2 * r2)
    return p * nrm - [ox, oy] + PATCH / 2


def undistort_grid(fe):
    ys, xs = np.mgrid[0:PATCH, 0:PATCH].astype(np.float64)
    d = np.stack([xs, ys], -1)
    if fe is None:
        return d
    ox, oy, _, _ = fe
    nrm = FRAME_W / 2
    pd = (d - PATCH / 2 + [ox, oy]) / nrm
    rd = np.sqrt((pd ** 2).sum(-1, keepdims=True))
    ru_t, rd_t = _radial(fe)
    ru = np.interp(rd, rd_t, ru_t)
    pu = pd * np.where(rd > 1e-9, ru / np.maximum(rd, 1e-9), 1.0)
    return pu * nrm - [ox, oy] + PATCH / 2


def shoot(rng, face, quads, glyph_px):
    h, geo = pose(rng, face.shape, quads, glyph_px)
    fe = fisheye(rng)
    # The face is drawn far larger than the sensor sees it; filter first.
    minif = float(np.sqrt(abs(np.linalg.det(np.linalg.inv(h)[:2, :2]))))
    src = cv2.GaussianBlur(face, (0, 0), max(0.5 * minif, 0.01)) if minif > 1 else face
    und = undistort_grid(fe)
    hi = np.linalg.inv(h)
    pts = np.concatenate([und, np.ones(und.shape[:2] + (1,))], -1) @ hi.T
    mapx = (pts[..., 0] / pts[..., 2]).astype(np.float32)
    mapy = (pts[..., 1] / pts[..., 2]).astype(np.float32)
    bg = rng.uniform(0.05, 0.9)
    img = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    cell_quads = []
    for q in quads:
        p = np.concatenate([q, np.ones((4, 1))], 1) @ h.T
        cell_quads.append(distort_pts(p[:, :2] / p[:, 2:], fe))
    meta = dict(geo, fisheye=None if fe is None else [round(fe[2], 3), round(fe[3], 3)])
    return img.astype(np.float32), np.float32(cell_quads), meta


def light_and_sensor(rng, img):
    meta = {}
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / PATCH
    # uneven light, then glare: soft broad sheen and small hot spots (IR LEDs
    # reflected in the glass sit right on the drums at night)
    g = float(np.clip(np.exp(rng.normal(math.log(0.8), 0.35)), 0.2, 1.3))
    grad = 1 + rng.normal(0, 0.25) * (xx - .5) + rng.normal(0, 0.25) * (yy - .5)
    img = img * g * grad
    meta["light"] = round(g, 2)
    if rng.random() < 0.4:
        cx, cy, sx, sy = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(.1, .6), rng.uniform(.05, .3)
        img = img + rng.uniform(0.1, 0.6) * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        meta["sheen"] = True
    if rng.random() < 0.15:
        k = int(rng.integers(1, 6))
        for _ in range(k):
            cx, cy, r = rng.uniform(.2, .8), rng.uniform(.3, .7), rng.uniform(.01, .05)
            img = img + 2.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2))
        meta["hotspots"] = k
    # optics
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.35, 1.3))
    # shot and read noise, in 0..1 units of full scale
    a, b = 10 ** rng.uniform(-4, -2.3), 10 ** rng.uniform(-6, -3)
    img = img + rng.normal(0, 1, img.shape) * np.sqrt(np.clip(img, 0, None) * a + b)
    meta["noise"] = [round(math.log10(a), 2), round(math.log10(b), 2)]
    # ISP: denoise, sharpen (the halo real crops show), clip
    if rng.random() < 0.6:
        img = cv2.bilateralFilter(img.astype(np.float32), 5, rng.uniform(0.03, 0.15), 2)
    amt = rng.uniform(0, 1.6)
    img = img + amt * (img - cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 2.0)))
    img = np.clip(img, 0, 1)
    full = rng.random() < 0.5
    y = img * 255 if full else 16 + img * 219
    meta.update(sharpen=round(amt, 2), range="full" if full else "limited")
    return np.clip(y + 0.5, 0, 255).astype(np.uint8), meta


def codec_roundtrip(tiles, rng, tmp):
    """Tiles of one full frame through one codec at one quality."""
    frame = np.full((FRAME_H, FRAME_W), 128, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, COLS)
        frame[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = t
    kind = rng.choice(["h264", "h265", "jpeg"], p=[0.4, 0.4, 0.2])
    if kind == "jpeg":
        q = int(rng.integers(40, 95))
        out = cv2.imdecode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                           cv2.IMREAD_GRAYSCALE)
    else:
        q = int(rng.integers(22, 42))
        yuv = np.concatenate([frame.ravel(), np.full(FRAME_W * FRAME_H // 2, 128, np.uint8)])
        raw, enc = os.path.join(tmp, "f.yuv"), os.path.join(tmp, "f.mkv")
        yuv.tofile(raw)
        lib = ["-c:v", "libx264", "-qp", str(q)] if kind == "h264" else \
            ["-c:v", "libx265", "-x265-params", f"qp={q}:log-level=none"]
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                        "-s", f"{FRAME_W}x{FRAME_H}", "-i", raw, "-frames:v", "1", *lib,
                        "-preset", "veryfast", enc], check=True)
        dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", enc, "-f", "rawvideo",
                              "-pix_fmt", "yuv420p", "-"], check=True, capture_output=True).stdout
        out = np.frombuffer(dec, np.uint8)[:FRAME_W * FRAME_H].reshape(FRAME_H, FRAME_W)
    res = []
    for k in range(len(tiles)):
        r, c = divmod(k, COLS)
        res.append(out[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH])
    return res, {"codec": kind, "q": q}


def clicked(rng, q, cw):
    # A person's click: a shared slip and each corner's own.
    return q + rng.normal(0, 0.05 * cw, (1, 2)) + rng.normal(0, 0.04 * cw, (4, 2))


# ---------------------------------------------------------------- driver

def load_real_faces(cameras, wall):
    out = []
    for path in cameras:
        cam = load_camera(path)
        for rec, stem in harvested(wall, cam["camera"], luma_min=60):
            y = frame_luma(stem)
            for m in cam["meters"]:
                cells = cells_at(m, rec["at"])
                if cells is not None:
                    out.append((y, dict(m, cells=cells, _camera=cam["camera"])))
    return out


_real = None


def init_worker(cameras, wall):
    global _real
    _real = load_real_faces(cameras, wall) if cameras else []


def make_frame(job):
    seed, out, fontfit, keep_sheet = job
    rng = np.random.default_rng(seed)
    tiles, metas = [], []
    for k in range(COLS * ROWS):
        n = int(rng.integers(4, 10))
        if _real and rng.random() < 0.5:
            face, quads, pos, fm = real_face(rng, _real, fontfit)
        else:
            face, quads, pos, fm = proc_face(rng, n, fontfit)
        glyph_px = float(np.exp(rng.uniform(math.log(11), math.log(44))))
        img, cq, gm = shoot(rng, face, quads, glyph_px)
        y, sm = light_and_sensor(rng, img)
        tiles.append(y)
        metas.append((cq, pos, {**fm, **gm, **sm, "glyph_px": round(glyph_px, 1)}))
    with tempfile.TemporaryDirectory() as tmp:
        tiles, cm = codec_roundtrip(tiles, rng, tmp)
    lines = []
    for k, (t, (cq, pos, m)) in enumerate(zip(tiles, metas)):
        styles = m.pop("drums")
        for i, (q, p) in enumerate(zip(cq, pos)):
            cw = float(np.linalg.norm(q[1] - q[0]))
            if not np.all((q > -2) & (q < PATCH + 2)):
                continue  # a drum the pose put off the patch
            crop = rectify(t, clicked(rng, q, cw))
            name = f"{seed}_{k}_{i}.png"
            cv2.imwrite(os.path.join(out, "crops", name), crop)
            # A drum a reflection has washed out says nothing; it is kept,
            # marked, for a model that learns to say so.
            clipped = float((crop >= (250 if m["range"] == "full" else 232)).mean())
            p = float(p) % 10
            lines.append({"file": name, "pos": round(p, 3) % 10, "cls": pos_class(p),
                          "drum": styles[i], "occluded": clipped > 0.25, **m, **cm})
        if keep_sheet and k < 3:
            v = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            for q in cq:
                cv2.polylines(v, [np.int32(q)], True, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(out, "sheet", f"{seed}_{k}.png"), v)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("frames", type=int, help="full frames; each holds %d patches" % (COLS * ROWS))
    ap.add_argument("seed", type=int, nargs="?", default=20261021)
    ap.add_argument("--cameras", default="")
    ap.add_argument("--wall", default="")
    ap.add_argument("--fontfit", default="")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "sheet"), exist_ok=True)
    cams = [c for c in a.cameras.split(",") if c]
    jobs = [(a.seed * 100000 + i, a.out, a.fontfit, i % 50 == 0) for i in range(a.frames)]
    n = 0
    with Pool(a.jobs, initializer=init_worker, initargs=(cams, a.wall)) as pool, \
            open(os.path.join(a.out, "labels.jsonl"), "w") as f:
        for lines in pool.imap(make_frame, jobs):
            for l in lines:
                f.write(json.dumps(l) + "\n")
            n += len(lines)
    print(f"{n} crops from {a.frames} frames")


if __name__ == "__main__":
    main()
