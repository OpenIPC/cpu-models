# What the meter tools share: the cell crop the network is given, and the one
# way every crop is cut.
#
# A drum cell is cut by the homography from its quad (corners TL, TR, BR, BL
# as the digit stands) to a CELL_W x CELL_H rectangle, bilinear, from the
# luma plane in the limited range the VPSS delivers it (16..235). The camera
# will cut it the same way, so real test crops, synthetic training crops and
# the camera all see one geometry and one range.
import hashlib
import json
import os
import subprocess

import cv2
import numpy as np

CELL_W, CELL_H = 20, 32

# A drum's position: the digit fully in the window is d.0; d.5 is halfway to
# the next, which enters from below. Half-steps are the classes.
CLASSES = 20


def pos_class(pos):
    return int(round(pos * 2)) % CLASSES


def rectify(y, quad, w=CELL_W, h=CELL_H):
    m = cv2.getPerspectiveTransform(np.float32(quad),
                                    np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(y, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def to_limited(v):
    # 0..1 (black..white) to limited-range luma
    return np.clip(16 + 219 * v, 0, 255)


def decode_luma(hevc):
    # The Y plane exactly as decoded, not via RGB.
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0",
                            hevc], check=True, capture_output=True, text=True).stdout
    w, h = map(int, probe.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "hevc", "-i", hevc,
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8)[:w * h].reshape(h, w)


def load_camera(path):
    return json.load(open(path))


def harvested(root, cam, luma_min=None):
    # (record, path stem) for every kept frame of a camera, oldest first:
    # wall frames (harvest.py) are <at>_<id>, lab frames (labcam.py) <at>
    idx = [json.loads(l) for l in open(os.path.join(root, cam, "index.jsonl"))]
    idx.sort(key=lambda r: r["at"])
    for r in idx:
        if luma_min is not None and r["luma"] < luma_min:
            continue
        stem = f"{r['at']}_{r['id']}" if "id" in r else str(r["at"])
        yield r, os.path.join(root, cam, stem)


def frame_luma(stem):
    # A wall frame is the camera's own bitstream; a lab frame is its Y plane.
    if os.path.exists(stem + ".hevc"):
        return decode_luma(stem + ".hevc")
    return cv2.imread(stem + ".png", cv2.IMREAD_GRAYSCALE)


def cells_at(meter, at):
    # A camera that was moved has one set of cells per placement: "views",
    # each {from, to, cells}; a camera never moved has plain "cells".
    for v in meter.get("views", []):
        if v["from"] <= at and (v["to"] is None or at < v["to"]):
            return v["cells"]
    return meter.get("cells")


def face_homography(quads):
    """Frame pixels to the meter's face, in cell widths: the first drum cell
    has its top left at (0, 0), each cell is 1 wide and as tall as the boxes
    are, at their measured spacing. A dial or anything else printed on the
    face is annotated in these units once and found again in any frame and
    after any re-annotation of the cells. Returns (H, pitch, height)."""
    q = np.float32(quads)
    n = len(q)
    centres = q.mean(1)
    wbox = float(np.mean([np.linalg.norm(c[1] - c[0]) for c in q]))
    hbox = float(np.mean([np.linalg.norm(c[3] - c[0]) for c in q]))
    pitch = float(np.linalg.norm(centres[-1] - centres[0]) / max(n - 1, 1)) / wbox
    ch = hbox / wbox
    dst = []
    for i in range(n):
        x = i * pitch
        dst += [[x, 0], [x + 1, 0], [x + 1, ch], [x, ch]]
    h, _ = cv2.findHomography(q.reshape(-1, 2), np.float32(dst))
    return h, pitch, ch


def reading_at(meter, at):
    for r in meter["readings"]:
        if r["from"] <= at and (r["to"] is None or at < r["to"]):
            return r["drums"]
    return None


FONTS_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts.txt")


def pinned_fonts(fonts_dir):
    """The fonts fonts.txt pins, each checked against its hash, by file name
    in <fonts_dir>: a render or a fit is reproducible only from the bytes
    that were pinned, not from whatever .ttf is lying in the directory
    (scripts/fetch.sh tools/meters/fonts.txt <fonts_dir> fetches them)."""
    fonts = []
    for line in open(FONTS_LIST):
        if not line.strip() or line.startswith("#"):
            continue
        name, sha = line.split()[:2]
        path = os.path.join(fonts_dir, name)
        try:
            got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        except FileNotFoundError:
            raise SystemExit(f"{path}: missing; bash scripts/fetch.sh tools/meters/fonts.txt {fonts_dir}")
        if got != sha:
            raise SystemExit(f"{path}: sha256 {got}, fonts.txt pins {sha}")
        fonts.append(name)
    return sorted(fonts)
