# drum_crops.py <coco-dir> <out.raw> [n]
#
# Crops for checking the drum model's engine against the reference, bit for
# bit, made only from what the release build already holds hash-checked:
# COCO val2017. Exactness needs inputs that drive the arithmetic everywhere,
# not digits, so these are 20x32 luma patches of every scale cut from the
# photographs, after a few made to reach the edges of the input's
# quantization: a flat crop (its spread floored), crops whose standardised
# pixels saturate the int8 input, hard edges and smooth ramps. Written as
# <n> crops of 20x32 bytes back to back, the layout classify.c reads.
# Seeded: the same inputs give the same crops.
import os
import random
import sys

import cv2
import numpy as np

W, H = 20, 32


def edge_cases():
    yield np.full((H, W), 90, np.uint8)                     # flat: the floor
    one = np.full((H, W), 90, np.uint8)
    one[H // 2, W // 2] = 255                               # one hot pixel: saturates
    yield one
    yield ((np.indices((H, W)).sum(0) % 2) * 255).astype(np.uint8)   # checkerboard
    half = np.zeros((H, W), np.uint8)
    half[:, W // 2:] = 255                                  # a hard edge
    yield half
    yield np.tile(np.linspace(0, 255, W), (H, 1)).round().astype(np.uint8)   # ramps
    yield np.tile(np.linspace(0, 255, H)[:, None], (1, W)).round().astype(np.uint8)
    dim = np.full((H, W), 16, np.uint8)
    dim[8:24, 6:14] = 20                                    # a faint digit-like block
    yield dim


def main(coco, out, n=400):
    n = int(n)
    rng = random.Random(20261022)
    imgs = sorted(f for f in os.listdir(os.path.join(coco, "val2017")) if f.endswith(".jpg"))
    crops = list(edge_cases())
    while len(crops) < n:
        im = cv2.imread(os.path.join(coco, "val2017", rng.choice(imgs)), cv2.IMREAD_GRAYSCALE)
        ih, iw = im.shape
        h = rng.randint(H // 2, min(ih, 8 * H))
        w = max(1, round(h * W / H))
        if w > iw:
            continue
        y, x = rng.randint(0, ih - h), rng.randint(0, iw - w)
        crops.append(cv2.resize(im[y:y + h, x:x + w], (W, H), interpolation=cv2.INTER_AREA))
    np.stack(crops[:n]).astype(np.uint8).tofile(out)
    print(f"drum_crops: {n} crops to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
