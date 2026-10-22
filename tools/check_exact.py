# check_exact.py <model.onnx> <crop.nv21>...
#
# The engine must compute what the quantized model defines. For each crop,
# <crop>.heads (written by dump_heads, from qnet.c) must equal qref.py's
# reference arithmetic exactly, every value of all six head maps. Exits
# non-zero on any difference, so a release stops before it measures or
# publishes anything.
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import qref  # noqa: E402

W, H = 320, 192


def chw_from_nv21(y, vu):
    """The float model's input, as majestic's preprocessing defines it."""
    ky = np.float32(255 / 219 / 128)
    kc = np.float32(255 / 224 / 128)
    Y = (y.astype(np.float32) - np.float32(16)) * ky - np.float32(1)
    V = (vu[:, 0::2].astype(np.float32) - np.float32(128)) * kc
    U = (vu[:, 1::2].astype(np.float32) - np.float32(128)) * kc
    up = lambda p: np.repeat(np.repeat(p, 2, 0), 2, 1)
    return np.stack([Y, up(U), up(V)]).astype(np.float32)


g = qref.Graph(sys.argv[1])
crops = sys.argv[2:]
if not crops:
    sys.exit("check_exact: no crops")
bad = 0
for f in crops:
    raw = np.fromfile(f, np.uint8)
    ref = qref.run(g, chw_from_nv21(raw[:W * H].reshape(H, W), raw[W * H:].reshape(H // 2, W)),
                  head="float")
    got = np.fromfile(f + ".heads", np.float32)
    want = np.concatenate([r.transpose(1, 2, 0).reshape(-1) for r in ref]).astype(np.float32)
    if got.shape != want.shape or not np.array_equal(got, want):
        n = int(np.sum(got != want)) if got.shape == want.shape else -1
        print(f"check_exact: {f}: {n} values differ")
        bad += 1
if bad:
    sys.exit(f"check_exact: {bad} of {len(crops)} crops differ from the reference")
print(f"check_exact: all {len(crops)} crops identical to the reference")
