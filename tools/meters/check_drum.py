# check_drum.py <drum_int8.onnx> <crops.raw> [exact-n] [<drum_float.onnx> <labels.jsonl>]
#
# Two checks of the drum blob as the camera's engine runs it, from
# <crops.raw>.out (written by classify.c):
#
# exact: for the first exact-n crops (default 200; 0 for all), the engine's
#   int8 input must equal the reference's standardisation of the crop, and
#   its 20 logits must equal qref.py's arithmetic on the quantized graph,
#   every one. Exits non-zero on any difference. This is the release build's
#   check, on crops of anything (drum_crops.py).
#
# scores, given the float model and labels: the digit read, the int8
#   engine's against the float model's (onnxruntime on the float ONNX), per
#   test set in <labels.jsonl>: what quantization costs, on real crops only.
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")
import qref  # noqa: E402

FLOOR = 4.0
CLASSES = 20


def reference_input(crop):
    """The engine's standardisation, in double, before quantization."""
    c = crop.astype(np.float64)
    m = c.mean()
    s = max(np.sqrt(((c - m) ** 2).mean()), FLOOR)
    return (c - m) / s


def digit_ok(c, d):
    return int(round(c / 2)) % 10 == d or c in ((2 * d - 1) % CLASSES, 2 * d)


def main(q_onnx, raw, exact_n=200, f_onnx=None, labels=None):
    exact_n = int(exact_n)
    g = qref.Graph(q_onnx)
    _, _, h, w = g.input_shape
    crops = np.fromfile(raw, np.uint8).reshape(-1, h, w)
    rec = np.fromfile(raw + ".out", np.uint8).reshape(len(crops), -1)
    eng_in = rec[:, :h * w].view(np.int8).reshape(-1, h, w)
    eng_lg = rec[:, h * w:].copy().view(np.float32)
    qi = next(n for n in g.nodes if n.op_type == "QuantizeLinear" and n.input[0] == g.input)
    s, zp = g.qparams[qi.output[0]]
    if exact_n <= 0:
        exact_n = len(crops)
    bad = 0
    for i in range(min(exact_n, len(crops))):
        x = reference_input(crops[i])
        want_in = np.clip(np.round(x / s) + zp, -128, 127).astype(np.int8)
        want = qref.run(g, x[None], head="float")[0].reshape(-1).astype(np.float32)
        if not np.array_equal(eng_in[i], want_in) or not np.array_equal(eng_lg[i], want):
            bad += 1
            if bad <= 3:
                print(f"check_drum: crop {i}: input {int(np.sum(eng_in[i] != want_in))} differ, "
                      f"logits {int(np.sum(eng_lg[i] != want))} differ")
    if bad:
        sys.exit(f"check_drum: {bad} of {min(exact_n, len(crops))} crops differ from the reference")
    print(f"check_drum: {min(exact_n, len(crops))} crops: input and logits identical to the reference")

    if not f_onnx:
        return
    import onnxruntime as ort  # scoring only: the release build has no float model

    L = [json.loads(l) for l in open(labels)]
    sess = ort.InferenceSession(f_onnx, providers=["CPUExecutionProvider"])
    xs = np.stack([reference_input(c) for c in crops]).astype(np.float32)[:, None]
    # the float ONNX takes one crop at a time, as the camera does
    fl = np.concatenate([sess.run(None, {"input": xs[i:i + 1]})[0].reshape(-1, CLASSES)
                         for i in range(len(xs))])
    pe, pf = eng_lg.argmax(1), fl.argmax(1)
    rows = defaultdict(lambda: [0, 0, 0, 0])
    for l, a, b in zip(L, pe, pf):
        d = int(l["label"])
        r = rows[l["set"]]
        r[0] += 1
        r[1] += digit_ok(a, d)
        r[2] += digit_ok(b, d)
        r[3] += a == b
    for k, (n, e, f, same) in sorted(rows.items()):
        print(f"  {k:28s} n={n:5d}  int8 {e / n:.3f}  float {f / n:.3f}  same class {same / n:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
