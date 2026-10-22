# train_drum.py <synth-dir> <real-dirs> <out-dir> [epochs] [open-train-dirs]
#
# The realism gate, and the drum model's first shape: a classifier trained on
# synthetic crops, and optionally the train splits of open datasets, scored
# on real crops it never saw: <real-dirs> (comma-separated) each hold
# labels/*.jsonl and crops/, as realcells.py and opencells.py write them.
#
# The network is built from what majestic's int8 engine runs (engine/pdnet.c):
# 3x3 convolutions, depthwise 3x3, 1x1, ReLU; its last layer is a convolution
# the size of the last (2x2) map, which is a fully connected layer the engine
# needs no new op for. 20 classes, a half digit each. A drum between two half steps
# is taught as both, in proportion, so 2.3 is mostly "2.5" and some "2".
#
# Each crop is standardised by its own mean and spread before the network
# sees it: one camera's range is 16..235 and another's 0..255, a cupboard is
# darker than a stairwell, and that is not the network's business.
#
# Real scores are split the ways that matter: per camera, and for the wall
# camera whose faces the generator borrows, flagged as such; per light level;
# exact (the half step that was shown) and digit (the digit a reader takes).
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CLASSES

# Measured: on a shared 8-core host, 2 threads and channels-last train this
# net in a minute an epoch; all 8 took half an hour.
torch.set_num_threads(int(os.environ.get("THREADS", 2)))
CL = torch.channels_last


def standardise(x):
    x = x.astype(np.float32)
    m = x.mean(axis=(-1, -2), keepdims=True)
    s = x.std(axis=(-1, -2), keepdims=True)
    return (x - m) / np.maximum(s, 4.0)


def load_synth(d):
    L = [json.loads(l) for l in open(os.path.join(d, "labels.jsonl"))]
    L = [l for l in L if not l["occluded"]]
    x = np.stack([cv2.imread(os.path.join(d, "crops", l["file"]), cv2.IMREAD_GRAYSCALE) for l in L])
    pos = np.array([l["pos"] for l in L], np.float32)
    return standardise(x)[:, None], pos, L


def soft_targets(pos):
    t = (pos * 2) % CLASSES
    lo = np.floor(t).astype(int) % CLASSES
    hi = (lo + 1) % CLASSES
    w = t - np.floor(t)
    y = np.zeros((len(pos), CLASSES), np.float32)
    y[np.arange(len(pos)), lo] += 1 - w
    y[np.arange(len(pos)), hi] += w
    return y


def dw(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class DrumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),  # 32x20
            dw(16, 32, 2), dw(32, 32, 1),                                          # 16x10
            dw(32, 64, 2), dw(64, 64, 1),                                          # 8x5
            dw(64, 96, 2),                                                         # 4x3
            dw(96, 128, 2),                                                        # 2x2
        )
        # The FC, as a convolution over the whole 2x2 map: square, because the
        # engine's kernels are (a 4x3 one would need a new kernel shape).
        self.head = nn.Conv2d(128, CLASSES, 2)

    def forward(self, x):
        return self.head(self.body(x)).flatten(1)


def load_real(real_dirs):
    out = []
    for real_dir in real_dirs.split(","):
        for f in sorted(os.listdir(os.path.join(real_dir, "labels"))):
            for l in map(json.loads, open(os.path.join(real_dir, "labels", f))):
                if l["label"].isdigit():
                    l["img"] = cv2.imread(os.path.join(real_dir, "crops", l["file"]), cv2.IMREAD_GRAYSCALE)
                    out.append(l)
    return out


def evaluate(model, real, seen_cameras):
    model.eval()
    x = torch.from_numpy(standardise(np.stack([r["img"] for r in real]))[:, None]).contiguous(memory_format=CL)
    with torch.no_grad():
        p = model(x).argmax(1).numpy()
    rows = {}
    for r, c in zip(real, p):
        d = int(r["label"])
        light = "day" if r["frame_luma"] >= 60 else "dim" if r["frame_luma"] >= 30 else "night"
        cam = r["camera"] + (" (faces seen)" if r["camera"] in seen_cameras else "")
        # an unlit frame is no test of a reader; it is reported, not pooled
        for key in (f"{cam} / {light}",) + ((f"{cam} / lit",) if light != "night" else ()):
            s = rows.setdefault(key, [0, 0, 0])
            s[0] += 1
            s[1] += c == 2 * d
            s[2] += int(round(c / 2)) % 10 == d or c in ((2 * d - 1) % CLASSES, 2 * d)
    return {k: {"n": v[0], "exact": round(v[1] / v[0], 3), "digit": round(v[2] / v[0], 3)}
            for k, v in sorted(rows.items())}


def main(synth, real_dir, out, epochs=12, open_train=""):
    epochs = int(epochs)
    os.makedirs(out, exist_ok=True)
    x, pos, L = load_synth(synth)
    seen = {l["face"].split(":")[1].split("/")[0] for l in L if l["face"].startswith("real:")}
    n = len(x)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    val, tr = idx[:5000], idx[5000:]
    if open_train:
        # Real digits from open datasets' train splits: whole digits only,
        # so they teach what digits look like and the synthetic set teaches
        # how drums roll.
        o = load_real(open_train)
        xo = standardise(np.stack([r["img"] for r in o]))[:, None]
        tr = np.concatenate([tr, np.arange(n, n + len(o))])
        x = np.concatenate([x, xo])
        pos = np.concatenate([pos, np.array([float(r["label"]) for r in o], np.float32)])
        print(f"{len(o)} open training digits", flush=True)
    y = soft_targets(pos)
    real = load_real(real_dir)
    xt, yt = torch.from_numpy(x).contiguous(memory_format=CL), torch.from_numpy(y)
    model = DrumNet().to(memory_format=CL)
    print(f"{sum(p.numel() for p in model.parameters())} parameters, {len(tr)} train, "
          f"{len(val)} synthetic val, {len(real)} real", flush=True)
    opt = torch.optim.AdamW(model.parameters(), 3e-3, weight_decay=1e-4)
    steps = epochs * (len(tr) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    log = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(tr)
        for b in range(0, len(perm) - 255, 256):
            j = perm[b:b + 256]
            xb = xt[j]
            # a pixel of slip either way: the camera's crop is never exact
            dx, dy = rng.integers(-1, 2, 2)
            xb = torch.roll(xb, (int(dy), int(dx)), (2, 3))
            loss = -(yt[j] * F.log_softmax(model(xb), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xt[val]).argmax(1).numpy()
        tv = np.round(pos[val] * 2).astype(int) % CLASSES
        acc = float((pv == tv).mean())
        near = float((np.minimum((pv - tv) % CLASSES, (tv - pv) % CLASSES) <= 1).mean())
        ev = evaluate(model, real, seen)
        log.append({"epoch": ep, "loss": round(float(loss.detach()), 4), "synth_exact": round(acc, 3),
                    "synth_near": round(near, 3), "real": ev, "sec": round(time.time() - t0)})
        print(json.dumps(log[-1]), flush=True)
    torch.save(model.state_dict(), os.path.join(out, "drum.pt"))
    json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
