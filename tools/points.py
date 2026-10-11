# points.py <labels.json> <dets> <out.json> <out.md>
#
# What majestic's three object-detection presets mean for the person model:
# the score threshold of each, measured on the labelled crops (crops.py) from
# the engine's own detections (dets). A box is right when it overlaps a person
# by IoU 0.5; people marked ignore are neither a hit nor a miss. Matching is
# greedy by score, so one pass gives exact precision and recall at every
# threshold.
#
#   balanced          the F1-optimal threshold
#   fewerFalseAlarms  the F0.5 optimum (precision weighted)
#   catchEverything   the highest threshold with recall >= 0.90, else the
#                     F2 optimum
#
# Writes {"presets": {preset: {"human": threshold}}} for majestic to read
# beside the model, and a table of what each preset achieves, including the
# false alarms per person-free crop.
import json
import sys
from collections import defaultdict

import numpy as np

PRESETS = ["fewerFalseAlarms", "balanced", "catchEverything"]


def iou(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def main(labels_path, dets_path, out_json, out_md):
    labels = json.load(open(labels_path))
    dets = defaultdict(list)
    for line in open(dets_path):
        p = line.split()
        if len(p) == 6:
            dets[p[0]].append((int(p[5]), [float(v) for v in p[1:5]]))
    npos = sum(1 for g in labels.values() for b in g if not b["ignore"])
    empty = [c for c, g in labels.items() if not g]
    rec, bg = [], []
    for c, g in labels.items():
        used = [False] * len(g)
        for s, box in sorted(dets.get(c, []), key=lambda d: -d[0]):
            best, bi = 0.0, -1
            for i, b in enumerate(g):
                if used[i] and not b["ignore"]:
                    continue
                o = iou(box, b["box"])
                if o > best:
                    best, bi = o, i
            if best >= 0.5:
                if g[bi]["ignore"]:
                    continue
                used[bi] = True
                rec.append((s, True))
            else:
                rec.append((s, False))
            if not g:
                bg.append(s)
    s = np.array([r[0] for r in rec])
    hit = np.array([r[1] for r in rec], bool)
    bg = np.array(bg)
    ts = np.arange(1, 100)
    tp = np.array([(hit & (s >= t)).sum() for t in ts], float)
    fp = np.array([(~hit & (s >= t)).sum() for t in ts], float)
    some = tp + fp > 0
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=some)
    r = tp / max(npos, 1)
    f = lambda b: (1 + b * b) * p * r / np.maximum(b * b * p + r, 1e-9)
    ok = np.nonzero(r >= 0.90)[0]
    idx = {"fewerFalseAlarms": int(f(0.5).argmax()), "balanced": int(f(1).argmax()),
           "catchEverything": int(ok[-1]) if len(ok) else int(f(2).argmax())}
    presets, rows = {}, []
    for name in PRESETS:
        i = idx[name]
        t = ts[i] / 100
        presets[name] = {"human": round(float(t), 2)}
        rows.append(f"| {name} | {t:.2f} | {p[i]:.3f} | {r[i]:.3f} | "
                    f"{2 * p[i] * r[i] / max(p[i] + r[i], 1e-9):.3f} | "
                    f"{(bg >= ts[i]).sum() / max(len(empty), 1):.3f} |")
    with open(out_json, "w") as o:
        json.dump({"presets": presets}, o, indent=1, sort_keys=True)
    table = "\n".join([
        f"{len(labels)} crops of COCO val2017 ({len(empty)} with nobody in them), "
        f"{npos} people counted.",
        "",
        "| preset | threshold | precision | recall | F1 | false alarms per person-free crop |",
        "|---|---|---|---|---|---|"] + rows) + "\n"
    with open(out_md, "w") as o:
        o.write(table)
    print(table)


if __name__ == "__main__":
    main(*sys.argv[1:])
