# fuse.py <readings.jsonl> <camera.json> [...] [--out fused.jsonl]
#
# One reading per meter per moment, out of every camera that watches it and
# every moment before it.
#
# A reading is decoded whole, not drum by drum: a candidate H (the meter's
# count in half steps of its last drum) says where every drum must stand.
# The last drum turns with H; a drum above it stands on its digit, unless
# every drum below it reads 9 and the last is rolling, when it rolls too.
# A candidate scores the sum, over drums, of the log of the probability the
# model gave the class it implies, so an odometer that cannot exist (a carry
# half done on one drum and not the next) is never read.
#
# Cameras: frames of one meter (its meter_id) taken within STEP seconds are
# one moment, and a candidate's score there is the sum of its score in each
# camera, so a camera blinded by glare is outvoted by its hesitation, not by
# a vote.
#
# Time: a meter never runs backwards (its last drum may seem to, by BACK half
# steps at a cost: that is the reading's noise, not the meter), nor faster
# than max_per_hour of its last drum's units, and mostly it runs at about
# typical_per_hour: a step is charged its size over what that rate would turn
# in the time. The best path through the moments is found by Viterbi; a
# moment no candidate of which can follow is let through at a cost, so one
# bad frame cannot end the run.
#
# Trust: a camera is weighted by how often its own reading agreed with the
# fused one, found by iterating (EM, in effect). An IR-lit camera that reads
# an 8 as a 1 every frame is confident and wrong, and only the other cameras
# and time can tell it so; its weight is how much it is believed after.
#
# Scored against the cameras' labelled readings: each camera alone, cameras
# together, and cameras together over time, on the drums above the last.
import itertools
import json
import math
import sys

import numpy as np

from common import CLASSES, load_camera, reading_at

STEP = 300          # seconds within which frames of one meter are one moment
TOPK = 40           # candidates kept per camera frame
JUMP_COST = 100.0   # log-likelihood a path pays to break the rule once
ROBUST = 8.0        # the most one moment can count against a candidate
ROUNDS = 4          # trust re-estimations
TEMP = 2.0          # the model is overconfident on cameras it never saw
BACK = 4            # half steps the last drum may seem to go back: read noise
BACK_COST = 1.0     # per half step it does


def drum_classes(H, n):
    """Class (half step) of each of n drums, most significant first, for a
    count of H half steps of the last drum."""
    V, half = divmod(H, 2)
    out = [0] * n
    out[-1] = (2 * (V % 10) + half) % CLASSES
    lower_all_nine = True
    for k in range(1, n):
        lower_all_nine = lower_all_nine and (V // 10 ** (k - 1)) % 10 == 9
        d = (V // 10 ** k) % 10
        out[n - 1 - k] = (2 * d + (1 if (half and lower_all_nine) else 0)) % CLASSES
    return out


def score(H, logp):
    return float(sum(logp[i, c] for i, c in enumerate(drum_classes(H, len(logp)))))


def candidates(logp, k=3):
    """The counts worth scoring for one camera frame: each drum's most likely
    digits, combined."""
    n = len(logp)
    opts = []
    for i in range(n - 1):
        top = np.argsort(-logp[i])[:k]
        ds = sorted({int(c // 2) for c in top} | {int((c + 1) // 2) % 10 for c in top if c % 2})
        opts.append(ds)
    last = [int(c) for c in np.argsort(-logp[-1])[:k]]
    out = set()
    for ds in itertools.product(*opts):
        hi = 0
        for d in ds:
            hi = hi * 10 + d
        for c in last:
            out.add(hi * 20 + c)  # hi * 10 digits * 2 halves + the last drum's class
    return out


def viterbi(steps, weight, rate, typical):
    """The best path of counts through the moments. Every moment scores the
    same candidates (all the cameras' own, over all moments: the true count
    need not have been a bad frame's own guess), and no moment can hold a
    candidate more than ROBUST below its best, so one misread frame weighs
    as one frame and never more."""
    C = np.array(sorted(set().union(*(set(f["own"]) for st in steps for f in st))))
    sc = []
    for st in steps:
        s = np.zeros(len(C))
        for f in st:
            s += weight[f["camera"]] * np.array([score(int(H), f["logp"]) for H in C])
        sc.append(np.maximum(s, s.max() - ROBUST))
    D = C[None, :] - C[:, None]  # D[i, j]: from candidate i to candidate j
    best, back = sc[0].copy(), []
    for t in range(1, len(steps)):
        dt = (steps[t][0]["at"] - steps[t - 1][0]["at"]) / 3600
        limit = 2 * rate * dt + 2
        expect = 2 * typical * dt + 1
        ok = (D >= -BACK) & (D <= limit)
        cost = np.where(ok, np.where(D >= 0, D / expect, -D * BACK_COST), JUMP_COST)
        tot = best[:, None] - cost
        k = np.argmax(tot, 0)
        back.append(k)
        best = tot[k, np.arange(len(C))] + sc[t]
    path = [int(np.argmax(best))]
    for k in reversed(back):
        path.append(int(k[path[-1]]))
    path.reverse()
    return path, [list(C)] * len(steps), sc


def fmt(H, n):
    V, half = divmod(H, 2)
    return str(V).zfill(n)[-n:] + (".5" if half else "")


def main(path, *args):
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    meters = {}
    for c in args:
        cam = load_camera(c)
        for m in cam["meters"]:
            meters[(cam["camera"], m["name"])] = m
    frames = [json.loads(l) for l in open(path)]
    frames = [f for f in frames if f.get("readable", True)]
    by_meter = {}
    for f in frames:
        f["logp"] = np.log(np.clip(np.array(f["p"]), 1e-6, 1)) / TEMP
        by_meter.setdefault(f["meter_id"], []).append(f)

    report, fused_rows = {}, []
    for mid, fs in by_meter.items():
        fs.sort(key=lambda f: f["at"])
        n = len(fs[0]["p"])
        rate = max(meters[(f["camera"], f["meter"])].get("max_per_hour", 1e9) for f in fs)
        # moments
        steps, cur = [], []
        for f in fs:
            if cur and f["at"] - cur[0]["at"] > STEP:
                steps.append(cur)
                cur = []
            cur.append(f)
        if cur:
            steps.append(cur)
        typical = max(meters[(f["camera"], f["meter"])].get("typical_per_hour", rate / 10) for f in fs)
        cams = sorted({f["camera"] for f in fs})
        weight = {c: 1.0 for c in cams}
        for f in fs:
            own = sorted(candidates(f["logp"]), key=lambda H: -score(H, f["logp"]))[:TOPK]
            f["alone"], f["own"] = own[0], own
        # Trust needs a second opinion: one camera only agrees with itself.
        for _ in range(ROUNDS if len(cams) > 1 else 1):
            path, cand, sc = viterbi(steps, weight, rate, typical)
            # trust: agreement of each camera's own reading with the fused one
            agree = {c: [0, 0] for c in cams}
            for t, st in enumerate(steps):
                Hv = int(cand[t][path[t]])
                for f in st:
                    agree[f["camera"]][0] += 1
                    agree[f["camera"]][1] += fmt(f["alone"], n)[:n - 1] == fmt(Hv, n)[:n - 1]
            if len(cams) > 1:
                weight = {c: max(a[1] / a[0], 0.05) for c, a in agree.items()}
        print(mid, "trust", {c: round(w, 2) for c, w in weight.items()})

        def truth(f):
            r = reading_at(meters[(f["camera"], f["meter"])], f["at"])
            return None if r is None or "?" in r[:-1] or "r" in r[:-1] else r[:-1]

        rows = report.setdefault(mid, {"alone": [0, 0], "cameras": [0, 0], "cameras+time": [0, 0]})
        for t, st in enumerate(steps):
            plain = np.array([sum(score(int(H), f["logp"]) for f in st) for H in cand[t]])
            Ht = int(cand[t][int(np.argmax(plain))])
            Hv = int(cand[t][path[t]])
            s = np.sort(sc[t])[::-1]  # trusted scores
            margin = float(s[0] - s[1]) if len(s) > 1 else math.inf
            tr = truth(st[0])
            row = {"meter_id": mid, "at": st[0]["at"], "cameras": [f["camera"] for f in st],
                   "reading": fmt(Hv, n), "cameras_only": fmt(Ht, n), "margin": round(margin, 2),
                   "alone": {f["camera"]: fmt(f["alone"], n) for f in st}, "truth": tr}
            fused_rows.append(row)
            if tr is None:
                continue
            for f in st:
                rows["alone"][0] += 1
                rows["alone"][1] += fmt(f["alone"], n)[:n - 1] == tr
            rows["cameras"][0] += 1
            rows["cameras"][1] += fmt(Ht, n)[:n - 1] == tr
            rows["cameras+time"][0] += 1
            rows["cameras+time"][1] += fmt(Hv, n)[:n - 1] == tr
    for mid, r in report.items():
        print(mid, {k: f"{v[1]}/{v[0]}" for k, v in r.items()})
    if out_path:
        with open(out_path, "w") as f:
            for r in fused_rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
