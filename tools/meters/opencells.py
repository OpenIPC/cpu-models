# opencells.py <yolo-dataset-dir> <out-dir> <name> [splits=test,valid]
#
# Digit crops out of an open dataset whose digits are boxed one by one
# (YOLO labels, classes named "0".."9"), cut the way a camera cuts a drum
# cell: each box grown to the cell's shape around its centre (a digit's box
# is tighter than the drum's window), then through common.rectify from luma.
# Writes <out>/crops/<name>_*.png and <out>/labels/<name>.jsonl in the same
# form realcells.py does, so train_drum.py scores them like any camera.
#
# These are other people's photographs, many meters, many phones: a test of
# whether the model reads meters it was never shown, never training data.
import json
import os
import sys

import cv2
import numpy as np

from common import CELL_H, CELL_W, rectify

GROW = 1.15  # cell height over digit-box height


def names(d):
    for line in open(os.path.join(d, "data.yaml")):
        if line.startswith("names:"):
            return [n.strip().strip("'\"") for n in line.split("[", 1)[1].rsplit("]", 1)[0].split(",")]
    raise SystemExit("no names in data.yaml")


def main(d, out, name, splits="test,valid"):
    cls = names(d)
    os.makedirs(f"{out}/crops", exist_ok=True)
    os.makedirs(f"{out}/labels", exist_ok=True)
    n = 0
    with open(f"{out}/labels/{name}.jsonl", "w") as lab:
        for split in splits.split(","):
            for f in sorted(os.listdir(f"{d}/{split}/images")):
                stem = os.path.splitext(f)[0]
                lp = f"{d}/{split}/labels/{stem}.txt"
                if not os.path.exists(lp):
                    continue
                bgr = cv2.imread(f"{d}/{split}/images/{f}")
                if bgr is None:
                    continue
                y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
                H, W = y.shape
                rows = [r.split() for r in open(lp)]
                rows = [r for r in rows if len(r) == 5 and cls[int(r[0])].isdigit()]
                # The row the digits sit on, as the camera's annotation would
                # give it: a line through their centres. Its direction is
                # taken left to right, so a meter photographed upside down
                # stays upside down; nothing here can know.
                c = np.array([[float(r[1]) * W, float(r[2]) * H] for r in rows])
                ang = 0.0
                hmed = float(np.median([float(r[4]) * H for r in rows])) if rows else 0
                if len(c) >= 3:
                    vx, vy, x0, y0 = cv2.fitLine(c.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                    off = np.abs((c[:, 0] - x0) * vy - (c[:, 1] - y0) * vx)
                    # several numbers in one photo (a dashboard): no one row
                    if np.median(off) < 0.5 * hmed:
                        if vx < 0:
                            vx, vy = -vx, -vy
                        ang = float(np.arctan2(vy, vx))
                u = np.array([np.cos(ang), np.sin(ang)])
                v = np.array([-u[1], u[0]])
                # A cell ends where the next drum begins, as a camera's
                # annotation splits a window: never wider than the distance
                # to the nearest digit.
                dist = np.linalg.norm(c[:, None] - c[None], axis=-1) if len(c) else np.zeros((0, 0))
                np.fill_diagonal(dist, np.inf)
                for k, r in enumerate(rows):
                    cx, cy, bw, bh = (float(t) for t in r[1:])
                    # the box's extent across the row, from its axis-aligned size
                    ext = abs(v[0]) * bw * W + abs(v[1]) * bh * H
                    h = ext * GROW
                    w = min(h * CELL_W / CELL_H, 0.95 * float(dist[k].min()) if len(c) > 1 else np.inf)
                    ctr = np.array([cx * W, cy * H])
                    quad = [(ctr - u * w / 2 - v * h / 2).tolist(), (ctr + u * w / 2 - v * h / 2).tolist(),
                            (ctr + u * w / 2 + v * h / 2).tolist(), (ctr - u * w / 2 + v * h / 2).tolist()]
                    out_name = f"{name}_{split}_{stem[:40]}_{k}.png"
                    cv2.imwrite(f"{out}/crops/{out_name}", rectify(y, quad))
                    lab.write(json.dumps({"file": out_name, "camera": name, "meter": stem[:40], "cell": k,
                                          "at": 0, "label": cls[int(r[0])], "row_deg": round(float(np.degrees(ang)), 1),
                                          "frame_luma": round(float(y.mean()), 1)}) + "\n")
                    n += 1
    print(f"{name}: {n} digits")


if __name__ == "__main__":
    main(*sys.argv[1:])
