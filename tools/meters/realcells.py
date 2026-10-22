# realcells.py <camera.json> <harvest-dir> <out-dir>
#
# Cuts every annotated drum cell out of every kept frame of a camera, through
# the same rectifier the camera will run (common.rectify), into:
#   <out>/crops/*.png    CELL_W x CELL_H luma, the real test set
#   <out>/labels/<camera>.jsonl
#                        what each crop shows, and the frame's mean luma so a
#                        score can be split by light
#   <out>/bank/<d>/*.png the cells of daylight frames whose digit is known,
#                        cut 3x oversampled: the glyph style the synthetic
#                        digits are matched to
#
# <harvest-dir> holds <camera>/ as harvest.py or labcam.py keep it.
#
# Real frames are only ever a test set and a style reference, never training
# samples: a model scored on the frames it was trained on says nothing.
import json
import os
import sys

import cv2

from common import CELL_H, CELL_W, cells_at, frame_luma, harvested, load_camera, reading_at, rectify

DAYLIGHT = 60  # mean frame luma above which a frame is daylight here


def main(camera, root, out):
    cam = load_camera(camera)
    os.makedirs(f"{out}/crops", exist_ok=True)
    os.makedirs(f"{out}/labels", exist_ok=True)
    n = 0
    with open(f"{out}/labels/{cam['camera']}.jsonl", "w") as lab:
        for rec, stem in harvested(root, cam["camera"]):
            y = frame_luma(stem)
            for m in cam["meters"]:
                drums = reading_at(m, rec["at"])
                cells = cells_at(m, rec["at"])
                if cells is None:
                    continue  # a placement nobody has annotated yet
                for i, quad in enumerate(cells):
                    name = f"{cam['camera']}_{rec['at']}_{m['name']}_{i}.png"
                    cv2.imwrite(f"{out}/crops/{name}", rectify(y, quad))
                    label = drums[i] if drums else "?"
                    lab.write(json.dumps({"file": name, "camera": cam["camera"], "meter": m["name"],
                                          "cell": i, "at": rec["at"], "label": label,
                                          "frame_luma": rec["luma"]}) + "\n")
                    n += 1
                    if label.isdigit() and rec["luma"] >= DAYLIGHT:
                        d = f"{out}/bank/{label}"
                        os.makedirs(d, exist_ok=True)
                        cv2.imwrite(f"{d}/{name}", rectify(y, quad, CELL_W * 3, CELL_H * 3))
    print(f"{n} cells cut")


if __name__ == "__main__":
    main(*sys.argv[1:])
