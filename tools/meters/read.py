# read.py <model.pt> <out.jsonl> <camera.json>:<harvest-dir> [...]
#
# What the drum model makes of every annotated frame: for each meter, each
# drum's whole distribution over the 20 half-step classes, not just its best
# guess. A fusion that only sees argmaxes cannot weigh one camera's
# hesitation against another's certainty; this keeps the hesitation.
#
# A frame with nothing in it is marked so, not read: a dark room at night
# makes a picture of pure gain noise (structure no larger than the pixel
# noise), and an unlit cupboard a black one (no contrast on the drums). The
# model would read digits into either, with confidence.
#
# A meter with red dials (its "dials", see dials.py) gets each needle's value
# and how clear it was, alongside its drums.
#
# The probabilities are the float model's (the training checkpoint), so what
# read.py and fuse.py report is the reading's arithmetic over that model.
# The camera runs the int8 export; what quantization costs is measured
# against the same float model by check_drum.py (0.2 points on the open
# water-meter split), and is not folded in here.
import json
import sys

import cv2
import numpy as np
import torch

from common import cells_at, frame_luma, harvested, load_camera, rectify
from dials import read_dial
from train_drum import DrumNet, standardise


MIN_SNR = 2.0       # 9x9-smoothed spread over the median pixel step
MIN_CONTRAST = 6.0  # luma spread over a meter's drum crops


def snr(y):
    y = y.astype(np.float32)
    return float(cv2.blur(y, (9, 9)).std() / (np.median(np.abs(np.diff(y, axis=1))) + 1))


def main(model, out, *sources):
    net = DrumNet()
    net.load_state_dict(torch.load(model))
    net.eval()
    n = 0
    with open(out, "w") as f:
        for src in sources:
            camera, root = src.split(":", 1)
            cam = load_camera(camera)
            for rec, stem in harvested(root, cam["camera"]):
                y = None
                for m in cam["meters"]:
                    cells = cells_at(m, rec["at"])
                    if cells is None:
                        continue
                    if y is None:
                        y = frame_luma(stem)
                        ratio = snr(y)
                    crops = np.stack([rectify(y, q) for q in cells])
                    contrast = float(crops.std())
                    x = standardise(crops)[:, None]
                    with torch.no_grad():
                        p = torch.softmax(net(torch.from_numpy(x)), 1).numpy()
                    f.write(json.dumps({"camera": cam["camera"], "meter": m["name"],
                                        "meter_id": m.get("meter_id", cam["camera"] + "/" + m["name"]),
                                        "at": rec["at"], "frame_luma": rec["luma"],
                                        "snr": round(ratio, 2), "contrast": round(contrast, 1),
                                        "readable": ratio >= MIN_SNR and contrast >= MIN_CONTRAST,
                                        "p": np.round(p, 5).tolist(),
                                        "dials": [[round(v, 3), round(c, 2), d["mult"]] for d in m.get("dials", [])
                                                  for v, c in [read_dial(y, cells, d)[:2]]]}) + "\n")
                    n += 1
    print(f"{n} meter frames read")


if __name__ == "__main__":
    main(*sys.argv[1:])
