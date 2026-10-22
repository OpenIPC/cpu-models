# export_drum.py <drum.pt> <synth-dir> <open-train-dir> <out-dir>
#
# The drum model as the camera's engine runs it: BatchNorm folded into the
# convolutions, exported to ONNX ending in the head's 1x1 map (no flatten:
# the engine's last op must be a convolution), quantized statically to QDQ
# int8 by onnxruntime (per-channel symmetric weights, asymmetric
# activations, calibrated on crops of both kinds the model trained on), and
# written as a blob by tools/export_qnet.py.
#
# Out: drum_float.onnx, drum_int8.onnx, drum.qnet in <out-dir>.
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)

from common import CELL_H, CELL_W
from train_drum import DrumNet, load_real, standardise

CALIB = 3000
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def fold(conv, bn):
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
    s = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    out = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, groups=conv.groups, bias=True)
    out.weight.data = w * s.reshape(-1, 1, 1, 1)
    out.bias.data = (b - bn.running_mean) * s + bn.bias.detach()
    return out


def folded(model):
    """The same network with every Conv+BatchNorm one Conv, as a flat
    Sequential ending in the head's map."""
    mods = []
    flat = []
    for m in model.body.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
            flat.append(m)
    i = 0
    while i < len(flat):
        m = flat[i]
        if isinstance(m, nn.Conv2d) and i + 1 < len(flat) and isinstance(flat[i + 1], nn.BatchNorm2d):
            mods.append(fold(m, flat[i + 1]))
            i += 2
        else:
            mods.append(m)
            i += 1
    mods.append(model.head)
    return nn.Sequential(*mods).eval()


def inline_identities(path):
    """The TorchScript exporter hands some biases to their Conv through an
    Identity of an initializer; the quantizer then cannot see them as
    constants and leaves them float. Make each such Identity's output the
    initializer itself."""
    m = onnx.load(path)
    init = {t.name: t for t in m.graph.initializer}
    keep = []
    for n in m.graph.node:
        if n.op_type == "Identity" and n.input[0] in init:
            t = onnx.TensorProto()
            t.CopyFrom(init[n.input[0]])
            t.name = n.output[0]
            m.graph.initializer.append(t)
        else:
            keep.append(n)
    del m.graph.node[:]
    m.graph.node.extend(keep)
    onnx.checker.check_model(m)
    onnx.save(m, path)


class Reader(CalibrationDataReader):
    def __init__(self, x):
        self.it = iter([{"input": x[i:i + 1]} for i in range(len(x))])

    def get_next(self):
        return next(self.it, None)


def calibration(synth, open_train, rng):
    L = [json.loads(l) for l in open(os.path.join(synth, "labels.jsonl"))]
    L = [l for l in L if not l["occluded"]]
    pick = rng.choice(len(L), CALIB * 2 // 3, replace=False)
    xs = [cv2.imread(os.path.join(synth, "crops", L[i]["file"]), cv2.IMREAD_GRAYSCALE) for i in pick]
    o = load_real(open_train)
    pick = rng.choice(len(o), CALIB - len(xs), replace=False)
    xs += [o[i]["img"] for i in pick]
    return standardise(np.stack(xs))[:, None].astype(np.float32)


def main(pt, synth, open_train, out):
    os.makedirs(out, exist_ok=True)
    model = DrumNet()
    model.load_state_dict(torch.load(pt))
    model.eval()
    net = folded(model)
    x = torch.randn(4, 1, CELL_H, CELL_W)
    with torch.no_grad():
        err = (net(x).flatten(1) - model(x)).abs().max().item()
    if err > 1e-4:
        raise SystemExit(f"folding changed the network by {err}")
    fp = os.path.join(out, "drum_float.onnx")
    torch.onnx.export(net, torch.zeros(1, 1, CELL_H, CELL_W), fp, input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    inline_identities(fp)
    q = os.path.join(out, "drum_int8.onnx")
    data = calibration(synth, open_train, np.random.default_rng(20261022))
    quantize_static(fp, q, Reader(data), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    ops = sorted({n.op_type for n in onnx.load(q).graph.node})
    print("int8 graph ops:", ops)
    subprocess.run([sys.executable, os.path.join(TOOLS, "export_qnet.py"), q,
                    os.path.join(out, "drum.qnet")], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
