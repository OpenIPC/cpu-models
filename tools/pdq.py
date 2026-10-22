# pdq.py: the CPU person detector's int8 graph, as integer arithmetic.
#
# Reads the statically quantized ONNX (QDQ) model and runs it the way a C
# engine on the camera would: int8 NHWC activations with zero points, int8
# per-channel weights, int32 accumulators and bias, and a fixed-point
# multiply-shift to requantize -- TFLite's arithmetic. The point is a
# reference the C engine can match bit for bit, checked here against
# onnxruntime's own execution of the QDQ graph.
#
# The detection heads differ on purpose (head="float"): their int32
# accumulators are taken straight to float rather than through the int8
# tensor the QDQ graph ends in, whose calibrated range caps the stride-8
# confidence at logit 0 (score 0.5) in steps of ~1 logit. head="int8" keeps
# the QDQ graph's behaviour, for the comparison.
import numpy as np
import onnx
from onnx import numpy_helper as nh


def quantize_multiplier(m):
    """m > 0 as (int32 mult, shift): m == mult * 2^(shift - 31)."""
    if m == 0:
        return 0, 0
    mant, exp = np.frexp(m)
    q = int(round(mant * (1 << 31)))
    if q == 1 << 31:
        q //= 2
        exp += 1
    return q, int(exp)


def requant(acc, mult, shift):
    """acc * mult * 2^(shift - 31), rounded once (half up).

    Not TFLite's MultiplyByQuantizedMultiplier, which rounds twice -- once in
    the doubling high multiply, again in the power-of-two divide -- and so
    lands one off the exact result for a quarter of negative values at some
    shifts. One 64-bit product and one rounding shift agree with exact
    rounding except at exact halves; ARMv7 NEON has the 32x32->64 multiply
    this needs (vmull_s32)."""
    n = (31 - np.asarray(shift, np.int64))
    prod = acc.astype(np.int64) * np.asarray(mult, np.int64)
    return (prod + (np.int64(1) << (n - 1))) >> n


class Graph:
    def __init__(self, path):
        m = onnx.load(path)
        self.init = {t.name: nh.to_array(t) for t in m.graph.initializer}
        for n in m.graph.node:
            if n.op_type == "Constant":
                self.init[n.output[0]] = nh.to_array(n.attribute[0].t)
        self.nodes = [n for n in m.graph.node if n.op_type != "Constant"]
        self.prod = {o: n for n in self.nodes for o in n.output}
        self.outputs = [o.name for o in m.graph.output]
        self.input = m.graph.input[0].name
        self.input_shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
        self.qparams = {}  # quantized tensor name -> (scale, zp)
        for n in self.nodes:
            if n.op_type == "QuantizeLinear":
                self.qparams[n.output[0]] = (float(self.init[n.input[1]]),
                                             int(self.init[n.input[2]]))

    def dq_source(self, name):
        """A DequantizeLinear's quantized input and its (scale, zp)."""
        n = self.prod[name]
        assert n.op_type == "DequantizeLinear", (name, n.op_type)
        src = n.input[0]
        s = self.init[n.input[1]]
        zp = self.init[n.input[2]] if len(n.input) > 2 else np.zeros_like(s, np.int32)
        return src, s, zp


def run(g, x_chw, head="float", keep=None, rq="fixed"):
    """x_chw: float32 [3, H, W] as the float model takes it."""
    t = {}  # tensor name -> int8 NHWC (H, W, C) for quantized activations
    qi = next(n for n in g.nodes if n.op_type == "QuantizeLinear" and n.input[0] == g.input)
    s, zp = g.qparams[qi.output[0]]
    t[qi.output[0]] = np.clip(np.round(x_chw.transpose(1, 2, 0) / s) + zp, -128, 127).astype(np.int32)
    outs = {}
    for n in g.nodes:
        if n.op_type == "Conv":
            src, s_in, zp_in = g.dq_source(n.input[0])
            wq, s_w, zp_w = g.dq_source(n.input[1])
            w = g.init[wq].astype(np.int32)                      # [Co, Ci/g, k, k]
            assert not np.any(zp_w)
            bias = np.zeros(w.shape[0], np.int64)
            if len(n.input) > 2:
                bq, s_b, _ = g.dq_source(n.input[2])
                bias = g.init[bq].astype(np.int64)
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
            k, st, grp = a["kernel_shape"][0], a["strides"][0], a.get("group", 1)
            pads = a.get("pads", [0, 0, 0, 0])
            x = t[src] - int(zp_in)
            H, W, C = x.shape
            xp = np.zeros((H + pads[0] + pads[2], W + pads[1] + pads[3], C), np.int32)
            xp[pads[0]:pads[0] + H, pads[1]:pads[1] + W] = x
            Ho = (xp.shape[0] - k) // st + 1
            Wo = (xp.shape[1] - k) // st + 1
            Co = w.shape[0]
            acc = np.zeros((Ho, Wo, Co), np.int64)
            cpg_in, cpg_out = C // grp, Co // grp
            for ky in range(k):
                for kx in range(k):
                    patch = xp[ky:ky + st * Ho:st, kx:kx + st * Wo:st]  # Ho, Wo, C
                    if grp == 1:
                        acc += patch.astype(np.int64) @ w[:, :, ky, kx].T.astype(np.int64)
                    elif cpg_in == 1 and cpg_out == 1:
                        acc += patch.astype(np.int64) * w[:, 0, ky, kx].astype(np.int64)
                    else:
                        raise NotImplementedError("grouped conv")
            acc += bias
            real_scale = float(s_in) * s_w.astype(np.float64)  # per output channel
            q = next(c for c in g.nodes if n.output[0] in c.input)
            final = any(o in g.outputs for o in
                        next(c for c in g.nodes if q.output[0] in c.input).output)
            if final and head == "float":
                # the scale as the blob stores it, float32, so the C engine
                # can match this exactly
                outs[q.output[0]] = acc * real_scale.astype(np.float32).astype(np.float64)
                continue
            s_out, zp_out = g.qparams[q.output[0]]
            if rq == "float":   # onnxruntime's way: float, round half to even
                y = np.round((acc * real_scale).astype(np.float32) / np.float32(s_out)).astype(np.int64) + zp_out
            else:
                mults = [quantize_multiplier(r / s_out) for r in real_scale]
                mult = np.array([m for m, _ in mults], np.int64)
                shift = np.array([e for _, e in mults], np.int64)
                y = requant(acc, mult, shift) + zp_out
            t[q.output[0]] = np.clip(y, -128, 127).astype(np.int32)
        elif n.op_type == "Resize":
            src, s_in, zp_in = g.dq_source(n.input[0])
            q = next(c for c in g.nodes if n.output[0] in c.input)
            s_out, zp_out = g.qparams[q.output[0]]
            assert abs(float(s_in) - s_out) < 1e-9 and int(zp_in) == zp_out
            t[q.output[0]] = t[src].repeat(2, 0).repeat(2, 1)   # nearest, x2
        elif n.op_type == "Add":
            (sa, s_a, zp_a), (sb, s_b, zp_b) = (g.dq_source(i) for i in n.input)
            q = next(c for c in g.nodes if n.output[0] in c.input)
            s_out, zp_out = g.qparams[q.output[0]]
            # TFLite's quantized add: both sides to a shared 2^20 grid
            ls = 20
            twice = 2 * max(float(s_a), float(s_b))
            ma, ea = quantize_multiplier(float(s_a) / twice)
            mb, eb = quantize_multiplier(float(s_b) / twice)
            mo, eo = quantize_multiplier(twice / ((1 << ls) * s_out))
            A = (t[sa] - int(zp_a)).astype(np.int64) << ls
            B = (t[sb] - int(zp_b)).astype(np.int64) << ls
            y = requant(requant(A, ma, ea) + requant(B, mb, eb), mo, eo) + zp_out
            t[q.output[0]] = np.clip(y, -128, 127).astype(np.int32)
    if keep is not None:
        keep.update(t)
    res = []
    for o in g.outputs:
        dq = g.prod[o]
        if dq.input[0] in outs:
            v = outs[dq.input[0]]
        else:
            s, zp = g.qparams[dq.input[0]]
            v = (t[dq.input[0]] - zp) * s
        res.append(v.transpose(2, 0, 1).astype(np.float32))  # C, H, W
    return res
