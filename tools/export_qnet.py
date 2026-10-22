# export_qnet.py <model_int8.onnx> <out.qnet>
#
# A QDQ graph as the blob qnet.c runs: the same walk, constants and
# arithmetic as qref.run(head="float"), which is the reference the C engine is
# checked against bit for bit. Two kinds:
#
#   a detector (kind 1): the person detector, an NV21 frame in, three levels
#   of (class, box) heads out -- six graph outputs;
#   a classifier (kind 2): a one-channel crop in, standardised by its own
#   mean and spread as the model was trained, its outputs the graph's.
#
# A node the engine has no op for is refused here, not dropped: a graph that
# kept a ReLU or a reshape would otherwise export as some other network.
#
# Layout, little-endian, every section 4-byte aligned:
#   "MAJQNET1" u32 format=1  u32 kind (1 detector, 2 classifier)
#   u32 in_w in_h in_c  f32 in_scale  i32 in_zp
#   classifier: u32 prep (1: (y - mean) / max(std, floor) over the crop)  f32 floor
#   u32 n_tensors, then per tensor u32 h w c
#   u32 n_ops, then per op u32 type, and:
#     CONV(1):   u32 in out k stride pad depthwise cin cout float_out
#                i32 in_zp out_zp
#                float_out ? f32 scale[cout] : i32 (mult, shift)[cout]
#                i32 bias[cout]
#                i8 w[...] then pad to 4 -- [cout][cin] for 1x1,
#                [cout][k][k][cin] for kxk, [k][k][c] for depthwise
#     RESIZE2X(2): u32 in out          (nearest, same quantization)
#     ADD(3):    u32 a b out  i32 zp_a zp_b zp_out
#                i32 mult_a shift_a mult_b shift_b mult_out shift_out
#   detector:   u32 n_levels, then per level u32 stride cls_tensor reg_tensor
#   classifier: u32 n_outputs, then per output u32 tensor
import struct
import sys

import numpy as np
import onnx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import qref  # noqa: E402

CONV, RESIZE2X, ADD = 1, 2, 3
ADD_SHIFT = 20
STD_FLOOR = 4.0   # tools/meters/train_drum.py standardise()
# Nodes that are not ops: quantization itself, and the shape arithmetic the
# person detector's Resize carries (its scales are fixed at 2).
PLUMBING = {"QuantizeLinear", "DequantizeLinear", "Constant", "Shape", "Slice", "Concat"}


def main(src, dst):
    g = qref.Graph(src)
    shapes = {}            # tensor name -> (h, w, c)
    ids = {}
    out = bytearray()

    def tid(name, shape):
        if name not in ids:
            ids[name] = len(ids)
            shapes[name] = shape
        return ids[name]

    unknown = sorted({n.op_type for n in g.nodes} - PLUMBING - {"Conv", "Resize", "Add"})
    if unknown:
        raise SystemExit(f"{src}: no engine op for {unknown}")
    _, in_c, in_h, in_w = g.input_shape
    detector = len(g.outputs) == 6
    if detector and in_c != 3:
        raise SystemExit(f"{src}: a detector takes NV21 as 3 channels, not {in_c}")
    qi = next(n for n in g.nodes if n.op_type == "QuantizeLinear" and n.input[0] == g.input)
    in_scale, in_zp = g.qparams[qi.output[0]]
    tid(qi.output[0], (in_h, in_w, in_c))
    ops = []
    final = {}
    for n in g.nodes:
        if n.op_type == "Conv":
            src_t, s_in, zp_in = g.dq_source(n.input[0])
            wq, s_w, _ = g.dq_source(n.input[1])
            w = g.init[wq].astype(np.int8)
            bias = np.zeros(w.shape[0], np.int32)
            if len(n.input) > 2:
                bq, _, _ = g.dq_source(n.input[2])
                bias = g.init[bq].astype(np.int32)
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
            k, st, grp = a["kernel_shape"][0], a["strides"][0], a.get("group", 1)
            pad = a.get("pads", [0, 0, 0, 0])[0]
            # The engine's convolutions are square, strided and padded alike
            # both ways: anything else would be read as something it is not.
            if (len(set(a["kernel_shape"])) != 1 or len(set(a["strides"])) != 1 or
                    len(set(a.get("pads", [0, 0, 0, 0]))) != 1 or
                    any(d != 1 for d in a.get("dilations", [1, 1]))):
                raise SystemExit(f"{src}: {n.name}: kernel {a['kernel_shape']} strides "
                                 f"{a['strides']} pads {a.get('pads')}: not one the engine runs")
            ih, iw, ic = shapes[src_t]
            oh, ow = (ih + 2 * pad - k) // st + 1, (iw + 2 * pad - k) // st + 1
            co = w.shape[0]
            dw = grp > 1
            real = float(s_in) * s_w.astype(np.float64)
            q = next(c for c in g.nodes if n.output[0] in c.input)
            dq = next(c for c in g.nodes if q.output[0] in c.input)
            is_final = any(o in g.outputs for o in dq.output)
            o = tid(q.output[0], (oh, ow, co))
            if is_final:
                final[dq.output[0]] = o
                zp_out = 0
            else:
                s_out, zp_out = g.qparams[q.output[0]]
            rec = struct.pack("<10I2i", CONV, ids[src_t], o, k, st, pad, int(dw), ic, co,
                              int(is_final), int(zp_in), int(zp_out))
            if is_final:
                rec += real.astype(np.float32).tobytes()
            else:
                ms = [qref.quantize_multiplier(r / s_out) for r in real]
                rec += np.array(ms, np.int32).reshape(-1).tobytes()
            rec += bias.tobytes()
            if dw:
                wb = w[:, 0].transpose(1, 2, 0)          # k, k, c
            elif k == 1:
                wb = w[:, :, 0, 0]                        # co, ci
            else:
                wb = w.transpose(0, 2, 3, 1)              # co, k, k, ci
            wb = np.ascontiguousarray(wb).tobytes()
            rec += wb + bytes(-len(wb) % 4)
            ops.append(rec)
        elif n.op_type == "Resize":
            src_t, _, _ = g.dq_source(n.input[0])
            q = next(c for c in g.nodes if n.output[0] in c.input)
            h, w_, c = shapes[src_t]
            ops.append(struct.pack("<3I", RESIZE2X, ids[src_t], tid(q.output[0], (2 * h, 2 * w_, c))))
        elif n.op_type == "Add":
            (ta, s_a, zp_a), (tb, s_b, zp_b) = (g.dq_source(i) for i in n.input)
            q = next(c for c in g.nodes if n.output[0] in c.input)
            s_out, zp_out = g.qparams[q.output[0]]
            twice = 2 * max(float(s_a), float(s_b))
            ma = qref.quantize_multiplier(float(s_a) / twice)
            mb = qref.quantize_multiplier(float(s_b) / twice)
            mo = qref.quantize_multiplier(twice / ((1 << ADD_SHIFT) * s_out))
            assert shapes[ta] == shapes[tb]
            o = tid(q.output[0], shapes[ta])
            ops.append(struct.pack("<4I9i", ADD, ids[ta], ids[tb], o, int(zp_a), int(zp_b),
                                   int(zp_out), *ma, *mb, *mo))
    out += b"MAJQNET1" + struct.pack("<2I", 1, 1 if detector else 2)
    out += struct.pack("<3Ifi", in_w, in_h, in_c, in_scale, in_zp)
    if not detector:
        out += struct.pack("<If", 1, STD_FLOOR)
    names = sorted(ids, key=ids.get)
    out += struct.pack("<I", len(names))
    for nm in names:
        out += struct.pack("<3I", *shapes[nm])
    out += struct.pack("<I", len(ops))
    for r in ops:
        out += r
    if detector:
        levels = []
        for lv, stride in enumerate((8, 16, 32)):
            cls_o, reg_o = g.outputs[2 * lv], g.outputs[2 * lv + 1]
            levels.append((stride, final[cls_o], final[reg_o]))
        out += struct.pack("<I", len(levels))
        for lv in levels:
            out += struct.pack("<3I", *lv)
    else:
        missing = [o for o in g.outputs if o not in final]
        if missing:
            raise SystemExit(f"{src}: outputs {missing} are not a convolution's float head")
        out += struct.pack("<I", len(g.outputs))
        for o in g.outputs:
            out += struct.pack("<I", final[o])
    open(dst, "wb").write(out)
    print(f"{dst}: {len(out)} bytes, {len(names)} tensors, {len(ops)} ops")


if __name__ == "__main__":
    main(*sys.argv[1:])
