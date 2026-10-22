#include "qnet.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define QNET_NEON 1
#endif

#define QNET_MAGIC "MAJQNET1"
#define QNET_FORMAT 1
#define MAX_TENSORS 64
#define MAX_OPS 64
#define MAX_LEVELS 3
#define MAX_OUTPUTS 4
#define NMS_IOU 0.55f

/* What a blob's network is, and so what it is given and gives. */
enum { KIND_DETECTOR = 1, KIND_CLASSIFIER = 2 };
enum { PREP_STANDARDISE = 1 };

/* A blob's activations, all tensors together, may not ask for more than
 * this: the real network's are a few hundred KB, and a camera has a
 * hundred-odd MB for everything. */
#define MAX_ACTIVATION_BYTES (8u << 20)

enum { OP_CONV = 1, OP_RESIZE2X = 2, OP_ADD = 3 };

enum { T_NONE = 0, T_INT8, T_FLOAT };

typedef struct {
    int h, w, c;
    int kind;    /* T_INT8 or T_FLOAT once something produces it */
    int8_t *q;   /* int8 activation, HWC */
    float *f;    /* float head output, HWC */
} Tensor;

typedef struct {
    int type;
    int in, in2, out;
    /* conv */
    int k, stride, pad, dw, cin, cout, float_out;
    int32_t in_zp, out_zp;
    const int32_t *ms;     /* (mult, shift) pairs */
    const float *scale;    /* float_out */
    const int32_t *bias;
    const int8_t *w;
    int32_t *bias_folded;  /* 1x1: bias - in_zp * sum(w) */
    /* For the NEON kernels, prepared at load: weights widened to int16 with
     * the output channel innermost (1x1: [cin][cout]; kxk: [k][k][cin][cout];
     * depthwise: [k][k][c]), and each channel's requantization as a
     * multiplier and the negated right shift vrshlq_s64 wants. */
    int16_t *w16;
    int32_t *mult;
    int64_t *nshift;
    int kpad;              /* kxk: k*k*cin rounded up to 8, the im2col row */
    /* add */
    int32_t zp_a, zp_b, zp_o, m_a, s_a, m_b, s_b, m_o, s_o;
} Op;

struct QNet {
    uint8_t *blob;
    int kind;
    int in_w, in_h, in_c;
    float in_scale;        /* classifier: the input's quantization ... */
    int32_t in_zp;
    float floor;           /* ... and the least spread a crop is divided by */
    int8_t lut_y[256], lut_c[256];
    int n_t, n_ops, n_lv, n_out;
    Tensor t[MAX_TENSORS];
    Op op[MAX_OPS];
    struct { int stride, cls, reg; } lv[MAX_LEVELS];
    int out[MAX_OUTPUTS];  /* classifier: its float outputs */
    int32_t *acc;
    int16_t *col;          /* im2col rows for the NEON kxk */
};

/* ---- arithmetic ---------------------------------------------------------- */

/* acc * mult * 2^(shift - 31), rounded once, half up. */
static inline int32_t requant(int32_t acc, int32_t mult, int32_t shift) {
    const int n = 31 - shift;
    const int64_t p = (int64_t)acc * mult;
    return (int32_t)((p + ((int64_t)1 << (n - 1))) >> n);
}

static inline int8_t clamp8(int32_t v) {
    return (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
}

/* A quotient already rounded onto the input's int8 grid, with its zero point.
 * Bounded before the conversion: a blob may carry any positive scale, and a
 * tiny one takes the quotient past what an int32 holds, or to infinity.
 * Every in-range value is unchanged. */
static inline int8_t quant_in(double q, int zp) {
    q = q < -512 ? -512 : q > 512 ? 512 : q;   /* NaN cannot reach here */
    return clamp8((int32_t)q + zp);
}

/* ---- loading --------------------------------------------------------------- */

typedef struct {
    const uint8_t *p, *end;
    bool bad;
} Rd;

static const void *take(Rd *r, size_t n) {
    if (r->bad || (size_t)(r->end - r->p) < n) {
        r->bad = true;
        return NULL;
    }
    const void *v = r->p;
    r->p += n;
    return v;
}

static uint32_t u32(Rd *r) {
    const uint8_t *b = take(r, 4);
    return b ? (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 |
                   (uint32_t)b[3] << 24
             : 0;
}

static int32_t i32(Rd *r) { return (int32_t)u32(r); }

static float f32(Rd *r) {
    uint32_t v = u32(r);
    float f;
    memcpy(&f, &v, 4);
    return f;
}

static bool tensor_ok(const QNet *n, uint32_t i) { return i < (uint32_t)n->n_t; }

static bool zp_ok(int32_t zp) { return zp >= -128 && zp <= 127; }

/* What requant() can shift by without undefined behaviour. */
static bool shift_ok(int32_t shift) { return shift >= -31 && shift <= 30; }

/* An op reads only tensors something earlier produced as int8, and writes a
 * tensor nothing has written yet: the order is the execution order, and every
 * buffer is read as what it is. */
static bool reads_int8(const QNet *n, uint32_t i) {
    return tensor_ok(n, i) && n->t[i].kind == T_INT8;
}
static bool writes_new(const QNet *n, uint32_t i) {
    return tensor_ok(n, i) && n->t[i].kind == T_NONE;
}

static bool parse(QNet *n, Rd *r) {
    const void *m = take(r, 8);
    if (!m || memcmp(m, QNET_MAGIC, 8) || u32(r) != QNET_FORMAT)
        return false;
    n->kind = (int)u32(r);
    if (n->kind != KIND_DETECTOR && n->kind != KIND_CLASSIFIER)
        return false;
    n->in_w = (int)u32(r);
    n->in_h = (int)u32(r);
    n->in_c = (int)u32(r);
    const float in_scale = f32(r);
    const int32_t in_zp = i32(r);
    if (r->bad || n->in_w <= 0 || n->in_w > 4096 || n->in_h <= 0 || n->in_h > 4096 ||
        !(in_scale > 0) || !isfinite(in_scale) || !zp_ok(in_zp))
        return false;
    n->in_scale = in_scale;
    n->in_zp = in_zp;
    if (n->kind == KIND_CLASSIFIER) {
        /* A crop of luma, standardised by its own mean and spread: the one
         * preprocessing a classifier here is trained with. */
        const uint32_t prep = u32(r);
        n->floor = f32(r);
        if (r->bad || n->in_c != 1 || prep != PREP_STANDARDISE || !(n->floor > 0) ||
            !isfinite(n->floor))
            return false;
    }
    /* A detector takes NV21 as Y, U and V; even, because NV21 carries
     * chroma for pixel pairs and row pairs. */
    if (n->kind == KIND_DETECTOR && (n->in_c != 3 || (n->in_w & 1) || (n->in_h & 1)))
        return false;
    /* The frame's bytes to the network's int8 input, as the float model's
     * preprocessing defines it: Y from BT.709 limited range to -1..1, U and
     * V from their offset to about -1..1. Two tables of 256. */
    const float ky = (float)(255.0 / 219.0 / 128.0), kc = (float)(255.0 / 224.0 / 128.0);
    for (int v = 0; v < 256; v++) {
        const float y = ((float)v - 16.0f) * ky - 1.0f;
        const float c = ((float)v - 128.0f) * kc;
        n->lut_y[v] = quant_in(rintf(y / in_scale), in_zp);
        n->lut_c[v] = quant_in(rintf(c / in_scale), in_zp);
    }
    n->n_t = (int)u32(r);
    if (r->bad || n->n_t <= 0 || n->n_t > MAX_TENSORS)
        return false;
    uint64_t total = 0;
    for (int i = 0; i < n->n_t; i++) {
        n->t[i].h = (int)u32(r);
        n->t[i].w = (int)u32(r);
        n->t[i].c = (int)u32(r);
        if (r->bad || n->t[i].h <= 0 || n->t[i].h > 4096 || n->t[i].w <= 0 ||
            n->t[i].w > 4096 || n->t[i].c <= 0 || n->t[i].c > 4096)
            return false;
        /* Each dimension bounded is not enough: their product is 2^36. Count
         * every tensor as float, the larger of its two possible storages. */
        total += (uint64_t)n->t[i].h * n->t[i].w * n->t[i].c * sizeof(float);
        if (total > MAX_ACTIVATION_BYTES)
            return false;
    }
    n->t[0].kind = T_INT8; /* the input, which qnet_detect() fills */
    n->n_ops = (int)u32(r);
    if (r->bad || n->n_ops <= 0 || n->n_ops > MAX_OPS)
        return false;
    for (int i = 0; i < n->n_ops; i++) {
        Op *o = &n->op[i];
        o->type = (int)u32(r);
        if (o->type == OP_CONV) {
            o->in = (int)u32(r);
            o->out = (int)u32(r);
            o->k = (int)u32(r);
            o->stride = (int)u32(r);
            const uint32_t pad = u32(r);
            o->pad = pad < 8 ? (int)pad : 8;
            o->dw = (int)u32(r);
            o->cin = (int)u32(r);
            o->cout = (int)u32(r);
            o->float_out = (int)u32(r);
            o->in_zp = i32(r);
            o->out_zp = i32(r);
            /* A 1x1 is a plain per-pixel multiply: no stride, no padding. */
            if (r->bad || !reads_int8(n, o->in) || !writes_new(n, o->out) || o->k < 1 ||
                o->k > 7 || o->stride < 1 || o->stride > 4 || o->pad >= o->k ||
                (o->k == 1 && o->stride != 1) || !zp_ok(o->in_zp) || !zp_ok(o->out_zp) ||
                (o->float_out != 0 && o->float_out != 1) || (o->dw != 0 && o->dw != 1))
                return false;
            const Tensor *ti = &n->t[o->in], *to = &n->t[o->out];
            if (ti->c != o->cin || to->c != o->cout || (o->dw && o->cin != o->cout) ||
                to->h != (ti->h + 2 * o->pad - o->k) / o->stride + 1 ||
                to->w != (ti->w + 2 * o->pad - o->k) / o->stride + 1)
                return false;
            if (o->float_out)
                o->scale = take(r, (size_t)o->cout * 4);
            else
                o->ms = take(r, (size_t)o->cout * 8);
            o->bias = take(r, (size_t)o->cout * 4);
            const size_t wn = o->dw ? (size_t)o->k * o->k * o->cout
                                    : (size_t)o->cout * o->k * o->k * o->cin;
            o->w = take(r, wn);
            take(r, (4 - wn % 4) % 4);
            if (r->bad)
                return false;
            if (!o->float_out)
                for (int c = 0; c < o->cout; c++)
                    if (!shift_ok(o->ms[2 * c + 1]))
                        return false;
            n->t[o->out].kind = o->float_out ? T_FLOAT : T_INT8;
        } else if (o->type == OP_RESIZE2X) {
            o->in = (int)u32(r);
            o->out = (int)u32(r);
            if (r->bad || !reads_int8(n, o->in) || !writes_new(n, o->out))
                return false;
            const Tensor *ti = &n->t[o->in], *to = &n->t[o->out];
            if (to->h != 2 * ti->h || to->w != 2 * ti->w || to->c != ti->c)
                return false;
            n->t[o->out].kind = T_INT8;
        } else if (o->type == OP_ADD) {
            o->in = (int)u32(r);
            o->in2 = (int)u32(r);
            o->out = (int)u32(r);
            o->zp_a = i32(r); o->zp_b = i32(r); o->zp_o = i32(r);
            o->m_a = i32(r); o->s_a = i32(r);
            o->m_b = i32(r); o->s_b = i32(r);
            o->m_o = i32(r); o->s_o = i32(r);
            if (r->bad || !reads_int8(n, o->in) || !reads_int8(n, o->in2) ||
                !writes_new(n, o->out) || !zp_ok(o->zp_a) || !zp_ok(o->zp_b) ||
                !zp_ok(o->zp_o) || !shift_ok(o->s_a) || !shift_ok(o->s_b) ||
                !shift_ok(o->s_o))
                return false;
            const Tensor *a = &n->t[o->in], *b = &n->t[o->in2], *to = &n->t[o->out];
            if (a->h != b->h || a->w != b->w || a->c != b->c || to->h != a->h ||
                to->w != a->w || to->c != a->c)
                return false;
            n->t[o->out].kind = T_INT8;
        } else {
            return false;
        }
    }
    if (n->kind == KIND_CLASSIFIER) {
        n->n_out = (int)u32(r);
        if (r->bad || n->n_out <= 0 || n->n_out > MAX_OUTPUTS)
            return false;
        for (int i = 0; i < n->n_out; i++) {
            n->out[i] = (int)u32(r);
            if (r->bad || !tensor_ok(n, (uint32_t)n->out[i]) || n->t[n->out[i]].kind != T_FLOAT)
                return false;
        }
        return !r->bad && n->t[0].h == n->in_h && n->t[0].w == n->in_w && n->t[0].c == n->in_c;
    }
    n->n_lv = (int)u32(r);
    if (r->bad || n->n_lv <= 0 || n->n_lv > MAX_LEVELS)
        return false;
    for (int i = 0; i < n->n_lv; i++) {
        n->lv[i].stride = (int)u32(r);
        n->lv[i].cls = (int)u32(r);
        n->lv[i].reg = (int)u32(r);
        if (r->bad || !tensor_ok(n, n->lv[i].cls) || !tensor_ok(n, n->lv[i].reg))
            return false;
        /* Both float, as the decode reads them, on the same grid. */
        const Tensor *tc = &n->t[n->lv[i].cls], *tr = &n->t[n->lv[i].reg];
        const int st = n->lv[i].stride;
        if (tc->kind != T_FLOAT || tr->kind != T_FLOAT || tc->c != 1 || tr->c != 4 ||
            tc->h != tr->h || tc->w != tr->w || st < 1 || st > 64)
            return false;
    }
    return !r->bad && n->t[0].h == n->in_h && n->t[0].w == n->in_w && n->t[0].c == 3;
}

/* Widened, channel-innermost weights and vector requantization constants
 * for a convolution's NEON kernel. Harmless without NEON. */
static bool prepare(Op *o) {
    if (o->type != OP_CONV)
        return true;
    const int k = o->k, ci = o->cin, co = o->cout;
    const size_t wn = o->dw ? (size_t)k * k * co : (size_t)co * k * k * ci;
    if (!(o->w16 = malloc(wn * sizeof(int16_t))))
        return false;
    if (o->dw) {
        for (size_t i = 0; i < wn; i++)
            o->w16[i] = o->w[i];
    } else {
        /* [k][k][cin] rows, cout innermost; a kxk's rows padded with zeros to
         * a multiple of 8, the im2col row its kernel reads 8 at a time. */
        o->kpad = (k * k * ci + 7) & ~7;
        free(o->w16);
        if (!(o->w16 = calloc((size_t)o->kpad * co, sizeof(int16_t))))
            return false;
        for (int c = 0; c < co; c++)
            for (int t = 0; t < k * k; t++)
                for (int i = 0; i < ci; i++)
                    o->w16[((size_t)t * ci + i) * co + c] =
                        o->w[((size_t)c * k * k + t) * ci + i];
    }
    if (o->float_out)
        return true;
    if (!(o->mult = malloc((size_t)co * sizeof(int32_t))) ||
        !(o->nshift = malloc((size_t)co * sizeof(int64_t))))
        return false;
    for (int c = 0; c < co; c++) {
        o->mult[c] = o->ms[2 * c];
        o->nshift[c] = -(int64_t)(31 - o->ms[2 * c + 1]);
    }
    return true;
}

QNet *qnet_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    QNet *n = calloc(1, sizeof(*n));
    long len = -1;
    if (n && !fseek(f, 0, SEEK_END) && (len = ftell(f)) > 0 && len < (16 << 20) &&
        !fseek(f, 0, SEEK_SET) && (n->blob = malloc((size_t)len)) &&
        fread(n->blob, 1, (size_t)len, f) == (size_t)len) {
        Rd r = {n->blob, n->blob + len, false};
        if (parse(n, &r)) {
            fclose(f);
            f = NULL;
            /* Buffers: every tensor its own, which for this network is a few
             * hundred KB and spares a liveness analysis. */
            bool ok = true;
            for (int i = 0; ok && i < n->n_t; i++) {
                const size_t e = (size_t)n->t[i].h * n->t[i].w * n->t[i].c;
                if (n->t[i].kind == T_FLOAT)
                    ok = (n->t[i].f = malloc(e * sizeof(float))) != NULL;
                else
                    ok = (n->t[i].q = calloc(e, 1)) != NULL;
            }
            int max_c = 0;
            for (int j = 0; j < n->n_ops; j++)
                if (n->op[j].type == OP_CONV && n->op[j].cout > max_c)
                    max_c = n->op[j].cout;
            ok = ok && (n->acc = malloc((size_t)max_c * sizeof(int32_t)));
            for (int j = 0; ok && j < n->n_ops; j++) {
                Op *o = &n->op[j];
                if (o->type != OP_CONV || o->k != 1 || o->dw)
                    continue;
                if (!(o->bias_folded = malloc((size_t)o->cout * sizeof(int32_t)))) {
                    ok = false;
                    break;
                }
                for (int c = 0; c < o->cout; c++) {
                    int32_t s = 0;
                    for (int i = 0; i < o->cin; i++)
                        s += o->w[(size_t)c * o->cin + i];
                    o->bias_folded[c] = o->bias[c] - o->in_zp * s;
                }
            }
            size_t col = 0;
            for (int j = 0; ok && j < n->n_ops; j++) {
                ok = prepare(&n->op[j]);
                const Op *o = &n->op[j];
                if (ok && o->type == OP_CONV && !o->dw && o->k > 1) {
                    const Tensor *to = &n->t[o->out];
                    const size_t e = (size_t)to->w * o->kpad;
                    if (e > col)
                        col = e;
                }
            }
            ok = ok && (!col || (n->col = malloc(col * sizeof(int16_t))));
            if (ok)
                return n;
        }
    }
    if (f)
        fclose(f);
    qnet_close(n);
    return NULL;
}

void qnet_close(QNet *n) {
    if (!n)
        return;
    for (int i = 0; i < n->n_t; i++) {
        free(n->t[i].q);
        free(n->t[i].f);
    }
    for (int j = 0; j < n->n_ops; j++) {
        free(n->op[j].bias_folded);
        free(n->op[j].w16);
        free(n->op[j].mult);
        free(n->op[j].nshift);
    }
    free(n->acc);
    free(n->col);
    free(n->blob);
    free(n);
}

int qnet_width(const QNet *n) { return n->in_w; }
int qnet_height(const QNet *n) { return n->in_h; }

/* ---- the layers -------------------------------------------------------------- */

static inline void emit(const Op *o, int32_t acc, int c, int8_t *q, float *f) {
    if (o->float_out)
        *f = (float)((double)acc * (double)o->scale[c]);
    else
        *q = clamp8(requant(acc, o->ms[2 * c], o->ms[2 * c + 1]) + o->out_zp);
}

/* Sum of a[i] * w[i] over n int8 pairs. */
static inline int32_t dot_i8(const int8_t *a, const int8_t *w, int n) {
    int i = 0;
    int32_t s = 0;
#ifdef QNET_NEON
    int32x4_t c = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
        const int8x16_t av = vld1q_s8(a + i), wv = vld1q_s8(w + i);
        c = vpadalq_s16(c, vmull_s8(vget_low_s8(av), vget_low_s8(wv)));
        c = vpadalq_s16(c, vmull_s8(vget_high_s8(av), vget_high_s8(wv)));
    }
    for (; i + 8 <= n; i += 8)
        c = vpadalq_s16(c, vmull_s8(vld1_s8(a + i), vld1_s8(w + i)));
    s = vgetq_lane_s32(c, 0) + vgetq_lane_s32(c, 1) + vgetq_lane_s32(c, 2) +
        vgetq_lane_s32(c, 3);
#endif
    for (; i < n; i++)
        s += a[i] * w[i];
    return s;
}

#ifdef QNET_NEON
/* Eight channels' accumulators to int8: acc * mult rounded once by the
 * channel's shift (vrshlq_s64 adds the half before shifting, as requant()
 * does), plus the zero point, saturated. Bit for bit requant() + clamp8(). */
static inline int8x8_t requant8(const Op *o, int c, int32x4_t lo, int32x4_t hi) {
    const int32x4_t m0 = vld1q_s32(o->mult + c), m1 = vld1q_s32(o->mult + c + 4);
    const int64_t *ns = o->nshift + c;
    int64x2_t p0 = vrshlq_s64(vmull_s32(vget_low_s32(lo), vget_low_s32(m0)), vld1q_s64(ns));
    int64x2_t p1 = vrshlq_s64(vmull_s32(vget_high_s32(lo), vget_high_s32(m0)), vld1q_s64(ns + 2));
    int64x2_t p2 = vrshlq_s64(vmull_s32(vget_low_s32(hi), vget_low_s32(m1)), vld1q_s64(ns + 4));
    int64x2_t p3 = vrshlq_s64(vmull_s32(vget_high_s32(hi), vget_high_s32(m1)), vld1q_s64(ns + 6));
    const int32x4_t zp = vdupq_n_s32(o->out_zp);
    const int32x4_t r0 = vaddq_s32(vcombine_s32(vqmovn_s64(p0), vqmovn_s64(p1)), zp);
    const int32x4_t r1 = vaddq_s32(vcombine_s32(vqmovn_s64(p2), vqmovn_s64(p3)), zp);
    return vqmovn_s16(vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1)));
}

/* The multiply both 1x1 and kxk come down to: npix rows of K inputs (K a
 * multiple of 8) against a [K][cout] int16 matrix, cout a multiple of 8.
 * Four pixels share every weight load; inputs come in 8 at a time and are
 * used lane by lane (vmlal_lane_s16), so nothing crosses from the NEON
 * registers to the core ones in the inner loop. IN is int8_t (a 1x1's
 * activations, zero point folded into its bias) or int16_t (an im2col row,
 * zero point already taken out). */
#define GEMM_LOAD8_int8_t(p) vmovl_s8(vld1_s8(p))
#define GEMM_LOAD8_int16_t(p) vld1q_s16(p)
#define DEFINE_GEMM(IN)                                                          \
    static void gemm_##IN(const Op *o, const IN *a, int K, const int32_t *bias, \
                          int8_t *out, int npix) {                               \
        const int cout = o->cout;                                                \
        int m = 0;                                                               \
        for (; m + 4 <= npix; m += 4) {                                          \
            const IN *a0 = a + (size_t)m * K, *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K; \
            for (int c = 0; c < cout; c += 8) {                                  \
                const int32x4_t b0 = vld1q_s32(bias + c), b1 = vld1q_s32(bias + c + 4); \
                int32x4_t l0 = b0, h0 = b1, l1 = b0, h1 = b1;                    \
                int32x4_t l2 = b0, h2 = b1, l3 = b0, h3 = b1;                    \
                const int16_t *w = o->w16 + c;                                   \
                for (int i = 0; i < K; i += 8) {                                 \
                    const int16x8_t x0 = GEMM_LOAD8_##IN(a0 + i);                \
                    const int16x8_t x1 = GEMM_LOAD8_##IN(a1 + i);                \
                    const int16x8_t x2 = GEMM_LOAD8_##IN(a2 + i);                \
                    const int16x8_t x3 = GEMM_LOAD8_##IN(a3 + i);                \
                    GEMM_STEP(0, vget_low_s16) GEMM_STEP(1, vget_low_s16)        \
                    GEMM_STEP(2, vget_low_s16) GEMM_STEP(3, vget_low_s16)        \
                    GEMM_STEP(0, vget_high_s16) GEMM_STEP(1, vget_high_s16)      \
                    GEMM_STEP(2, vget_high_s16) GEMM_STEP(3, vget_high_s16)      \
                }                                                                \
                vst1_s8(out + (size_t)(m + 0) * cout + c, requant8(o, c, l0, h0)); \
                vst1_s8(out + (size_t)(m + 1) * cout + c, requant8(o, c, l1, h1)); \
                vst1_s8(out + (size_t)(m + 2) * cout + c, requant8(o, c, l2, h2)); \
                vst1_s8(out + (size_t)(m + 3) * cout + c, requant8(o, c, l3, h3)); \
            }                                                                    \
        }                                                                        \
        for (; m < npix; m++) {                                                  \
            const IN *a0 = a + (size_t)m * K;                                    \
            for (int c = 0; c < cout; c += 8) {                                  \
                int32x4_t lo = vld1q_s32(bias + c), hi = vld1q_s32(bias + c + 4); \
                const int16_t *w = o->w16 + c;                                   \
                for (int i = 0; i < K; i++, w += cout) {                         \
                    const int16x8_t wv = vld1q_s16(w);                           \
                    lo = vmlal_n_s16(lo, vget_low_s16(wv), a0[i]);               \
                    hi = vmlal_n_s16(hi, vget_high_s16(wv), a0[i]);              \
                }                                                                \
                vst1_s8(out + (size_t)m * cout + c, requant8(o, c, lo, hi));     \
            }                                                                    \
        }                                                                        \
    }
/* One half (4 inputs) of an 8-input step: four weight rows, each against
 * its input lane, for all four pixels. */
#define GEMM_LANE(P, HALF, L)                                                    \
    { const int16x8_t wv = vld1q_s16(w); w += cout;                              \
      l0 = vmlal_lane_s16(l0, vget_low_s16(wv), HALF(x0), L);                    \
      h0 = vmlal_lane_s16(h0, vget_high_s16(wv), HALF(x0), L);                   \
      l1 = vmlal_lane_s16(l1, vget_low_s16(wv), HALF(x1), L);                    \
      h1 = vmlal_lane_s16(h1, vget_high_s16(wv), HALF(x1), L);                   \
      l2 = vmlal_lane_s16(l2, vget_low_s16(wv), HALF(x2), L);                    \
      h2 = vmlal_lane_s16(h2, vget_high_s16(wv), HALF(x2), L);                   \
      l3 = vmlal_lane_s16(l3, vget_low_s16(wv), HALF(x3), L);                    \
      h3 = vmlal_lane_s16(h3, vget_high_s16(wv), HALF(x3), L); }
/* GEMM_STEP(lane, half): the step order above walks lanes 0..3 of the low
 * half, then 0..3 of the high half, matching the weight rows i..i+7. */
#define GEMM_STEP(L, HALF) GEMM_LANE(0, HALF, L)
DEFINE_GEMM(int8_t)
DEFINE_GEMM(int16_t)

static void conv1x1_neon(const Op *o, const Tensor *ti, Tensor *to) {
    gemm_int8_t(o, ti->q, o->cin, o->bias_folded, to->q, to->h * to->w);
}

/* kxk (the stem) as im2col and the same multiply: each output pixel's k*k*cin
 * inputs, zero point taken out, padding as 0, rows padded to a multiple of 8. */
static void conv_kxk_neon(const Op *o, const Tensor *ti, Tensor *to, int16_t *col) {
    const int k = o->k, st = o->stride, pad = o->pad, cin = o->cin, K = o->kpad;
    /* One output row at a time, so the rows stay in L1 between being written
     * and being read: the first layer's whole im2col is a megabyte. */
    for (int oy = 0; oy < to->h; oy++) {
        int16_t *r = col;
        for (int ox = 0; ox < to->w; ox++, r += K) {
            int16_t *q = r;
            for (int ky = 0; ky < k; ky++) {
                const int iy = oy * st + ky - pad;
                for (int kx = 0; kx < k; kx++) {
                    const int ix = ox * st + kx - pad;
                    if (iy < 0 || iy >= ti->h || ix < 0 || ix >= ti->w) {
                        for (int i = 0; i < cin; i++)
                            *q++ = 0;
                        continue;
                    }
                    const int8_t *ip = ti->q + ((size_t)iy * ti->w + ix) * cin;
                    for (int i = 0; i < cin; i++)
                        *q++ = (int16_t)(ip[i] - o->in_zp);
                }
            }
            for (; q < r + K; q++)
                *q = 0;
        }
        gemm_int16_t(o, col, K, o->bias, to->q + (size_t)oy * to->w * o->cout, to->w);
    }
}

/* Depthwise: eight channels at a time, the chunk's nine weight vectors held
 * in registers across every pixel, and the accumulators across the taps.
 * Interior pixels skip the bounds tests; the border takes them. */
static void depthwise_neon(const Op *o, const Tensor *ti, Tensor *to) {
    const int k = o->k, st = o->stride, pad = o->pad, c = o->cout;
    const int16x8_t zp = vdupq_n_s16((int16_t)o->in_zp);
    const size_t rs = (size_t)ti->w * c; /* input row stride */
    for (int ch = 0; ch < c; ch += 8) {
        int16x8_t w[9];
        if (k == 3)
            for (int t = 0; t < 9; t++)
                w[t] = vld1q_s16(o->w16 + (size_t)t * c + ch);
        const int32x4_t b0 = vld1q_s32(o->bias + ch), b1 = vld1q_s32(o->bias + ch + 4);
        for (int oy = 0; oy < to->h; oy++) {
            const int y0 = oy * st - pad;
            for (int ox = 0; ox < to->w; ox++) {
                const int x0 = ox * st - pad;
                int32x4_t lo = b0, hi = b1;
                if (k == 3 && y0 >= 0 && y0 + 2 < ti->h && x0 >= 0 && x0 + 2 < ti->w) {
                    const int8_t *p = ti->q + (size_t)y0 * rs + (size_t)x0 * c + ch;
#define DW_TAP(T, OFF)                                                            \
    { const int16x8_t a = vsubq_s16(vmovl_s8(vld1_s8(p + (OFF))), zp);           \
      lo = vmlal_s16(lo, vget_low_s16(a), vget_low_s16(w[T]));                   \
      hi = vmlal_s16(hi, vget_high_s16(a), vget_high_s16(w[T])); }
                    DW_TAP(0, 0) DW_TAP(1, c) DW_TAP(2, 2 * c)
                    DW_TAP(3, rs) DW_TAP(4, rs + c) DW_TAP(5, rs + 2 * c)
                    DW_TAP(6, 2 * rs) DW_TAP(7, 2 * rs + c) DW_TAP(8, 2 * rs + 2 * c)
#undef DW_TAP
                } else {
                    for (int ky = 0; ky < k; ky++) {
                        const int iy = y0 + ky;
                        if (iy < 0 || iy >= ti->h)
                            continue;
                        for (int kx = 0; kx < k; kx++) {
                            const int ix = x0 + kx;
                            if (ix < 0 || ix >= ti->w)
                                continue;
                            const int16x8_t a = vsubq_s16(
                                vmovl_s8(vld1_s8(ti->q + (size_t)iy * rs + (size_t)ix * c + ch)), zp);
                            const int16x8_t wv = vld1q_s16(o->w16 + (size_t)(ky * k + kx) * c + ch);
                            lo = vmlal_s16(lo, vget_low_s16(a), vget_low_s16(wv));
                            hi = vmlal_s16(hi, vget_high_s16(a), vget_high_s16(wv));
                        }
                    }
                }
                vst1_s8(to->q + ((size_t)oy * to->w + ox) * c + ch, requant8(o, ch, lo, hi));
            }
        }
    }
}
#endif

/* 1x1: every output pixel is a row of cin bytes against cout weight rows.
 * The zero point is folded into the bias at load. */
static void conv1x1(const Op *o, const Tensor *ti, Tensor *to) {
    const int hw = to->h * to->w, cin = o->cin, cout = o->cout;
    for (int m = 0; m < hw; m++) {
        const int8_t *a = ti->q + (size_t)m * cin;
        for (int c = 0; c < cout; c++) {
            const int32_t acc = o->bias_folded[c] + dot_i8(a, o->w + (size_t)c * cin, cin);
            emit(o, acc, c, to->q ? to->q + (size_t)m * cout + c : NULL,
                 to->f ? to->f + (size_t)m * cout + c : NULL);
        }
    }
}

/* kxk, all channels mixed: the stem. Taps outside the picture are padding,
 * which contributes (zp - zp) * w = 0 and is skipped. */
static void conv_kxk(const Op *o, const Tensor *ti, Tensor *to, int32_t *acc) {
    const int k = o->k, st = o->stride, pad = o->pad, cin = o->cin, cout = o->cout;
    for (int oy = 0; oy < to->h; oy++)
        for (int ox = 0; ox < to->w; ox++) {
            for (int c = 0; c < cout; c++)
                acc[c] = o->bias[c];
            for (int ky = 0; ky < k; ky++) {
                const int iy = oy * st + ky - pad;
                if (iy < 0 || iy >= ti->h)
                    continue;
                for (int kx = 0; kx < k; kx++) {
                    const int ix = ox * st + kx - pad;
                    if (ix < 0 || ix >= ti->w)
                        continue;
                    const int8_t *ip = ti->q + ((size_t)iy * ti->w + ix) * cin;
                    for (int c = 0; c < cout; c++) {
                        const int8_t *wp = o->w + (((size_t)c * k + ky) * k + kx) * cin;
                        int32_t s = 0;
                        for (int i = 0; i < cin; i++)
                            s += ((int32_t)ip[i] - o->in_zp) * wp[i];
                        acc[c] += s;
                    }
                }
            }
            const size_t base = ((size_t)oy * to->w + ox) * cout;
            for (int c = 0; c < cout; c++)
                emit(o, acc[c], c, to->q ? to->q + base + c : NULL,
                     to->f ? to->f + base + c : NULL);
        }
}

/* Depthwise kxk: channels innermost, so each tap is one contiguous run. */
static void depthwise(const Op *o, const Tensor *ti, Tensor *to, int32_t *acc) {
    const int k = o->k, st = o->stride, pad = o->pad, c = o->cout;
    for (int oy = 0; oy < to->h; oy++)
        for (int ox = 0; ox < to->w; ox++) {
            for (int ch = 0; ch < c; ch++)
                acc[ch] = o->bias[ch];
            for (int ky = 0; ky < k; ky++) {
                const int iy = oy * st + ky - pad;
                if (iy < 0 || iy >= ti->h)
                    continue;
                for (int kx = 0; kx < k; kx++) {
                    const int ix = ox * st + kx - pad;
                    if (ix < 0 || ix >= ti->w)
                        continue;
                    const int8_t *ip = ti->q + ((size_t)iy * ti->w + ix) * c;
                    const int8_t *wp = o->w + ((size_t)ky * k + kx) * c;
                    int ch = 0;
#ifdef QNET_NEON
                    const int16x8_t zp = vdupq_n_s16((int16_t)o->in_zp);
                    for (; ch + 8 <= c; ch += 8) {
                        const int16x8_t a = vsubq_s16(vmovl_s8(vld1_s8(ip + ch)), zp);
                        const int16x8_t w = vmovl_s8(vld1_s8(wp + ch));
                        int32x4_t lo = vld1q_s32(acc + ch), hi = vld1q_s32(acc + ch + 4);
                        lo = vmlal_s16(lo, vget_low_s16(a), vget_low_s16(w));
                        hi = vmlal_s16(hi, vget_high_s16(a), vget_high_s16(w));
                        vst1q_s32(acc + ch, lo);
                        vst1q_s32(acc + ch + 4, hi);
                    }
#endif
                    for (; ch < c; ch++)
                        acc[ch] += ((int32_t)ip[ch] - o->in_zp) * wp[ch];
                }
            }
            const size_t base = ((size_t)oy * to->w + ox) * c;
            for (int ch = 0; ch < c; ch++)
                emit(o, acc[ch], ch, to->q ? to->q + base + ch : NULL,
                     to->f ? to->f + base + ch : NULL);
        }
}

static void resize2x(const Tensor *ti, Tensor *to) {
    const int c = ti->c;
    for (int y = 0; y < to->h; y++)
        for (int x = 0; x < to->w; x++)
            memcpy(to->q + ((size_t)y * to->w + x) * c,
                   ti->q + ((size_t)(y / 2) * ti->w + x / 2) * c, (size_t)c);
}

static void add(const Op *o, const Tensor *a, const Tensor *b, Tensor *to) {
    const size_t e = (size_t)to->h * to->w * to->c;
    size_t i = 0;
#ifdef QNET_NEON
    /* The same three roundings, two lanes of 64 bits at a time. */
    const int64x2_t na = vdupq_n_s64(-(int64_t)(31 - o->s_a));
    const int64x2_t nb = vdupq_n_s64(-(int64_t)(31 - o->s_b));
    const int64x2_t no = vdupq_n_s64(-(int64_t)(31 - o->s_o));
    const int32x2_t ma = vdup_n_s32(o->m_a), mb = vdup_n_s32(o->m_b), mo = vdup_n_s32(o->m_o);
    const int32x4_t za = vdupq_n_s32(o->zp_a), zb = vdupq_n_s32(o->zp_b), zo = vdupq_n_s32(o->zp_o);
#define ADD_RQ(x, m, nsh)                                                     \
    vcombine_s32(vqmovn_s64(vrshlq_s64(vmull_s32(vget_low_s32(x), m), nsh)),   \
                 vqmovn_s64(vrshlq_s64(vmull_s32(vget_high_s32(x), m), nsh)))
    for (; i + 8 <= e; i += 8) {
        const int16x8_t va = vmovl_s8(vld1_s8(a->q + i)), vb = vmovl_s8(vld1_s8(b->q + i));
        int32x4_t out[2];
        for (int h = 0; h < 2; h++) {
            const int16x4_t ha = h ? vget_high_s16(va) : vget_low_s16(va);
            const int16x4_t hb = h ? vget_high_s16(vb) : vget_low_s16(vb);
            const int32x4_t xa = vshlq_n_s32(vsubq_s32(vmovl_s16(ha), za), 20);
            const int32x4_t xb = vshlq_n_s32(vsubq_s32(vmovl_s16(hb), zb), 20);
            const int32x4_t sum = vaddq_s32(ADD_RQ(xa, ma, na), ADD_RQ(xb, mb, nb));
            out[h] = vaddq_s32(ADD_RQ(sum, mo, no), zo);
        }
        vst1_s8(to->q + i, vqmovn_s16(vcombine_s16(vqmovn_s32(out[0]), vqmovn_s32(out[1]))));
    }
#undef ADD_RQ
#endif
    for (; i < e; i++) {
        const int32_t ra = requant(((int32_t)a->q[i] - o->zp_a) * (1 << 20), o->m_a, o->s_a);
        const int32_t rb = requant(((int32_t)b->q[i] - o->zp_b) * (1 << 20), o->m_b, o->s_b);
        to->q[i] = clamp8(requant(ra + rb, o->m_o, o->s_o) + o->zp_o);
    }
}

/* ---- the frame in, the people out ------------------------------------------- */

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

static float iou(const QBox *a, const QBox *b) {
    const int x0 = a->x > b->x ? a->x : b->x, y0 = a->y > b->y ? a->y : b->y;
    const int x1 = a->x + a->w < b->x + b->w ? a->x + a->w : b->x + b->w;
    const int y1 = a->y + a->h < b->y + b->h ? a->y + a->h : b->y + b->h;
    if (x1 <= x0 || y1 <= y0)
        return 0;
    const float i = (float)(x1 - x0) * (y1 - y0);
    return i / ((float)a->w * a->h + (float)b->w * b->h - i);
}

static int by_score(const void *a, const void *b) {
    return ((const QBox *)b)->score - ((const QBox *)a)->score;
}

/* Every op in order, on whatever the input tensor holds. */
static void run_ops(QNet *n) {
    for (int j = 0; j < n->n_ops; j++) {
        const Op *o = &n->op[j];
        Tensor *to = &n->t[o->out];
        if (o->type == OP_CONV) {
            const Tensor *ti = &n->t[o->in];
#ifdef QNET_NEON
            /* Eight output channels a time; the heads (1 and 4 channels, and
             * float) take the plain path, which costs them nothing. */
            if (!o->float_out && o->cout % 8 == 0 &&
                (o->dw || o->k > 1 || o->cin % 8 == 0)) {
                if (o->dw)
                    depthwise_neon(o, ti, to);
                else if (o->k == 1)
                    conv1x1_neon(o, ti, to);
                else
                    conv_kxk_neon(o, ti, to, n->col);
            } else
#endif
            if (o->dw)
                depthwise(o, ti, to, n->acc);
            else if (o->k == 1)
                conv1x1(o, ti, to);
            else
                conv_kxk(o, ti, to, n->acc);
        } else if (o->type == OP_RESIZE2X) {
            resize2x(&n->t[o->in], to);
        } else {
            add(o, &n->t[o->in], &n->t[o->in2], to);
        }
    }
}

int qnet_detect(
    QNet *n, const uint8_t *y, int ystride, const uint8_t *vu, int vustride,
    QBox *out, int max, int score_min) {
    if (!n || n->kind != KIND_DETECTOR || !y || !vu || !out || max <= 0)
        return -1;
    int8_t *in = n->t[0].q;
    for (int j = 0; j < n->in_h; j++) {
        const uint8_t *yr = y + (size_t)j * ystride, *cr = vu + (size_t)(j / 2) * vustride;
        int8_t *p = in + (size_t)j * n->in_w * 3;
        for (int i = 0; i < n->in_w; i++) {
            const int ci = i & ~1;
            p[3 * i + 0] = n->lut_y[yr[i]];
            p[3 * i + 1] = n->lut_c[cr[ci + 1]]; /* U; NV21 puts V first */
            p[3 * i + 2] = n->lut_c[cr[ci]];     /* V */
        }
    }
    run_ops(n);

    /* Decode as the TVM shim did: an anchor-free head, a box per cell whose
     * score clears the floor, distances in stride units, then NMS. */
    static QBox cand[4096];
    int nc = 0;
    for (int l = 0; l < n->n_lv; l++) {
        const Tensor *tc = &n->t[n->lv[l].cls], *tr = &n->t[n->lv[l].reg];
        const int st = n->lv[l].stride;
        for (int cy = 0; cy < tc->h; cy++)
            for (int cx = 0; cx < tc->w; cx++) {
                const int i = cy * tc->w + cx;
                const int s = (int)(sigmoidf(tc->f[i]) * 100.0f + 0.5f);
                if (s < score_min || nc == (int)(sizeof(cand) / sizeof(cand[0])))
                    continue;
                /* Distances bounded before they become ints, and only there:
                 * at a stride of at most 64, 1e6 strides keeps every edge
                 * inside an int, and no box that fits one is changed. */
                float r[4];
                for (int s4 = 0; s4 < 4; s4++)
                    r[s4] = fminf(fmaxf(0.0f, tr->f[(size_t)i * 4 + s4]), 1e6f);
                const float ccx = (cx + 0.5f) * st, ccy = (cy + 0.5f) * st;
                const int x0 = (int)(ccx - r[0] * st);
                const int y0 = (int)(ccy - r[1] * st);
                const int x1 = (int)(ccx + r[2] * st);
                const int y1 = (int)(ccy + r[3] * st);
                if (x1 <= x0 || y1 <= y0)
                    continue;
                cand[nc++] = (QBox){x0, y0, x1 - x0, y1 - y0, s};
            }
    }
    qsort(cand, (size_t)nc, sizeof(cand[0]), by_score);
    int kept = 0;
    for (int i = 0; i < nc && kept < max; i++) {
        bool dup = false;
        for (int j = 0; j < kept && !dup; j++)
            dup = iou(&cand[i], &out[j]) > NMS_IOU;
        if (!dup)
            out[kept++] = cand[i];
    }
    return kept;
}

const float *qnet_head(const QNet *n, int level, int reg, int *h, int *w) {
    if (!n || level < 0 || level >= n->n_lv)
        return NULL;
    const Tensor *t = &n->t[reg ? n->lv[level].reg : n->lv[level].cls];
    *h = t->h;
    *w = t->w;
    return t->f;
}


int qnet_classify(QNet *n, const uint8_t *y, int ystride, float *out, int max) {
    if (!n || n->kind != KIND_CLASSIFIER || !y || !out || max <= 0)
        return -1;
    /* Standardised as in training: the crop's own mean and spread, the
     * spread floored so a flat crop is not blown up into noise. In double,
     * and rounded half to even, so the reference can say the same. */
    const int w = n->in_w, h = n->in_h;
    double sum = 0, sq = 0;
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            sum += y[(size_t)j * ystride + i];
    const double mean = sum / ((double)w * h);
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++) {
            const double d = y[(size_t)j * ystride + i] - mean;
            sq += d * d;
        }
    double sd = sqrt(sq / ((double)w * h));
    if (sd < n->floor)
        sd = n->floor;
    int8_t *in = n->t[0].q;
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++) {
            const double v = (y[(size_t)j * ystride + i] - mean) / sd / n->in_scale;
            in[(size_t)j * w + i] = quant_in(rint(v), n->in_zp);
        }
    run_ops(n);
    int k = 0;
    for (int o = 0; o < n->n_out; o++) {
        const Tensor *t = &n->t[n->out[o]];
        const size_t e = (size_t)t->h * t->w * t->c;
        for (size_t i = 0; i < e && k < max; i++)
            out[k++] = t->f[i];
    }
    return k;
}

const int8_t *qnet_input(const QNet *n) { return n ? n->t[0].q : NULL; }
