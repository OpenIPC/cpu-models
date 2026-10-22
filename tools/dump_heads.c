/* dump_heads <blob> <crop.nv21>...: each crop's six head maps, raw float32,
 * to <crop>.heads, for comparison with qref.py. */
#include <stdio.h>
#include <stdlib.h>
#include "qnet.h"
int main(int argc, char **argv) {
    QNet *n = qnet_open(argv[1]);
    if (!n) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    const int w = qnet_width(n), h = qnet_height(n);
    unsigned char *buf = malloc((size_t)w * h * 3 / 2);
    QBox box[64];
    for (int a = 2; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb");
        if (!f || fread(buf, 1, (size_t)w * h * 3 / 2, f) != (size_t)w * h * 3 / 2) return 1;
        fclose(f);
        qnet_detect(n, buf, w, buf + (size_t)w * h, w, box, 64, 101);
        char out[1024]; snprintf(out, sizeof out, "%s.heads", argv[a]);
        FILE *o = fopen(out, "wb");
        for (int l = 0; l < 3; l++)
            for (int r = 0; r < 2; r++) {
                int hh, ww; const float *p = qnet_head(n, l, r, &hh, &ww);
                fwrite(p, sizeof(float), (size_t)hh * ww * (r ? 4 : 1), o);
            }
        fclose(o);
    }
    qnet_close(n);
    return 0;
}
