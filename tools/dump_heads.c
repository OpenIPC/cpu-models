/* dump_heads <blob> <crop.nv21>...: each crop's six head maps, raw float32,
 * to <crop>.heads, for comparison with pdq.py. */
#include <stdio.h>
#include <stdlib.h>
#include "pdnet.h"
int main(int argc, char **argv) {
    PdNet *n = pdnet_open(argv[1]);
    if (!n) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    const int w = pdnet_width(n), h = pdnet_height(n);
    unsigned char *buf = malloc((size_t)w * h * 3 / 2);
    PdBox box[64];
    for (int a = 2; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb");
        if (!f || fread(buf, 1, (size_t)w * h * 3 / 2, f) != (size_t)w * h * 3 / 2) return 1;
        fclose(f);
        pdnet_detect(n, buf, w, buf + (size_t)w * h, w, box, 64, 101);
        char out[1024]; snprintf(out, sizeof out, "%s.heads", argv[a]);
        FILE *o = fopen(out, "wb");
        for (int l = 0; l < 3; l++)
            for (int r = 0; r < 2; r++) {
                int hh, ww; const float *p = pdnet_head(n, l, r, &hh, &ww);
                fwrite(p, sizeof(float), (size_t)hh * ww * (r ? 4 : 1), o);
            }
        fclose(o);
    }
    pdnet_close(n);
    return 0;
}
