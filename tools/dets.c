#define _POSIX_C_SOURCE 200809L
/* dets <blob> <crops-dir>: every box the engine finds on every crop, as
 * "<crop> x y w h score" (input pixels, score 0..100), at a score floor of
 * 1, so points.py can sweep every threshold. The crops in name order. */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pdnet.h"

static int cmp(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <blob> <crops-dir>\n", argv[0]);
        return 2;
    }
    PdNet *n = pdnet_open(argv[1]);
    if (!n) {
        fprintf(stderr, "dets: cannot open %s\n", argv[1]);
        return 1;
    }
    const int w = pdnet_width(n), h = pdnet_height(n);
    const size_t frame = (size_t)w * h * 3 / 2;
    DIR *d = opendir(argv[2]);
    if (!d)
        return 1;
    char **names = NULL;
    size_t nn = 0, cap = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        const size_t l = strlen(e->d_name);
        if (l <= 5 || strcmp(e->d_name + l - 5, ".nv21"))
            continue;
        if (nn == cap && !(names = realloc(names, (cap = cap ? 2 * cap : 1024) * sizeof(*names))))
            return 1;
        names[nn++] = strdup(e->d_name);
    }
    closedir(d);
    qsort(names, nn, sizeof(*names), cmp);
    unsigned char *buf = malloc(frame);
    PdBox box[256];
    for (size_t i = 0; i < nn; i++) {
        char path[4096];
        snprintf(path, sizeof path, "%s/%s", argv[2], names[i]);
        FILE *f = fopen(path, "rb");
        if (!f || fread(buf, 1, frame, f) != frame) {
            fprintf(stderr, "dets: cannot read %s\n", path);
            return 1;
        }
        fclose(f);
        const int k = pdnet_detect(n, buf, w, buf + (size_t)w * h, w, box, 256, 1);
        if (k < 0)
            return 1;
        names[i][strlen(names[i]) - 5] = 0;
        for (int j = 0; j < k; j++)
            printf("%s %d %d %d %d %d\n", names[i], box[j].x, box[j].y, box[j].w,
                   box[j].h, box[j].score);
    }
    fprintf(stderr, "dets: %zu crops\n", nn);
    pdnet_close(n);
    return 0;
}
