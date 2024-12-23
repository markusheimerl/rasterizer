#ifndef GIF_H
#define GIF_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

static const uint8_t DEFAULT_PALETTE[16 * 3] = {
    0x00, 0x00, 0x00,  // Black
    0xFF, 0xFF, 0xFF,  // White
    0xFF, 0xA5, 0x00,  // Orange
    0x00, 0xFF, 0x00,  // Green
    0x00, 0x00, 0xFF,  // Blue
    0x8B, 0x45, 0x13,  // Saddle Brown
    0xD2, 0x69, 0x1E,  // Chocolate
    0xCD, 0x85, 0x3F,  // Peru
    0xDE, 0xB8, 0x87,  // Burlywood
    0xD2, 0xB4, 0x8C,  // Tan
    0xBC, 0x8F, 0x8F,  // Rosy Brown
    0x80, 0x80, 0x80,  // Gray
    0x40, 0x40, 0x40,  // Dark Gray
    0xFF, 0x80, 0x00,  // Dark Orange
    0x00, 0x80, 0x00,  // Dark Green
    0x00, 0x00, 0x80   // Dark Blue
};

typedef struct {
    uint16_t w, h;
    int depth;
    int bgindex;
    int fd;
    int offset;
    int nframes;
    uint8_t *frame, *back;
    uint32_t partial;
    uint8_t buffer[0xFF];
    uint8_t palette[16 * 3];
} ge_GIF;

typedef struct Node {
    uint16_t key;
    struct Node *children[];
} Node;

static const uint8_t bayer_matrix[4][4] = {
    { 0,  8,  2, 10},
    {12,  4, 14,  6},
    { 3, 11,  1,  9},
    {15,  7, 13,  5}
};

static Node *new_node(uint16_t key, int degree) {
    Node *node = calloc(1, sizeof(*node) + degree * sizeof(Node *));
    if (node) node->key = key;
    return node;
}

static Node *new_trie(int degree, int *nkeys) {
    Node *root = new_node(0, degree);
    *nkeys = 0;
    for (; *nkeys < degree; (*nkeys)++)
        root->children[*nkeys] = new_node(*nkeys, degree);
    *nkeys += 2;
    return root;
}

static void del_trie(Node *root, int degree) {
    if (!root) return;
    for (int i = 0; i < degree; i++)
        del_trie(root->children[i], degree);
    free(root);
}

static void safe_write(int fd, const void *buf, size_t count) {
    if (write(fd, buf, count) != (ssize_t)count) {
        perror("Write error");
        exit(EXIT_FAILURE);
    }
}

ge_GIF *ge_new_gif(const char *fname, uint16_t width, uint16_t height, int depth, int bgindex, int loop) {
    ge_GIF *gif = calloc(1, sizeof(*gif) + (bgindex < 0 ? 2 : 1) * width * height);
    if (!gif) return NULL;

    gif->w = width; gif->h = height;
    gif->depth = abs(depth) > 1 ? abs(depth) : 2;
    gif->bgindex = bgindex;
    gif->frame = (uint8_t *)&gif[1];
    gif->back = &gif->frame[width * height];
    
    if ((gif->fd = creat(fname, 0666)) == -1) {
        free(gif);
        return NULL;
    }

    memcpy(gif->palette, DEFAULT_PALETTE, sizeof(gif->palette));
    
    safe_write(gif->fd, "GIF89a", 6);
    safe_write(gif->fd, (uint8_t[]){width & 0xFF, width >> 8, height & 0xFF, height >> 8}, 4);
    safe_write(gif->fd, (uint8_t[]){0xF0 | (gif->depth - 1), (uint8_t)bgindex, 0x00}, 3);
    safe_write(gif->fd, gif->palette, 3 << gif->depth);

    if (loop >= 0 && loop <= 0xFFFF) {
        safe_write(gif->fd, "!\xFF\x0BNETSCAPE2.0\x03\x01", 16);
        safe_write(gif->fd, (uint8_t[]){loop & 0xFF, loop >> 8, 0}, 3);
    }
    return gif;
}

static void put_key(ge_GIF *gif, uint16_t key, int key_size) {
    int byte_offset = gif->offset / 8;
    int bit_offset = gif->offset % 8;
    gif->partial |= ((uint32_t)key) << bit_offset;
    int bits_to_write = bit_offset + key_size;

    while (bits_to_write >= 8) {
        gif->buffer[byte_offset++] = gif->partial & 0xFF;
        if (byte_offset == 0xFF) {
            safe_write(gif->fd, "\xFF", 1);
            safe_write(gif->fd, gif->buffer, 0xFF);
            byte_offset = 0;
        }
        gif->partial >>= 8;
        bits_to_write -= 8;
    }
    gif->offset = (gif->offset + key_size) % (0xFF * 8);
}

static void end_key(ge_GIF *gif) {
    int byte_offset = gif->offset / 8;
    if (gif->offset % 8) gif->buffer[byte_offset++] = gif->partial & 0xFF;
    if (byte_offset) {
        safe_write(gif->fd, (uint8_t[]){byte_offset}, 1);
        safe_write(gif->fd, gif->buffer, byte_offset);
    }
    safe_write(gif->fd, "\0", 1);
    gif->offset = gif->partial = 0;
}

static void put_image(ge_GIF *gif, uint16_t w, uint16_t h, uint16_t x, uint16_t y) {
    int nkeys = 0, key_size;
    Node *node, *root = new_trie(1 << gif->depth, &nkeys);
    
    safe_write(gif->fd, ",", 1);
    safe_write(gif->fd, (uint8_t[]){x & 0xFF, x >> 8, y & 0xFF, y >> 8}, 4);
    safe_write(gif->fd, (uint8_t[]){w & 0xFF, w >> 8, h & 0xFF, h >> 8}, 4);
    safe_write(gif->fd, (uint8_t[]){0x00, gif->depth}, 2);

    key_size = gif->depth + 1;
    put_key(gif, 1 << gif->depth, key_size);
    node = root;

    for (int i = y; i < y + h; i++) {
        for (int j = x; j < x + w; j++) {
            uint8_t pixel = gif->frame[i * gif->w + j] & ((1 << gif->depth) - 1);
            Node *child = node->children[pixel];

            if (child) {
                node = child;
            } else {
                put_key(gif, node->key, key_size);
                if (nkeys < 0x1000) {
                    if (nkeys == (1 << key_size)) key_size++;
                    node->children[pixel] = new_node(nkeys++, 1 << gif->depth);
                } else {
                    put_key(gif, 1 << gif->depth, key_size);
                    del_trie(root, 1 << gif->depth);
                    root = new_trie(1 << gif->depth, &nkeys);
                    key_size = gif->depth + 1;
                }
                node = root->children[pixel];
            }
        }
    }

    put_key(gif, node->key, key_size);
    put_key(gif, (1 << gif->depth) + 1, key_size);
    end_key(gif);
    del_trie(root, 1 << gif->depth);
}

void ge_add_frame(ge_GIF *gif, uint8_t *input, uint16_t delay) {    
    uint8_t *flipped_buffer = calloc(gif->w * gif->h * 3, sizeof(uint8_t));

    // Flip image vertically
    for (int y = 0; y < gif->h; y++) {
        for (int x = 0; x < gif->w; x++) {
            int src_idx = (y * gif->w + x) * 3;
            int dst_idx = ((gif->h - 1 - y) * gif->w + x) * 3;
            flipped_buffer[dst_idx] = input[src_idx];
            flipped_buffer[dst_idx + 1] = input[src_idx + 1];
            flipped_buffer[dst_idx + 2] = input[src_idx + 2];
        }
    }

    for (int y = 0; y < gif->h; y++) {
        for (int x = 0; x < gif->w; x++) {
            int i = (y * gif->w + x) * 3;
            int r = flipped_buffer[i], g = flipped_buffer[i + 1], b = flipped_buffer[i + 2];
            uint8_t best_color = 0;
            uint32_t min_dist = UINT32_MAX;
            
            for (int j = 0; j < (1 << gif->depth); j++) {
                uint32_t dr = r - gif->palette[j * 3];
                uint32_t dg = g - gif->palette[j * 3 + 1];
                uint32_t db = b - gif->palette[j * 3 + 2];
                uint32_t dist = (dr * dr * 77 + dg * dg * 150 + db * db * 29) >> 8;
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_color = j;
                }
            }

            if (min_dist > 100) {
                int threshold = bayer_matrix[y & 3][x & 3];
                r = (r + ((threshold - 8) * 8));
                g = (g + ((threshold - 8) * 8));
                b = (b + ((threshold - 8) * 8));
                
                r = r < 0 ? 0 : (r > 255 ? 255 : r);
                g = g < 0 ? 0 : (g > 255 ? 255 : g);
                b = b < 0 ? 0 : (b > 255 ? 255 : b);
                
                min_dist = UINT32_MAX;
                for (int j = 0; j < (1 << gif->depth); j++) {
                    uint32_t dr = r - gif->palette[j * 3];
                    uint32_t dg = g - gif->palette[j * 3 + 1];
                    uint32_t db = b - gif->palette[j * 3 + 2];
                    uint32_t dist = (dr * dr * 77 + dg * dg * 150 + db * db * 29) >> 8;
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_color = j;
                    }
                }
            }
            gif->frame[y * gif->w + x] = best_color;
        }
    }

    if (delay || (gif->bgindex >= 0)) {
        safe_write(gif->fd, "!\xF9\x04", 3);
        safe_write(gif->fd, (uint8_t[]){((gif->bgindex >= 0 ? 2 : 1) << 2) + 1}, 1);
        safe_write(gif->fd, (uint8_t[]){delay & 0xFF, delay >> 8, (uint8_t)gif->bgindex, 0}, 4);
    }

    put_image(gif, gif->w, gif->h, 0, 0);
    gif->nframes++;

    if (gif->bgindex < 0) {
        uint8_t *tmp = gif->back;
        gif->back = gif->frame;
        gif->frame = tmp;
    }

    free(flipped_buffer);
}

void ge_close_gif(ge_GIF* gif) {
    safe_write(gif->fd, ";", 1);
    close(gif->fd);
    free(gif);
}

#endif /* GIF_H */