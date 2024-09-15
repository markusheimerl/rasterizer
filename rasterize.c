#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIDTH 800
#define HEIGHT 600
#define MAX_VERTICES 1000
#define MAX_FACES 2000

typedef struct {
    float x, y, z;
} Vector3;

Vector3 vertices[MAX_VERTICES];
int faces[MAX_FACES][3];
int vertex_count = 0;
int face_count = 0;

char frame_buffer[HEIGHT][WIDTH];

void clear_frame_buffer() {
    memset(frame_buffer, ' ', sizeof(frame_buffer));
}

void draw_line(int x1, int y1, int x2, int y2) {
    int dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
    int dy = -abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
    int err = dx + dy, e2;

    while (1) {
        if (x1 >= 0 && x1 < WIDTH && y1 >= 0 && y1 < HEIGHT)
            frame_buffer[y1][x1] = '#';
        if (x1 == x2 && y1 == y2) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x1 += sx; }
        if (e2 <= dx) { err += dx; y1 += sy; }
    }
}

void load_obj(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            sscanf(line, "v %f %f %f", &vertices[vertex_count].x, &vertices[vertex_count].y, &vertices[vertex_count].z);
            vertex_count++;
        } else if (line[0] == 'f' && line[1] == ' ') {
            sscanf(line, "f %d %d %d", &faces[face_count][0], &faces[face_count][1], &faces[face_count][2]);
            faces[face_count][0]--; faces[face_count][1]--; faces[face_count][2]--;
            face_count++;
        }
    }

    fclose(file);
}

void render() {
    clear_frame_buffer();

    for (int i = 0; i < face_count; i++) {
        for (int j = 0; j < 3; j++) {
            int v1 = faces[i][j];
            int v2 = faces[i][(j + 1) % 3];
            int x1 = (vertices[v1].x + 1) * WIDTH / 2;
            int y1 = (1 - vertices[v1].y) * HEIGHT / 2;
            int x2 = (vertices[v2].x + 1) * WIDTH / 2;
            int y2 = (1 - vertices[v2].y) * HEIGHT / 2;
            draw_line(x1, y1, x2, y2);
        }
    }
}

void save_bmp(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Failed to open file for writing: %s\n", filename);
        return;
    }

    int padding = (4 - (WIDTH * 3) % 4) % 4;
    int filesize = 54 + (3 * WIDTH + padding) * HEIGHT;

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(WIDTH    );
    bmpinfoheader[ 5] = (unsigned char)(WIDTH>> 8);
    bmpinfoheader[ 6] = (unsigned char)(WIDTH>>16);
    bmpinfoheader[ 7] = (unsigned char)(WIDTH>>24);
    bmpinfoheader[ 8] = (unsigned char)(HEIGHT    );
    bmpinfoheader[ 9] = (unsigned char)(HEIGHT>> 8);
    bmpinfoheader[10] = (unsigned char)(HEIGHT>>16);
    bmpinfoheader[11] = (unsigned char)(HEIGHT>>24);

    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);

    for (int i = HEIGHT - 1; i >= 0; i--) {
        for (int j = 0; j < WIDTH; j++) {
            unsigned char color = (frame_buffer[i][j] == '#') ? 255 : 0;
            unsigned char pixel[3] = {color, color, color};
            fwrite(pixel, 1, 3, f);
        }
        for (int p = 0; p < padding; p++) {
            unsigned char pad = 0;
            fwrite(&pad, 1, 1, f);
        }
    }

    fclose(f);
}

int main() {
    load_obj("drone.obj");
    render();
    save_bmp("drone_render.bmp");
    return 0;
}