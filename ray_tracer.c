#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// stb_image_write.h included here
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants
#define WIDTH 640
#define HEIGHT 480
#define MAX_DEPTH 3
#define N_RAYS 10

// Vector macros
#define VEC_SUB(a,b,r) { r[0]=a[0]-b[0]; r[1]=a[1]-b[1]; r[2]=a[2]-b[2]; }
#define VEC_ADD(a,b,r) { r[0]=a[0]+b[0]; r[1]=a[1]+b[1]; r[2]=a[2]+b[2]; }
#define VEC_SCALE(v,s,r) { r[0]=v[0]*(s); r[1]=v[1]*(s); r[2]=v[2]*(s); }
#define VEC_DOT(a,b) (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); if(l>0){ v[0]/=l; v[1]/=l; v[2]/=l; } }

// Random number between -1 and 1
#define RAND_UNIFORM() (((double)rand() / RAND_MAX) * 2.0 - 1.0)

// Ambient color
double ambient_color[3] = {0.5, 0.5, 0.5};

// Sphere intersection
int sphere_intersect(const double center[3], double radius, const double ro[3], const double rd[3],
                     double *t_out, double point[3], double normal[3]) {
    double oc[3]; VEC_SUB(ro, center, oc);
    double b = VEC_DOT(oc, rd);
    double c = VEC_DOT(oc, oc) - radius * radius;
    double delta = b * b - c;
    if (delta < 0) return 0;
    double sqrt_delta = sqrt(delta);
    double t = -b - sqrt_delta;
    if (t < 1e-3) {
        t = -b + sqrt_delta;
        if (t < 1e-3) return 0;
    }
    *t_out = t;
    VEC_SCALE(rd, t, point); VEC_ADD(ro, point, point);
    VEC_SUB(point, center, normal); VEC_NORM(normal);
    return 1;
}

// Box intersection
int box_intersect(const double bmin[3], const double bmax[3], const double ro[3], const double rd[3],
                  double *t_out, double point[3], double normal[3]) {
    double tmin = -INFINITY, tmax = INFINITY;
    int hit_axis = -1;
    for (int i = 0; i < 3; i++) {
        double invD = 1.0 / rd[i];
        double t0 = (bmin[i] - ro[i]) * invD;
        double t1 = (bmax[i] - ro[i]) * invD;
        if (invD < 0) { double tmp = t0; t0 = t1; t1 = tmp; }
        if (t0 > tmin) { tmin = t0; hit_axis = i; }
        if (t1 < tmax) tmax = t1;
        if (tmin > tmax) return 0;
    }
    if (tmin < 1e-3) return 0;
    *t_out = tmin;
    VEC_SCALE(rd, tmin, point); VEC_ADD(ro, point, point);
    normal[0] = normal[1] = normal[2] = 0;
    normal[hit_axis] = (rd[hit_axis] > 0) ? -1 : 1;
    return 1;
}

// Trace function
void trace(const double ro[3], const double rd[3], int depth, double color_out[3]) {
    if (depth > MAX_DEPTH) {
        color_out[0] = ambient_color[0];
        color_out[1] = ambient_color[1];
        color_out[2] = ambient_color[2];
        return;
    }
    double point[3], normal[3], color[3];
    int hot = 0, hit = 0;
    double nearest = INFINITY, t;
    double tmp_point[3], tmp_normal[3];

    // Scene description
    // Spheres
    double spheres[3][7] = {
        {6, 0, 7, 2, 1.0, 0.4, 0.6},
        {2.8, 1.1, 7, 0.9, 1.0, 1.0, 0.3},
        {5, -10, -7, 8, 1.0, 1.0, 1.0}
    };
    // Boxes
    double boxes[2][9] = {
        {3, -4, 11, 7, 2, 13, 0.4, 0.7, 1.0},
        {0, 2, 6, 11, 2.2, 16, 0.6, 0.7, 0.6}
    };

    // Check spheres
    for (int i = 0; i < 3; i++) {
        if (sphere_intersect(&spheres[i][0], spheres[i][3], ro, rd, &t, tmp_point, tmp_normal)) {
            if (t < nearest) {
                nearest = t; hit = 1; hot = (i == 2);
                color[0] = spheres[i][4]; color[1] = spheres[i][5]; color[2] = spheres[i][6];
                // Update point and normal
                point[0] = tmp_point[0]; point[1] = tmp_point[1]; point[2] = tmp_point[2];
                normal[0] = tmp_normal[0]; normal[1] = tmp_normal[1]; normal[2] = tmp_normal[2];
            }
        }
    }

    // Check boxes
    for (int i = 0; i < 2; i++) {
        if (box_intersect(&boxes[i][0], &boxes[i][3], ro, rd, &t, tmp_point, tmp_normal)) {
            if (t < nearest) {
                nearest = t; hit = 1; hot = 0;
                color[0] = boxes[i][6]; color[1] = boxes[i][7]; color[2] = boxes[i][8];
                // Update point and normal
                point[0] = tmp_point[0]; point[1] = tmp_point[1]; point[2] = tmp_point[2];
                normal[0] = tmp_normal[0]; normal[1] = tmp_normal[1]; normal[2] = tmp_normal[2];
            }
        }
    }

    if (!hit) {
        color_out[0] = ambient_color[0];
        color_out[1] = ambient_color[1];
        color_out[2] = ambient_color[2];
        return;
    }
    if (hot) {
        color_out[0] = color[0];
        color_out[1] = color[1];
        color_out[2] = color[2];
        return;
    }

    // Reflect
    double reflected_dir[3];
    double dot = VEC_DOT(rd, normal);
    VEC_SCALE(normal, 2 * dot, reflected_dir);
    VEC_SUB(rd, reflected_dir, reflected_dir);
    // Add small random perturbation
    reflected_dir[0] += RAND_UNIFORM() / 6.0;
    reflected_dir[1] += RAND_UNIFORM() / 6.0;
    reflected_dir[2] += RAND_UNIFORM() / 6.0;
    VEC_NORM(reflected_dir);

    double new_color[3];
    trace(point, reflected_dir, depth + 1, new_color);
    color_out[0] = color[0] * new_color[0];
    color_out[1] = color[1] * new_color[1];
    color_out[2] = color[2] * new_color[2];
}

int main() {
    srand(42);
    double focal = 500, azimuth = 30 * M_PI / 180.0;
    double eye[3] = {0, 0, 0};
    double *image = calloc(WIDTH * HEIGHT * 3, sizeof(double));

    for (int r = 0; r < N_RAYS; r++) {
        printf("Pass %d/%d\n", r + 1, N_RAYS);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                double x_screen = x - WIDTH / 2, y_screen = y - HEIGHT / 2, z_screen = focal;
                double dir[3] = {
                    cos(azimuth) * x_screen + sin(azimuth) * z_screen,
                    y_screen,
                    -sin(azimuth) * x_screen + cos(azimuth) * z_screen
                };
                VEC_NORM(dir);
                double color[3];
                trace(eye, dir, 0, color);
                int idx = (y * WIDTH + x) * 3;
                image[idx + 0] += color[0];
                image[idx + 1] += color[1];
                image[idx + 2] += color[2];
            }
        }
    }

    // Prepare image data
    unsigned char *output_image = malloc(WIDTH * HEIGHT * 3);
    for (int i = 0; i < WIDTH * HEIGHT * 3; i++)
        output_image[i] = (unsigned char)(fmin(image[i] / N_RAYS, 1.0) * 255);

    // Save the image
    if (stbi_write_png("result.png", WIDTH, HEIGHT, 3, output_image, WIDTH * 3))
        printf("Image saved to 'result.png'\n");
    else
        fprintf(stderr, "Failed to save image\n");

    free(image);
    free(output_image);
    return 0;
}
