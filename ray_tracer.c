#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Include the stb_image_write header
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants
#define WIDTH 640
#define HEIGHT 480
#define MAX_DEPTH 3
#define N_RAYS 10

// Ambient color
double ambient_color[3] = {0.5, 0.5, 0.5};

// Define object types
typedef enum {
    SPHERE,
    BOX
} ObjectType;

// Object structure
typedef struct {
    ObjectType type;
    double center[3]; // For spheres
    double radius;    // For spheres
    double min[3];    // For boxes
    double max[3];    // For boxes
    double color[3];
    int hot;
} Object;

// Vector operations
void vector_subtract(const double *a, const double *b, double *result) {
    result[0] = a[0] - b[0];
    result[1] = a[1] - b[1];
    result[2] = a[2] - b[2];
}

void vector_add(const double *a, const double *b, double *result) {
    result[0] = a[0] + b[0];
    result[1] = a[1] + b[1];
    result[2] = a[2] + b[2];
}

void vector_scale(const double *v, double scale, double *result) {
    result[0] = v[0] * scale;
    result[1] = v[1] * scale;
    result[2] = v[2] * scale;
}

double vector_dot(const double *a, const double *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void vector_normalize(double *v) {
    double norm = sqrt(vector_dot(v, v));
    if (norm > 0) {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

void vector_copy(const double *src, double *dest) {
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

// Random number between -1 and 1
double rand_uniform() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Sphere intersection
int sphere_intersect(const double center[3], double radius, const double ray_origin[3], const double ray_direction[3],
                     double *t_out, double point[3], double normal[3]) {
    double z[3];
    vector_subtract(center, ray_origin, z);
    double proj = vector_dot(ray_direction, z);
    double z_dot_z = vector_dot(z, z);
    double delta = radius*radius + proj*proj - z_dot_z;
    if (delta < 0) return 0;
    double sqrt_delta = sqrt(delta);
    double t = proj - sqrt_delta;
    if (t < 1e-3) return 0;
    *t_out = t;
    vector_scale(ray_direction, t, point);
    vector_add(ray_origin, point, point);
    vector_subtract(point, center, normal);
    vector_scale(normal, 1.0 / radius, normal);
    return 1;
}

// Box intersection
int box_intersect(const double bmin[3], const double bmax[3], const double ray_origin[3], double ray_direction[3],
                  double *t_out, double point[3], double normal[3]) {
    double tmin = -INFINITY, tmax = INFINITY;
    int hit_axis = -1;
    for (int i = 0; i < 3; i++) {
        double direction = ray_direction[i];
        if (fabs(direction) < 1e-3) direction = (direction >= 0) ? 1e-3 : -1e-3;
        double invD = 1.0 / direction;
        double t0 = (bmin[i] - ray_origin[i]) * invD;
        double t1 = (bmax[i] - ray_origin[i]) * invD;
        int sign = (invD < 0);
        if (sign) {
            double tmp = t0; t0 = t1; t1 = tmp;
        }
        if (t0 > tmin) { tmin = t0; hit_axis = i; }
        if (t1 < tmax) tmax = t1;
        if (tmin > tmax) return 0;
    }
    if (tmin < 1e-3) return 0;
    *t_out = tmin;
    vector_scale(ray_direction, tmin, point);
    vector_add(ray_origin, point, point);
    memset(normal, 0, 3 * sizeof(double));
    normal[hit_axis] = -((ray_direction[hit_axis] > 0) ? 1 : -1);
    return 1;
}

// Scene intersection
int scene_intersect(const double ray_origin[3], const double ray_direction[3],
                    double point[3], double normal[3], double color[3], int *hot) {
    int hit = 0;
    double nearest = INFINITY;
    double tmp_point[3], tmp_normal[3];
    double t;
    // Scene description
    Object objects[] = {
        {SPHERE, {6, 0, 7}, 2, {0}, {0}, {1.0, 0.4, 0.6}, 0},
        {SPHERE, {2.8, 1.1, 7}, 0.9, {0}, {0}, {1.0, 1.0, 0.3}, 0},
        {SPHERE, {5, -10, -7}, 8, {0}, {0}, {1.0, 1.0, 1.0}, 1},
        {BOX, {0}, 0, {3, -4, 11}, {7, 2, 13}, {0.4, 0.7, 1.0}, 0},
        {BOX, {0}, 0, {0, 2, 6}, {11, 2.2, 16}, {0.6, 0.7, 0.6}, 0}
    };
    int num_objects = sizeof(objects) / sizeof(Object);
    for (int i = 0; i < num_objects; i++) {
        Object *o = &objects[i];
        int current_hit = 0;
        if (o->type == SPHERE) {
            current_hit = sphere_intersect(o->center, o->radius, ray_origin, ray_direction, &t, tmp_point, tmp_normal);
        } else {
            current_hit = box_intersect(o->min, o->max, ray_origin, ray_direction, &t, tmp_point, tmp_normal);
        }
        if (current_hit && t < nearest) {
            nearest = t;
            vector_copy(tmp_point, point);
            vector_copy(tmp_normal, normal);
            vector_copy(o->color, color);
            *hot = o->hot;
            hit = 1;
        }
    }
    return hit;
}

// Reflect function
void reflect(const double vector[3], const double normal[3], double result[3]) {
    double dot = vector_dot(vector, normal);
    double tmp[3];
    vector_scale(normal, 2 * dot, tmp);
    vector_subtract(vector, tmp, result);
    // Add small random perturbation
    double random_vec[3] = {rand_uniform() / 6.0, rand_uniform() / 6.0, rand_uniform() / 6.0};
    vector_add(result, random_vec, result);
    vector_normalize(result);
}

// Trace function
void trace(const double ray_origin[3], const double ray_direction[3], int depth, double color_out[3]) {
    if (depth > MAX_DEPTH) {
        vector_copy(ambient_color, color_out);
        return;
    }
    double point[3], normal[3], color[3];
    int hot = 0;
    int hit = scene_intersect(ray_origin, ray_direction, point, normal, color, &hot);
    if (!hit) {
        vector_copy(ambient_color, color_out);
        return;
    }
    if (hot) {
        vector_copy(color, color_out);
        return;
    }
    double reflected_dir[3];
    reflect(ray_direction, normal, reflected_dir);
    double new_color[3];
    trace(point, reflected_dir, depth + 1, new_color);
    color_out[0] = color[0] * new_color[0];
    color_out[1] = color[1] * new_color[1];
    color_out[2] = color[2] * new_color[2];
}

// Main function
int main() {
    // Set random seed for reproducibility
    srand(42);

    double focal = 500;
    double azimuth = 30 * M_PI / 180.0;

    double eye[3] = {0, 0, 0};
    double *image = (double *)calloc(WIDTH * HEIGHT * 3, sizeof(double));

    for (int r = 0; r < N_RAYS; r++) {
        printf("Pass %d/%d\n", r + 1, N_RAYS);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                double x_screen = x - WIDTH / 2;
                double y_screen = y - HEIGHT / 2;
                double z_screen = focal;

                double dir[3];
                dir[0] = cos(azimuth) * x_screen + sin(azimuth) * z_screen;
                dir[1] = y_screen;
                dir[2] = -sin(azimuth) * x_screen + cos(azimuth) * z_screen;
                vector_normalize(dir);

                double color[3];
                trace(eye, dir, 0, color);

                int idx = (y * WIDTH + x) * 3;
                image[idx + 0] += color[0];
                image[idx + 1] += color[1];
                image[idx + 2] += color[2];
            }
        }
    }

    // Prepare image data for saving
    unsigned char *output_image = (unsigned char *)malloc(WIDTH * HEIGHT * 3);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        int r = (int)(fmin(image[i * 3 + 0] / N_RAYS, 1.0) * 255);
        int g = (int)(fmin(image[i * 3 + 1] / N_RAYS, 1.0) * 255);
        int b = (int)(fmin(image[i * 3 + 2] / N_RAYS, 1.0) * 255);
        output_image[i * 3 + 0] = (unsigned char)r;
        output_image[i * 3 + 1] = (unsigned char)g;
        output_image[i * 3 + 2] = (unsigned char)b;
    }

    // Save the image using stb_image_write
    if (stbi_write_png("result.png", WIDTH, HEIGHT, 3, output_image, WIDTH * 3)) {
        printf("Image saved to 'result.png'\n");
    } else {
        fprintf(stderr, "Failed to save image\n");
    }

    // Clean up
    free(image);
    free(output_image);

    return 0;
}
