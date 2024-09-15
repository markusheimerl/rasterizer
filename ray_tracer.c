#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// Include stb_image_write for saving the output image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants
#define WIDTH 640
#define HEIGHT 480
#define MAX_DEPTH 3
#define N_RAYS 10

// Maximum number of vertices and triangles
#define MAX_VERTICES 100000
#define MAX_TRIANGLES 200000

// Vector macros
#define VEC_SUB(a,b,r) { r[0]=(a)[0]-(b)[0]; r[1]=(a)[1]-(b)[1]; r[2]=(a)[2]-(b)[2]; }
#define VEC_ADD(a,b,r) { r[0]=(a)[0]+(b)[0]; r[1]=(a)[1]+(b)[1]; r[2]=(a)[2]+(b)[2]; }
#define VEC_SCALE(v,s,r) { r[0]=(v)[0]*(s); r[1]=(v)[1]*(s); r[2]=(v)[2]*(s); }
#define VEC_DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); if(l>0){ (v)[0]/=l; (v)[1]/=l; (v)[2]/=l; } }
#define VEC_CROSS(a,b,r) { \
    (r)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; \
    (r)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; \
    (r)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; \
}

// Random number between -1 and 1
#define RAND_UNIFORM() (((double)rand() / RAND_MAX) * 2.0 - 1.0)

// Ambient color
double ambient_color[3] = {0.5, 0.5, 0.5};

// Global arrays for vertices and triangles
double vertices[MAX_VERTICES][3];
int triangles[MAX_TRIANGLES][3];
int num_vertices = 0;
int num_triangles = 0;

// BVH Node structure
typedef struct BVHNode {
    double bbox_min[3];
    double bbox_max[3];
    int start; // Start index in triangles array
    int count; // Number of triangles in this node
    struct BVHNode *left;
    struct BVHNode *right;
} BVHNode;

BVHNode *bvh_root = NULL;

// Function to compute bounding box for a triangle
void compute_triangle_bbox(int tri_index, double bbox_min[3], double bbox_max[3]) {
    int *tri = triangles[tri_index];
    for (int i = 0; i < 3; i++) {
        bbox_min[i] = bbox_max[i] = vertices[tri[0]][i];
        if (vertices[tri[1]][i] < bbox_min[i]) bbox_min[i] = vertices[tri[1]][i];
        if (vertices[tri[2]][i] < bbox_min[i]) bbox_min[i] = vertices[tri[2]][i];
        if (vertices[tri[1]][i] > bbox_max[i]) bbox_max[i] = vertices[tri[1]][i];
        if (vertices[tri[2]][i] > bbox_max[i]) bbox_max[i] = vertices[tri[2]][i];
    }
}

// Function to compute bounding box for a set of triangles
void compute_bbox(int start, int end, double bbox_min[3], double bbox_max[3]) {
    bbox_min[0] = bbox_min[1] = bbox_min[2] = DBL_MAX;
    bbox_max[0] = bbox_max[1] = bbox_max[2] = -DBL_MAX;
    for (int i = start; i < end; i++) {
        double tri_bbox_min[3], tri_bbox_max[3];
        compute_triangle_bbox(i, tri_bbox_min, tri_bbox_max);
        for (int j = 0; j < 3; j++) {
            if (tri_bbox_min[j] < bbox_min[j]) bbox_min[j] = tri_bbox_min[j];
            if (tri_bbox_max[j] > bbox_max[j]) bbox_max[j] = tri_bbox_max[j];
        }
    }
}

// Function to build BVH recursively
BVHNode* build_bvh(int start, int end) {
    BVHNode *node = (BVHNode*)malloc(sizeof(BVHNode));
    node->start = start;
    node->count = end - start;
    node->left = node->right = NULL;
    compute_bbox(start, end, node->bbox_min, node->bbox_max);

    if (node->count <= 4) {
        // Leaf node
        return node;
    } else {
        // Determine axis to split
        double extent[3] = {
            node->bbox_max[0] - node->bbox_min[0],
            node->bbox_max[1] - node->bbox_min[1],
            node->bbox_max[2] - node->bbox_min[2]
        };
        int axis = 0;
        if (extent[1] > extent[0]) axis = 1;
        if (extent[2] > extent[axis]) axis = 2;

        // Sort triangles based on centroid along axis
        int mid = (start + end) / 2;
        // Compute centroids
        double *centroids = (double*)malloc((end - start) * sizeof(double));
        for (int i = start; i < end; i++) {
            int *tri = triangles[i];
            double centroid = (vertices[tri[0]][axis] + vertices[tri[1]][axis] + vertices[tri[2]][axis]) / 3.0;
            centroids[i - start] = centroid;
        }

        // Partition triangles
        for (int i = start; i < end - 1; i++) {
            for (int j = i + 1; j < end; j++) {
                if (centroids[i - start] > centroids[j - start]) {
                    // Swap centroids
                    double temp_centroid = centroids[i - start];
                    centroids[i - start] = centroids[j - start];
                    centroids[j - start] = temp_centroid;
                    // Swap triangles
                    int temp_tri[3];
                    temp_tri[0] = triangles[i][0];
                    temp_tri[1] = triangles[i][1];
                    temp_tri[2] = triangles[i][2];
                    triangles[i][0] = triangles[j][0];
                    triangles[i][1] = triangles[j][1];
                    triangles[i][2] = triangles[j][2];
                    triangles[j][0] = temp_tri[0];
                    triangles[j][1] = temp_tri[1];
                    triangles[j][2] = temp_tri[2];
                }
            }
        }
        free(centroids);

        // Recursively build child nodes
        node->left = build_bvh(start, mid);
        node->right = build_bvh(mid, end);
    }
    return node;
}

// Function to intersect ray with BVH node
int bvh_intersect(BVHNode *node, const double ro[3], const double rd[3],
                  double *t_out, double point[3], double normal[3], double *nearest,
                  double hit_color[3]) {
    // AABB-Ray intersection
    double tmin = 0, tmax = *nearest;
    for (int i = 0; i < 3; i++) {
        double invD = 1.0 / rd[i];
        double t0 = (node->bbox_min[i] - ro[i]) * invD;
        double t1 = (node->bbox_max[i] - ro[i]) * invD;
        if (invD < 0.0) {
            double temp = t0; t0 = t1; t1 = temp;
        }
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax <= tmin)
            return 0;
    }

    int hit = 0;
    if (node->left == NULL && node->right == NULL) {
        // Leaf node, test triangles
        for (int i = node->start; i < node->start + node->count; i++) {
            int *tri = triangles[i];
            const double *v0 = vertices[tri[0]];
            const double *v1 = vertices[tri[1]];
            const double *v2 = vertices[tri[2]];
            double t_tmp, point_tmp[3], normal_tmp[3];
            if (triangle_intersect(v0, v1, v2, ro, rd, &t_tmp, point_tmp, normal_tmp)) {
                if (t_tmp < *nearest) {
                    *nearest = t_tmp;
                    *t_out = t_tmp;
                    point[0] = point_tmp[0]; point[1] = point_tmp[1]; point[2] = point_tmp[2];
                    normal[0] = normal_tmp[0]; normal[1] = normal_tmp[1]; normal[2] = normal_tmp[2];
                    hit_color[0] = 0.7; hit_color[1] = 0.7; hit_color[2] = 0.7; // Mesh color
                    hit = 1;
                }
            }
        }
    } else {
        // Recurse into child nodes
        int hit_left = 0, hit_right = 0;
        if (node->left)
            hit_left = bvh_intersect(node->left, ro, rd, t_out, point, normal, nearest, hit_color);
        if (node->right)
            hit_right = bvh_intersect(node->right, ro, rd, t_out, point, normal, nearest, hit_color);
        hit = hit_left || hit_right;
    }
    return hit;
}

// Function to parse OBJ file
void parse_obj_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open OBJ file %s\n", filename);
        exit(1);
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            // Vertex position
            double x, y, z;
            sscanf(line + 2, "%lf %lf %lf", &x, &y, &z);
            if (num_vertices >= MAX_VERTICES) {
                fprintf(stderr, "Exceeded maximum number of vertices\n");
                exit(1);
            }
            vertices[num_vertices][0] = x;
            vertices[num_vertices][1] = y;
            vertices[num_vertices][2] = z;
            num_vertices++;
        } else if (line[0] == 'f' && line[1] == ' ') {
            // Face
            int vi[3];
            int matches = sscanf(line + 2, "%d %d %d", &vi[0], &vi[1], &vi[2]);
            if (matches != 3) {
                // Try to parse format with slashes
                matches = sscanf(line + 2, "%d/%*d/%*d %d/%*d/%*d %d/%*d/%*d", &vi[0], &vi[1], &vi[2]);
                if (matches != 3) {
                    matches = sscanf(line + 2, "%d//%*d %d//%*d %d//%*d", &vi[0], &vi[1], &vi[2]);
                    if (matches != 3) {
                        fprintf(stderr, "Failed to parse face: %s", line);
                        continue;
                    }
                }
            }
            if (num_triangles >= MAX_TRIANGLES) {
                fprintf(stderr, "Exceeded maximum number of triangles\n");
                exit(1);
            }
            // OBJ indices start from 1
            triangles[num_triangles][0] = vi[0] - 1;
            triangles[num_triangles][1] = vi[1] - 1;
            triangles[num_triangles][2] = vi[2] - 1;
            num_triangles++;
        }
    }

    fclose(file);
}

// Triangle intersection using Möller–Trumbore algorithm
int triangle_intersect(const double v0[3], const double v1[3], const double v2[3],
                       const double ro[3], const double rd[3],
                       double *t_out, double point[3], double normal[3]) {
    const double EPSILON = 1e-8;
    double edge1[3], edge2[3], h[3], s[3], q[3];
    double a, f, u, v;
    VEC_SUB(v1, v0, edge1);
    VEC_SUB(v2, v0, edge2);
    VEC_CROSS(rd, edge2, h);
    a = VEC_DOT(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return 0;    // Ray is parallel to triangle
    f = 1.0 / a;
    VEC_SUB(ro, v0, s);
    u = f * VEC_DOT(s, h);
    if (u < 0.0 || u > 1.0)
        return 0;
    VEC_CROSS(s, edge1, q);
    v = f * VEC_DOT(rd, q);
    if (v < 0.0 || u + v > 1.0)
        return 0;
    double t = f * VEC_DOT(edge2, q);
    if (t > EPSILON) { // Ray intersection
        *t_out = t;
        VEC_SCALE(rd, t, point);
        VEC_ADD(ro, point, point);
        VEC_CROSS(edge1, edge2, normal);
        VEC_NORM(normal);
        return 1;
    } else
        return 0;
}

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

// Function to rotate a point around the Y-axis
void rotate_y(double angle, double point[3]) {
    double s = sin(angle);
    double c = cos(angle);
    double x = point[0];
    double z = point[2];
    point[0] = c * x + s * z;
    point[2] = -s * x + c * z;
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
    double hit_color[3];

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

    // Check BVH for mesh intersections
    if (bvh_root) {
        if (bvh_intersect(bvh_root, ro, rd, &t, tmp_point, tmp_normal, &nearest, hit_color)) {
            nearest = t; hit = 1; hot = 0;
            color[0] = hit_color[0]; color[1] = hit_color[1]; color[2] = hit_color[2];
            // Update point and normal
            point[0] = tmp_point[0]; point[1] = tmp_point[1]; point[2] = tmp_point[2];
            normal[0] = tmp_normal[0]; normal[1] = tmp_normal[1]; normal[2] = tmp_normal[2];
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

    // Parse OBJ file
    parse_obj_file("african_head.obj");

    // Apply transformations to the mesh vertices
    double scale_factor = 0.8;
    double translation[3] = {4.0, -1.0, 10.0}; // Adjusted translation
    double rotation_y = M_PI;      // Rotate 180 degrees around Y-axis

    for (int i = 0; i < num_vertices; i++) {
        // Scale
        vertices[i][0] *= scale_factor;
        vertices[i][1] *= scale_factor;
        vertices[i][2] *= scale_factor;

        // Rotate around Y-axis
        rotate_y(rotation_y, vertices[i]);

        // Translate
        vertices[i][0] += translation[0];
        vertices[i][1] += translation[1];
        vertices[i][2] += translation[2];
    }

    // Build BVH for the mesh
    bvh_root = build_bvh(0, num_triangles);

    for (int r = 0; r < N_RAYS; r++) {
        printf("Pass %d/%d\n", r + 1, N_RAYS);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                double x_screen = x - WIDTH / 2 + RAND_UNIFORM();
                double y_screen = y - HEIGHT / 2 + RAND_UNIFORM();
                double z_screen = focal;
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
