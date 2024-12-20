#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "gif.h"
#include "bmp.h"
#include "obj.h"

#define WIDTH 640
#define HEIGHT 480
#define FRAMES 60
#define ASPECT_RATIO ((double)WIDTH / (double)HEIGHT)
#define FOV_Y 60.0
#define NEAR_PLANE 0.1
#define FAR_PLANE 100.0

#define VEC_DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VEC_CROSS(a,b,r) { (r)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; \
                          (r)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; \
                          (r)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; }
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); \
                     if(l>0) { (v)[0]/=l; (v)[1]/=l; (v)[2]/=l; } }

typedef struct {
    double (*vertices)[3];
    double (*initial_vertices)[3];
    double (*transformed_vertices)[3];
    double (*texcoords)[2];
    int (*triangles)[3];
    int (*texcoord_indices)[3];
    int num_vertices, num_texcoords, num_triangles;
    struct {
        unsigned char *data;
        int width, height, channels;
    } texture;
    double model_matrix[4][4];
} Object3D;

void transform_object(Object3D* obj, double translate[3], double scale, double rotate_y) {
    double c = cos(rotate_y), s = sin(rotate_y);
    memset(obj->model_matrix, 0, 16 * sizeof(double));

    // Combine scale, rotation, and translation into a single matrix
    obj->model_matrix[0][0] = c * scale;
    obj->model_matrix[0][2] = s * scale;
    obj->model_matrix[1][1] = scale;
    obj->model_matrix[2][0] = -s * scale;
    obj->model_matrix[2][2] = c * scale;
    obj->model_matrix[3][3] = 1.0;

    obj->model_matrix[0][3] = translate[0];
    obj->model_matrix[1][3] = translate[1];
    obj->model_matrix[2][3] = translate[2];
}

Object3D* create_object(const char* obj_file, const char* texture_file) {
    Object3D* obj = calloc(1, sizeof(Object3D));
    if (!obj) return NULL;

    const int MAX_VERTICES = 100000;
    const int MAX_TRIANGLES = 200000;

    obj->vertices = malloc(MAX_VERTICES * sizeof(*obj->vertices) * 3);
    obj->initial_vertices = obj->vertices + MAX_VERTICES;
    obj->transformed_vertices = obj->vertices + (MAX_VERTICES * 2);
    obj->texcoords = malloc(MAX_VERTICES * sizeof(*obj->texcoords));
    obj->triangles = malloc(MAX_TRIANGLES * sizeof(*obj->triangles) * 2);
    obj->texcoord_indices = obj->triangles + MAX_TRIANGLES;

    load_obj(obj_file, obj->vertices, obj->initial_vertices, obj->texcoords,
                   obj->triangles, obj->texcoord_indices, &obj->num_vertices,
                   &obj->num_texcoords, &obj->num_triangles);

    obj->texture.data = load_bmp(texture_file, &obj->texture.width,
                                 &obj->texture.height, &obj->texture.channels);

    memset(obj->model_matrix, 0, 16 * sizeof(double));
    obj->model_matrix[0][0] = obj->model_matrix[1][1] =
    obj->model_matrix[2][2] = obj->model_matrix[3][3] = 1.0;

    return obj;
}

void update_vertices(Object3D* obj, double view_matrix[4][4]) {
    // Create perspective projection matrix
    double f = (1.0 / tan(FOV_Y * M_PI / 360.0));
    double proj_matrix[4][4] = {
        {f/ASPECT_RATIO, 0, 0, 0},
        {0, f, 0, 0},
        {0, 0, (FAR_PLANE+NEAR_PLANE)/(NEAR_PLANE-FAR_PLANE), (2*FAR_PLANE*NEAR_PLANE)/(NEAR_PLANE-FAR_PLANE)},
        {0, 0, -1, 0}
    };

    // Transform each vertex
    for (int i = 0; i < obj->num_vertices; i++) {
        // Start with initial vertex position
        double vertex[4] = {
            obj->initial_vertices[i][0],
            obj->initial_vertices[i][1],
            obj->initial_vertices[i][2],
            1.0
        };
        
        // Apply model transformation
        double model_vertex[4] = {0};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                model_vertex[row] += obj->model_matrix[row][col] * vertex[col];
            }
        }
        
        // Apply view transformation
        double view_vertex[4] = {0};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                view_vertex[row] += view_matrix[row][col] * model_vertex[col];
            }
        }
        
        // Apply perspective projection
        double proj_vertex[4] = {0};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                proj_vertex[row] += proj_matrix[row][col] * view_vertex[col];
            }
        }
        
        // Perform perspective divide and viewport transformation
        if (proj_vertex[3] != 0) {
            obj->transformed_vertices[i][0] = ((proj_vertex[0] / proj_vertex[3] + 1.0) * 0.5) * WIDTH;
            obj->transformed_vertices[i][1] = ((proj_vertex[1] / proj_vertex[3] + 1.0) * 0.5) * HEIGHT;
            obj->transformed_vertices[i][2] = proj_vertex[2] / proj_vertex[3];
        } else {
            obj->transformed_vertices[i][0] = 0;
            obj->transformed_vertices[i][1] = 0;
            obj->transformed_vertices[i][2] = 0;
        }
    }
}

void render_scene(uint8_t *image, Object3D **objects, int num_objects) {
    double *depth_buffer = calloc(WIDTH * HEIGHT, sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) depth_buffer[i] = -DBL_MAX;

    for (int obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        Object3D *obj = objects[obj_idx];

        for (int tri_idx = 0; tri_idx < obj->num_triangles; tri_idx++) {
            double triangle[3][4], uv[3][2];
            for (int v = 0; v < 3; v++) {
                memcpy(triangle[v], obj->transformed_vertices[obj->triangles[tri_idx][v]], 3 * sizeof(double));
                triangle[v][3] = 1.0 / triangle[v][2];
                memcpy(uv[v], obj->texcoords[obj->texcoord_indices[tri_idx][v]], 2 * sizeof(double));
            }

            // Get triangle bounds
            int min_x = fmax(0, floor(fmin(fmin(triangle[0][0], triangle[1][0]), triangle[2][0])));
            int min_y = fmax(0, floor(fmin(fmin(triangle[0][1], triangle[1][1]), triangle[2][1])));
            int max_x = fmin(WIDTH - 1, ceil(fmax(fmax(triangle[0][0], triangle[1][0]), triangle[2][0])));
            int max_y = fmin(HEIGHT - 1, ceil(fmax(fmax(triangle[0][1], triangle[1][1]), triangle[2][1])));

            // Rasterize triangle
            for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                    double bary_denom = ((triangle[1][1] - triangle[2][1]) * (triangle[0][0] - triangle[2][0]) +
                                       (triangle[2][0] - triangle[1][0]) * (triangle[0][1] - triangle[2][1]));

                    if (fabs(bary_denom) < 1e-6) continue;

                    double lambda0 = ((triangle[1][1] - triangle[2][1]) * (x - triangle[2][0]) +
                                     (triangle[2][0] - triangle[1][0]) * (y - triangle[2][1])) / bary_denom;
                    double lambda1 = ((triangle[2][1] - triangle[0][1]) * (x - triangle[2][0]) +
                                     (triangle[0][0] - triangle[2][0]) * (y - triangle[2][1])) / bary_denom;
                    double lambda2 = 1.0 - lambda0 - lambda1;

                    if (lambda0 >= 0 && lambda0 <= 1 && lambda1 >= 0 && lambda1 <= 1 && lambda2 >= 0 && lambda2 <=1) {
                        int idx = y * WIDTH + x;
                        double z = lambda0 * triangle[0][2] + lambda1 * triangle[1][2] + lambda2 * triangle[2][2];

                        if (z > depth_buffer[idx]) {
                            depth_buffer[idx] = z;

                            double u = (lambda0 * uv[0][0] * triangle[0][3] +
                                       lambda1 * uv[1][0] * triangle[1][3] +
                                       lambda2 * uv[2][0] * triangle[2][3]) * z;
                            double v = (lambda0 * uv[0][1] * triangle[0][3] +
                                       lambda1 * uv[1][1] * triangle[1][3] +
                                       lambda2 * uv[2][1] * triangle[2][3]) * z;

                            u = u - floor(u);
                            v = 1.0 - (v - floor(v));

                            int tx = (int)(u * obj->texture.width);
                            int ty = (int)(v * obj->texture.height);
                            tx = fmin(fmax(tx, 0), obj->texture.width - 1);
                            ty = fmin(fmax(ty, 0), obj->texture.height - 1);

                            int tex_idx = (ty * obj->texture.width + tx) * 3;
                            for (int i = 0; i < 3; i++) {
                                image[idx * 3 + i] = obj->texture.data[tex_idx + i];
                            }
                        }
                    }
                }
            }
        }
    }
    free(depth_buffer);
}

int main() {
    Object3D* objects[] = {
        create_object("drone.obj", "drone.bmp"),
        create_object("ground.obj", "ground.bmp")
    };

    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, 4, -1, 0);

    double camera_pos[3] = {-2.0, 1.0, -2.0};
    double camera_target[3] = {0.0, 0.0, 0.0};
    double camera_up[3] = {0.0, 1.0, 0.0};

    for (int frame = 0; frame < FRAMES; frame++) {
        memset(frame_buffer, 0, WIDTH * HEIGHT * 3);

        // Update transforms
        double drone_pos[3] = {0.0, 0.5, 0.0};
        double ground_pos[3] = {0.0, -0.5, 0.0};
        transform_object(objects[0], drone_pos, 0.5, frame * (2.0 * M_PI) / FRAMES);
        transform_object(objects[1], ground_pos, 1.0, 0.0);

        // Calculate view matrix
        double z[3] = {camera_target[0]-camera_pos[0], camera_target[1]-camera_pos[1], camera_target[2]-camera_pos[2]};
        VEC_NORM(z);
        double x[3], y[3];
        VEC_CROSS(camera_up, z, x);
        VEC_NORM(x);
        VEC_CROSS(z, x, y);

        double view_matrix[4][4] = {
            {x[0], x[1], x[2], -VEC_DOT(x, camera_pos)},
            {y[0], y[1], y[2], -VEC_DOT(y, camera_pos)},
            {z[0], z[1], z[2], -VEC_DOT(z, camera_pos)},
            {0, 0, 0, 1}
        };

        for (int i = 0; i < 2; i++) {
            update_vertices(objects[i], view_matrix);
        }

        render_scene(frame_buffer, objects, 2);

        camera_pos[0] += 0.05;
        camera_pos[2] += 0.05;
        camera_target[0] += 0.05;
        camera_target[2] += 0.05;

        ge_add_frame(gif, frame_buffer, 6);
        printf("Rendered frame %d/%d\n", frame + 1, FRAMES);
    }

    ge_close_gif(gif);
    free(frame_buffer);
    for (int i = 0; i < 2; i++) {
        if (objects[i]) {
            free(objects[i]->vertices);
            free(objects[i]->texcoords);
            free(objects[i]->triangles);
            free(objects[i]->texture.data);
            free(objects[i]);
        }
    }

    return 0;
}