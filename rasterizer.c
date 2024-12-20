#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "gif.h"
#include "bmp.h"
#include "obj.h"

// gcc -O3 rasterizer.c -lm && ./a.out

#define WIDTH 640
#define HEIGHT 480
#define FRAMES 60
#define ASPECT_RATIO ((double)WIDTH / (double)HEIGHT)
#define FOV_Y 60.0
#define NEAR_PLANE 0.1
#define FAR_PLANE 100.0

// Core vector operations
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

void matrix_identity(double m[4][4]) {
    memset(m, 0, 16 * sizeof(double));
    m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0;
}

void matrix_multiply(double a[4][4], double b[4][4], double result[4][4]) {
    double temp[4][4] = {0};
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            for(int k = 0; k < 4; k++) {
                temp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    memcpy(result, temp, 16 * sizeof(double));
}

void matrix_translate(double m[4][4], double x, double y, double z) {
    double t[4][4] = {{1,0,0,x}, {0,1,0,y}, {0,0,1,z}, {0,0,0,1}};
    matrix_multiply(m, t, m);
}

void matrix_scale(double m[4][4], double s) {
    double scale[4][4] = {{s,0,0,0}, {0,s,0,0}, {0,0,s,0}, {0,0,0,1}};
    matrix_multiply(m, scale, m);
}

void matrix_rotate_y(double m[4][4], double angle) {
    double c = cos(angle), s = sin(angle);
    double r[4][4] = {{c,0,s,0}, {0,1,0,0}, {-s,0,c,0}, {0,0,0,1}};
    matrix_multiply(m, r, m);
}

void build_transform_matrix(double matrix[4][4], double translate[3], double scale, double rotate_y) {
    matrix_identity(matrix);
    if (translate) matrix_translate(matrix, translate[0], translate[1], translate[2]);
    if (scale != 1.0) matrix_scale(matrix, scale);
    if (rotate_y != 0.0) matrix_rotate_y(matrix, rotate_y);
}

void calculate_view_matrix(const double pos[3], const double target[3], 
                         const double up[3], double view[4][4]) {
    double z[3] = {target[0]-pos[0], target[1]-pos[1], target[2]-pos[2]};
    VEC_NORM(z);
    
    double x[3];
    VEC_CROSS(up, z, x);
    VEC_NORM(x);
    
    double y[3];
    VEC_CROSS(z, x, y);
    
    view[0][0] = x[0]; view[0][1] = x[1]; view[0][2] = x[2];
    view[1][0] = y[0]; view[1][1] = y[1]; view[1][2] = y[2];
    view[2][0] = z[0]; view[2][1] = z[1]; view[2][2] = z[2];
    
    view[0][3] = -VEC_DOT(x, pos);
    view[1][3] = -VEC_DOT(y, pos);
    view[2][3] = -VEC_DOT(z, pos);
    
    view[3][0] = view[3][1] = view[3][2] = 0.0;
    view[3][3] = 1.0;
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

    parse_obj_file(obj_file, obj->vertices, obj->initial_vertices, obj->texcoords,
                  obj->triangles, obj->texcoord_indices, &obj->num_vertices,
                  &obj->num_texcoords, &obj->num_triangles);
    
    obj->texture.data = load_bmp(texture_file, &obj->texture.width,
                                &obj->texture.height, &obj->texture.channels);
    
    matrix_identity(obj->model_matrix);
    return obj;
}

void free_object(Object3D* obj) {
    if (!obj) return;
    free(obj->vertices);
    free(obj->texcoords);
    free(obj->triangles);
    free(obj->texture.data);
    free(obj);
}

void sample_texture(double u, double v, double color[3], unsigned char *texture_data, 
                   int texture_width, int texture_height) {
    u = u - floor(u);
    v = 1.0 - (v - floor(v));
    
    int x = (int)(u * texture_width);
    int y = (int)(v * texture_height);
    x = fmin(fmax(x, 0), texture_width - 1);
    y = fmin(fmax(y, 0), texture_height - 1);
    
    int idx = (y * texture_width + x) * 3;
    static const double inv255 = 1.0 / 255.0;
    for (int i = 0; i < 3; i++) {
        color[i] = texture_data[idx + i] * inv255;
    }
}

void get_triangle_bounds(double triangle[3][4], int *min_x, int *min_y, int *max_x, int *max_y) {
    *min_x = fmax(0, floor(fmin(fmin(triangle[0][0], triangle[1][0]), triangle[2][0])));
    *min_y = fmax(0, floor(fmin(fmin(triangle[0][1], triangle[1][1]), triangle[2][1])));
    *max_x = fmin(WIDTH - 1, ceil(fmax(fmax(triangle[0][0], triangle[1][0]), triangle[2][0])));
    *max_y = fmin(HEIGHT - 1, ceil(fmax(fmax(triangle[0][1], triangle[1][1]), triangle[2][1])));
}

void calculate_barycentric(double triangle[3][4], int x, int y, double *lambda0, double *lambda1) {
    double bary_denom = ((triangle[1][1] - triangle[2][1]) * (triangle[0][0] - triangle[2][0]) + 
                        (triangle[2][0] - triangle[1][0]) * (triangle[0][1] - triangle[2][1]));
    
    if (fabs(bary_denom) < 1e-6) {
        *lambda0 = *lambda1 = -1;
        return;
    }
    
    *lambda0 = ((triangle[1][1] - triangle[2][1]) * (x - triangle[2][0]) + 
                (triangle[2][0] - triangle[1][0]) * (y - triangle[2][1])) / bary_denom;
    *lambda1 = ((triangle[2][1] - triangle[0][1]) * (x - triangle[2][0]) + 
                (triangle[0][0] - triangle[2][0]) * (y - triangle[2][1])) / bary_denom;
}

void render_triangle(uint8_t *image, double *depth_buffer, double triangle[3][4], 
                    double uv[3][2], unsigned char *texture_data, 
                    int texture_width, int texture_height) {
    int min_x, min_y, max_x, max_y;
    get_triangle_bounds(triangle, &min_x, &min_y, &max_x, &max_y);

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            double lambda0, lambda1;
            calculate_barycentric(triangle, x, y, &lambda0, &lambda1);
            double lambda2 = 1.0 - lambda0 - lambda1;

            if (lambda0 >= 0 && lambda0 <= 1 && lambda1 >= 0 && lambda1 <= 1 && lambda2 >= 0) {
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
                    
                    double color[3];
                    sample_texture(u, v, color, texture_data, texture_width, texture_height);
                    
                    for (int i = 0; i < 3; i++) {
                        image[idx * 3 + i] = (uint8_t)(color[i] * 255.0);
                    }
                }
            }
        }
    }
}

void render_frame(uint8_t *image, Object3D **objects, int num_objects) {
    double *depth_buffer = calloc(WIDTH * HEIGHT, sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = -DBL_MAX;
    }

    for (int obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        Object3D *obj = objects[obj_idx];
        
        for (int tri_idx = 0; tri_idx < obj->num_triangles; tri_idx++) {
            double triangle[3][4];
            double uv[3][2];
            
            for (int v = 0; v < 3; v++) {
                memcpy(triangle[v], obj->transformed_vertices[obj->triangles[tri_idx][v]], 3 * sizeof(double));
                triangle[v][3] = 1.0 / triangle[v][2];
                memcpy(uv[v], obj->texcoords[obj->texcoord_indices[tri_idx][v]], 2 * sizeof(double));
            }

            render_triangle(image, depth_buffer, triangle, uv, obj->texture.data, 
                          obj->texture.width, obj->texture.height);
        }
    }
    free(depth_buffer);
}

void create_projection_matrix(double projection[4][4]) {
    const double f = 1.0 / tan(FOV_Y * M_PI / 360.0);
    memset(projection, 0, 16 * sizeof(double));
    projection[0][0] = f / ASPECT_RATIO;
    projection[1][1] = f;
    projection[2][2] = (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE);
    projection[2][3] = (2 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE);
    projection[3][2] = -1;
}

void update_object_vertices(Object3D* obj, double view_matrix[4][4]) {
    double projection[4][4], viewport[4][4], modelview[4][4], mvp[4][4], final_transform[4][4];
    
    create_projection_matrix(projection);
    
    memset(viewport, 0, 16 * sizeof(double));
    viewport[0][0] = WIDTH / 2.0;
    viewport[1][1] = HEIGHT / 2.0;
    viewport[0][3] = WIDTH / 2.0;
    viewport[1][3] = HEIGHT / 2.0;
    viewport[2][2] = viewport[3][3] = 1.0;
    
    matrix_multiply(view_matrix, obj->model_matrix, modelview);
    matrix_multiply(projection, modelview, mvp);
    matrix_multiply(viewport, mvp, final_transform);
    
    for (int i = 0; i < obj->num_vertices; i++) {
        const double* vertex = obj->initial_vertices[i];
        double* transformed = obj->transformed_vertices[i];
        double result[4] = {0};
        
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                result[j] += final_transform[j][k] * vertex[k];
            }
            result[j] += final_transform[j][3];
        }
        
        if (fabs(result[3]) > 1e-6) {
            double inv_w = 1.0 / result[3];
            for (int j = 0; j < 3; j++) {
                transformed[j] = result[j] * inv_w;
            }
        } else {
            memset(transformed, 0, 3 * sizeof(double));
        }
    }
}

int main() {
    Object3D* objects[] = {
        create_object("drone.obj", "drone.bmp"),
        create_object("ground.obj", "ground.bmp")
    };
    
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, 3, -1, 0);
    
    double camera_pos[3] = {-2.0, 1.0, -2.0};
    double camera_target[3] = {0.0, 0.0, 0.0};
    double camera_up[3] = {0.0, 1.0, 0.0};
    
    double ground_transform[3] = {0.0, -0.5, 0.0};
    build_transform_matrix(objects[1]->model_matrix, ground_transform, 1.0, 0.0);

    for (int frame = 0; frame < FRAMES; frame++) {
        memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
        
        double drone_transform[3] = {0.0, 0.5, 0.0};
        build_transform_matrix(objects[0]->model_matrix, drone_transform, 0.5, 
                             frame * (2.0 * M_PI) / FRAMES);
        
        double view_matrix[4][4];
        calculate_view_matrix(camera_pos, camera_target, camera_up, view_matrix);
        
        camera_pos[0] += 0.05;
        camera_pos[2] += 0.05;
        camera_target[0] += 0.05;
        camera_target[2] += 0.05;
        
        for (int i = 0; i < 2; i++) {
            update_object_vertices(objects[i], view_matrix);
        }
        render_frame(frame_buffer, objects, 2);
        
        ge_add_frame(gif, frame_buffer, 6);
        printf("Rendered frame %d/%d\n", frame + 1, FRAMES);
    }
    
    ge_close_gif(gif);
    free(frame_buffer);
    for (int i = 0; i < 2; i++) {
        free_object(objects[i]);
    }
    
    return 0;
}