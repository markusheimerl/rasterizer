#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "gif.h"
#include "bmp.h"
#include "obj.h"
#include <stdbool.h>

#define WIDTH 640
#define HEIGHT 480
#define FRAMES 60
#define ASPECT_RATIO ((double)WIDTH / (double)HEIGHT)
#define FOV_Y 60.0
#define NEAR_PLANE 3.0
#define FAR_PLANE 100.0

#define VEC_DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VEC_CROSS(a,b,r) { (r)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; (r)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; (r)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; }
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); if(l>0) { (v)[0]/=l; (v)[1]/=l; (v)[2]/=l; } }

typedef struct {
    double* vertices;
    double* initial_vertices;
    double* texcoords;
    int* triangles;
    int* texcoord_indices;
    unsigned char* texture_data;
    double transform[4][4];
    int counts[3];
    int texture_dims[2];
} Mesh;

void transform_mesh(Mesh* mesh, double translate[3], double scale, double rotate_y) {
    double c = cos(rotate_y), s = sin(rotate_y);
    memset(mesh->transform, 0, 16 * sizeof(double));

    mesh->transform[0][0] = c * scale;
    mesh->transform[0][2] = s * scale;
    mesh->transform[1][1] = scale;
    mesh->transform[2][0] = -s * scale;
    mesh->transform[2][2] = c * scale;
    mesh->transform[3][3] = 1.0;

    mesh->transform[0][3] = translate[0];
    mesh->transform[1][3] = translate[1];
    mesh->transform[2][3] = translate[2];
}

Mesh* create_mesh(const char* obj_file, const char* texture_file) {
    Mesh* mesh = calloc(1, sizeof(Mesh));
    if (!mesh) return NULL;

    const int MAX_VERTICES = 100000;
    const int MAX_TRIANGLES = 200000;

    // Allocate memory for mesh data
    mesh->vertices = malloc(MAX_VERTICES * 3 * sizeof(double));
    mesh->initial_vertices = malloc(MAX_VERTICES * 3 * sizeof(double));
    mesh->texcoords = malloc(MAX_VERTICES * 2 * sizeof(double));
    mesh->triangles = malloc(MAX_TRIANGLES * 3 * sizeof(int));
    mesh->texcoord_indices = malloc(MAX_TRIANGLES * 3 * sizeof(int));

    // Load mesh data
    load_obj(obj_file, 
            (double(*)[3])mesh->vertices,
            (double(*)[3])mesh->initial_vertices,
            (double(*)[2])mesh->texcoords,
            (int(*)[3])mesh->triangles,
            (int(*)[3])mesh->texcoord_indices,
            &mesh->counts[0],  // vertex_count
            &mesh->counts[1],  // texcoord_count
            &mesh->counts[2]); // triangle_count

    // Load texture
    int channels;
    mesh->texture_data = load_bmp(texture_file,
                                 &mesh->texture_dims[0],
                                 &mesh->texture_dims[1],
                                 &channels);

    // Initialize transform matrix to identity
    memset(mesh->transform, 0, 16 * sizeof(double));
    mesh->transform[0][0] = 1.0;
    mesh->transform[1][1] = 1.0;
    mesh->transform[2][2] = 1.0;
    mesh->transform[3][3] = 1.0;

    return mesh;
}

void update_vertices(Mesh* mesh, double camera_pos[3], double camera_target[3], double camera_up[3]) {
    // Allocate temporary storage for transformed vertices
    double* transformed_vertices = malloc(mesh->counts[0] * 3 * sizeof(double));
    
    // Create view matrix
    double z[3] = {
        camera_target[0] - camera_pos[0],
        camera_target[1] - camera_pos[1],
        camera_target[2] - camera_pos[2]
    };
    double x[3], y[3];
    
    VEC_NORM(z);
    VEC_CROSS(camera_up, z, x);
    VEC_NORM(x);
    VEC_CROSS(z, x, y);

    // Build view matrix directly
    double view_matrix[4][4] = {
        {x[0], x[1], x[2], -(x[0]*camera_pos[0] + x[1]*camera_pos[1] + x[2]*camera_pos[2])},
        {y[0], y[1], y[2], -(y[0]*camera_pos[0] + y[1]*camera_pos[1] + y[2]*camera_pos[2])},
        {z[0], z[1], z[2], -(z[0]*camera_pos[0] + z[1]*camera_pos[1] + z[2]*camera_pos[2])},
        {0, 0, 0, 1}
    };

    // Create projection matrix
    const double fov_rad = FOV_Y * M_PI / 360.0;
    const double f = 1.0 / tan(fov_rad);
    const double near_far_factor = (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE);
    const double near_far_term = (2 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE);
    
    double proj_matrix[4][4] = {
        {f/ASPECT_RATIO, 0, 0, 0},
        {0, f, 0, 0},
        {0, 0, near_far_factor, near_far_term},
        {0, 0, -1, 0}
    };

    double (*initial_vertices)[3] = (double(*)[3])mesh->initial_vertices;
    double (*transformed)[3] = (double(*)[3])transformed_vertices;

    for (int i = 0; i < mesh->counts[0]; i++) {
        double vertex[4] = {
            initial_vertices[i][0],
            initial_vertices[i][1],
            initial_vertices[i][2],
            1.0
        };
        
        double world_vertex[4] = {0};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                world_vertex[row] += mesh->transform[row][col] * vertex[col];
            }
        }
        
        double view_vertex[4] = {0};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                view_vertex[row] += view_matrix[row][col] * world_vertex[col];
            }
        }
        
        double clip_vertex[4] = {0};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                clip_vertex[row] += proj_matrix[row][col] * view_vertex[col];
            }
        }
        
        if (clip_vertex[3] != 0) {
            const double inv_w = 1.0 / clip_vertex[3];
            transformed[i][0] = ((clip_vertex[0] * inv_w + 1.0) * 0.5) * WIDTH;
            transformed[i][1] = ((clip_vertex[1] * inv_w + 1.0) * 0.5) * HEIGHT;
            transformed[i][2] = clip_vertex[2] * inv_w;
        } else {
            transformed[i][0] = 0;
            transformed[i][1] = 0;
            transformed[i][2] = 0;
        }
    }
    
    // Copy transformed vertices back to mesh->vertices
    memcpy(mesh->vertices, transformed_vertices, mesh->counts[0] * 3 * sizeof(double));
    free(transformed_vertices);
}

void render_scene(uint8_t *image, Mesh **meshes, int num_meshes) {
    double *depth_buffer = calloc(WIDTH * HEIGHT, sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = -DBL_MAX;
    }

    for (int mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
        Mesh *mesh = meshes[mesh_idx];
        double (*vertices)[3] = (double(*)[3])mesh->vertices;
        double (*texcoords)[2] = (double(*)[2])mesh->texcoords;
        int (*triangles)[3] = (int(*)[3])mesh->triangles;
        int (*texcoord_indices)[3] = (int(*)[3])mesh->texcoord_indices;

        for (int tri_idx = 0; tri_idx < mesh->counts[2]; tri_idx++) {
            double vertex_pos[3][4];  // x, y, z, w(1/z)
            double vertex_uv[3][2];   // u, v
            
            // Check if any vertex is behind near plane
            bool skip_triangle = false;
            for (int v = 0; v < 3; v++) {
                const int vertex_idx = triangles[tri_idx][v];
                if (vertices[vertex_idx][2] < -NEAR_PLANE) {  // Check against near plane
                    skip_triangle = true;
                    printf("Triangle %d is behind near plane\n", tri_idx);
                    break;
                }
                vertex_pos[v][0] = vertices[vertex_idx][0];
                vertex_pos[v][1] = vertices[vertex_idx][1];
                vertex_pos[v][2] = vertices[vertex_idx][2];
                vertex_pos[v][3] = 1.0 / vertex_pos[v][2];
                
                const int uv_idx = texcoord_indices[tri_idx][v];
                vertex_uv[v][0] = texcoords[uv_idx][0];
                vertex_uv[v][1] = texcoords[uv_idx][1];
            }

            if (skip_triangle) continue;

            int min_x = fmax(0, floor(fmin(fmin(vertex_pos[0][0], vertex_pos[1][0]), vertex_pos[2][0])));
            int min_y = fmax(0, floor(fmin(fmin(vertex_pos[0][1], vertex_pos[1][1]), vertex_pos[2][1])));
            int max_x = fmin(WIDTH - 1, ceil(fmax(fmax(vertex_pos[0][0], vertex_pos[1][0]), vertex_pos[2][0])));
            int max_y = fmin(HEIGHT - 1, ceil(fmax(fmax(vertex_pos[0][1], vertex_pos[1][1]), vertex_pos[2][1])));

            double bary_denom = ((vertex_pos[1][1] - vertex_pos[2][1]) * (vertex_pos[0][0] - vertex_pos[2][0]) +
                                (vertex_pos[2][0] - vertex_pos[1][0]) * (vertex_pos[0][1] - vertex_pos[2][1]));
            
            if (fabs(bary_denom) < 1e-6) continue;

            for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                    double lambda0 = ((vertex_pos[1][1] - vertex_pos[2][1]) * (x - vertex_pos[2][0]) +
                                    (vertex_pos[2][0] - vertex_pos[1][0]) * (y - vertex_pos[2][1])) / bary_denom;
                    double lambda1 = ((vertex_pos[2][1] - vertex_pos[0][1]) * (x - vertex_pos[2][0]) +
                                    (vertex_pos[0][0] - vertex_pos[2][0]) * (y - vertex_pos[2][1])) / bary_denom;
                    double lambda2 = 1.0 - lambda0 - lambda1;

                    if (lambda0 >= 0 && lambda0 <= 1 && 
                        lambda1 >= 0 && lambda1 <= 1 && 
                        lambda2 >= 0 && lambda2 <= 1) {
                        
                        int pixel_idx = y * WIDTH + x;
                        double z = lambda0 * vertex_pos[0][2] + 
                                 lambda1 * vertex_pos[1][2] + 
                                 lambda2 * vertex_pos[2][2];

                        if (z > depth_buffer[pixel_idx]) {
                            depth_buffer[pixel_idx] = z;

                            double u = (lambda0 * vertex_uv[0][0] * vertex_pos[0][3] +
                                      lambda1 * vertex_uv[1][0] * vertex_pos[1][3] +
                                      lambda2 * vertex_uv[2][0] * vertex_pos[2][3]) * z;
                            double v = (lambda0 * vertex_uv[0][1] * vertex_pos[0][3] +
                                      lambda1 * vertex_uv[1][1] * vertex_pos[1][3] +
                                      lambda2 * vertex_uv[2][1] * vertex_pos[2][3]) * z;

                            u = u - floor(u);
                            v = 1.0 - (v - floor(v));

                            int tx = (int)(u * mesh->texture_dims[0]);
                            int ty = (int)(v * mesh->texture_dims[1]);
                            
                            tx = fmin(fmax(tx, 0), mesh->texture_dims[0] - 1);
                            ty = fmin(fmax(ty, 0), mesh->texture_dims[1] - 1);

                            int tex_idx = (ty * mesh->texture_dims[0] + tx) * 3;
                            image[pixel_idx * 3 + 0] = mesh->texture_data[tex_idx + 0];
                            image[pixel_idx * 3 + 1] = mesh->texture_data[tex_idx + 1];
                            image[pixel_idx * 3 + 2] = mesh->texture_data[tex_idx + 2];
                        }
                    }
                }
            }
        }
    }
    
    free(depth_buffer);
}

int main() {
    Mesh* meshes[] = {
        create_mesh("drone.obj", "drone.bmp"),
        create_mesh("ground.obj", "ground.bmp")
    };

    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, 4, -1, 0);

    double camera_pos[3] = {-2.0, 1.0, -2.0};
    double camera_target[3] = {0.0, 0.0, 0.0};
    double camera_up[3] = {0.0, 1.0, 0.0};

    for (int frame = 0; frame < FRAMES; frame++) {
        memset(frame_buffer, 0, WIDTH * HEIGHT * 3);

        double drone_pos[3] = {0.0, 0.5, 0.0};
        double ground_pos[3] = {0.0, -0.5, 0.0};
        transform_mesh(meshes[0], drone_pos, 0.5, frame * (2.0 * M_PI) / FRAMES);
        transform_mesh(meshes[1], ground_pos, 1.0, 0.0);

        for (int i = 0; i < 2; i++) {
            update_vertices(meshes[i], camera_pos, camera_target, camera_up);
        }

        render_scene(frame_buffer, meshes, 2);

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
        if (meshes[i]) {
            free(meshes[i]->vertices);
            free(meshes[i]->initial_vertices);
            free(meshes[i]->texcoords);
            free(meshes[i]->triangles);
            free(meshes[i]->texcoord_indices);
            free(meshes[i]->texture_data);
            free(meshes[i]);
        }
    }

    return 0;
}