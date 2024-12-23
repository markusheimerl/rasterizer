#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include "bmp.h"
#include "obj.h"

#define WIDTH 640
#define HEIGHT 480
#define ASPECT_RATIO ((double)WIDTH / (double)HEIGHT)
#define FOV_Y 60.0
#define NEAR_PLANE 0.1
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

void transform_mesh(Mesh* mesh, double translate[3], double scale, double rotation_matrix[9]) {
    // Build transformation matrix
    memset(mesh->transform, 0, 16 * sizeof(double));
    
    // Scale and rotation combined
    mesh->transform[0][0] = scale * rotation_matrix[0];
    mesh->transform[0][1] = scale * rotation_matrix[1];
    mesh->transform[0][2] = scale * rotation_matrix[2];
    
    mesh->transform[1][0] = scale * rotation_matrix[3];
    mesh->transform[1][1] = scale * rotation_matrix[4];
    mesh->transform[1][2] = scale * rotation_matrix[5];
    
    mesh->transform[2][0] = scale * rotation_matrix[6];
    mesh->transform[2][1] = scale * rotation_matrix[7];
    mesh->transform[2][2] = scale * rotation_matrix[8];
    
    // Translation
    mesh->transform[0][3] = translate[0];
    mesh->transform[1][3] = translate[1];
    mesh->transform[2][3] = translate[2];
    
    // Homogeneous coordinate
    mesh->transform[3][3] = 1.0;
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

void free_meshes(Mesh** meshes, int num_meshes) {
    for (int i = 0; i < num_meshes; i++) {
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
}

void vertex_shader(Mesh** meshes, int num_meshes, double camera_pos[3], double camera_target[3]) {
    // Calculate view matrix
    double forward[3] = {
        camera_target[0] - camera_pos[0],
        camera_target[1] - camera_pos[1],
        camera_target[2] - camera_pos[2]
    };
    VEC_NORM(forward);
    
    double right[3];
    double camera_up[3] = {0, 1, 0};
    VEC_CROSS(forward, camera_up, right);
    VEC_NORM(right);
    
    double up[3];
    VEC_CROSS(right, forward, up);
    
    double view_matrix[4][4] = {
        {right[0], right[1], right[2], -VEC_DOT(right, camera_pos)},
        {up[0], up[1], up[2], -VEC_DOT(up, camera_pos)},
        {-forward[0], -forward[1], -forward[2], VEC_DOT(forward, camera_pos)},
        {0, 0, 0, 1}
    };
    
    // Calculate projection matrix
    double fovy_rad = FOV_Y * M_PI / 180.0;
    double f = 1.0 / tan(fovy_rad / 2.0);
    double projection_matrix[4][4] = {
        {f / ASPECT_RATIO, 0, 0, 0},
        {0, f, 0, 0},
        {0, 0, (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE), 
            (2 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE)},
        {0, 0, -1, 0}
    };
    
    // Transform all vertices
    for (int i = 0; i < num_meshes; i++) {
        for (int j = 0; j < meshes[i]->counts[0]; j++) {
            // Apply model transform
            double pos[4] = {
                meshes[i]->initial_vertices[j * 3],
                meshes[i]->initial_vertices[j * 3 + 1],
                meshes[i]->initial_vertices[j * 3 + 2],
                1.0
            };
            double transformed[4] = {0};
            
            // Model transform
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    transformed[row] += meshes[i]->transform[row][col] * pos[col];
                }
            }
            
            // View transform
            double viewed[4] = {0};
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    viewed[row] += view_matrix[row][col] * transformed[col];
                }
            }
            
            // Projection transform
            double projected[4] = {0};
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    projected[row] += projection_matrix[row][col] * viewed[col];
                }
            }
            
            // Perspective divide
            if (projected[3] != 0) {
                projected[0] /= projected[3];
                projected[1] /= projected[3];
                projected[2] /= projected[3];
            }
            
            // Store NDC coordinates
            meshes[i]->vertices[j * 3] = projected[0];
            meshes[i]->vertices[j * 3 + 1] = projected[1];
            meshes[i]->vertices[j * 3 + 2] = projected[2];
        }
    }
}

void rasterize(uint8_t *image, Mesh **meshes, int num_meshes) {
    // Create and initialize z-buffer
    double *z_buffer = malloc(WIDTH * HEIGHT * sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        z_buffer[i] = DBL_MAX;
    }

    // For each mesh
    for (int m = 0; m < num_meshes; m++) {
        Mesh *mesh = meshes[m];
        
        // For each triangle
        for (int t = 0; t < mesh->counts[2]; t++) {
            // Get vertex indices
            int v0_idx = mesh->triangles[t * 3];
            int v1_idx = mesh->triangles[t * 3 + 1];
            int v2_idx = mesh->triangles[t * 3 + 2];

            // Check if any vertex is behind the near plane
            if (mesh->vertices[v0_idx * 3 + 2] > 1.0 ||
                mesh->vertices[v1_idx * 3 + 2] > 1.0 ||
                mesh->vertices[v2_idx * 3 + 2] > 1.0) {
                continue;  // Skip triangle if any vertex is behind near plane
            }

            // Get texture coordinate indices
            int t0_idx = mesh->texcoord_indices[t * 3];
            int t1_idx = mesh->texcoord_indices[t * 3 + 1];
            int t2_idx = mesh->texcoord_indices[t * 3 + 2];

            // Get vertex positions and apply viewport transform
            double x0 = (mesh->vertices[v0_idx * 3] + 1.0) * WIDTH * 0.5;
            double y0 = (mesh->vertices[v0_idx * 3 + 1] + 1.0) * HEIGHT * 0.5;
            double z0 = mesh->vertices[v0_idx * 3 + 2];
            
            double x1 = (mesh->vertices[v1_idx * 3] + 1.0) * WIDTH * 0.5;
            double y1 = (mesh->vertices[v1_idx * 3 + 1] + 1.0) * HEIGHT * 0.5;
            double z1 = mesh->vertices[v1_idx * 3 + 2];
            
            double x2 = (mesh->vertices[v2_idx * 3] + 1.0) * WIDTH * 0.5;
            double y2 = (mesh->vertices[v2_idx * 3 + 1] + 1.0) * HEIGHT * 0.5;
            double z2 = mesh->vertices[v2_idx * 3 + 2];

            // Skip triangles that are too close to the camera
            if (z0 < -0.1 || z1 < -0.1 || z2 < -0.1) {
                continue;
            }

            // Get texture coordinates
            double u0 = mesh->texcoords[t0_idx * 2];
            double v0 = mesh->texcoords[t0_idx * 2 + 1];
            
            double u1 = mesh->texcoords[t1_idx * 2];
            double v1 = mesh->texcoords[t1_idx * 2 + 1];
            
            double u2 = mesh->texcoords[t2_idx * 2];
            double v2 = mesh->texcoords[t2_idx * 2 + 1];

            // Calculate bounding box
            int min_x = (int)fmax(0, fmin(fmin(x0, x1), x2));
            int max_x = (int)fmin(WIDTH - 1, fmax(fmax(x0, x1), x2));
            int min_y = (int)fmax(0, fmin(fmin(y0, y1), y2));
            int max_y = (int)fmin(HEIGHT - 1, fmax(fmax(y0, y1), y2));

            // Triangle area
            double area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            if (fabs(area) < 1e-8) continue;  // Skip tiny triangles

            for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                    // Barycentric coordinates
                    double w0 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / area;
                    double w1 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) / area;
                    double w2 = 1.0 - w0 - w1;

                    if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                        double z = w0 * z0 + w1 * z1 + w2 * z2;
                        int pixel_idx = y * WIDTH + x;
                        
                        if (z < z_buffer[pixel_idx]) {
                            z_buffer[pixel_idx] = z;

                            double w0_perspective = w0 / z0;
                            double w1_perspective = w1 / z1;
                            double w2_perspective = w2 / z2;
                            double w_sum = w0_perspective + w1_perspective + w2_perspective;

                            double u = (w0_perspective * u0 + w1_perspective * u1 + w2_perspective * u2) / w_sum;
                            double v = (w0_perspective * v0 + w1_perspective * v1 + w2_perspective * v2) / w_sum;

                            int tx = (int)(u * (mesh->texture_dims[0] - 1));
                            int ty = (int)(v * (mesh->texture_dims[1] - 1));
                            
                            tx = fmax(0, fmin(tx, mesh->texture_dims[0] - 1));
                            ty = fmax(0, fmin(ty, mesh->texture_dims[1] - 1));

                            int texel_idx = (ty * mesh->texture_dims[0] + tx) * 3;
                            int pixel_offset = (y * WIDTH + x) * 3;

                            image[pixel_offset] = mesh->texture_data[texel_idx];
                            image[pixel_offset + 1] = mesh->texture_data[texel_idx + 1];
                            image[pixel_offset + 2] = mesh->texture_data[texel_idx + 2];
                        }
                    }
                }
            }
        }
    }

    free(z_buffer);
}