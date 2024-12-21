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
    // Create rotation matrix around Y axis
    double cos_y = cos(rotate_y);
    double sin_y = sin(rotate_y);
    
    // Build transformation matrix
    memset(mesh->transform, 0, 16 * sizeof(double));
    
    // Scale
    mesh->transform[0][0] = scale * cos_y;
    mesh->transform[0][2] = scale * sin_y;
    mesh->transform[1][1] = scale;
    mesh->transform[2][0] = -scale * sin_y;
    mesh->transform[2][2] = scale * cos_y;
    
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

void update_vertices(Mesh* mesh, double camera_pos[3], double camera_target[3], double camera_up[3]) {
    // Calculate view matrix
    double forward[3] = {
        camera_target[0] - camera_pos[0],
        camera_target[1] - camera_pos[1],
        camera_target[2] - camera_pos[2]
    };
    VEC_NORM(forward);
    
    double right[3];
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
    for (int i = 0; i < mesh->counts[0]; i++) {
        // Apply model transform
        double pos[4] = {
            mesh->initial_vertices[i * 3],
            mesh->initial_vertices[i * 3 + 1],
            mesh->initial_vertices[i * 3 + 2],
            1.0
        };
        double transformed[4] = {0};
        
        // Model transform
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                transformed[row] += mesh->transform[row][col] * pos[col];
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
        
        // Viewport transform
        mesh->vertices[i * 3] = (projected[0] + 1.0) * WIDTH * 0.5;
        mesh->vertices[i * 3 + 1] = (projected[1] + 1.0) * HEIGHT * 0.5;
        mesh->vertices[i * 3 + 2] = projected[2];
    }
}

void render_scene(uint8_t *image, Mesh **meshes, int num_meshes) {
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

            // Get texture coordinate indices
            int t0_idx = mesh->texcoord_indices[t * 3];
            int t1_idx = mesh->texcoord_indices[t * 3 + 1];
            int t2_idx = mesh->texcoord_indices[t * 3 + 2];

            // Get vertex positions
            double x0 = mesh->vertices[v0_idx * 3];
            double y0 = mesh->vertices[v0_idx * 3 + 1];
            double z0 = mesh->vertices[v0_idx * 3 + 2];
            
            double x1 = mesh->vertices[v1_idx * 3];
            double y1 = mesh->vertices[v1_idx * 3 + 1];
            double z1 = mesh->vertices[v1_idx * 3 + 2];
            
            double x2 = mesh->vertices[v2_idx * 3];
            double y2 = mesh->vertices[v2_idx * 3 + 1];
            double z2 = mesh->vertices[v2_idx * 3 + 2];

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
            if (fabs(area) < 1e-8) continue;  // Skip degenerate triangles

            // Rasterize triangle
            for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                    // Barycentric coordinates
                    double w0 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / area;
                    double w1 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) / area;
                    double w2 = 1.0 - w0 - w1;

                    // Check if point is inside triangle
                    if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                        // Interpolate z
                        double z = w0 * z0 + w1 * z1 + w2 * z2;
                        
                        // Z-buffer test
                        int pixel_idx = y * WIDTH + x;
                        if (z < z_buffer[pixel_idx]) {
                            z_buffer[pixel_idx] = z;

                            // Interpolate texture coordinates
                            double u = w0 * u0 + w1 * u1 + w2 * u2;
                            double v = w0 * v0 + w1 * v1 + w2 * v2;

                            // Sample texture
                            int tx = (int)(u * (mesh->texture_dims[0] - 1));
                            int ty = (int)(v * (mesh->texture_dims[1] - 1));
                            
                            // Clamp texture coordinates
                            tx = fmax(0, fmin(tx, mesh->texture_dims[0] - 1));
                            ty = fmax(0, fmin(ty, mesh->texture_dims[1] - 1));

                            int texel_idx = (ty * mesh->texture_dims[0] + tx) * 3;
                            int pixel_offset = (y * WIDTH + x) * 3;

                            // Write pixel color
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

int main() {
    Mesh* meshes[] = {
        create_mesh("drone.obj", "drone.bmp"),
        create_mesh("ground.obj", "ground.bmp")
    };

    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    uint8_t *flipped_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
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

        // Flip the image vertically
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int src_idx = (y * WIDTH + x) * 3;
                int dst_idx = ((HEIGHT - 1 - y) * WIDTH + x) * 3;
                flipped_buffer[dst_idx] = frame_buffer[src_idx];
                flipped_buffer[dst_idx + 1] = frame_buffer[src_idx + 1];
                flipped_buffer[dst_idx + 2] = frame_buffer[src_idx + 2];
            }
        }

        camera_pos[0] += 0.05;
        camera_pos[2] += 0.05;
        camera_target[0] += 0.05;
        camera_target[2] += 0.05;

        ge_add_frame(gif, flipped_buffer, 6);
        printf("Rendered frame %d/%d\n", frame + 1, FRAMES);
    }

    ge_close_gif(gif);
    free(frame_buffer);
    free(flipped_buffer);
    
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