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
    // Create and initialize depth buffer
    double *depth_buffer = malloc(WIDTH * HEIGHT * sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = 1.0;
    }

    // For each mesh
    for (int m = 0; m < num_meshes; m++) {
        Mesh *mesh = meshes[m];
        
        // For each triangle
        for (int t = 0; t < mesh->counts[2]; t++) {
            int *tri = &mesh->triangles[t * 3];
            int *tex_tri = &mesh->texcoord_indices[t * 3];
            
            // Get triangle vertices
            double x1 = mesh->vertices[tri[0] * 3];
            double y1 = mesh->vertices[tri[0] * 3 + 1];
            double z1 = mesh->vertices[tri[0] * 3 + 2];
            
            double x2 = mesh->vertices[tri[1] * 3];
            double y2 = mesh->vertices[tri[1] * 3 + 1];
            double z2 = mesh->vertices[tri[1] * 3 + 2];
            
            double x3 = mesh->vertices[tri[2] * 3];
            double y3 = mesh->vertices[tri[2] * 3 + 1];
            double z3 = mesh->vertices[tri[2] * 3 + 2];

            // Basic backface culling
            if ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) < 0) continue;

            // Get texture coordinates and prepare for perspective correction
            double u1 = mesh->texcoords[tex_tri[0] * 2];
            double v1 = mesh->texcoords[tex_tri[0] * 2 + 1];
            double u2 = mesh->texcoords[tex_tri[1] * 2];
            double v2 = mesh->texcoords[tex_tri[1] * 2 + 1];
            double u3 = mesh->texcoords[tex_tri[2] * 2];
            double v3 = mesh->texcoords[tex_tri[2] * 2 + 1];

            // Prepare perspective-correct values
            // We multiply texture coordinates by 1/z for perspective correction
            double w1 = 1.0 / (z1 + 1e-8); // Add small epsilon to prevent division by zero
            double w2 = 1.0 / (z2 + 1e-8);
            double w3 = 1.0 / (z3 + 1e-8);

            double u1_p = u1 * w1;
            double v1_p = v1 * w1;
            double u2_p = u2 * w2;
            double v2_p = v2 * w2;
            double u3_p = u3 * w3;
            double v3_p = v3 * w3;

            // Calculate bounding box
            int minX = (int)fmax(0, fmin(fmin(x1, x2), x3));
            int maxX = (int)fmin(WIDTH - 1, fmax(fmax(x1, x2), x3));
            int minY = (int)fmax(0, fmin(fmin(y1, y2), y3));
            int maxY = (int)fmin(HEIGHT - 1, fmax(fmax(y1, y2), y3));

            // Rasterize triangle
            for (int y = minY; y <= maxY; y++) {
                for (int x = minX; x <= maxX; x++) {
                    // Calculate barycentric coordinates
                    double denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
                    if (fabs(denominator) < 1e-8) continue;

                    double w1_b = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator;
                    double w2_b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator;
                    double w3_b = 1.0 - w1_b - w2_b;

                    // Check if point is inside triangle
                    if (w1_b >= 0 && w2_b >= 0 && w3_b >= 0) {
                        // Interpolate Z
                        double z = w1_b * z1 + w2_b * z2 + w3_b * z3;
                        
                        // Depth test
                        if (z < depth_buffer[y * WIDTH + x]) {
                            depth_buffer[y * WIDTH + x] = z;

                            // Perspective-correct texture coordinate interpolation
                            double w = 1.0 / (w1_b * w1 + w2_b * w2 + w3_b * w3);
                            double u = (w1_b * u1_p + w2_b * u2_p + w3_b * u3_p) * w;
                            double v = (w1_b * v1_p + w2_b * v2_p + w3_b * v3_p) * w;

                            // Sample texture
                            int tx = (int)(u * (mesh->texture_dims[0] - 1));
                            int ty = (int)(v * (mesh->texture_dims[1] - 1));
                            
                            // Clamp texture coordinates
                            tx = fmax(0, fmin(tx, mesh->texture_dims[0] - 1));
                            ty = fmax(0, fmin(ty, mesh->texture_dims[1] - 1));

                            int texel_idx = (ty * mesh->texture_dims[0] + tx) * 3;
                            int pixel_idx = (y * WIDTH + x) * 3;

                            // Write pixel
                            image[pixel_idx] = mesh->texture_data[texel_idx];
                            image[pixel_idx + 1] = mesh->texture_data[texel_idx + 1];
                            image[pixel_idx + 2] = mesh->texture_data[texel_idx + 2];
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