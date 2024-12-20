#ifndef OBJECT3D_H
#define OBJECT3D_H

typedef struct {
    double (*vertices)[3];            // Current vertex positions
    double (*initial_vertices)[3];    // Original vertex positions
    double (*transformed_vertices)[3]; // After transformation
    double (*texcoords)[2];          // Texture coordinates
    
    int (*triangles)[3];             // Triangle indices
    int (*texcoord_indices)[3];      // Texture coordinate indices
    
    int num_vertices;
    int num_texcoords;
    int num_triangles;
    
    struct {                         // Texture information
        unsigned char *data;
        int width;
        int height;
        int channels;
    } texture;
    
    double model_matrix[4][4];
} Object3D;

static inline Object3D* create_object(const char* obj_file, const char* texture_file) {
    Object3D* obj = calloc(1, sizeof(Object3D));  // Use calloc to zero-initialize
    if (!obj) return NULL;
    
    // Define array sizes as constants
    const int MAX_VERTICES = 100000;
    const int MAX_TRIANGLES = 200000;
    
    // Allocate all vertex-related arrays
    obj->vertices = malloc(MAX_VERTICES * sizeof(*obj->vertices));
    obj->initial_vertices = malloc(MAX_VERTICES * sizeof(*obj->initial_vertices));
    obj->transformed_vertices = malloc(MAX_VERTICES * sizeof(*obj->transformed_vertices));
    obj->texcoords = malloc(MAX_VERTICES * sizeof(*obj->texcoords));
    
    // Allocate all triangle-related arrays
    obj->triangles = malloc(MAX_TRIANGLES * sizeof(*obj->triangles));
    obj->texcoord_indices = malloc(MAX_TRIANGLES * sizeof(*obj->texcoord_indices));
    
    // Load model and texture data
    parse_obj_file(obj_file, obj->vertices, obj->initial_vertices, obj->texcoords,
                  obj->triangles, obj->texcoord_indices, &obj->num_vertices,
                  &obj->num_texcoords, &obj->num_triangles);
    
    obj->texture.data = load_bmp(texture_file, &obj->texture.width,
                                &obj->texture.height, &obj->texture.channels);
    
    matrix_identity(obj->model_matrix);
    return obj;
}

static inline void free_object(Object3D* obj) {
    if (!obj) return;
    
    free(obj->vertices);
    free(obj->initial_vertices);
    free(obj->transformed_vertices);
    free(obj->texcoords);
    free(obj->triangles);
    free(obj->texcoord_indices);
    free(obj->texture.data);
    free(obj);
}

#endif