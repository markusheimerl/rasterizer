#ifndef OBJECT3D_H
#define OBJECT3D_H

typedef struct {
    double (*vertices)[3];
    double (*initial_vertices)[3];
    double (*transformed_vertices)[3];
    double (*texcoords)[2];
    int (*triangles)[3];
    int (*texcoord_indices)[3];
    int num_vertices;
    int num_texcoords;
    int num_triangles;
    unsigned char *texture_data;
    int texture_width;
    int texture_height;
    int texture_channels;
    double model_matrix[4][4];
} Object3D;


Object3D* create_object(const char* obj_file, const char* texture_file) {
    Object3D* obj = malloc(sizeof(Object3D));
    
    obj->vertices = malloc(100000 * sizeof(*obj->vertices));
    obj->initial_vertices = malloc(100000 * sizeof(*obj->initial_vertices));
    obj->transformed_vertices = malloc(100000 * sizeof(*obj->transformed_vertices));
    obj->texcoords = malloc(100000 * sizeof(*obj->texcoords));
    obj->triangles = malloc(200000 * sizeof(*obj->triangles));
    obj->texcoord_indices = malloc(200000 * sizeof(*obj->texcoord_indices));
    obj->num_vertices = 0;
    obj->num_texcoords = 0;
    obj->num_triangles = 0;
    
    parse_obj_file(obj_file, obj->vertices, obj->initial_vertices, obj->texcoords, 
                  obj->triangles, obj->texcoord_indices, &obj->num_vertices, 
                  &obj->num_texcoords, &obj->num_triangles);

    obj->texture_data = load_bmp(texture_file, &obj->texture_width, 
                                &obj->texture_height, &obj->texture_channels);
    
    matrix_identity(obj->model_matrix);
    
    return obj;
}

void free_object(Object3D* obj) {
    free(obj->vertices);
    free(obj->initial_vertices);
    free(obj->transformed_vertices);
    free(obj->texcoords);
    free(obj->triangles);
    free(obj->texcoord_indices);
    free(obj->texture_data);
    free(obj);
}

#endif /* OBJECT3D_H */