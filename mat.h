#ifndef MAT_H
#define MAT_H

#define VEC_SUB(a,b,r) { (r)[0]=(a)[0]-(b)[0]; (r)[1]=(a)[1]-(b)[1]; (r)[2]=(a)[2]-(b)[2]; }
#define VEC_DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VEC_CROSS(a,b,r) { (r)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; (r)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; (r)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; }
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); if(l>0){ (v)[0]/=l; (v)[1]/=l; (v)[2]/=l; } }
#define VEC_SCALE(v,s,r) { (r)[0]=(v)[0]*(s); (r)[1]=(v)[1]*(s); (r)[2]=(v)[2]*(s); }
#define VEC_ADD(a,b,r) { (r)[0]=(a)[0]+(b)[0]; (r)[1]=(a)[1]+(b)[1]; (r)[2]=(a)[2]+(b)[2]; }

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
    double t[4][4] = {
        {1, 0, 0, x},
        {0, 1, 0, y},
        {0, 0, 1, z},
        {0, 0, 0, 1}
    };
    matrix_multiply(m, t, m);
}

void matrix_scale(double m[4][4], double s) {
    double scale[4][4] = {
        {s, 0, 0, 0},
        {0, s, 0, 0},
        {0, 0, s, 0},
        {0, 0, 0, 1}
    };
    matrix_multiply(m, scale, m);
}

void matrix_rotate_y(double m[4][4], double angle) {
    double c = cos(angle);
    double s = sin(angle);
    double r[4][4] = {
        {c,  0, s, 0},
        {0,  1, 0, 0},
        {-s, 0, c, 0},
        {0,  0, 0, 1}
    };
    matrix_multiply(m, r, m);
}

void normalize_vector(double v[3]) {
    double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-6) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

void cross_product(const double a[3], const double b[3], double result[3]) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

double dot_product(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void calculate_view_matrix(const double camera_pos[3], 
                         const double camera_target[3],
                         const double camera_up[3],
                         double view_matrix[4][4]) {
    // Calculate camera axes
    double z_axis[3] = {
        camera_target[0] - camera_pos[0],
        camera_target[1] - camera_pos[1],
        camera_target[2] - camera_pos[2]
    };
    normalize_vector(z_axis);
    
    double x_axis[3];
    cross_product(camera_up, z_axis, x_axis);
    normalize_vector(x_axis);
    
    double y_axis[3];
    cross_product(z_axis, x_axis, y_axis);
    
    // Build view matrix
    view_matrix[0][0] = x_axis[0];
    view_matrix[0][1] = x_axis[1];
    view_matrix[0][2] = x_axis[2];
    view_matrix[0][3] = -dot_product(x_axis, camera_pos);
    
    view_matrix[1][0] = y_axis[0];
    view_matrix[1][1] = y_axis[1];
    view_matrix[1][2] = y_axis[2];
    view_matrix[1][3] = -dot_product(y_axis, camera_pos);
    
    view_matrix[2][0] = z_axis[0];
    view_matrix[2][1] = z_axis[1];
    view_matrix[2][2] = z_axis[2];
    view_matrix[2][3] = -dot_product(z_axis, camera_pos);
    
    view_matrix[3][0] = 0.0;
    view_matrix[3][1] = 0.0;
    view_matrix[3][2] = 0.0;
    view_matrix[3][3] = 1.0;
}

#endif /* MAT_H */