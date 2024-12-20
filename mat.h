#ifndef MAT_H
#define MAT_H

// Vector operations as macros for performance-critical operations
#define VEC_SUB(a,b,r) { (r)[0]=(a)[0]-(b)[0]; (r)[1]=(a)[1]-(b)[1]; (r)[2]=(a)[2]-(b)[2]; }
#define VEC_ADD(a,b,r) { (r)[0]=(a)[0]+(b)[0]; (r)[1]=(a)[1]+(b)[1]; (r)[2]=(a)[2]+(b)[2]; }
#define VEC_SCALE(v,s,r) { (r)[0]=(v)[0]*(s); (r)[1]=(v)[1]*(s); (r)[2]=(v)[2]*(s); }
#define VEC_DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VEC_CROSS(a,b,r) { (r)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; \
                          (r)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; \
                          (r)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; }
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); \
                     if(l>0) { (v)[0]/=l; (v)[1]/=l; (v)[2]/=l; } }

// Basic matrix operations
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

// Transformation matrices
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

// Camera view matrix calculation
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

#endif /* MAT_H */