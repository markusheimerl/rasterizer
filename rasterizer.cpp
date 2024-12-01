#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <cmath>
#include <thread>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float f) const { return Vec3(x * f, y * f, z * f); }
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    Vec3 normalize() const { float l = sqrt(x * x + y * y + z * z); return Vec3(x / l, y / l, z / l); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
};

struct Vec2 {
    float u, v;
    Vec2(float u = 0, float v = 0) : u(u), v(v) {}
    Vec2 operator*(float f) const { return Vec2(u * f, v * f); }
    Vec2 operator+(const Vec2& other) const { return Vec2(u + other.u, v + other.v); }
};

struct Mat3 {
    float m[9];
    Mat3() { for (int i = 0; i < 9; i++) m[i] = (i % 4 == 0) ? 1.0f : 0.0f; }
    Mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22) {
        m[0] = m00; m[1] = m01; m[2] = m02; m[3] = m10; m[4] = m11; m[5] = m12; m[6] = m20; m[7] = m21; m[8] = m22;
    }
    static Mat3 identity() { return Mat3(); }
    static Mat3 diag(float a, float b, float c) { Mat3 result; result.m[0] = a; result.m[4] = b; result.m[8] = c; return result; }
    Mat3 operator+(const Mat3& other) const { Mat3 result; for (int i = 0; i < 9; i++) result.m[i] = m[i] + other.m[i]; return result; }
    Mat3 operator*(const Mat3& other) const {
        Mat3 result;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            result.m[i*3 + j] = 0;
            for (int k = 0; k < 3; k++) result.m[i*3 + j] += m[i*3 + k] * other.m[k*3 + j];
        }
        return result;
    }
    Vec3 operator*(const Vec3& v) const { 
        return Vec3(m[0]*v.x + m[1]*v.y + m[2]*v.z, 
                   m[3]*v.x + m[4]*v.y + m[5]*v.z, 
                   m[6]*v.x + m[7]*v.y + m[8]*v.z); 
    }
    Mat3& operator+=(const Mat3& other) { for (int i = 0; i < 9; i++) m[i] += other.m[i]; return *this; }
    Mat3 operator*(float scalar) const {
        Mat3 result;
        for (int i = 0; i < 9; i++) result.m[i] = m[i] * scalar;
        return result;
    }
    Vec3 multiplyVector(const Vec3& v) const {
        return Vec3(m[0] * v.x + m[1] * v.y + m[2] * v.z,
                   m[3] * v.x + m[4] * v.y + m[5] * v.z,
                   m[6] * v.x + m[7] * v.y + m[8] * v.z);
    }
};

struct Mat4 {
    float m[16];
    Mat4() { for (int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.0f : 0.0f; }
    Mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0] = m00; m[1] = m01; m[2] = m02; m[3] = m03;
        m[4] = m10; m[5] = m11; m[6] = m12; m[7] = m13;
        m[8] = m20; m[9] = m21; m[10] = m22; m[11] = m23;
        m[12] = m30; m[13] = m31; m[14] = m32; m[15] = m33;
    }
    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
            result.m[i*4 + j] = 0;
            for (int k = 0; k < 4; k++) 
                result.m[i*4 + j] += m[i*4 + k] * other.m[k*4 + j];
        }
        return result;
    }
    Vec3 multiplyPoint(const Vec3& v) const {
        float w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15];
        return Vec3((m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3]) / w,
                   (m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7]) / w,
                   (m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]) / w);
    }
    Vec3 multiplyVector(const Vec3& v) const {
        return Vec3(m[0] * v.x + m[1] * v.y + m[2] * v.z,
                   m[4] * v.x + m[5] * v.y + m[6] * v.z,
                   m[8] * v.x + m[9] * v.y + m[10] * v.z);
    }
    static Mat4 identity() { return Mat4(); }
    static Mat4 rotationY(float angle) {
        float c = cos(angle), s = sin(angle);
        return Mat4(c, 0, -s, 0,
                   0, 1, 0, 0,
                   s, 0, c, 0,
                   0, 0, 0, 1);
    }
    void setTranslation(const Vec3& t) { m[3] = t.x; m[7] = t.y; m[11] = t.z; }
    void setRotation(const Mat3& r) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m[i*4 + j] = r.m[i*3 + j];
    }
    static Mat4 scale(float x, float y, float z) {
        return Mat4(x, 0, 0, 0,
                   0, y, 0, 0,
                   0, 0, z, 0,
                   0, 0, 0, 1);
    }
};

struct Triangle {
    Vec3 v[3];
    Vec2 uv[3];
    Vec3 n[3];
};

Mat4 create_projection_matrix(float fov, float aspect, float near, float far) {
    float tanHalfFov = tan(fov / 2.0f);
    float f = 1.0f / tanHalfFov;
    float nf = 1.0f / (near - far);
    return Mat4(f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (far + near) * nf, 2.0f * far * near * nf,
                0, 0, -1, 0);
}

Mat4 create_model_matrix_random() {
    static std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> dis_pos(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dis_scale(0.8f, 1.3f);
    std::uniform_real_distribution<float> dis_rot(0.0f, 2.0f * M_PI);
    
    float tx = dis_pos(gen), ty = dis_pos(gen), tz = dis_pos(gen) - 5.0f;
    float scale = dis_scale(gen), rotation = dis_rot(gen);
    float cos_r = cos(rotation), sin_r = sin(rotation);
    
    return Mat4(cos_r * scale, 0, -sin_r * scale, tx,
                0, scale, 0, ty,
                sin_r * scale, 0, cos_r * scale, tz,
                0, 0, 0, 1);
}

void load_obj(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filename << std::endl;
        return;
    }

    std::vector<Vec3> vertices, normals;
    std::vector<Vec2> texcoords;
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            Vec3 v;
            iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        }
        else if (type == "vt") {
            Vec2 vt;
            iss >> vt.u >> vt.v;
            texcoords.push_back(vt);
        }
        else if (type == "vn") {
            Vec3 vn;
            iss >> vn.x >> vn.y >> vn.z;
            normals.push_back(vn);
        }
        else if (type == "f") {
            Triangle tri;
            for (int i = 0; i < 3; i++) {
                int v, vt, vn;
                char slash;
                iss >> v >> slash >> vt >> slash >> vn;
                tri.v[i] = vertices[v - 1];
                tri.uv[i] = texcoords[vt - 1];
                tri.n[i] = normals[vn - 1];
            }
            triangles.push_back(tri);
        }
    }
}

Mat3 skew(const Vec3& v) {
    return Mat3(0, -v.z, v.y,
                v.z, 0, -v.x,
                -v.y, v.x, 0);
}

void render_triangle(const Triangle& tri, unsigned char* output, float* zbuffer,
                    const unsigned char* texture, int tex_width, int tex_height,
                    int width, int height) {
    Vec3 screen_coords[3];
    for (int i = 0; i < 3; i++) {
        screen_coords[i] = Vec3((tri.v[i].x + 1.0f) * width / 2.0f,
                               (1.0f - tri.v[i].y) * height / 2.0f,
                               tri.v[i].z);
    }

    // Calculate bounding box
    int minX = std::max(0, static_cast<int>(std::min({screen_coords[0].x, screen_coords[1].x, screen_coords[2].x})));
    int maxX = std::min(width - 1, static_cast<int>(std::max({screen_coords[0].x, screen_coords[1].x, screen_coords[2].x})));
    int minY = std::max(0, static_cast<int>(std::min({screen_coords[0].y, screen_coords[1].y, screen_coords[2].y})));
    int maxY = std::min(height - 1, static_cast<int>(std::max({screen_coords[0].y, screen_coords[1].y, screen_coords[2].y})));

    Vec3 light_dir = Vec3(1, 1, 1).normalize();

    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            Vec3 edge1 = screen_coords[1] - screen_coords[0];
            Vec3 edge2 = screen_coords[2] - screen_coords[0];
            Vec3 h = Vec3(x, y, 0) - screen_coords[0];

            float det = edge1.x * edge2.y - edge1.y * edge2.x;
            if (std::abs(det) < 1e-6) continue;

            float u = (h.x * edge2.y - h.y * edge2.x) / det;
            float v = (edge1.x * h.y - edge1.y * h.x) / det;
            if (u < 0 || v < 0 || u + v > 1) continue;

            int idx = (y * width + x);
            float z = screen_coords[0].z + u * (screen_coords[1].z - screen_coords[0].z) +
                     v * (screen_coords[2].z - screen_coords[0].z);

            if (z < zbuffer[idx]) {
                zbuffer[idx] = z;

                Vec2 uv = tri.uv[0] * (1-u-v) + tri.uv[1] * u + tri.uv[2] * v;
                int tex_x = std::min(static_cast<int>(uv.u * tex_width), tex_width - 1);
                int tex_y = std::min(static_cast<int>((1.0f - uv.v) * tex_height), tex_height - 1);
                int tex_idx = (tex_y * tex_width + tex_x) * 3;

                Vec3 normal = (tri.n[0] * (1-u-v) + tri.n[1] * u + tri.n[2] * v).normalize();
                float diffuse = std::max(0.0f, normal.dot(light_dir));
                
                Vec3 color(texture[tex_idx] / 255.0f,
                          texture[tex_idx + 1] / 255.0f,
                          texture[tex_idx + 2] / 255.0f);
                color = color * (0.3f + 0.7f * diffuse);

                output[idx * 3 + 0] = static_cast<unsigned char>(std::min(color.x * 255.0f, 255.0f));
                output[idx * 3 + 1] = static_cast<unsigned char>(std::min(color.y * 255.0f, 255.0f));
                output[idx * 3 + 2] = static_cast<unsigned char>(std::min(color.z * 255.0f, 255.0f));
            }
        }
    }
}

void transform_and_render_scene(const std::vector<Triangle>& triangles,
                              const std::vector<int>& offsets,
                              const std::vector<int>& counts,
                              const std::vector<unsigned char>& textures,
                              const std::vector<int>& tex_widths,
                              const std::vector<int>& tex_heights,
                              const std::vector<Mat4>& model_matrices,
                              const Mat4& projection,
                              unsigned char* output,
                              float* zbuffer,
                              int width, int height,
                              int num_objects,
                              int scene) {
    std::fill(zbuffer, zbuffer + width * height, FLT_MAX);
    std::fill(output, output + width * height * 3, 0);

    size_t texture_offset = 0;
    for (int obj = 0; obj < num_objects; obj++) {
        int offset = offsets[scene * num_objects + obj];
        int count = counts[scene * num_objects + obj];
        Mat4 mp = projection * model_matrices[scene * num_objects + obj];

        for (int i = 0; i < count; i++) {
            Triangle transformed_tri = triangles[offset + i];
            for (int j = 0; j < 3; j++) {
                transformed_tri.v[j] = mp.multiplyPoint(transformed_tri.v[j]);
                transformed_tri.n[j] = model_matrices[scene * num_objects + obj].multiplyVector(transformed_tri.n[j]).normalize();
            }

            render_triangle(transformed_tri,
                          output,
                          zbuffer,
                          textures.data() + texture_offset,
                          tex_widths[scene * num_objects + obj],
                          tex_heights[scene * num_objects + obj],
                          width, height);
        }
        texture_offset += tex_widths[scene * num_objects + obj] * 
                         tex_heights[scene * num_objects + obj] * 3;
    }
}

int main() {
    const int width = 400, height = 300;
    const int num_objects = 2, num_scenes = 1;
    const int num_frames = 10;

    // Load models and textures
    std::vector<std::vector<Triangle>> triangles(num_objects);
    std::vector<unsigned char*> textures(num_objects);
    std::vector<int> tex_widths(num_objects), tex_heights(num_objects);

    load_obj("african_head.obj", triangles[0]);
    load_obj("drone.obj", triangles[1]);

    textures[0] = stbi_load("african_head_diffuse.tga", &tex_widths[0], &tex_heights[0], nullptr, 3);
    textures[1] = stbi_load("drone.png", &tex_widths[1], &tex_heights[1], nullptr, 3);

    // Prepare combined data structures
    std::vector<Triangle> all_triangles;
    std::vector<int> offsets(num_scenes * num_objects);
    std::vector<int> counts(num_scenes * num_objects);
    std::vector<unsigned char> all_textures;
    std::vector<int> all_tex_widths(num_scenes * num_objects);
    std::vector<int> all_tex_heights(num_scenes * num_objects);

    // Combine triangles and textures
    for (int scene = 0; scene < num_scenes; scene++) {
        for (int obj = 0; obj < num_objects; obj++) {
            offsets[scene * num_objects + obj] = all_triangles.size();
            counts[scene * num_objects + obj] = triangles[obj].size();
            all_triangles.insert(all_triangles.end(), triangles[obj].begin(), triangles[obj].end());
            
            all_tex_widths[scene * num_objects + obj] = tex_widths[obj];
            all_tex_heights[scene * num_objects + obj] = tex_heights[obj];
            
            size_t tex_size = tex_widths[obj] * tex_heights[obj] * 3;
            all_textures.insert(all_textures.end(), 
                              textures[obj], 
                              textures[obj] + tex_size);
        }
    }

    // Initialize matrices and buffers
    Mat4 projection = create_projection_matrix(M_PI / 4.0f, static_cast<float>(width) / height, 0.1f, 100.0f);
    std::vector<Mat4> model_matrices(num_scenes * num_objects);
    for (int scene = 0; scene < num_scenes; scene++) {
        for (int obj = 0; obj < num_objects; obj++) {
            model_matrices[scene * num_objects + obj] = create_model_matrix_random();
        }
    }

    // Prepare rendering buffers
    std::vector<float> zbuffer(width * height);
    std::vector<unsigned char> output(width * height * 3);

    // Setup video writers
    std::vector<cv::VideoWriter> video_writers(num_scenes);
    for (int scene = 0; scene < num_scenes; scene++) {
        std::string filename = "output_scene" + std::to_string(scene) + ".mp4";
        video_writers[scene].open(filename, 
                                cv::VideoWriter::fourcc('a','v','c','1'), 
                                30, 
                                cv::Size(width, height));
    }

    // Main rendering loop
    for (int frame = 0; frame < num_frames; frame++) {
        // Update model matrices (animation)
        for (int scene = 0; scene < num_scenes; scene++) {
            float rotation = 0.1f;
            Mat4 rotation_matrix = Mat4::rotationY(rotation);
            for (int obj = 0; obj < num_objects; obj++) {
                model_matrices[scene * num_objects + obj] = 
                    rotation_matrix * model_matrices[scene * num_objects + obj];
            }
        }

        // Render each scene
        for (int scene = 0; scene < num_scenes; scene++) {
            transform_and_render_scene(
                all_triangles,
                offsets,
                counts,
                all_textures,
                all_tex_widths,
                all_tex_heights,
                model_matrices,
                projection,
                output.data(),
                zbuffer.data(),
                width,
                height,
                num_objects,
                scene
            );
            
            cv::Mat frame(height, width, CV_8UC3, output.data());
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            video_writers[scene].write(frame);
        }
    }

    // Cleanup
    for (auto& writer : video_writers) {
        writer.release();
    }
    for (auto texture : textures) {
        stbi_image_free(texture);
    }

    return 0;
}