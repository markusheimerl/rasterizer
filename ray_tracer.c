#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

// Include stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Include FFmpeg headers
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

// Constants
#define WIDTH 640
#define HEIGHT 480
#define N_RAYS 10
#define N_FRAMES 60

// Maximum counts
#define MAX_VERTICES 100000
#define MAX_TEXCOORDS 100000
#define MAX_TRIANGLES 200000

// Vector macros
#define VEC_SUB(a,b,r) { (r)[0]=(a)[0]-(b)[0]; (r)[1]=(a)[1]-(b)[1]; (r)[2]=(a)[2]-(b)[2]; }
#define VEC_DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VEC_CROSS(a,b,r) { \
    (r)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; \
    (r)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; \
    (r)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; \
}
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); if(l>0){ (v)[0]/=l; (v)[1]/=l; (v)[2]/=l; } }
#define VEC_SCALE(v,s,r) { (r)[0]=(v)[0]*(s); (r)[1]=(v)[1]*(s); (r)[2]=(v)[2]*(s); }
#define VEC_ADD(a,b,r) { (r)[0]=(a)[0]+(b)[0]; (r)[1]=(a)[1]+(b)[1]; (r)[2]=(a)[2]+(b)[2]; }

// Random number between -1 and 1
#define RAND_UNIFORM() (((double)rand() / RAND_MAX) * 2.0 - 1.0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global variables
double vertices[MAX_VERTICES][3];
double initial_vertices[MAX_VERTICES][3];
double texcoords[MAX_TEXCOORDS][2];
int triangles[MAX_TRIANGLES][3];
int texcoord_indices[MAX_TRIANGLES][3];
int num_vertices = 0, num_texcoords = 0, num_triangles = 0;

// BVH Node structure
typedef struct BVHNode {
    double bbox_min[3], bbox_max[3];
    int start, count;
    struct BVHNode *left, *right;
} BVHNode;

BVHNode *bvh_root = NULL;

// Texture data
unsigned char *texture_data = NULL;
int texture_width, texture_height, texture_channels;

// Function to parse OBJ file
void parse_obj_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) { fprintf(stderr, "Failed to open OBJ file %s\n", filename); exit(1); }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "v ", 2) == 0) {
            double x, y, z;
            sscanf(line + 2, "%lf %lf %lf", &x, &y, &z);
            vertices[num_vertices][0] = x; vertices[num_vertices][1] = y; vertices[num_vertices][2] = z;
            initial_vertices[num_vertices][0] = x; initial_vertices[num_vertices][1] = y; initial_vertices[num_vertices][2] = z;
            num_vertices++;
        } else if (strncmp(line, "vt ", 3) == 0) {
            double u, v;
            sscanf(line + 3, "%lf %lf", &u, &v);
            texcoords[num_texcoords][0] = u; texcoords[num_texcoords][1] = v;
            num_texcoords++;
        } else if (strncmp(line, "f ", 2) == 0) {
            int vi[3], ti[3];
            int matches = sscanf(line + 2, "%d/%d %d/%d %d/%d", &vi[0], &ti[0], &vi[1], &ti[1], &vi[2], &ti[2]);
            if (matches != 6) {
                matches = sscanf(line + 2, "%d/%d/%*d %d/%d/%*d %d/%d/%*d", &vi[0], &ti[0], &vi[1], &ti[1], &vi[2], &ti[2]);
                if (matches != 6) {
                    matches = sscanf(line + 2, "%d %d %d", &vi[0], &vi[1], &vi[2]);
                    if (matches != 3) { fprintf(stderr, "Failed to parse face: %s", line); continue; }
                    else { ti[0] = ti[1] = ti[2] = -1; }
                }
            }
            triangles[num_triangles][0] = vi[0] - 1; triangles[num_triangles][1] = vi[1] - 1; triangles[num_triangles][2] = vi[2] - 1;
            texcoord_indices[num_triangles][0] = (ti[0] != -1) ? ti[0] - 1 : -1;
            texcoord_indices[num_triangles][1] = (ti[1] != -1) ? ti[1] - 1 : -1;
            texcoord_indices[num_triangles][2] = (ti[2] != -1) ? ti[2] - 1 : -1;
            num_triangles++;
        }
    }
    fclose(file);
}

// Function to compute bounding box for a set of triangles
void compute_bbox(int start, int end, double bbox_min[3], double bbox_max[3]) {
    bbox_min[0] = bbox_min[1] = bbox_min[2] = DBL_MAX;
    bbox_max[0] = bbox_max[1] = bbox_max[2] = -DBL_MAX;
    for (int i = start; i < end; i++) {
        int *tri = triangles[i];
        for (int j = 0; j < 3; j++) {
            double *v = vertices[tri[j]];
            for (int k = 0; k < 3; k++) {
                if (v[k] < bbox_min[k]) bbox_min[k] = v[k];
                if (v[k] > bbox_max[k]) bbox_max[k] = v[k];
            }
        }
    }
}

// Function to build BVH recursively
BVHNode* build_bvh(int start, int end) {
    BVHNode *node = malloc(sizeof(BVHNode));
    node->start = start; node->count = end - start; node->left = node->right = NULL;
    compute_bbox(start, end, node->bbox_min, node->bbox_max);

    if (node->count <= 4) return node;

    int axis = 0;
    double extent[3] = { node->bbox_max[0] - node->bbox_min[0], node->bbox_max[1] - node->bbox_min[1], node->bbox_max[2] - node->bbox_min[2] };
    if (extent[1] > extent[axis]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;

    double *centroids = malloc(node->count * sizeof(double));
    for (int i = 0; i < node->count; i++) {
        int *tri = triangles[start + i];
        centroids[i] = (vertices[tri[0]][axis] + vertices[tri[1]][axis] + vertices[tri[2]][axis]) / 3.0;
    }

    // Simple bubble sort for median split
    for (int i = 0; i < node->count - 1; i++) {
        for (int j = 0; j < node->count - i - 1; j++) {
            if (centroids[j] > centroids[j + 1]) {
                double temp_c = centroids[j]; centroids[j] = centroids[j + 1]; centroids[j + 1] = temp_c;

                int temp_tri[3], temp_tex[3];
                memcpy(temp_tri, triangles[start + j], 3 * sizeof(int));
                memcpy(triangles[start + j], triangles[start + j + 1], 3 * sizeof(int));
                memcpy(triangles[start + j + 1], temp_tri, 3 * sizeof(int));

                memcpy(temp_tex, texcoord_indices[start + j], 3 * sizeof(int));
                memcpy(texcoord_indices[start + j], texcoord_indices[start + j + 1], 3 * sizeof(int));
                memcpy(texcoord_indices[start + j + 1], temp_tex, 3 * sizeof(int));
            }
        }
    }
    free(centroids);

    int mid = start + node->count / 2;
    node->left = build_bvh(start, mid);
    node->right = build_bvh(mid, end);
    return node;
}

// Triangle intersection with barycentric coordinates
int triangle_intersect(const double v0[3], const double v1[3], const double v2[3],
                       const double ro[3], const double rd[3], double *t_out, double *u_out, double *v_out, double normal[3]) {
    double edge1[3], edge2[3], h[3], s[3], q[3];
    VEC_SUB(v1, v0, edge1); VEC_SUB(v2, v0, edge2);
    VEC_CROSS(rd, edge2, h);
    double a = VEC_DOT(edge1, h);
    if (fabs(a) < 1e-8) return 0;
    double f = 1.0 / a;
    VEC_SUB(ro, v0, s);
    double u = f * VEC_DOT(s, h);
    if (u < 0.0 || u > 1.0) return 0;
    VEC_CROSS(s, edge1, q);
    double v = f * VEC_DOT(rd, q);
    if (v < 0.0 || u + v > 1.0) return 0;
    double t = f * VEC_DOT(edge2, q);
    if (t > 1e-8) {
        *t_out = t; *u_out = u; *v_out = v;
        VEC_CROSS(edge1, edge2, normal); VEC_NORM(normal);
        return 1;
    }
    return 0;
}

// Function to intersect ray with BVH node
int bvh_intersect(BVHNode *node, const double ro[3], const double rd[3],
                  double *t_out, double *u_out, double *v_out, double normal[3], int *hit_tri_idx) {
    // AABB-Ray intersection
    double tmin = -INFINITY, tmax = INFINITY;
    for (int i = 0; i < 3; i++) {
        double invD = 1.0 / rd[i];
        double t0 = (node->bbox_min[i] - ro[i]) * invD;
        double t1 = (node->bbox_max[i] - ro[i]) * invD;
        if (invD < 0.0) { double temp = t0; t0 = t1; t1 = temp; }
        if (t0 > tmin) tmin = t0; if (t1 < tmax) tmax = t1;
        if (tmax <= tmin) return 0;
    }

    if (node->left == NULL && node->right == NULL) {
        int hit = 0;
        double nearest = *t_out;
        for (int i = node->start; i < node->start + node->count; i++) {
            int *tri = triangles[i];
            double t, u, v, n[3];
            if (triangle_intersect(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]], ro, rd, &t, &u, &v, n)) {
                if (t < nearest) {
                    nearest = t; *t_out = t; *u_out = u; *v_out = v;
                    normal[0] = n[0]; normal[1] = n[1]; normal[2] = n[2];
                    *hit_tri_idx = i; hit = 1;
                }
            }
        }
        return hit;
    }

    int hit_left = 0, hit_right = 0;
    if (node->left) hit_left = bvh_intersect(node->left, ro, rd, t_out, u_out, v_out, normal, hit_tri_idx);
    if (node->right) hit_right = bvh_intersect(node->right, ro, rd, t_out, u_out, v_out, normal, hit_tri_idx);
    return hit_left || hit_right;
}

// Rotate point around Y-axis
void rotate_y(double angle, double point[3]) {
    double s = sin(angle), c = cos(angle), x = point[0], z = point[2];
    point[0] = c * x + s * z; point[2] = -s * x + c * z;
}

// Sample texture
void sample_texture(double u, double v, double color[3]) {
    u = fmin(fmax(u, 0.0), 1.0); v = fmin(fmax(v, 0.0), 1.0);
    int x = (int)(u * (texture_width - 1)), y = (int)((1.0 - v) * (texture_height - 1));
    int idx = (y * texture_width + x) * texture_channels;
    color[0] = texture_data[idx] / 255.0;
    color[1] = texture_data[idx + 1] / 255.0;
    color[2] = texture_data[idx + 2] / 255.0;
}

// Trace function
void trace(const double ro[3], const double rd[3], double color_out[3]) {
    double t = INFINITY, u, v, normal[3];
    int hit_tri_idx = -1;
    if (bvh_intersect(bvh_root, ro, rd, &t, &u, &v, normal, &hit_tri_idx)) {
        int *tri = triangles[hit_tri_idx];
        int *tex_idx = texcoord_indices[hit_tri_idx];
        if (tex_idx[0] >= 0) {
            double *uv0 = texcoords[tex_idx[0]], *uv1 = texcoords[tex_idx[1]], *uv2 = texcoords[tex_idx[2]];
            double w = 1.0 - u - v;
            double tex_u = w * uv0[0] + u * uv1[0] + v * uv2[0];
            double tex_v = w * uv0[1] + u * uv1[1] + v * uv2[1];
            double tex_color[3];
            sample_texture(tex_u, tex_v, tex_color);
            double shade = fabs(VEC_DOT(normal, rd));
            color_out[0] = tex_color[0] * shade;
            color_out[1] = tex_color[1] * shade;
            color_out[2] = tex_color[2] * shade;
        } else {
            double shade = fabs(VEC_DOT(normal, rd));
            color_out[0] = color_out[1] = color_out[2] = shade;
        }
    } else {
        color_out[0] = color_out[1] = color_out[2] = 0.5;
    }
}

// Function to free BVH tree
void free_bvh(BVHNode *node) {
    if (!node) return;
    free_bvh(node->left);
    free_bvh(node->right);
    free(node);
}

// Main function
int main() {
    srand(42);

    // Read the OBJ file
    parse_obj_file("african_head.obj");

    // Load the texture
    texture_data = stbi_load("african_head_diffuse.tga", &texture_width, &texture_height, &texture_channels, 3);
    if (!texture_data) { fprintf(stderr, "Failed to load texture image\n"); return 1; }

    // Allocate image buffers
    double *image = calloc(WIDTH * HEIGHT * 3, sizeof(double));
    unsigned char *output_image = malloc(WIDTH * HEIGHT * 3);

    // Initialize FFmpeg
    avformat_network_init();

    // Set up FFmpeg encoding
    AVFormatContext *oc = NULL;
    AVOutputFormat *fmt = NULL;
    AVStream *video_st = NULL;
    AVCodecContext *c = NULL;
    AVCodec *codec = NULL;

    // Allocate the output media context
    avformat_alloc_output_context2(&oc, NULL, NULL, "output.mp4");
    if (!oc) {
        fprintf(stderr, "Could not deduce output format from file extension.\n");
        exit(1);
    }
    fmt = oc->oformat;

    // Add the video stream using the default format codecs and initialize the codec
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    video_st = avformat_new_stream(oc, NULL);
    if (!video_st) {
        fprintf(stderr, "Could not allocate stream\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    // Set codec parameters
    c->codec_id = AV_CODEC_ID_H264;
    c->bit_rate = 400000;
    c->width = WIDTH;
    c->height = HEIGHT;
    video_st->time_base = (AVRational){1, 30};
    c->time_base = video_st->time_base;
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // Open the codec
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    // Copy the stream parameters to the muxer
    if (avcodec_parameters_from_context(video_st->codecpar, c) < 0) {
        fprintf(stderr, "Could not copy the stream parameters\n");
        exit(1);
    }

    // Open the output file
    if (!(fmt->flags & AVFMT_NOFILE)) {
        if (avio_open(&oc->pb, "output.mp4", AVIO_FLAG_WRITE) < 0) {
            fprintf(stderr, "Could not open 'output.mp4'\n");
            exit(1);
        }
    }

    // Write the stream header
    if (avformat_write_header(oc, NULL) < 0) {
        fprintf(stderr, "Error occurred when opening output file\n");
        exit(1);
    }

    // For image conversion
    struct SwsContext *sws_ctx = sws_getContext(
        WIDTH, HEIGHT, AV_PIX_FMT_RGB24,
        WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    AVFrame *frame = av_frame_alloc();
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;

    // Allocate the buffers for the frame data
    if (av_image_alloc(frame->data, frame->linesize, c->width, c->height, c->pix_fmt, 32) < 0) {
        fprintf(stderr, "Could not allocate raw picture buffer\n");
        exit(1);
    }

    AVPacket pkt;
    int frame_count = 0;

    // Define constants
    double eye[3] = {0, 0, 0};
    double focal = 500;
    double scale_factor = 1.0;
    double translation[3] = {-1.0, 0.0, 4.0};
    double initial_rotation = 0.0;
    double angle_per_frame = (2.0 * M_PI) / N_FRAMES;

    for (int frame_num = 0; frame_num < N_FRAMES; frame_num++) {
        printf("Rendering frame %d/%d\n", frame_num + 1, N_FRAMES);

        // Copy initial_vertices to vertices
        for (int i = 0; i < num_vertices; i++) {
            vertices[i][0] = initial_vertices[i][0];
            vertices[i][1] = initial_vertices[i][1];
            vertices[i][2] = initial_vertices[i][2];
        }

        // Apply transformations
        double rotation_y_angle = initial_rotation + frame_num * angle_per_frame;
        for (int i = 0; i < num_vertices; i++) {
            vertices[i][0] *= scale_factor;
            vertices[i][1] *= -scale_factor;
            vertices[i][2] *= scale_factor;

            rotate_y(rotation_y_angle, vertices[i]);

            vertices[i][0] += translation[0];
            vertices[i][1] += translation[1];
            vertices[i][2] += translation[2];
        }

        // Build BVH
        bvh_root = build_bvh(0, num_triangles);

        // Initialize image buffer
        memset(image, 0, WIDTH * HEIGHT * 3 * sizeof(double));

        for (int r = 0; r < N_RAYS; r++) {
            printf("Pass %d/%d\n", r + 1, N_RAYS);
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    double x_screen = x - WIDTH / 2 + RAND_UNIFORM();
                    double y_screen = y - HEIGHT / 2 + RAND_UNIFORM();
                    double dir[3] = { x_screen, y_screen, focal };
                    VEC_NORM(dir);
                    double color[3];
                    trace(eye, dir, color);
                    int idx = (y * WIDTH + x) * 3;
                    image[idx] += color[0];
                    image[idx + 1] += color[1];
                    image[idx + 2] += color[2];
                }
            }
        }

        // Prepare frame data
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int idx = (y * WIDTH + x) * 3;
                output_image[idx] = (unsigned char)(fmin(image[idx] / N_RAYS, 1.0) * 255);
                output_image[idx + 1] = (unsigned char)(fmin(image[idx + 1] / N_RAYS, 1.0) * 255);
                output_image[idx + 2] = (unsigned char)(fmin(image[idx + 2] / N_RAYS, 1.0) * 255);
            }
        }

        // Convert RGB to YUV and encode
        const uint8_t *inData[1] = { output_image };
        int inLinesize[1] = { 3 * WIDTH };
        sws_scale(sws_ctx, inData, inLinesize, 0, HEIGHT, frame->data, frame->linesize);

        frame->pts = frame_count++;

        // Encode the frame
        av_init_packet(&pkt);
        pkt.data = NULL;
        pkt.size = 0;

        int ret = avcodec_send_frame(c, frame);
        if (ret < 0) {
            fprintf(stderr, "Error sending a frame for encoding\n");
            exit(1);
        }

        while (ret >= 0) {
            ret = avcodec_receive_packet(c, &pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            else if (ret < 0) {
                fprintf(stderr, "Error during encoding\n");
                exit(1);
            }

            // Rescale output packet timestamp values from codec to stream timebase
            av_packet_rescale_ts(&pkt, c->time_base, video_st->time_base);
            pkt.stream_index = video_st->index;

            // Write the compressed frame to the media file
            ret = av_interleaved_write_frame(oc, &pkt);
            if (ret < 0) {
                fprintf(stderr, "Error while writing output packet\n");
                exit(1);
            }
            av_packet_unref(&pkt);
        }

        // Free BVH
        free_bvh(bvh_root);
        bvh_root = NULL;
    }

    // Flush the encoder
    avcodec_send_frame(c, NULL);
    while (avcodec_receive_packet(c, &pkt) == 0) {
        av_packet_rescale_ts(&pkt, c->time_base, video_st->time_base);
        pkt.stream_index = video_st->index;
        av_interleaved_write_frame(oc, &pkt);
        av_packet_unref(&pkt);
    }

    // Write the trailer
    av_write_trailer(oc);

    // Clean up
    avcodec_close(c);
    avcodec_free_context(&c);
    av_frame_free(&frame);
    sws_freeContext(sws_ctx);
    if (!(fmt->flags & AVFMT_NOFILE))
        avio_closep(&oc->pb);
    avformat_free_context(oc);

    // Free resources
    free(image);
    free(output_image);
    stbi_image_free(texture_data);

    printf("Video saved to 'output.mp4'\n");

    return 0;
}
