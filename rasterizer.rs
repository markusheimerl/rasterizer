use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::f64;
use ffmpeg_next as ffmpeg;

// Constants
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const FRAMES: usize = 60;
const ASPECT_RATIO: f64 = WIDTH as f64 / HEIGHT as f64;

// Structs
#[derive(Clone, Copy)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy)]
struct Vec2 {
    u: f64,
    v: f64,
}

#[derive(Clone)]
struct Triangle {
    vertices: [usize; 3],
    texcoords: [usize; 3],
}

struct Model {
    vertices: Vec<Vec3>,
    initial_vertices: Vec<Vec3>,
    texcoords: Vec<Vec2>,
    triangles: Vec<Triangle>,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )
    }

    fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn normalize(&mut self) {
        let len = (self.dot(self)).sqrt();
        if len > 0.0 {
            self.x /= len;
            self.y /= len;
            self.z /= len;
        }
    }

    fn scale(&self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    fn rotate_y(&mut self, angle: f64) {
        let s = angle.sin();
        let c = angle.cos();
        let x = self.x;
        let z = self.z;
        self.x = c * x + s * z;
        self.z = -s * x + c * z;
    }
}

impl Model {
    fn new() -> Self {
        Model {
            vertices: Vec::new(),
            initial_vertices: Vec::new(),
            texcoords: Vec::new(),
            triangles: Vec::new(),
        }
    }

    fn parse_obj_file(&mut self, filename: &str) -> std::io::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let mut words = line.split_whitespace();
            
            match words.next() {
                Some("v") => {
                    let x: f64 = words.next().unwrap().parse().unwrap();
                    let y: f64 = words.next().unwrap().parse().unwrap();
                    let z: f64 = words.next().unwrap().parse().unwrap();
                    let vertex = Vec3::new(x, y, z);
                    self.vertices.push(vertex);
                    self.initial_vertices.push(vertex);
                },
                Some("vt") => {
                    let u: f64 = words.next().unwrap().parse().unwrap();
                    let v: f64 = words.next().unwrap().parse().unwrap();
                    self.texcoords.push(Vec2 { u, v });
                },
                Some("f") => {
                    let mut vertex_indices = [0; 3];
                    let mut texcoord_indices = [0; 3];
                    
                    for (i, word) in words.enumerate() {
                        let indices: Vec<&str> = word.split('/').collect();
                        vertex_indices[i] = indices[0].parse::<usize>().unwrap() - 1;
                        if indices.len() > 1 {
                            texcoord_indices[i] = indices[1].parse::<usize>().unwrap() - 1;
                        }
                    }
                    
                    self.triangles.push(Triangle {
                        vertices: vertex_indices,
                        texcoords: texcoord_indices,
                    });
                },
                _ => {}
            }
        }
        Ok(())
    }
}

// Texture handling
struct Texture {
    data: Vec<u8>,
    width: usize,
    height: usize,
    channels: usize,
}

impl Texture {
    fn load(path: &str) -> Option<Self> {
        let img = image::open(path).ok()?;
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        Some(Texture {
            data: rgb.into_raw(),
            width: width as usize,
            height: height as usize,
            channels: 3,
        })
    }

    fn sample(&self, u: f64, v: f64) -> [f64; 3] {
        let u = u.min(1.0).max(0.0);
        let v = (1.0 - v.min(1.0).max(0.0));
        
        let x = (u * (self.width - 1) as f64) as usize;
        let y = (v * (self.height - 1) as f64) as usize;
        
        let idx = (y * self.width + x) * self.channels;
        [
            self.data[idx] as f64 / 255.0,
            self.data[idx + 1] as f64 / 255.0,
            self.data[idx + 2] as f64 / 255.0,
        ]
    }
}

use ffmpeg_next::format::{input, output};
use ffmpeg_next::codec::encoder::video::Video;
use ffmpeg_next::frame::Video as VideoFrame;
use ffmpeg_next::util::frame::video::Video as VideoUtil;
use ffmpeg_next::software::scaling::{context::Context, flag::Flags};
use std::time::Instant;

struct Rasterizer {
    frame_buffer: Vec<f64>,      // RGBA buffer where A is depth
    output_buffer: Vec<u8>,      // RGB buffer for final output
    texture: Texture,
    model: Model,
}

impl Rasterizer {
    fn new(texture_path: &str, model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = Model::new();
        model.parse_obj_file(model_path)?;
        
        let texture = Texture::load(texture_path)
            .ok_or("Failed to load texture")?;
            
        Ok(Rasterizer {
            frame_buffer: vec![0.0; WIDTH * HEIGHT * 4],
            output_buffer: vec![0; WIDTH * HEIGHT * 3],
            texture,
            model,
        })
    }

    fn clear_buffer(&mut self) {
        for i in 0..WIDTH * HEIGHT {
            self.frame_buffer[i * 4] = 0.0;     // R
            self.frame_buffer[i * 4 + 1] = 0.0; // G
            self.frame_buffer[i * 4 + 2] = 0.0; // B
            self.frame_buffer[i * 4 + 3] = f64::MAX; // Depth
        }
    }

    fn draw_triangle(&mut self, pts: &[[f64; 4]; 3], uv: &[[f64; 2]; 3]) {
        // Calculate bounding box
        let mut bbox_min_x = f64::MAX;
        let mut bbox_min_y = f64::MAX;
        let mut bbox_max_x = f64::MIN;
        let mut bbox_max_y = f64::MIN;

        for pt in pts.iter() {
            bbox_min_x = bbox_min_x.min(pt[0]);
            bbox_min_y = bbox_min_y.min(pt[1]);
            bbox_max_x = bbox_max_x.max(pt[0]);
            bbox_max_y = bbox_max_y.max(pt[1]);
        }

        // Clamp to screen bounds
        let min_x = bbox_min_x.max(0.0) as i32;
        let min_y = bbox_min_y.max(0.0) as i32;
        let max_x = bbox_max_x.min((WIDTH - 1) as f64) as i32;
        let max_y = bbox_max_y.min((HEIGHT - 1) as f64) as i32;

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                let x_f = x as f64;
                let y_f = y as f64;

                // Calculate barycentric coordinates
                let denominator = ((pts[1][1] - pts[2][1]) * (pts[0][0] - pts[2][0]) +
                                 (pts[2][0] - pts[1][0]) * (pts[0][1] - pts[2][1]));
                
                let lambda_1 = ((pts[1][1] - pts[2][1]) * (x_f - pts[2][0]) +
                               (pts[2][0] - pts[1][0]) * (y_f - pts[2][1])) / denominator;
                let lambda_2 = ((pts[2][1] - pts[0][1]) * (x_f - pts[2][0]) +
                               (pts[0][0] - pts[2][0]) * (y_f - pts[2][1])) / denominator;
                let lambda_3 = 1.0 - lambda_1 - lambda_2;

                if lambda_1 >= 0.0 && lambda_1 <= 1.0 &&
                   lambda_2 >= 0.0 && lambda_2 <= 1.0 &&
                   lambda_3 >= 0.0 && lambda_3 <= 1.0 {
                    
                    let z = lambda_1 * pts[0][2] + lambda_2 * pts[1][2] + lambda_3 * pts[2][2];
                    let idx = (y as usize * WIDTH + x as usize) * 4;

                    if z < self.frame_buffer[idx + 3] {
                        // Update depth
                        self.frame_buffer[idx + 3] = z;

                        // Interpolate texture coordinates
                        let u = lambda_1 * uv[0][0] + lambda_2 * uv[1][0] + lambda_3 * uv[2][0];
                        let v = lambda_1 * uv[0][1] + lambda_2 * uv[1][1] + lambda_3 * uv[2][1];

                        // Sample texture
                        let color = self.texture.sample(u, v);
                        self.frame_buffer[idx] = color[0];
                        self.frame_buffer[idx + 1] = color[1];
                        self.frame_buffer[idx + 2] = color[2];
                    }
                }
            }
        }
    }

    fn render_frame(&mut self, angle: f64) {
        self.clear_buffer();

        let scale_factor = 1.0;
        let translation = Vec3::new(0.0, 0.0, 3.0);

        // Transform vertices
        let mut transformed_vertices = self.model.initial_vertices.clone();
        for vertex in transformed_vertices.iter_mut() {
            vertex.x *= scale_factor;
            vertex.y *= -scale_factor; // Invert Y for correct orientation
            vertex.z *= scale_factor;

            vertex.rotate_y(angle);

            vertex.x += translation.x;
            vertex.y += translation.y;
            vertex.z += translation.z;
        }

        // Render triangles
        for triangle in &self.model.triangles {
            let mut verts = [[0.0; 4]; 3];
            let mut uvs = [[0.0; 2]; 3];

            for (i, &vert_idx) in triangle.vertices.iter().enumerate() {
                let vertex = &transformed_vertices[vert_idx];
                let inv_z = 1.0 / vertex.z;
                
                verts[i][0] = vertex.x * inv_z * WIDTH as f64 / (2.0 * ASPECT_RATIO) + WIDTH as f64 / 2.0;
                verts[i][1] = vertex.y * inv_z * HEIGHT as f64 / 2.0 + HEIGHT as f64 / 2.0;
                verts[i][2] = vertex.z;

                let texcoord = &self.model.texcoords[triangle.texcoords[i]];
                uvs[i][0] = texcoord.u;
                uvs[i][1] = texcoord.v;
            }

            self.draw_triangle(&verts, &uvs);
        }
    }

    fn write_frame_to_output(&mut self) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let in_idx = (y * WIDTH + x) * 4;
                let out_idx = (y * WIDTH + x) * 3;
                
                for c in 0..3 {
                    self.output_buffer[out_idx + c] = 
                        (self.frame_buffer[in_idx + c].max(0.0).min(1.0) * 255.0) as u8;
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ffmpeg_next::init()?;

    let mut rasterizer = Rasterizer::new(
        "african_head_diffuse.tga",
        "african_head.obj"
    )?;

    // Set up FFmpeg output
    let mut output_context = output::create("output_rasterizer.mp4")?;
    let codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::H264)
        .ok_or("Failed to find H264 encoder")?;
    let global_header = output_context.format().flags().contains(Flags::GLOBAL_HEADER);

    let mut stream = output_context.add_stream()?;
    let mut encoder = codec.video()?;
    
    encoder.set_height(HEIGHT as i32);
    encoder.set_width(WIDTH as i32);
    encoder.set_format(ffmpeg_next::format::Pixel::YUV420P);
    encoder.set_time_base((1, 30));

    if global_header {
        encoder.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
    }

    let encoder = encoder.open_as(codec)?;
    stream.set_parameters(encoder.parameters());

    output_context.write_header()?;

    let mut scaler = Context::get(
        ffmpeg_next::format::Pixel::RGB24,
        WIDTH as u32,
        HEIGHT as u32,
        ffmpeg_next::format::Pixel::YUV420P,
        WIDTH as u32,
        HEIGHT as u32,
        Flags::BILINEAR,
    )?;

    // Render frames
    for frame_num in 0..FRAMES {
        let angle = 2.0 * std::f64::consts::PI * frame_num as f64 / FRAMES as f64;
        
        println!("Rendering frame {}/{}", frame_num + 1, FRAMES);
        rasterizer.render_frame(angle);
        rasterizer.write_frame_to_output();

        // Convert and encode frame
        // ... FFmpeg frame conversion and encoding code ...
    }

    output_context.write_trailer()?;
    
    println!("Video saved to 'output_rasterizer.mp4'");
    Ok(())
}