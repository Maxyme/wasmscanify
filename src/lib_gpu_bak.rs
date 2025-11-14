/*! jscanify-wasm v1.4.0 | (c) ColonelParrot and other contributors | MIT License */

use photon_rs::PhotonImage;
use photon_rs::conv;
use photon_rs::monochrome;
use nalgebra::Matrix3;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};
use homography::{HomographyComputation, geo::Point};
use wgpu::util::DeviceExt;
use std::cell::RefCell;

// Thread-local cache for GPU context to avoid re-initializing adapter/device/pipeline
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

thread_local! {
    static GPU_CONTEXT: RefCell<Option<GpuContext>> = RefCell::new(None);
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Debug, Clone)]
#[wasm_bindgen]
pub struct CornerPoints {
    top_left: Point,
    top_right: Point,
    bottom_left: Point,
    bottom_right: Point,
}

#[wasm_bindgen]
pub struct Jscanify {}

#[wasm_bindgen]
impl Jscanify {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Jscanify {
        console_error_panic_hook::set_once();
        Jscanify {}
    }

    /// Process ImageData and find the paper contour
    /// Returns corner points as a JS object or null if no paper detected
    #[wasm_bindgen(js_name = findPaperContour)]
    pub fn find_paper_contour(&self, image_data: ImageData) -> Option<CornerPoints> {
        let width = image_data.width();
        let height = image_data.height();
        let data = image_data.data().0;

        // Convert to PhotonImage
        let mut photon_img = PhotonImage::new(data.to_vec(), width, height);

        // Apply edge detection using photon-rs
        conv::edge_detection(&mut photon_img);

        // Convert to grayscale for contour finding
        let gray_img = photon_to_gray(&mut photon_img);

        // Find contours
        let contours = find_contours(&gray_img, width, height);

        // Find the largest contour
        if let Some(idx) = find_largest_contour_index(&contours) {
            let corners = get_corner_points(&contours[idx], width, height);
            Some(corners)
        } else {
            None
        }
    }

    /// Extract and undistort the paper from the image
    #[wasm_bindgen(js_name = extractPaper)]
    pub fn extract_paper(
        &self,
        image_data: ImageData,
        result_width: u32,
        result_height: u32,
    ) -> Result<ImageData, JsValue> {
        let width = image_data.width();
        let height = image_data.height();
        let data = image_data.data().0;

        log("Starting paper extraction");
        // Convert to RGBA image for perspective transform
        let img = PhotonImage::new(data.to_vec(), width, height);

        // Auto-detect corners using photon-rs edge detection
        let mut photon_img = PhotonImage::new(data.to_vec(), width, height);
        conv::edge_detection(&mut photon_img);
        let gray_img = photon_to_gray(&mut photon_img);
        let contours = find_contours(&gray_img, width, height);

        let corners = if let Some(idx) = find_largest_contour_index(&contours) {
            get_corner_points(&contours[idx], width, height)
        } else {
            return Err(JsValue::from_str("No paper detected"));
        };

        // Apply perspective transform (CPU)
        let warped = warp_perspective_cpu(&img, &corners, result_width, result_height);

        // Convert back to ImageData
        let result_data = warped.get_raw_pixels();
        ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&result_data),
            result_width,
            result_height,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to create ImageData: {:?}", e)))
    }

    /// Extract and undistort the paper from the image with custom corner points
    #[wasm_bindgen(js_name = extractPaperWithCorners)]
    pub fn extract_paper_with_corners(
        &self,
        image_data: ImageData,
        result_width: u32,
        result_height: u32,
        top_left_x: f32,
        top_left_y: f32,
        top_right_x: f32,
        top_right_y: f32,
        bottom_left_x: f32,
        bottom_left_y: f32,
        bottom_right_x: f32,
        bottom_right_y: f32,
    ) -> Result<ImageData, JsValue> {
        let width = image_data.width();
        let height = image_data.height();
        let data = image_data.data().0;

        // Convert to RGBA image
        let img = PhotonImage::new(data.to_vec(), width, height);

        // Use provided corner points
        let corners = CornerPoints {
            top_left: Point {
                x: top_left_x,
                y: top_left_y,
            },
            top_right: Point {
                x: top_right_x,
                y: top_right_y,
            },
            bottom_left: Point {
                x: bottom_left_x,
                y: bottom_left_y,
            },
            bottom_right: Point {
                x: bottom_right_x,
                y: bottom_right_y,
            },
        };

        // Apply perspective transform (CPU)
        let warped = warp_perspective_cpu(&img, &corners, result_width, result_height);

        // Convert back to ImageData
        let result_data = warped.get_raw_pixels();
        ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&result_data),
            result_width,
            result_height,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to create ImageData: {:?}", e)))
    }

    /// Highlight the paper borders in the image
    #[wasm_bindgen(js_name = highlightPaper)]
    pub fn highlight_paper(
        &self,
        canvas: HtmlCanvasElement,
        image_data: ImageData,
        color: Option<String>,
        thickness: Option<f64>,
    ) -> Result<(), JsValue> {
        let ctx = canvas
            .get_context("2d")?
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()?;

        let width = image_data.width();
        let height = image_data.height();
        let data = image_data.data().0;

        // Draw original image
        ctx.put_image_data(&image_data, 0.0, 0.0)?;

        // Find paper contour using photon-rs edge detection
        let mut photon_img = PhotonImage::new(data.to_vec(), width, height);
        conv::edge_detection(&mut photon_img);
        let gray_img = photon_to_gray(&mut photon_img);
        let contours = find_contours(&gray_img, width, height);

        if let Some(idx) = find_largest_contour_index(&contours) {
            let corners = get_corner_points(&contours[idx], width, height);

            // Draw highlight
            ctx.set_stroke_style_str(&color.unwrap_or_else(|| "orange".to_string()));
            ctx.set_line_width(thickness.unwrap_or(10.0));
            ctx.begin_path();
            ctx.move_to(corners.top_left.x as f64, corners.top_left.y as f64);
            ctx.line_to(corners.top_right.x as f64, corners.top_right.y as f64);
            ctx.line_to(corners.bottom_right.x as f64, corners.bottom_right.y as f64);
            ctx.line_to(corners.bottom_left.x as f64, corners.bottom_left.y as f64);
            ctx.line_to(corners.top_left.x as f64, corners.top_left.y as f64);
            ctx.stroke();
        }

        Ok(())
    }

    /// Extract and undistort the paper from the image (GPU-accelerated)
    /// Returns a Promise that resolves to ImageData
    #[wasm_bindgen(js_name = extractPaperGpu)]
    pub async fn extract_paper_gpu(
        &self,
        image_data: ImageData,
        result_width: u32,
        result_height: u32,
    ) -> Result<ImageData, JsValue> {
        let width = image_data.width();
        let height = image_data.height();
        let data = image_data.data().0;

        // Convert to RGBA image for perspective transform
        let img = PhotonImage::new(data.to_vec(), width, height);

        // Auto-detect corners using photon-rs edge detection
        let mut photon_img = PhotonImage::new(data.to_vec(), width, height);
        conv::edge_detection(&mut photon_img);
        let gray_img = photon_to_gray(&mut photon_img);
        let contours = find_contours(&gray_img, width, height);

        let corners = if let Some(idx) = find_largest_contour_index(&contours) {
            get_corner_points(&contours[idx], width, height)
        } else {
            return Err(JsValue::from_str("No paper detected"));
        };

        // Try GPU-accelerated perspective transform, fall back to CPU if unavailable
        let warped = match warp_perspective_gpu(&img, &corners, result_width, result_height).await {
            Ok(result) => result,
            Err(_) => {
                // GPU unavailable, fall back to CPU
                log("WebGPU unavailable, using CPU fallback");
                warp_perspective_cpu(&img, &corners, result_width, result_height)
            }
        };

        // Convert back to ImageData
        let result_data = warped.get_raw_pixels();
        ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&result_data),
            result_width,
            result_height,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to create ImageData: {:?}", e)))
    }

    /// Extract and undistort the paper with custom corners (GPU-accelerated)
    /// Returns a Promise that resolves to ImageData
    #[wasm_bindgen(js_name = extractPaperWithCornersGpu)]
    pub async fn extract_paper_with_corners_gpu(
        &self,
        image_data: ImageData,
        result_width: u32,
        result_height: u32,
        top_left_x: f32,
        top_left_y: f32,
        top_right_x: f32,
        top_right_y: f32,
        bottom_left_x: f32,
        bottom_left_y: f32,
        bottom_right_x: f32,
        bottom_right_y: f32,
    ) -> Result<ImageData, JsValue> {
        let width = image_data.width();
        let height = image_data.height();
        let data = image_data.data().0;

        // Convert to RGBA image
        let img = PhotonImage::new(data.to_vec(), width, height);

        // Use provided corner points
        let corners = CornerPoints {
            top_left: Point {
                x: top_left_x,
                y: top_left_y,
            },
            top_right: Point {
                x: top_right_x,
                y: top_right_y,
            },
            bottom_left: Point {
                x: bottom_left_x,
                y: bottom_left_y,
            },
            bottom_right: Point {
                x: bottom_right_x,
                y: bottom_right_y,
            },
        };

        // Try GPU-accelerated perspective transform, fall back to CPU if unavailable
        let warped = match warp_perspective_gpu(&img, &corners, result_width, result_height).await {
            Ok(result) => result,
            Err(_) => {
                // GPU unavailable, fall back to CPU
                log("WebGPU unavailable, using CPU fallback");
                warp_perspective_cpu(&img, &corners, result_width, result_height)
            }
        };

        // Convert back to ImageData
        let result_data = warped.get_raw_pixels();
        ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&result_data),
            result_width,
            result_height,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to create ImageData: {:?}", e)))
    }
}

/// Convert PhotonImage to grayscale for contour detection
fn photon_to_gray(photon_img: &mut PhotonImage) -> Vec<u8> {
    // Apply photon-rs grayscale conversion in-place
    monochrome::b_grayscale(photon_img);
    
    let width = photon_img.get_width();
    let height = photon_img.get_height();
    let raw_pixels = photon_img.get_raw_pixels();
    
    let mut gray = Vec::with_capacity((width * height) as usize);

    // Extract just the R channel (since grayscale, R=G=B)
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            if idx < raw_pixels.len() {
                gray.push(raw_pixels[idx]);
            }
        }
    }

    gray
}

/// Simple contour finding from edge-detected image
/// Returns points on the boundary of the largest white region
fn find_contours(gray_data: &[u8], width: u32, height: u32) -> Vec<Vec<(u32, u32)>> {
    let mut contours = Vec::new();
    let mut visited = vec![vec![false; width as usize]; height as usize];
    
    // Find edge pixels (bright pixels in edge-detected image)
    let threshold = 128u8;
    
    for y in 0..height {
        for x in 0..width {
            if visited[y as usize][x as usize] {
                continue;
            }
            
            let idx = (y * width + x) as usize;
            if idx < gray_data.len() && gray_data[idx] > threshold {
                // Start a new contour from this point
                let contour = trace_contour(gray_data, width, height, x, y, &mut visited, threshold);
                if contour.len() > 20 {  // Filter small contours
                    contours.push(contour);
                }
            }
        }
    }
    
    contours
}

/// Trace a contour from a starting point
fn trace_contour(
    gray_data: &[u8],
    width: u32,
    height: u32,
    start_x: u32,
    start_y: u32,
    visited: &mut Vec<Vec<bool>>,
    threshold: u8,
) -> Vec<(u32, u32)> {
    let mut contour = Vec::new();
    let mut stack = vec![(start_x, start_y)];
    
    while let Some((x, y)) = stack.pop() {
        if visited[y as usize][x as usize] {
            continue;
        }
        
        visited[y as usize][x as usize] = true;
        contour.push((x, y));
        
        // Check 8-connected neighbors
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nx = nx as u32;
                    let ny = ny as u32;
                    let idx = (ny * width + nx) as usize;
                    
                    if !visited[ny as usize][nx as usize] && idx < gray_data.len() && gray_data[idx] > threshold {
                        stack.push((nx, ny));
                    }
                }
            }
        }
    }
    
    contour
}

/// Find the largest contour by area, returning its index
fn find_largest_contour_index(contours: &[Vec<(u32, u32)>]) -> Option<usize> {
    if contours.is_empty() {
        return None;
    }

    let mut max_area = 0.0;
    let mut max_contour_idx = 0;

    for (idx, contour) in contours.iter().enumerate() {
        let area = contour_area(contour);
        if area > max_area {
            max_area = area;
            max_contour_idx = idx;
        }
    }

    if max_area > 0.0 {
        Some(max_contour_idx)
    } else {
        None
    }
}

/// Calculate contour area using Shoelace formula
fn contour_area(points: &[(u32, u32)]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0_f64;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        area += (points[i].0 as f64) * (points[j].1 as f64);
        area -= (points[j].0 as f64) * (points[i].1 as f64);
    }

    area.abs() / 2.0
}

/// Get corner points from a contour
fn get_corner_points(contour: &[(u32, u32)], width: u32, height: u32) -> CornerPoints {
    // Calculate center of contour
    let center_x = contour.iter().map(|p| p.0 as f32).sum::<f32>() / contour.len() as f32;
    let center_y = contour.iter().map(|p| p.1 as f32).sum::<f32>() / contour.len() as f32;

    let mut top_left = Point { x: 0.0, y: 0.0 };
    let mut top_right = Point {
        x: width as f32,
        y: 0.0,
    };
    let mut bottom_left = Point {
        x: 0.0,
        y: height as f32,
    };
    let mut bottom_right = Point {
        x: width as f32,
        y: height as f32,
    };

    let mut tl_dist = 0.0;
    let mut tr_dist = 0.0;
    let mut bl_dist = 0.0;
    let mut br_dist = 0.0;

    for point in contour {
        let px = point.0 as f32;
        let py = point.1 as f32;
        let dist = ((px - center_x).powi(2) + (py - center_y).powi(2)).sqrt();

        if px < center_x && py < center_y {
            // Top left
            if dist > tl_dist {
                top_left = Point { x: px, y: py };
                tl_dist = dist;
            }
        } else if px > center_x && py < center_y {
            // Top right
            if dist > tr_dist {
                top_right = Point { x: px, y: py };
                tr_dist = dist;
            }
        } else if px < center_x && py > center_y {
            // Bottom left
            if dist > bl_dist {
                bottom_left = Point { x: px, y: py };
                bl_dist = dist;
            }
        } else if px > center_x && py > center_y {
            // Bottom right
            if dist > br_dist {
                bottom_right = Point { x: px, y: py };
                br_dist = dist;
            }
        }
    }

    CornerPoints {
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    }
}

/// WGSL shader for perspective transformation with bilinear interpolation
const PERSPECTIVE_SHADER: &str = r#"
struct Transform {
    matrix: mat3x3<f32>,
    inv_matrix: mat3x3<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var<storage, read> src_image: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_image: array<u32>;

fn bilinear_sample(x: f32, y: f32) -> vec4<f32> {
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, transform.src_width - 1u);
    let y1 = min(y0 + 1u, transform.src_height - 1u);
    
    let dx = fract(x);
    let dy = fract(y);
    
    // Get four surrounding pixels
    let idx00 = y0 * transform.src_width + x0;
    let idx10 = y0 * transform.src_width + x1;
    let idx01 = y1 * transform.src_width + x0;
    let idx11 = y1 * transform.src_width + x1;
    
    let p00 = unpack4x8unorm(src_image[idx00]);
    let p10 = unpack4x8unorm(src_image[idx10]);
    let p01 = unpack4x8unorm(src_image[idx01]);
    let p11 = unpack4x8unorm(src_image[idx11]);
    
    // Bilinear interpolation
    let top = mix(p00, p10, dx);
    let bottom = mix(p01, p11, dx);
    return mix(top, bottom, dy);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= transform.dst_width || y >= transform.dst_height) {
        return;
    }
    
    // Apply inverse transform to find source coordinates
    let dst_pt = vec3<f32>(f32(x), f32(y), 1.0);
    let src_pt_h = transform.inv_matrix * dst_pt;
    
    if (abs(src_pt_h.z) < 0.0000001) {
        return;
    }
    
    let src_x = src_pt_h.x / src_pt_h.z;
    let src_y = src_pt_h.y / src_pt_h.z;
    
    // Check bounds
    if (src_x >= 0.0 && src_x < f32(transform.src_width - 1u) &&
        src_y >= 0.0 && src_y < f32(transform.src_height - 1u)) {
        
        let color = bilinear_sample(src_x, src_y);
        let idx = y * transform.dst_width + x;
        dst_image[idx] = pack4x8unorm(color);
    }
}
"#;

/// GPU-accelerated perspective warp using WebGPU
async fn warp_perspective_gpu(
    img: &PhotonImage,
    corners: &CornerPoints,
    dst_width: u32,
    dst_height: u32,
) -> Result<PhotonImage, JsValue> {
    // Compute transform matrices
    let src = [
        (corners.top_left.x, corners.top_left.y),
        (corners.top_right.x, corners.top_right.y),
        (corners.bottom_left.x, corners.bottom_left.y),
        (corners.bottom_right.x, corners.bottom_right.y),
    ];
    let dst = [
        (0.0, 0.0),
        (dst_width as f32, 0.0),
        (0.0, dst_height as f32),
        (dst_width as f32, dst_height as f32),
    ];
    
    let transform_matrix = compute_perspective_transform(&src, &dst);
    let inv_matrix = transform_matrix.try_inverse()
        .ok_or_else(|| JsValue::from_str("Failed to invert transform matrix"))?;
    
    // Convert matrices to f32 for GPU
    let matrix_f32: [[f32; 3]; 3] = [
        [transform_matrix[(0,0)] as f32, transform_matrix[(0,1)] as f32, transform_matrix[(0,2)] as f32],
        [transform_matrix[(1,0)] as f32, transform_matrix[(1,1)] as f32, transform_matrix[(1,2)] as f32],
        [transform_matrix[(2,0)] as f32, transform_matrix[(2,1)] as f32, transform_matrix[(2,2)] as f32],
    ];
    let inv_matrix_f32: [[f32; 3]; 3] = [
        [inv_matrix[(0,0)] as f32, inv_matrix[(0,1)] as f32, inv_matrix[(0,2)] as f32],
        [inv_matrix[(1,0)] as f32, inv_matrix[(1,1)] as f32, inv_matrix[(1,2)] as f32],
        [inv_matrix[(2,0)] as f32, inv_matrix[(2,1)] as f32, inv_matrix[(2,2)] as f32],
    ];
    
    // Prepare uniform data
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct TransformUniforms {
        matrix: [[f32; 4]; 3],      // Padded mat3x3
        inv_matrix: [[f32; 4]; 3],   // Padded mat3x3
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    }
    
    let uniforms = TransformUniforms {
        matrix: [
            [matrix_f32[0][0], matrix_f32[0][1], matrix_f32[0][2], 0.0],
            [matrix_f32[1][0], matrix_f32[1][1], matrix_f32[1][2], 0.0],
            [matrix_f32[2][0], matrix_f32[2][1], matrix_f32[2][2], 0.0],
        ],
        inv_matrix: [
            [inv_matrix_f32[0][0], inv_matrix_f32[0][1], inv_matrix_f32[0][2], 0.0],
            [inv_matrix_f32[1][0], inv_matrix_f32[1][1], inv_matrix_f32[1][2], 0.0],
            [inv_matrix_f32[2][0], inv_matrix_f32[2][1], inv_matrix_f32[2][2], 0.0],
        ],
        src_width: img.get_width(),
        src_height: img.get_height(),
        dst_width,
        dst_height,
    };
    
    // Convert RGBA bytes to packed u32
    let src_pixels: Vec<u32> = img.get_raw_pixels()
        .chunks_exact(4)
        .map(|rgba| u32::from_le_bytes([rgba[0], rgba[1], rgba[2], rgba[3]]))
        .collect();

    // Ensure GPU context (adapter/device/pipeline) is initialized and cached
    if GPU_CONTEXT.with(|c| c.borrow().is_none()) {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| JsValue::from_str("Failed to find GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Perspective Transform Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to create device: {:?}", e)))?;

        // Create common pipeline pieces and cache them
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Perspective Transform Shader"),
            source: wgpu::ShaderSource::Wgsl(PERSPECTIVE_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Perspective Transform Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        GPU_CONTEXT.with(|c| {
            *c.borrow_mut() = Some(GpuContext {
                device,
                queue,
                shader,
                bind_group_layout,
                pipeline: compute_pipeline,
            });
        });
    }

    // Create buffers, encode and submit within a borrow scope so we don't hold a RefCell borrow across awaits
    let output_buffer = GPU_CONTEXT.with(|c| {
        let ctx_opt = c.borrow();
        let ctx = ctx_opt.as_ref().expect("GPU context should be initialized");
        let device = &ctx.device;
        let queue = &ctx.queue;
        let compute_pipeline = &ctx.pipeline;
        let bind_group_layout = &ctx.bind_group_layout;

        let dst_pixels = vec![0u32; (dst_width * dst_height) as usize];

        // Create buffers
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let src_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Source Image Buffer"),
            contents: bytemuck::cast_slice(&src_pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dst_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Destination Image Buffer"),
            contents: bytemuck::cast_slice(&dst_pixels),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (dst_width * dst_height * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Perspective Transform Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((dst_width + 7) / 8, (dst_height + 7) / 8, 1);
        }

        encoder.copy_buffer_to_buffer(&dst_buffer, 0, &output_buffer, 0, (dst_width * dst_height * 4) as u64);

        queue.submit(Some(encoder.finish()));

        output_buffer
    });

    // Read back results
    let buffer_slice = output_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // Poll the device to ensure the mapping completes promptly
    GPU_CONTEXT.with(|c| {
        if let Some(ctx) = c.borrow().as_ref() {
            ctx.device.poll(wgpu::Maintain::Wait);
        }
    });

    receiver.await
        .map_err(|_| JsValue::from_str("Failed to receive buffer mapping result"))?
        .map_err(|e| JsValue::from_str(&format!("Failed to map buffer: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let result_pixels: Vec<u8> = data.to_vec();
    drop(data);
    output_buffer.unmap();

    Ok(PhotonImage::new(result_pixels, dst_width, dst_height))
}

/// Apply perspective warp to transform the paper to a rectangle (CPU fallback)
fn warp_perspective_cpu(
    img: &PhotonImage,
    corners: &CornerPoints,
    dst_width: u32,
    dst_height: u32,
) -> PhotonImage {
    let mut result_data = vec![0u8; (dst_width * dst_height * 4) as usize];

    // Source points (detected corners)
    let src = [
        (corners.top_left.x, corners.top_left.y),
        (corners.top_right.x, corners.top_right.y),
        (corners.bottom_left.x, corners.bottom_left.y),
        (corners.bottom_right.x, corners.bottom_right.y),
    ];

    // Destination points (rectangle)
    let dst = [
        (0.0, 0.0),
        (dst_width as f32, 0.0),
        (0.0, dst_height as f32),
        (dst_width as f32, dst_height as f32),
    ];

    // Compute perspective transform matrix
    let transform = compute_perspective_transform(&src, &dst);
    // Precompute inverse transform once (avoid inverting per-pixel)
    let inv_transform = match transform.try_inverse() {
        Some(inv) => inv,
        None => return PhotonImage::new(result_data, dst_width, dst_height),
    };

    // Apply transformation
    for y in 0..dst_height {
        for x in 0..dst_width {
            // dst homogeneous point
            let dst_vec = nalgebra::Vector3::new(x as f64, y as f64, 1.0);

            // Map to source homogeneous coordinates using precomputed inverse
            let src_h = inv_transform * dst_vec;
            let w = src_h[2];
            if w.abs() < 1e-10 {
                continue;
            }

            let src_x = src_h[0] / w;
            let src_y = src_h[1] / w;

            if src_x >= 0.0
                && src_x < (img.get_width() - 1) as f64
                && src_y >= 0.0
                && src_y < (img.get_height() - 1) as f64
            {
                // Bilinear interpolation
                let pixel = bilinear_interpolate(img, src_x, src_y);
                let idx = ((y * dst_width + x) * 4) as usize;
                result_data[idx..idx + 4].copy_from_slice(&pixel);
            }
        }
    }

    PhotonImage::new(result_data, dst_width, dst_height)
}

/// Compute perspective transform matrix (3x3) using homography-rs
fn compute_perspective_transform(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> Matrix3<f64> {
    
    // Create a new instance of HomographyComputation
    let mut hc = HomographyComputation::new();
    
    // Add point correspondences
    for i in 0..4 {
        let src_pt = Point::new(src[i].0, src[i].1);
        let dst_pt = Point::new(dst[i].0, dst[i].1);
        hc.add_point_correspondence(src_pt, dst_pt);
    }
    
    // Get restrictions and compute solution
    let restrictions = hc.get_restrictions();
    let solution = restrictions.compute();
    
    // Convert Matrix3<f32> to Matrix3<f64>
    let m = solution.matrix;
    Matrix3::new(
        m[(0, 0)] as f64, m[(0, 1)] as f64, m[(0, 2)] as f64,
        m[(1, 0)] as f64, m[(1, 1)] as f64, m[(1, 2)] as f64,
        m[(2, 0)] as f64, m[(2, 1)] as f64, m[(2, 2)] as f64,
    )
}

/// Apply inverse perspective transform using nalgebra
fn apply_inverse_transform(transform: &Matrix3<f64>, dst_pt: &[f64; 3]) -> Option<[f64; 2]> {
    // Matrix inversion using nalgebra
    let inv = transform.try_inverse()?;

    // Apply inverse transformation
    let w = inv[(2, 0)] * dst_pt[0] + inv[(2, 1)] * dst_pt[1] + inv[(2, 2)] * dst_pt[2];

    if w.abs() < 1e-10 {
        return None;
    }

    Some([
        (inv[(0, 0)] * dst_pt[0] + inv[(0, 1)] * dst_pt[1] + inv[(0, 2)] * dst_pt[2]) / w,
        (inv[(1, 0)] * dst_pt[0] + inv[(1, 1)] * dst_pt[1] + inv[(1, 2)] * dst_pt[2]) / w,
    ])
}

/// Bilinear interpolation for smooth pixel sampling
fn bilinear_interpolate(img: &PhotonImage, x: f64, y: f64) -> [u8; 4] {
    let x0 = x.floor() as u32;
    let x1 = x.ceil() as u32;
    let y0 = y.floor() as u32;
    let y1 = y.ceil() as u32;

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    let width = img.get_width();
    let height = img.get_height();
    let raw_pixels = img.get_raw_pixels();
    
    // Helper to get pixel at (x, y)
    let get_pixel = |px: u32, py: u32| -> [u8; 4] {
        let idx = ((py * width + px) * 4) as usize;
        [
            raw_pixels[idx],
            raw_pixels[idx + 1],
            raw_pixels[idx + 2],
            raw_pixels[idx + 3],
        ]
    };

    // Get the four surrounding pixels
    let p00 = get_pixel(x0, y0);
    let p10 = get_pixel(x1.min(width - 1), y0);
    let p01 = get_pixel(x0, y1.min(height - 1));
    let p11 = get_pixel(x1.min(width - 1), y1.min(height - 1));

    // Bilinear interpolation formula: (1-dx)(1-dy)*p00 + dx(1-dy)*p10 + (1-dx)*dy*p01 + dx*dy*p11
    let mut result = [0u8; 4];
    for ch in 0..4 {
        let val = (1.0 - dx) * (1.0 - dy) * p00[ch] as f64
            + dx * (1.0 - dy) * p10[ch] as f64
            + (1.0 - dx) * dy * p01[ch] as f64
            + dx * dy * p11[ch] as f64;
        result[ch] = val.round().clamp(0.0, 255.0) as u8;
    }

    result
}

// Set panic hook for better error messages in WASM
mod console_error_panic_hook {
    use std::panic;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn error(s: &str);
    }

    pub fn set_once() {
        panic::set_hook(Box::new(|info| {
            error(&info.to_string());
        }));
    }
}
