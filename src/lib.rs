use imageproc::image::GrayImage;
use imageproc::image::ImageBuffer;
use photon_rs::PhotonImage;
use photon_rs::conv;
use photon_rs::monochrome;
use nalgebra::{Matrix3, SMatrix, SVector};
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};
//use kornia::image::{Image as KorniaImage, ImageSize};
// use kornia::warp::warp_perspective as kornia_warp_perspective;
// use kornia::interpolation::InterpolationMode;
//use kornia::imgproc::interpolation::InterpolationMode::Bilinear;
use imgproc::interpolation::InterpolationMode;
use std::cell::RefCell;
use std::rc::Rc;
use kornia::{image::{Image, ImageSize}, imgproc};
use kornia::io::functional as F;
use imageproc::contours::find_contours;
mod find_contours;
mod detect_document;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct Point {
    pub x: f64,
    pub y: f64,
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

        /// Extract and undistort the paper from the image
    #[wasm_bindgen(js_name = extractPaperHough)]
    pub fn extract_paper_hough(
        &self,
        image_data: ImageData,
        result_width: u32,
        result_height: u32,
    ) -> Result<ImageData, JsValue> {
        detect_document::extract_paper_hough(image_data, result_width, result_height)
    }

    #[wasm_bindgen(js_name = extractPaperHough2)]
    pub fn extract_paper_hough2(
        &self,
        image_data: ImageData,
        result_width: u32,
        result_height: u32,
    ) -> Result<ImageData, JsValue> {
        detect_document::extract_paper_hough2(image_data, result_width, result_height)
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

        log("Converting ImageData to Kornia Image");
        
        // Convert RGBA to RGB (ImageData is RGBA, kornia needs RGB)
        let rgb_data: Vec<u8> = data
            .chunks(4)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        
        // Read the image using kornia-rs
        let img = Image::<u8, 3>::new(
            ImageSize { width: width as usize, height: height as usize },
            rgb_data,
        ).map_err(|e| JsValue::from_str(&format!("Failed to create Kornia image: {:?}", e)))?;

        log("Image loaded into Kornia");
        
        // resize the image
        let new_size = ImageSize {
            width: result_width as usize,
            height: result_height as usize,
        };

        let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0).map_err(|e| JsValue::from_str(&format!("Failed to resize image: {:?}", e)))?;
        imgproc::resize::resize_fast(
            &img, &mut image_resized,
            InterpolationMode::Nearest,
        ).map_err(|e| JsValue::from_str(&format!("Failed to resize image: {:?}", e)))?;

        //let image_f32: Image<f32, 3> = img.cast_and_scale::<f32>(1.0 / 255.0).map_err(|e| JsValue::from_str(&format!("Failed to cast and scale image: {:?}", e)))?;



        // let mut gray_resized = Image::<f32, 1>::from_size_val(new_size, 0.0).map_err(|e| JsValue::from_str(&format!("Failed to create resized gray image: {:?}", e)))?;
        // imgproc::resize::resize_native(
        //     &gray, &mut gray_resized,
        //     Bilinear,
        // ).map_err(|e| JsValue::from_str(&format!("Failed to resize gray image: {:?}", e)))?;
        // convert to grayscale and scale
        // ops::cast_and_scale(&img, &mut img_f32, 1. / 255.)?;
        log("Resize done");

        // Convert to f32 for image processing
        let image_f32: Image<f32, 3> = image_resized.cast_and_scale::<f32>(1.0 / 255.0)
            .map_err(|e| JsValue::from_str(&format!("Failed to cast to f32: {:?}", e)))?;

        // convert the image to grayscale
        let mut gray = Image::<f32, 1>::from_size_val(image_f32.size(), 0.0)
            .map_err(|e| JsValue::from_str(&format!("Failed to create gray image: {:?}", e)))?;
        imgproc::color::gray_from_rgb(&image_f32, &mut gray)
            .map_err(|e| JsValue::from_str(&format!("Failed to convert to grayscale: {:?}", e)))?;

        log("GS conversion done");
        // let gray_image = GrayImage::new(
        //     width,
        //     height,
        //     // gray.as_slice()
        //     //     .ok_or_else(|| JsValue::from_str("Failed to get gray image data slice"))?
        //     //     .to_vec(),
        // );
        // let gray_image = ImageBuffer::from_raw(
        //     width,
        //     height,
        //     gray.as_slice(),
        // ).ok_or_else(|| JsValue::from_str("Failed to create GrayImage from raw data"))?;
        //let gray_image = GrayImage::from_raw(240, 360, gray.as_slice()).unwrap();
        //let contours = find_contours(&gray_image);

        // Apply Gaussian blur to reduce noise
        let mut blurred = Image::<f32, 1>::from_size_val(gray.size(), 0.0)
            .map_err(|e| JsValue::from_str(&format!("Failed to create blurred image: {:?}", e)))?;
        imgproc::filter::gaussian_blur(&gray, &mut blurred, (5, 5), (1.0, 1.0))
            .map_err(|e| JsValue::from_str(&format!("Failed to blur image: {:?}", e)))?;

        log("Blur done");

        // Apply Sobel edge detection
        let mut edges = Image::<f32, 1>::from_size_val(blurred.size(), 0.0)
            .map_err(|e| JsValue::from_str(&format!("Failed to create edges image: {:?}", e)))?;
        imgproc::filter::sobel(&blurred, &mut edges, 3)
            .map_err(|e| JsValue::from_str(&format!("Failed to apply Sobel: {:?}", e)))?;

        log("Edge detection done");

        // Get the longest edge to find the contour
        let max_edge = edges.as_slice().iter().cloned().fold(0.0, f32::max);
        log(&format!("Max edge value: {}", max_edge));

        // Threshold the edges to get binary image
        let mut binary = Image::<f32, 1>::from_size_val(edges.size(), 0.0)
            .map_err(|e| JsValue::from_str(&format!("Failed to create binary image: {:?}", e)))?;
        imgproc::threshold::threshold_binary(&edges, &mut binary, 0.1, 1.0)
            .map_err(|e| JsValue::from_str(&format!("Failed to threshold: {:?}", e)))?;

        log("Thresholding done");

        // Create an image u8
        let mut binary_u8 = Image::<u8, 1>::from_size_val(binary.size(), 0)
            .map_err(|e| JsValue::from_str(&format!("Failed to create binary_u8 image: {:?}", e)))?;

        let ct = find_contours::find_contours(&binary_u8, 128);

        // Highlight the image with it's contours
        let width = binary_u8.width();
        let height = binary_u8.height();
        for contour in ct {
            for &(x, y) in contour.points() {
                let x = x as usize;
                let y = y as usize;
                if x < width && y < height {
                    binary_u8.as_slice_mut()[y * width + x] = 255;
                }
            }
        }

        // Get the longest contour from the binary image
        //let contours = find_contours(&binary);

        //let image_viz = img.clone();

        // convert the image to grayscale
        // let image_f32: Image<f32, 3, _> = img.cast_and_scale::<f32>(1.0 / 255.0)?;
        // let mut gray = Image::<f32, 1, _>::from_size_val(image_f32.size(), 0.0)?;
        // imgproc::color::gray_from_rgb(&image_f32, &mut gray)?;


        // // Convert back to image data 
        // ImageData::new_with_u8_clamped_array_and_sh(
        //     Clamped(&gray.data.as_slice().ok_or_else(|| JsValue::from_str("Failed to get gray image data slice"))?),
        //     width,
        //     height,
        // )
        // .map_err(|e| JsValue::from_str(&format!("Failed to create ImageData: {:?}", e)))

        // let image_f32: Image<f32, 3, _> = img.cast_and_scale::<f32>(1.0 / 255.0)?;

        // // convert the image to grayscale
        // let mut gray = Image::<f32, 1, _>::from_size_val(image_f32.size(), 0.0)?;
        
        // let contours = find_contours(&gray_img, width, height);

        // let corners = if let Some(largest_contour) = find_largest_contour(&contours) {
        //     get_corner_points(&largest_contour, width, height)
        // } else {
        //     return Err(JsValue::from_str("No paper detected"));
        // };

        // // Apply perspective transform
        // let warped = warp_perspective(&img, &corners, result_width, result_height);

        // Convert grayscale back to RGBA for ImageData (using binary edges)
        let rgba_data = binary_u8
            .as_slice()
            .iter()
            .flat_map(|&gray_value| {
                let byte_value = (gray_value * 255_u8).clamp(0_u8, 255_u8);
                [byte_value, byte_value, byte_value, 255u8] // R, G, B, A
            })
            .collect::<Vec<u8>>();
        
        // Convert back to ImageData using the result dimensions
        ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&rgba_data),
            result_width,
            result_height,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to create ImageData: {:?}", e)))
    }
}

/// Simple contour finding from edge-detected image
/// Returns points on the boundary of the largest white region
// fn find_contours(gray_data: &[u8], width: u32, height: u32) -> Vec<Vec<(u32, u32)>> {
//     let mut contours = Vec::new();
//     let mut visited = vec![vec![false; width as usize]; height as usize];
    
//     // Find edge pixels (bright pixels in edge-detected image)
//     let threshold = 128u8;
    
//     for y in 0..height {
//         for x in 0..width {
//             if visited[y as usize][x as usize] {
//                 continue;
//             }
            
//             let idx = (y * width + x) as usize;
//             if idx < gray_data.len() && gray_data[idx] > threshold {
//                 // Start a new contour from this point
//                 let contour = trace_contour(gray_data, width, height, x, y, &mut visited, threshold);
//                 if contour.len() > 20 {  // Filter small contours
//                     contours.push(contour);
//                 }
//             }
//         }
//     }
    
//     contours
// }

// /// Trace a contour from a starting point
// fn trace_contour(
//     gray_data: &[u8],
//     width: u32,
//     height: u32,
//     start_x: u32,
//     start_y: u32,
//     visited: &mut Vec<Vec<bool>>,
//     threshold: u8,
// ) -> Vec<(u32, u32)> {
//     let mut contour = Vec::new();
//     let mut stack = vec![(start_x, start_y)];
    
//     while let Some((x, y)) = stack.pop() {
//         if visited[y as usize][x as usize] {
//             continue;
//         }
        
//         visited[y as usize][x as usize] = true;
//         contour.push((x, y));
        
//         // Check 8-connected neighbors
//         for dy in -1i32..=1 {
//             for dx in -1i32..=1 {
//                 if dx == 0 && dy == 0 {
//                     continue;
//                 }
                
//                 let nx = x as i32 + dx;
//                 let ny = y as i32 + dy;
                
//                 if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
//                     let nx = nx as u32;
//                     let ny = ny as u32;
//                     let idx = (ny * width + nx) as usize;
                    
//                     if !visited[ny as usize][nx as usize] && idx < gray_data.len() && gray_data[idx] > threshold {
//                         stack.push((nx, ny));
//                     }
//                 }
//             }
//         }
//     }
    
//     contour
// }

// /// Find the largest contour by area
// fn find_largest_contour(contours: &[Vec<(u32, u32)>]) -> Option<Vec<(u32, u32)>> {
//     if contours.is_empty() {
//         return None;
//     }

//     let mut max_area = 0.0;
//     let mut max_contour_idx = 0;

//     for (idx, contour) in contours.iter().enumerate() {
//         let area = contour_area(contour);
//         if area > max_area {
//             max_area = area;
//             max_contour_idx = idx;
//         }
//     }

//     if max_area > 0.0 {
//         Some(contours[max_contour_idx].clone())
//     } else {
//         None
//     }
// }

// /// Calculate contour area using Shoelace formula
// fn contour_area(points: &[(u32, u32)]) -> f64 {
//     if points.len() < 3 {
//         return 0.0;
//     }

//     let mut area = 0.0_f64;
//     for i in 0..points.len() {
//         let j = (i + 1) % points.len();
//         area += (points[i].0 as f64) * (points[j].1 as f64);
//         area -= (points[j].0 as f64) * (points[i].1 as f64);
//     }

//     area.abs() / 2.0
// }

// /// Get corner points from a contour
// fn get_corner_points(contour: &[(u32, u32)], width: u32, height: u32) -> CornerPoints {
//     // Calculate center of contour
//     let center_x = contour.iter().map(|p| p.0 as f64).sum::<f64>() / contour.len() as f64;
//     let center_y = contour.iter().map(|p| p.1 as f64).sum::<f64>() / contour.len() as f64;

//     let mut top_left = Point { x: 0.0, y: 0.0 };
//     let mut top_right = Point {
//         x: width as f64,
//         y: 0.0,
//     };
//     let mut bottom_left = Point {
//         x: 0.0,
//         y: height as f64,
//     };
//     let mut bottom_right = Point {
//         x: width as f64,
//         y: height as f64,
//     };

//     let mut tl_dist = 0.0;
//     let mut tr_dist = 0.0;
//     let mut bl_dist = 0.0;
//     let mut br_dist = 0.0;

//     for point in contour {
//         let px = point.0 as f64;
//         let py = point.1 as f64;
//         let dist = ((px - center_x).powi(2) + (py - center_y).powi(2)).sqrt();

//         if px < center_x && py < center_y {
//             // Top left
//             if dist > tl_dist {
//                 top_left = Point { x: px, y: py };
//                 tl_dist = dist;
//             }
//         } else if px > center_x && py < center_y {
//             // Top right
//             if dist > tr_dist {
//                 top_right = Point { x: px, y: py };
//                 tr_dist = dist;
//             }
//         } else if px < center_x && py > center_y {
//             // Bottom left
//             if dist > bl_dist {
//                 bottom_left = Point { x: px, y: py };
//                 bl_dist = dist;
//             }
//         } else if px > center_x && py > center_y {
//             // Bottom right
//             if dist > br_dist {
//                 bottom_right = Point { x: px, y: py };
//                 br_dist = dist;
//             }
//         }
//     }

//     CornerPoints {
//         top_left,
//         top_right,
//         bottom_left,
//         bottom_right,
//     }
// }

/// Apply perspective warp to transform the paper to a rectangle
// fn warp_perspective(
//     img: &PhotonImage,
//     corners: &CornerPoints,
//     dst_width: u32,
//     dst_height: u32,
// ) -> PhotonImage {
//     // Build source and destination point arrays for homography
//     let src_pts = [
//         (corners.top_left.x, corners.top_left.y),
//         (corners.top_right.x, corners.top_right.y),
//         (corners.bottom_left.x, corners.bottom_left.y),
//         (corners.bottom_right.x, corners.bottom_right.y),
//     ];

//     let dst_pts = [
//         (0.0, 0.0),
//         (dst_width as f64, 0.0),
//         (0.0, dst_height as f64),
//         (dst_width as f64, dst_height as f64),
//     ];

//     let transform = compute_perspective_transform_cpu(&src_pts, &dst_pts);
    
//     // Convert transform to row-major f32 array expected by kornia-rs
//     let m: [f32; 9] = [
//         transform[(0, 0)] as f32,
//         transform[(0, 1)] as f32,
//         transform[(0, 2)] as f32,
//         transform[(1, 0)] as f32,
//         transform[(1, 1)] as f32,
//         transform[(1, 2)] as f32,
//         transform[(2, 0)] as f32,
//         transform[(2, 1)] as f32,
//         transform[(2, 2)] as f32,
//     ];

//     // Convert PhotonImage (u8 RGBA) to Kornia Image<f32, 4>
//     let src_w = img.get_width();
//     let src_h = img.get_height();
//     let raw = img.get_raw_pixels();
//     let src_f32: Vec<f32> = raw.iter().map(|&b| b as f32).collect();
//     let src_size = ImageSize { width: src_w as usize, height: src_h as usize };
//     let src_img = KorniaImage::<f32, 4>::new(src_size, src_f32).expect("Failed to create Kornia image");

//     // Destination size
//     let dst_size = ImageSize { width: dst_width as usize, height: dst_height as usize };

//     // Perform warp using kornia-rs
//     let dst_img = kornia_warp_perspective(&src_img, m, dst_size, InterpolationMode::Bilinear)
//         .expect("kornia warp_perspective failed");

//     // Convert resulting Kornia image (f32) back to u8 RGBA
//     // Extract raw f32 data from the Kornia image via the public ndarray storage
//     let dst_data_f32: Vec<f32> = dst_img
//         .data
//         .as_slice()
//         .expect("Failed to get dst image data slice")
//         .to_vec();

//     let mut dst_bytes = Vec::with_capacity((dst_width * dst_height * 4) as usize);
//     for val in dst_data_f32 {
//         let b = (val.clamp(0.0, 255.0).round()) as u8;
//         dst_bytes.push(b);
//     }

//     PhotonImage::new(dst_bytes, dst_width, dst_height)
// }

/// GPU-accelerated perspective warp using wgpu for matrix computation
// async fn warp_perspective_gpu(
//     img: &PhotonImage,
//     corners: &CornerPoints,
//     dst_width: u32,
//     dst_height: u32,
// ) -> Result<PhotonImage, JsValue> {
//     // Build source and destination point arrays for homography
//     let src_pts = [
//         (corners.top_left.x, corners.top_left.y),
//         (corners.top_right.x, corners.top_right.y),
//         (corners.bottom_left.x, corners.bottom_left.y),
//         (corners.bottom_right.x, corners.bottom_right.y),
//     ];

//     let dst_pts = [
//         (0.0, 0.0),
//         (dst_width as f64, 0.0),
//         (0.0, dst_height as f64),
//         (dst_width as f64, dst_height as f64),
//     ];

//     // Compute homography matrix using GPU if available
//     log("Computing perspective transform matrix (GPU)...");
    
//     // Clone the Rc<GpuContext> to use outside the borrow scope
//     let gpu_ctx_opt = GPU_CONTEXT.with(|ctx| ctx.borrow().clone());
    
//     let transform = if let Some(gpu_ctx) = gpu_ctx_opt {
//         log("Using GPU-accelerated perspective transform computation");
//         match gpu_ctx.compute_perspective_transform(&src_pts, &dst_pts).await {
//             Ok(mat) => {
//                 log("GPU computation successful");
//                 mat
//             }
//             Err(e) => {
//                 log(&format!("GPU computation failed: {:?}, falling back to CPU", e));
//                 compute_perspective_transform_cpu(&src_pts, &dst_pts)
//             }
//         }
//     } else {
//         log("GPU not initialized, using CPU");
//         compute_perspective_transform_cpu(&src_pts, &dst_pts)
//     };
//     log("Perspective transform matrix computed.");
    
//     // Convert transform to row-major f32 array expected by kornia-rs
//     let m: [f32; 9] = [
//         transform[(0, 0)] as f32,
//         transform[(0, 1)] as f32,
//         transform[(0, 2)] as f32,
//         transform[(1, 0)] as f32,
//         transform[(1, 1)] as f32,
//         transform[(1, 2)] as f32,
//         transform[(2, 0)] as f32,
//         transform[(2, 1)] as f32,
//         transform[(2, 2)] as f32,
//     ];

//     // Convert PhotonImage (u8 RGBA) to Kornia Image<f32, 4>
//     let src_w = img.get_width();
//     let src_h = img.get_height();
//     let raw = img.get_raw_pixels();
//     let src_f32: Vec<f32> = raw.iter().map(|&b| b as f32).collect();
//     let src_size = ImageSize { width: src_w as usize, height: src_h as usize };
//     let src_img = KorniaImage::<f32, 4>::new(src_size, src_f32)
//         .map_err(|e| JsValue::from_str(&format!("Failed to create Kornia image: {:?}", e)))?;

//     // Destination size
//     let dst_size = ImageSize { width: dst_width as usize, height: dst_height as usize };

//     // Perform warp using kornia-rs
//     let dst_img = kornia_warp_perspective(&src_img, m, dst_size, InterpolationMode::Bilinear)
//         .map_err(|e| JsValue::from_str(&format!("kornia warp_perspective failed: {:?}", e)))?;

//     // Convert resulting Kornia image (f32) back to u8 RGBA
//     let dst_data_f32: Vec<f32> = dst_img
//         .data
//         .as_slice()
//         .ok_or_else(|| JsValue::from_str("Failed to get dst image data slice"))?
//         .to_vec();

//     let mut dst_bytes = Vec::with_capacity((dst_width * dst_height * 4) as usize);
//     for val in dst_data_f32 {
//         let b = (val.clamp(0.0, 255.0).round()) as u8;
//         dst_bytes.push(b);
//     }

//     Ok(PhotonImage::new(dst_bytes, dst_width, dst_height))
// }

/// Compute perspective transform matrix (3x3) using nalgebra (CPU version)
/// Uses Direct Linear Transform (DLT) algorithm for homography estimation
// fn compute_perspective_transform_cpu(src: &[(f64, f64); 4], dst: &[(f64, f64); 4]) -> Matrix3<f64> {
//     // Build linear system: A * h = b
//     // We need to solve for the 8 parameters of the homography matrix
//     let mut a = SMatrix::<f64, 8, 8>::zeros();
//     let mut b = SVector::<f64, 8>::zeros();

//     for i in 0..4 {
//         let (sx, sy) = src[i];
//         let (dx, dy) = dst[i];

//         // Row for x coordinate
//         let row_idx = i * 2;
//         a.row_mut(row_idx).copy_from_slice(&[sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy]);
//         b[row_idx] = dx;

//         // Row for y coordinate  
//         let row_idx = i * 2 + 1;
//         a.row_mut(row_idx).copy_from_slice(&[0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy]);
//         b[row_idx] = dy;
//     }

//     // Solve linear system using nalgebra's LU decomposition
//     let lu = a.lu();
//     let h = lu.solve(&b).expect("Failed to compute perspective transform: singular matrix");

//     // Construct 3x3 transformation matrix
//     Matrix3::new(
//         h[0], h[1], h[2],
//         h[3], h[4], h[5],
//         h[6], h[7], 1.0,
//     )
// }

/// Apply inverse perspective transform using nalgebra
// fn apply_inverse_transform(transform: &Matrix3<f64>, dst_pt: &[f64; 3]) -> Option<[f64; 2]> {
//     // Matrix inversion using nalgebra
//     let inv = transform.try_inverse()?;

//     // Apply inverse transformation
//     let w = inv[(2, 0)] * dst_pt[0] + inv[(2, 1)] * dst_pt[1] + inv[(2, 2)] * dst_pt[2];

//     if w.abs() < 1e-10 {
//         return None;
//     }

//     Some([
//         (inv[(0, 0)] * dst_pt[0] + inv[(0, 1)] * dst_pt[1] + inv[(0, 2)] * dst_pt[2]) / w,
//         (inv[(1, 0)] * dst_pt[0] + inv[(1, 1)] * dst_pt[1] + inv[(1, 2)] * dst_pt[2]) / w,
//     ])
// }

/// Bilinear interpolation for smooth pixel sampling
// fn bilinear_interpolate(img: &PhotonImage, x: f64, y: f64) -> [u8; 4] {
//     let x0 = x.floor() as u32;
//     let x1 = x.ceil() as u32;
//     let y0 = y.floor() as u32;
//     let y1 = y.ceil() as u32;

//     let dx = x - x0 as f64;
//     let dy = y - y0 as f64;

//     let width = img.get_width();
//     let height = img.get_height();
//     let raw_pixels = img.get_raw_pixels();
    
//     // Helper to get pixel at (x, y)
//     let get_pixel = |px: u32, py: u32| -> [u8; 4] {
//         let idx = ((py * width + px) * 4) as usize;
//         [
//             raw_pixels[idx],
//             raw_pixels[idx + 1],
//             raw_pixels[idx + 2],
//             raw_pixels[idx + 3],
//         ]
//     };

//     // Get the four surrounding pixels
//     let p00 = get_pixel(x0, y0);
//     let p10 = get_pixel(x1.min(width - 1), y0);
//     let p01 = get_pixel(x0, y1.min(height - 1));
//     let p11 = get_pixel(x1.min(width - 1), y1.min(height - 1));

//     // Bilinear interpolation formula: (1-dx)(1-dy)*p00 + dx(1-dy)*p10 + (1-dx)*dy*p01 + dx*dy*p11
//     let mut result = [0u8; 4];
//     for ch in 0..4 {
//         let val = (1.0 - dx) * (1.0 - dy) * p00[ch] as f64
//             + dx * (1.0 - dy) * p10[ch] as f64
//             + (1.0 - dx) * dy * p01[ch] as f64
//             + dx * dy * p11[ch] as f64;
//         result[ch] = val.round().clamp(0.0, 255.0) as u8;
//     }

//     result
// }

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
