use imageproc::image::{GrayImage, RgbImage, Rgb};
use imageproc::image;
use imageproc::edges::canny;
use imageproc::hough::{detect_lines, LineDetectionOptions, PolarLine};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use imageproc::rect::Rect;
use imageproc::drawing::draw_polygon_mut;
use std::f32::consts::PI;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::ImageData;

/// Represents a corner point in 2D space
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// Represents a line in image space (converted from polar form)
#[derive(Debug, Clone, Copy)]
struct Line {
    // Line equation: ax + by + c = 0
    a: f32,
    b: f32,
    c: f32,
}

impl Line {
    /// Create a line from polar coordinates (r, theta)
    fn from_polar(polar: &PolarLine) -> Self {
        let r = polar.r as f32;
        let theta = polar.angle_in_degrees as f32 * PI / 180.0;
        
        // Line equation in polar form: x*cos(theta) + y*sin(theta) = r
        // Converting to ax + by + c = 0 form:
        // cos(theta)*x + sin(theta)*y - r = 0
        Line {
            a: theta.cos(),
            b: theta.sin(),
            c: -r,
        }
    }
    
    /// Compute intersection point with another line
    fn intersect(&self, other: &Line) -> Option<Point> {
        let det = self.a * other.b - self.b * other.a;
        
        // Lines are parallel or coincident
        if det.abs() < 1e-6 {
            return None;
        }
        
        let x = (self.b * other.c - self.c * other.b) / det;
        let y = (self.c * other.a - self.a * other.c) / det;
        
        Some(Point { x, y })
    }
    
    /// Calculate angle between two lines in degrees
    fn angle_with(&self, other: &Line) -> f32 {
        // Use dot product formula: cos(angle) = (a1*a2 + b1*b2) / (|v1| * |v2|)
        let dot = self.a * other.a + self.b * other.b;
        let mag1 = (self.a * self.a + self.b * self.b).sqrt();
        let mag2 = (other.a * other.a + other.b * other.b).sqrt();
        
        let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
        cos_angle.acos() * 180.0 / PI
    }
}

/// Represents a quadrilateral formed by 4 corner points
#[derive(Debug, Clone)]
pub struct Quadrilateral {
    pub corners: [Point; 4],
    pub score: f32,
}

impl Quadrilateral {
    /// Create a new quadrilateral and compute its score based on edge strength
    fn new(corners: [Point; 4], edge_image: &GrayImage) -> Self {
        let score = Self::compute_score(&corners, edge_image);
        Quadrilateral { corners, score }
    }
    
    /// Score quadrilateral by summing edge strength along its perimeter
    fn compute_score(corners: &[Point; 4], edge_image: &GrayImage) -> f32 {
        let width = edge_image.width() as f32;
        let height = edge_image.height() as f32;
        let mut total_score = 0.0;
        let mut sample_count = 0;
        
        // Sample points along each edge of the quadrilateral
        for i in 0..4 {
            let p1 = corners[i];
            let p2 = corners[(i + 1) % 4];
            
            // Calculate edge length
            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let length = (dx * dx + dy * dy).sqrt();
            
            // Sample points along the edge (every pixel approximately)
            let num_samples = length.ceil() as usize;
            
            for j in 0..num_samples {
                let t = j as f32 / num_samples as f32;
                let x = p1.x + t * dx;
                let y = p1.y + t * dy;
                
                // Check if point is within image bounds
                if x >= 0.0 && x < width && y >= 0.0 && y < height {
                    let pixel = edge_image.get_pixel(x as u32, y as u32);
                    total_score += pixel[0] as f32;
                    sample_count += 1;
                }
            }
        }
        
        // Return average edge strength
        if sample_count > 0 {
            total_score / sample_count as f32
        } else {
            0.0
        }
    }
    
    /// Check if quadrilateral is valid (non-self-intersecting, reasonable area)
    fn is_valid(&self, min_area: f32) -> bool {
        let area = self.area();
        area > min_area && self.is_convex()
    }
    
    /// Calculate area using Shoelace formula
    fn area(&self) -> f32 {
        let mut area = 0.0;
        for i in 0..4 {
            let j = (i + 1) % 4;
            area += self.corners[i].x * self.corners[j].y;
            area -= self.corners[j].x * self.corners[i].y;
        }
        (area / 2.0).abs()
    }
    
    /// Check if quadrilateral is convex
    fn is_convex(&self) -> bool {
        // A quadrilateral is convex if all cross products have the same sign
        let mut sign = 0;
        
        for i in 0..4 {
            let p1 = self.corners[i];
            let p2 = self.corners[(i + 1) % 4];
            let p3 = self.corners[(i + 2) % 4];
            
            let dx1 = p2.x - p1.x;
            let dy1 = p2.y - p1.y;
            let dx2 = p3.x - p2.x;
            let dy2 = p3.y - p2.y;
            
            let cross = dx1 * dy2 - dy1 * dx2;
            
            if cross.abs() < 1e-6 {
                continue; // Ignore nearly collinear points
            }
            
            let current_sign = if cross > 0.0 { 1 } else { -1 };
            
            if sign == 0 {
                sign = current_sign;
            } else if sign != current_sign {
                return false; // Cross products have different signs
            }
        }
        
        true
    }
}

/// Detect document edges using Hough transform approach
/// 
/// This implements the algorithm described in the Dropbox blog post:
/// https://dropbox.tech/machine-learning/fast-and-accurate-document-detection-for-scanning
/// 
/// Steps:
/// 1. Load and preprocess image (grayscale, blur)
/// 2. Edge detection (Canny)
/// 3. Hough transform to detect lines
/// 4. Compute line intersections as potential corners
/// 5. Score all possible quadrilaterals
/// 6. Return the best quadrilateral
pub fn detect_document_hough(image_path: &str) -> Result<Quadrilateral, String> {
    // Load image
    let img = image::open(image_path)
        .map_err(|e| format!("Failed to load image: {}", e))?;
    
    // Convert to grayscale
    let gray = img.to_luma8();
    let img_width = gray.width();
    let img_height = gray.height();
    
    println!("Image loaded: {}x{}", img_width, img_height);
    
    // Apply Gaussian blur to reduce noise
    let blurred = imageproc::filter::gaussian_blur_f32(&gray, 2.0);
    
    // Apply Canny edge detection
    // Using moderate thresholds to get strong edges
    let edges = canny(&blurred, 50.0, 150.0);
    
    println!("Edge detection complete");
    
    // Detect lines using Hough transform
    let options = LineDetectionOptions {
        vote_threshold: 100,        // Minimum votes for a line to be detected
        suppression_radius: 20,     // Suppress nearby lines in Hough space
    };
    
    let detected_lines = detect_lines(&edges, options);
    println!("Detected {} lines", detected_lines.len());
    
    if detected_lines.len() < 4 {
        return Err(format!("Not enough lines detected: {}", detected_lines.len()));
    }
    
    // Convert polar lines to Cartesian form
    let lines: Vec<Line> = detected_lines.iter()
        .map(|pl| Line::from_polar(pl))
        .collect();
    
    // Find all line intersections with geometric constraints
    let mut corners = Vec::new();
    
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            if let Some(point) = lines[i].intersect(&lines[j]) {
                // Check if intersection is within image bounds (with some margin)
                let margin = -50.0; // Allow points slightly outside
                if point.x >= margin && point.x <= (img_width as f32 + margin) 
                    && point.y >= margin && point.y <= (img_height as f32 + margin) {
                    
                    // Filter out very acute angles (likely false positives)
                    let angle = lines[i].angle_with(&lines[j]);
                    let angle_diff = (angle - 90.0).abs();
                    
                    // Only keep intersections with angles between 30 and 150 degrees
                    // (i.e., angle difference from 90 degrees is less than 60)
                    if angle_diff < 60.0 {
                        corners.push(point);
                    }
                }
            }
        }
    }
    
    println!("Found {} potential corners", corners.len());
    
    if corners.len() < 4 {
        return Err(format!("Not enough corners found: {}", corners.len()));
    }
    
    // Generate and score all possible quadrilaterals
    let min_area = (img_width * img_height) as f32 * 0.1; // At least 10% of image area
    let mut best_quad: Option<Quadrilateral> = None;
    let mut max_score = 0.0;
    
    // Enumerate quadrilaterals (this can be expensive with many corners)
    // Limit to top N corners or use a more efficient algorithm for production
    let max_corners = corners.len().min(15); // Limit to avoid combinatorial explosion
    
    for i in 0..max_corners {
        for j in (i + 1)..max_corners {
            for k in (j + 1)..max_corners {
                for l in (k + 1)..max_corners {
                    let quad_corners = [corners[i], corners[j], corners[k], corners[l]];
                    
                    // Try to order corners to form a proper quadrilateral
                    if let Some(ordered) = order_corners(&quad_corners) {
                        let quad = Quadrilateral::new(ordered, &edges);
                        
                        if quad.is_valid(min_area) && quad.score > max_score {
                            max_score = quad.score;
                            best_quad = Some(quad);
                        }
                    }
                }
            }
        }
    }
    
    best_quad.ok_or_else(|| "No valid quadrilateral found".to_string())
}

/// Order 4 points to form a proper quadrilateral
/// Orders points as: top-left, top-right, bottom-right, bottom-left
fn order_corners(points: &[Point; 4]) -> Option<[Point; 4]> {
    // Calculate centroid
    let cx = points.iter().map(|p| p.x).sum::<f32>() / 4.0;
    let cy = points.iter().map(|p| p.y).sum::<f32>() / 4.0;
    
    // Classify points by quadrant relative to centroid
    let mut top_left = None;
    let mut top_right = None;
    let mut bottom_left = None;
    let mut bottom_right = None;
    
    for &point in points {
        if point.x < cx && point.y < cy {
            top_left = Some(point);
        } else if point.x >= cx && point.y < cy {
            top_right = Some(point);
        } else if point.x < cx && point.y >= cy {
            bottom_left = Some(point);
        } else {
            bottom_right = Some(point);
        }
    }
    
    // Check if we have one point in each quadrant
    if let (Some(tl), Some(tr), Some(bl), Some(br)) = (top_left, top_right, bottom_left, bottom_right) {
        Some([tl, tr, br, bl])
    } else {
        None
    }
}

/// Example usage and test function
#[cfg(test)]
mod tests {
    use super::*;
    use imageproc::drawing::draw_filled_rect_mut;
    use imageproc::rect::Rect;
    use image::Rgb;

    #[test]
    fn test_line_intersection() {
        // Test perpendicular lines
        let line1 = Line { a: 1.0, b: 0.0, c: -5.0 }; // Vertical line at x=5
        let line2 = Line { a: 0.0, b: 1.0, c: -3.0 }; // Horizontal line at y=3
        
        let intersection = line1.intersect(&line2).unwrap();
        assert!((intersection.x - 5.0).abs() < 0.001);
        assert!((intersection.y - 3.0).abs() < 0.001);
    }
    
    #[test]
    fn test_quadrilateral_area() {
        // Test a simple 10x10 square
        let quad = Quadrilateral {
            corners: [
                Point { x: 0.0, y: 0.0 },
                Point { x: 10.0, y: 0.0 },
                Point { x: 10.0, y: 10.0 },
                Point { x: 0.0, y: 10.0 },
            ],
            score: 0.0,
        };
        
        assert!((quad.area() - 100.0).abs() < 0.001);
        assert!(quad.is_convex());
    }

    #[test]
    fn test_detect_and_warp_simple_rectangle() {
        // 1. Create a synthetic image with a rectangle
        let img_w = 300;
        let img_h = 400;
        let mut image = RgbImage::new(img_w, img_h); // Black background

        // Define the rectangle
        let rect_x = 50;
        let rect_y = 70;
        let rect_w = 150;
        let rect_h = 200;
        let rect = Rect::at(rect_x, rect_y).of_size(rect_w, rect_h);
        
        // Draw a white filled rectangle
        let white = Rgb([255u8, 255u8, 255u8]);
        draw_filled_rect_mut(&mut image, rect, white);

        // 2. Define expected output dimensions
        let out_w = rect_w;
        let out_h = rect_h;

        // 3. Call the function
        let result = detect_and_warp_document(&image, out_w, out_h);

        // 4. Assert the results
        assert!(result.is_ok(), "Function should return Ok, but got Err: {:?}", result.err());
        let warped_image = result.unwrap();

        assert_eq!(warped_image.width(), out_w, "Warped image width should match");
        assert_eq!(warped_image.height(), out_h, "Warped image height should match");

        // 5. Check the content of the warped image
        // It should be almost entirely white.
        let mut non_white_pixels = 0;
        for pixel in warped_image.pixels() {
            // Allow for some minor interpolation artifacts at the edges
            if pixel[0] < 250 || pixel[1] < 250 || pixel[2] < 250 {
                non_white_pixels += 1;
            }
        }

        // Allow a small percentage of non-white pixels due to anti-aliasing/interpolation
        let total_pixels = (out_w * out_h) as f32;
        let non_white_ratio = non_white_pixels as f32 / total_pixels;
        
        assert!(
            non_white_ratio < 0.05,
            "The warped image should be mostly white, but {}% of pixels were not.",
            non_white_ratio * 100.0
        );
    }
}

/// Main function to demonstrate usage
/// 
/// Example:
/// ```no_run
/// use lib_2::detect_document_hough;
/// 
/// let result = detect_document_hough("test_image.jpg");
/// match result {
///     Ok(quad) => {
///         println!("Document detected!");
///         println!("Corners: {:?}", quad.corners);
///         println!("Score: {}", quad.score);
///     }
///     Err(e) => println!("Error: {}", e),
/// }
/// ```
pub fn example_usage() {
    // This function demonstrates how to use the document detection
    let test_image = "test_document.jpg";
    
    match detect_document_hough(test_image) {
        Ok(quad) => {
            println!("Document detected successfully!");
            println!("Top-left corner: ({}, {})", quad.corners[0].x, quad.corners[0].y);
            println!("Top-right corner: ({}, {})", quad.corners[1].x, quad.corners[1].y);
            println!("Bottom-right corner: ({}, {})", quad.corners[2].x, quad.corners[2].y);
            println!("Bottom-left corner: ({}, {})", quad.corners[3].x, quad.corners[3].y);
            println!("Detection score: {}", quad.score);
            println!("Area: {} pixels", quad.area());
        }
        Err(e) => {
            eprintln!("Failed to detect document: {}", e);
        }
    }
}

/// WASM-compatible function to detect document from ImageData and return highlighted quadrilateral
/// 
/// This function:
/// 1. Converts ImageData (RGBA) to grayscale imageproc image
/// 2. Detects document using Hough transform
/// 3. Draws the detected quadrilateral on the original image
/// 4. Returns the result as ImageData for use with WASM/JS
pub fn extract_paper_hough(
    image_data: ImageData,
    result_width: u32,
    result_height: u32,
) -> Result<ImageData, JsValue> {
    // Get image dimensions
    let width = image_data.width();
    let height = image_data.height();
    let data = image_data.data().0;
    
    // Convert RGBA ImageData to RGB (skip alpha channel)
    let rgb_data: Vec<u8> = data
        .chunks(4)
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    
    // Create RGB image from the data
    let rgb_image = RgbImage::from_raw(width, height, rgb_data)
        .ok_or_else(|| JsValue::from_str("Failed to create RGB image from ImageData"))?;
    
    // Convert to grayscale for processing
    let gray = imageproc::image::imageops::grayscale(&rgb_image);
    
    // Apply Gaussian blur to reduce noise
    let blurred = imageproc::filter::gaussian_blur_f32(&gray, 2.0);
    
    // Apply Canny edge detection
    let edges = canny(&blurred, 50.0, 150.0);
    
    // Detect lines using Hough transform
    let options = LineDetectionOptions {
        vote_threshold: 80,         // Lower threshold for more sensitivity
        suppression_radius: 15,     // Suppress nearby lines in Hough space
    };
    
    let detected_lines = detect_lines(&edges, options);
    
    if detected_lines.len() < 4 {
        return Err(JsValue::from_str(&format!("Not enough lines detected: {}", detected_lines.len())));
    }
    
    // Convert polar lines to Cartesian form
    let lines: Vec<Line> = detected_lines.iter()
        .map(|pl| Line::from_polar(pl))
        .collect();
    
    // Find all line intersections with geometric constraints
    let mut corners = Vec::new();
    
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            if let Some(point) = lines[i].intersect(&lines[j]) {
                // Check if intersection is within image bounds (with some margin)
                let margin = -50.0;
                if point.x >= margin && point.x <= (width as f32 + margin) 
                    && point.y >= margin && point.y <= (height as f32 + margin) {
                    
                    // Filter out very acute angles
                    let angle = lines[i].angle_with(&lines[j]);
                    let angle_diff = (angle - 90.0).abs();
                    
                    if angle_diff < 60.0 {
                        corners.push(point);
                    }
                }
            }
        }
    }
    
    if corners.len() < 4 {
        return Err(JsValue::from_str(&format!("Not enough corners found: {}", corners.len())));
    }
    
    // Generate and score all possible quadrilaterals
    let min_area = (width * height) as f32 * 0.05; // At least 5% of image area
    let mut best_quad: Option<Quadrilateral> = None;
    let mut max_score = 0.0;
    
    let max_corners = corners.len().min(12);
    
    for i in 0..max_corners {
        for j in (i + 1)..max_corners {
            for k in (j + 1)..max_corners {
                for l in (k + 1)..max_corners {
                    let quad_corners = [corners[i], corners[j], corners[k], corners[l]];
                    
                    if let Some(ordered) = order_corners(&quad_corners) {
                        let quad = Quadrilateral::new(ordered, &edges);
                        
                        if quad.is_valid(min_area) && quad.score > max_score {
                            max_score = quad.score;
                            best_quad = Some(quad);
                        }
                    }
                }
            }
        }
    }
    
    let quad = best_quad.ok_or_else(|| JsValue::from_str("No valid quadrilateral found"))?;
    
    // Create output image with highlighted quadrilateral
    let mut output_image = rgb_image.clone();
    
    // Draw the quadrilateral on the image
    let polygon_points: Vec<imageproc::point::Point<i32>> = quad.corners
        .iter()
        .map(|p| imageproc::point::Point::new(p.x as i32, p.y as i32))
        .collect();

    // Log detected corners for debugging   
    web_sys::console::log_1(&format!(
        "Detected corners: TL({:.1},{:.1}) TR({:.1},{:.1}) BR({:.1},{:.1}) BL({:.1},{:.1})",
        quad.corners[0].x, quad.corners[0].y,
        quad.corners[1].x, quad.corners[1].y,
        quad.corners[2].x, quad.corners[2].y,
        quad.corners[3].x, quad.corners[3].y
    ).into());
    
    // Draw the quadrilateral edges in green
    let green = imageproc::image::Rgb([0u8, 255u8, 0u8]);
    for i in 0..4 {
        let p1 = polygon_points[i];
        let p2 = polygon_points[(i + 1) % 4];
        imageproc::drawing::draw_line_segment_mut(
            &mut output_image,
            (p1.x as f32, p1.y as f32),
            (p2.x as f32, p2.y as f32),
            green,
        );
    }
    
    // Draw corner points in red
    let red = imageproc::image::Rgb([255u8, 0u8, 0u8]);
    for point in &quad.corners {
        imageproc::drawing::draw_filled_circle_mut(
            &mut output_image,
            (point.x as i32, point.y as i32),
            5,
            red,
        );
    }
    
    // Resize output image to desired dimensions if different
    let final_image = if width != result_width || height != result_height {
        imageproc::image::imageops::resize(
            &output_image,
            result_width,
            result_height,
            imageproc::image::imageops::FilterType::Lanczos3,
        )
    } else {
        output_image
    };
    
    // Convert RGB back to RGBA for ImageData
    let rgba_data: Vec<u8> = final_image
        .pixels()
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255u8])
        .collect();
    
    // Create and return ImageData
    ImageData::new_with_u8_clamped_array_and_sh(
        Clamped(&rgba_data),
        result_width,
        result_height,
    )
    .map_err(|e| JsValue::from_str(&format!("Failed to create ImageData: {:?}", e)))
}

/// Compute perspective transform matrix for warping using homography crate
/// Maps source quadrilateral to destination rectangle
fn compute_homography_matrix(src_corners: &[Point; 4], dst_width: f32, dst_height: f32) -> Projection {
    // Create homography computation instance
    let mut hc = homography::HomographyComputation::<f32>::new();
    
    // Source points (document corners in original image)
    let src_points = [
        homography::geo::Point::new(src_corners[0].x, src_corners[0].y), // top-left
        homography::geo::Point::new(src_corners[1].x, src_corners[1].y), // top-right
        homography::geo::Point::new(src_corners[2].x, src_corners[2].y), // bottom-right
        homography::geo::Point::new(src_corners[3].x, src_corners[3].y), // bottom-left
    ];
    
    // Destination points (corners of output rectangle)
    let dst_points = [
        homography::geo::Point::new(0.0, 0.0),                          // top-left
        homography::geo::Point::new(dst_width, 0.0),             // top-right
        homography::geo::Point::new(dst_width, dst_height), // bottom-right
        homography::geo::Point::new(0.0, dst_height),            // bottom-left
    ];
    
    // Add point correspondences (from dst to src for inverse transform)
    // warp_into needs the inverse transformation (from dst to src)
    for i in 0..4 {
        hc.add_point_correspondence(dst_points[i].clone(), src_points[i].clone());
    }
    
    // Get restrictions and compute the homography solution
    let restrictions = hc.get_restrictions();
    let solution = restrictions.compute();
    
    // Extract the 3x3 matrix from the solution
    //let h_matrix = solution.matrix;
    let h_matrix = solution.matrix.try_inverse().expect("Homography matrix is not invertible");
    
    // Convert nalgebra Matrix3 to row-major f32 array for Projection
    let matrix = [
        h_matrix[(0, 0)] as f32, h_matrix[(0, 1)] as f32, h_matrix[(0, 2)] as f32,
        h_matrix[(1, 0)] as f32, h_matrix[(1, 1)] as f32, h_matrix[(1, 2)] as f32,
        h_matrix[(2, 0)] as f32, h_matrix[(2, 1)] as f32, h_matrix[(2, 2)] as f32,
    ];
    // let matrix_s = h_matrix.as_slice();
    // let mut array: [f32; 9] = [0.0; 9]; // Initialize with default values
    // array.copy_from_slice(matrix_s);
    //
    // web_sys::console::log_1(&format!(
    //     "Homography matrix: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.6}, {:.6}, {:.3}]",
    //     matrix[0], matrix[1], matrix[2],
    //     matrix[3], matrix[4], matrix[5],
    //     matrix[6], matrix[7], matrix[8]
    // ).into());
    
    Projection::from_matrix(matrix).expect("Failed to create projection")
}

/// Core function to detect document and warp it to a rectangle
/// 
/// This function:
/// 1. Detects document using Hough transform
/// 2. Warps the detected quadrilateral to a rectangle using perspective transform
/// 3. Returns the warped RgbImage
pub fn detect_and_warp_document(
    rgb_image: &RgbImage,
    result_width: u32,
    result_height: u32,
) -> Result<RgbImage, String> {
    let width = rgb_image.width();
    let height = rgb_image.height();
    
    // Convert to grayscale for processing
    let gray = imageproc::image::imageops::grayscale(rgb_image);
    
    // Apply Gaussian blur to reduce noise
    let blurred = imageproc::filter::gaussian_blur_f32(&gray, 2.0);
    
    // Apply Canny edge detection
    let edges = canny(&blurred, 50.0, 150.0);
    
    // Detect lines using Hough transform
    let options = LineDetectionOptions {
        vote_threshold: 80,         // Lower threshold for more sensitivity
        suppression_radius: 15,     // Suppress nearby lines in Hough space
    };
    
    let detected_lines = detect_lines(&edges, options);
    
    if detected_lines.len() < 4 {
        return Err(format!("Not enough lines detected: {}", detected_lines.len()));
    }
    
    // Convert polar lines to Cartesian form
    let lines: Vec<Line> = detected_lines.iter()
        .map(|pl| Line::from_polar(pl))
        .collect();
    
    // Find all line intersections with geometric constraints
    let mut corners = Vec::new();
    
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            if let Some(point) = lines[i].intersect(&lines[j]) {
                // Check if intersection is within image bounds (with some margin)
                let margin = -50.0;
                if point.x >= margin && point.x <= (width as f32 + margin) 
                    && point.y >= margin && point.y <= (height as f32 + margin) {
                    
                    // Filter out very acute angles
                    let angle = lines[i].angle_with(&lines[j]);
                    let angle_diff = (angle - 90.0).abs();
                    
                    if angle_diff < 60.0 {
                        corners.push(point);
                    }
                }
            }
        }
    }
    
    if corners.len() < 4 {
        return Err(format!("Not enough corners found: {}", corners.len()));
    }
    
    // Generate and score all possible quadrilaterals
    let min_area = (width * height) as f32 * 0.05; // At least 5% of image area
    let mut best_quad: Option<Quadrilateral> = None;
    let mut max_score = 0.0;
    
    let max_corners = corners.len().min(12);
    
    for i in 0..max_corners {
        for j in (i + 1)..max_corners {
            for k in (j + 1)..max_corners {
                for l in (k + 1)..max_corners {
                    let quad_corners = [corners[i], corners[j], corners[k], corners[l]];
                    
                    if let Some(ordered) = order_corners(&quad_corners) {
                        let quad = Quadrilateral::new(ordered, &edges);
                        
                        if quad.is_valid(min_area) && quad.score > max_score {
                            max_score = quad.score;
                            best_quad = Some(quad);
                        }
                    }
                }
            }
        }
    }
    
    let quad = best_quad.ok_or_else(|| "No valid quadrilateral found".to_string())?;
    
    // Compute homography for perspective transform
    let projection = compute_homography_matrix(&quad.corners, result_width as f32, result_height as f32);
    
    // Create output image with the desired dimensions
    let mut warped_image = RgbImage::new(result_width, result_height);
    
    // Warp the image using the computed homography
    warp_into(
        rgb_image,
        &projection,
        Interpolation::Bilinear,
        imageproc::image::Rgb([255u8, 255u8, 255u8]), // White background
        &mut warped_image,
    );
    
    Ok(warped_image)
}

/// WASM-compatible function to detect document and warp it to a rectangle
/// 
/// This function:
/// 1. Converts ImageData (RGBA) to RGB imageproc image
/// 2. Calls detect_and_warp_document for processing
/// 3. Converts result back to ImageData for WASM/JS
pub fn extract_paper_hough2(
    image_data: ImageData,
    result_width: u32,
    result_height: u32,
) -> Result<ImageData, JsValue> {
    // Get image dimensions
    let width = image_data.width();
    let height = image_data.height();
    let data = image_data.data().0;
    
    // Convert RGBA ImageData to RGB (skip alpha channel)
    let rgb_data: Vec<u8> = data
        .chunks(4)
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    
    // Create RGB image from the data
    let rgb_image = RgbImage::from_raw(width, height, rgb_data)
        .ok_or_else(|| JsValue::from_str("Failed to create RGB image from ImageData"))?;
    
    // Log detected corners for debugging
    // (moved logging here since detect_and_warp_document is library code without web_sys)
    
    // Process the image
    let warped_image = detect_and_warp_document(&rgb_image, result_width, result_height)
        .map_err(|e| JsValue::from_str(&e))?;
    
    // Convert RGB back to RGBA for ImageData
    let rgba_data: Vec<u8> = warped_image
        .pixels()
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255u8])
        .collect();
    
    // Create and return ImageData
    ImageData::new_with_u8_clamped_array_and_sh(
        Clamped(&rgba_data),
        result_width,
        result_height,
    )
    .map_err(|e| JsValue::from_str(&format!("Failed to create ImageData: {:?}", e)))
}
