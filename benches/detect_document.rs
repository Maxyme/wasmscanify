use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jscanify_wasm::detect_document::{find_best_quadrilateral, compute_homography_matrix};
use imageproc::image::{Rgb, RgbImage, GrayImage, Luma};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
use imageproc::geometric_transformations::{warp_into, Interpolation};

fn generate_gray_test_image(width: u32, height: u32) -> GrayImage {
    let mut image = GrayImage::new(width, height);
    // Draw a white rectangle in the middle
    let rect = Rect::at((width / 4) as i32, (height / 4) as i32)
        .of_size(width / 2, height / 2);
    draw_filled_rect_mut(&mut image, rect, Luma([255u8]));
    image
}

fn generate_rgb_test_image(width: u32, height: u32) -> RgbImage {
    let mut image = RgbImage::new(width, height);
    // Draw a white rectangle in the middle
    let rect = Rect::at((width / 4) as i32, (height / 4) as i32)
        .of_size(width / 2, height / 2);
    draw_filled_rect_mut(&mut image, rect, Rgb([255u8, 255u8, 255u8]));
    image
}

fn benchmark_detection(c: &mut Criterion) {
    let gray_image = generate_gray_test_image(800, 600);

    c.bench_function("detect_document_synthetic", |b| {
        b.iter(|| {
            // Benchmark the detection phase
            find_best_quadrilateral(black_box(&gray_image))
        })
    });
}

fn benchmark_warping(c: &mut Criterion) {
    let gray_image = generate_gray_test_image(800, 600);
    let rgb_image = generate_rgb_test_image(800, 600);
    
    // Pre-calculate quad for warping benchmark
    // We know this synthetic image has a quad
    let quad = find_best_quadrilateral(&gray_image).expect("Failed to detect document in synthetic image");
    
    let width = 800;
    let height = 600;
    
    c.bench_function("warp_document_synthetic", |b| {
        b.iter(|| {
            // Benchmark the warping phase
            let projection = compute_homography_matrix(black_box(&quad.corners), width as f32, height as f32);
            let mut warped_image = RgbImage::new(width, height);
            
            warp_into(
                black_box(&rgb_image),
                &projection,
                Interpolation::Bilinear,
                Rgb([255u8, 255u8, 255u8]),
                &mut warped_image,
            );
            black_box(warped_image)
        })
    });
}

fn benchmark_real_image(c: &mut Criterion) {
    // Load the real test image
    let img_path = "tests/test.png";
    // Use the image crate directly to open the file
    let img = image::open(img_path).expect("Failed to open test image");
    // Convert to grayscale as required by the detection algorithm
    let gray_image = img.to_luma8();

    c.bench_function("detect_document_real_image", |b| {
        b.iter(|| {
            find_best_quadrilateral(black_box(&gray_image))
        })
    });
}

criterion_group!(benches, benchmark_detection, benchmark_warping, benchmark_real_image);
criterion_main!(benches);
