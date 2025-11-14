use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use photon_rs::PhotonImage;
use image;
use std::time::Duration;

fn load_test_image() -> PhotonImage {
    let img = image::open("tests/test.png")
        .expect("Failed to load test image")
        .to_rgba8();
    
    let (width, height) = img.dimensions();
    let raw_pixels = img.into_raw();
    
    PhotonImage::new(raw_pixels, width, height)
}

fn benchmark_image_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_processing");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    
    let img = load_test_image();
    
    // Test different output sizes to show scaling behavior
    let output_sizes = vec![
        ("Small", 400, 600),
        ("Medium", 800, 1200),
        ("Large", 1200, 1800),
        ("XLarge", 1600, 2400),
    ];
    
    for (label, out_width, out_height) in output_sizes {
        group.bench_with_input(
            BenchmarkId::new("Resize", label),
            &(out_width, out_height),
            |b, &(w, h)| {
                b.iter(|| {
                    // Benchmark image resizing as a proxy for transformation work
                    let resized = photon_rs::transform::resize(
                        &img,
                        w,
                        h,
                        photon_rs::transform::SamplingFilter::Nearest
                    );
                    black_box(resized)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_edge_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_detection");
    group.measurement_time(Duration::from_secs(5));
    
    let img = load_test_image();
    
    group.bench_function("sobel_edge_detection", |b| {
        b.iter(|| {
            let mut img_clone = img.clone();
            photon_rs::conv::edge_detection(&mut img_clone);
            black_box(img_clone)
        });
    });
    
    group.finish();
}

fn benchmark_grayscale_conversion(c: &mut Criterion) {
    let img = load_test_image();
    
    c.bench_function("grayscale_conversion", |b| {
        b.iter(|| {
            let mut img_clone = img.clone();
            photon_rs::monochrome::b_grayscale(&mut img_clone);
            black_box(img_clone)
        });
    });
}

criterion_group!(
    benches,
    benchmark_image_sizes,
    benchmark_edge_detection,
    benchmark_grayscale_conversion
);
criterion_main!(benches);
