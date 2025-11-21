use wasmcanify::detect_document::find_best_quadrilateral;
use image::ImageReader;

fn main() {
    // Load the image
    let img_path = "tests/test.jpg";
    let img = ImageReader::open(img_path)
        .expect("Failed to open image file")
        .decode()
        .expect("Failed to decode image");
    
    let gray_image = img.to_luma8();
    
    println!("Image loaded: {}x{}", gray_image.width(), gray_image.height());
    println!("Starting detection loop for profiling...");

    // Run detection multiple times to capture a good profile
    // 100 iterations should be enough for a decent flamegraph
    for i in 0..100 {
        if i % 10 == 0 {
            println!("Iteration {}", i);
        }
        let _ = find_best_quadrilateral(&gray_image);
    }
    
    println!("Done.");
}
