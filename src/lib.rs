use wasm_bindgen::prelude::*;
use web_sys::ImageData;
pub mod detect_document;

pub use wasm_bindgen_rayon::init_thread_pool;

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
        warp_image: bool
    ) -> Result<ImageData, JsValue> {
        detect_document::extract_paper_hough(image_data, result_width, result_height, warp_image)
    }
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
