use imageproc::image::GrayImage;
use imageproc::image::ImageBuffer;
use photon_rs::PhotonImage;
use photon_rs::conv;
use photon_rs::monochrome;
use nalgebra::{Matrix3, SMatrix, SVector};
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};
use imgproc::interpolation::InterpolationMode;
use std::cell::RefCell;
use std::rc::Rc;
use kornia::{image::{Image, ImageSize}, imgproc};
use kornia::io::functional as F;
use imageproc::contours::find_contours;
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
