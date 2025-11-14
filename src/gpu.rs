use nalgebra::Matrix3;
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use std::sync::Arc;

/// GPU context for WGPU operations
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Check if GPU/WebGPU is available
    pub async fn is_available() -> bool {
        let instance = wgpu::Instance::default();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;
        
        adapter.is_some()
    }

    /// Initialize GPU context
    pub async fn new() -> Result<Self, JsValue> {
        let instance = wgpu::Instance::default();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| JsValue::from_str("Failed to find GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("jscanify-gpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to create device: {:?}", e)))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Compute perspective transform matrix using GPU (DLT algorithm)
    /// This solves the linear system Ax = b for homography estimation
    pub async fn compute_perspective_transform(
        &self,
        src: &[(f64, f64); 4],
        dst: &[(f64, f64); 4],
    ) -> Result<Matrix3<f64>, JsValue> {
        // Build the linear system matrix A (8x8) and vector b (8x1)
        let mut a_data = vec![0.0f32; 64]; // 8x8 matrix
        let mut b_data = vec![0.0f32; 8];  // 8x1 vector

        for i in 0..4 {
            let (sx, sy) = (src[i].0 as f32, src[i].1 as f32);
            let (dx, dy) = (dst[i].0 as f32, dst[i].1 as f32);

            // Row for x coordinate
            let row_idx = i * 2;
            a_data[row_idx * 8 + 0] = sx;
            a_data[row_idx * 8 + 1] = sy;
            a_data[row_idx * 8 + 2] = 1.0;
            a_data[row_idx * 8 + 6] = -dx * sx;
            a_data[row_idx * 8 + 7] = -dx * sy;
            b_data[row_idx] = dx;

            // Row for y coordinate
            let row_idx = i * 2 + 1;
            a_data[row_idx * 8 + 3] = sx;
            a_data[row_idx * 8 + 4] = sy;
            a_data[row_idx * 8 + 5] = 1.0;
            a_data[row_idx * 8 + 6] = -dy * sx;
            a_data[row_idx * 8 + 7] = -dy * sy;
            b_data[row_idx] = dy;
        }

        // Create GPU buffers
        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A Buffer"),
            contents: bytemuck::cast_slice(&a_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector b Buffer"),
            contents: bytemuck::cast_slice(&b_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size: 32, // 8 f32 values
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create compute shader for solving the linear system
        let shader_code = r#"
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> vector_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

// Solve 8x8 linear system using Gaussian elimination with partial pivoting
@compute @workgroup_size(1)
fn main() {
    // Copy A and b to local arrays for manipulation
    var a: array<f32, 64>;
    var b: array<f32, 8>;
    
    for (var i = 0u; i < 64u; i++) {
        a[i] = matrix_a[i];
    }
    for (var i = 0u; i < 8u; i++) {
        b[i] = vector_b[i];
    }
    
    // Gaussian elimination with partial pivoting
    for (var k = 0u; k < 8u; k++) {
        // Find pivot
        var max_val = abs(a[k * 8u + k]);
        var max_row = k;
        for (var i = k + 1u; i < 8u; i++) {
            let val = abs(a[i * 8u + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }
        
        // Swap rows if needed
        if (max_row != k) {
            for (var j = 0u; j < 8u; j++) {
                let temp = a[k * 8u + j];
                a[k * 8u + j] = a[max_row * 8u + j];
                a[max_row * 8u + j] = temp;
            }
            let temp_b = b[k];
            b[k] = b[max_row];
            b[max_row] = temp_b;
        }
        
        // Eliminate column
        let pivot = a[k * 8u + k];
        if (abs(pivot) > 1e-10) {
            for (var i = k + 1u; i < 8u; i++) {
                let factor = a[i * 8u + k] / pivot;
                for (var j = k; j < 8u; j++) {
                    a[i * 8u + j] -= factor * a[k * 8u + j];
                }
                b[i] -= factor * b[k];
            }
        }
    }
    
    // Back substitution
    for (var i = 7i; i >= 0; i--) {
        let ui = u32(i);
        var sum = b[ui];
        for (var j = ui + 1u; j < 8u; j++) {
            sum -= a[ui * 8u + j] * result[j];
        }
        let diag = a[ui * 8u + ui];
        result[ui] = select(0.0, sum / diag, abs(diag) > 1e-10);
    }
}
"#;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Linear System Solver Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Linear System Solver Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&result_buffer, 0, &output_buffer, 0, 32);

        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .await
            .map_err(|_| JsValue::from_str("Failed to receive buffer mapping result"))?
            .map_err(|e| JsValue::from_str(&format!("Failed to map buffer: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let h: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        output_buffer.unmap();

        // Construct 3x3 transformation matrix
        Ok(Matrix3::new(
            h[0] as f64, h[1] as f64, h[2] as f64,
            h[3] as f64, h[4] as f64, h[5] as f64,
            h[6] as f64, h[7] as f64, 1.0,
        ))
    }
}
