@group(0) @binding(0)
var<uniform> transform: mat3x3<f32>;

@group(0) @binding(1)
var src_texture: texture_2d<f32>;

@group(0) @binding(2)
var dst_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(3)
var a_sampler: sampler;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coords = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let dst_size = textureDimensions(dst_texture);

    if (dst_coords.x >= i32(dst_size.x) || dst_coords.y >= i32(dst_size.y)) {
        return;
    }

    let dst_uv = vec2<f32>(f32(dst_coords.x) + 0.5, f32(dst_coords.y) + 0.5);

    // Inverse transform
    let inv_transform = inverse(transform);
    let src_uv_h = inv_transform * vec3<f32>(dst_uv.x, dst_uv.y, 1.0);
    let src_uv = src_uv_h.xy / src_uv_h.z;

    let src_size = vec2<f32>(f32(textureDimensions(src_texture).x), f32(textureDimensions(src_texture).y));
    let normalized_src_uv = src_uv / src_size;

    if (normalized_src_uv.x < 0.0 || normalized_src_uv.x > 1.0 || normalized_src_uv.y < 0.0 || normalized_src_uv.y > 1.0) {
        textureStore(dst_texture, dst_coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    } else {
        let color = textureSample(src_texture, a_sampler, normalized_src_uv);
        textureStore(dst_texture, dst_coords, color);
    }
}
