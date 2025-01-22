struct VOutput{   
    @location(0) v_color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct UniformData {
    modelview: mat4x4<f32>,
    projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniform_data: UniformData;

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>) -> VOutput {    
    var out: VOutput;
    out.position = uniform_data.projection * (uniform_data.modelview * vec4<f32>(pos * 0.5, 1.0));
    out.v_color = vec4<f32>(color, 1.0);
    return out;
}

@fragment
fn fs_main(in: VOutput) -> @location(0) vec4<f32> {
    return in.v_color;
}