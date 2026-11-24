struct Params {
    m: u32,          // Número de instancias (filas)
    n: u32,          // Número de características (columnas + bias)
    alpha: f32,      // Tasa de aprendizaje
    _pad: u32,       // Alineación a 16 bytes (regla std140/Uniform)
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;      // Matriz [m x n] (row-major)
@group(0) @binding(2) var<storage, read> y: array<f32>;      // Vector [m]
@group(0) @binding(3) var<storage, read> theta: array<f32>;  // Vector [n]
@group(0) @binding(4) var<storage, read_write> error: array<f32>; // Vector [m]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.m) {
        return;
    }

    var prediction: f32 = 0.0;
    let row_offset = row * params.n;

    // Producto punto de la fila i de X con theta
    for (var j: u32 = 0u; j < params.n; j = j + 1u) {
        prediction = prediction + X[row_offset + j] * theta[j];
    }

    error[row] = prediction - y[row];
}