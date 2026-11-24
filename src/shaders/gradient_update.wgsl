struct Params {
    m: u32,
    n: u32,
    alpha: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> error: array<f32>;
@group(0) @binding(3) var<storage, read_write> theta: array<f32>;

const WG_SIZE: u32 = 256u;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let feature_idx = wg_id.x; // Cada workgroup calcula el gradiente de una característica j
    let tid = local_id.x;

    if (feature_idx >= params.n) {
        return;
    }

    // 1. Acumulación parcial (Grid-Stride Loop) para soportar m > 256
    var partial_sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < params.m) {
        // Acceso a X[i, feature_idx]
        partial_sum = partial_sum + X[i * params.n + feature_idx] * error[i];
        i = i + WG_SIZE;
    }

    shared_sum[tid] = partial_sum;
    workgroupBarrier();

    // 2. Reducción paralela en árbol dentro de Workgroup Memory
    var stride: u32 = WG_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // 3. El hilo 0 actualiza el peso theta[j] directamente en VRAM
    if (tid == 0u) {
        let gradient = shared_sum[0] / f32(params.m);
        theta[feature_idx] = theta[feature_idx] - params.alpha * gradient;
    }
}