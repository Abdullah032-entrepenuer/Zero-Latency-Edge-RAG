/**
 * similarity.wgsl - High-Performance WebGPU Compute Shader for Parallel Vector Cosine Similarity
 *
 * Computes Cosine Similarity between a query vector (Float32, 1536 dimensions) and 
 * 100,000+ quantized Int8 vectors stored in flat storage buffers.
 */

struct QueryUniform {
    vectorCount: u32,
    dimension: u32,
    queryNorm: f32,
    pad: u32,
    // 1536 Float32 query elements packed into 384 vec4<f32> elements (16-byte aligned)
    queryComponents: array<vec4<f32>, 384>,
};

// Storage Buffer 0: Flat array of packed u32 values (each u32 contains 4 Int8 dimensions)
@group(0) @binding(0) var<storage, read> vectorDatabase: array<u32>;

// Uniform Buffer 1: Query parameters and Float32 query vector
@group(0) @binding(1) var<uniform> query: QueryUniform;

// Storage Buffer 2: Output Float32 similarity scores per vector index
@group(0) @binding(2) var<storage, read_write> outputScores: array<f32>;

/**
 * Unpacks a single u32 word containing 4 packed Int8 values into a WGSL vec4<f32>.
 * Employs arithmetic right-shifts to preserve Int8 sign extension [-128, 127].
 */
fn unpack_i8_to_f32_x4(packed_u32: u32) -> vec4<f32> {
    let b0 = f32(i32(packed_u32 << 24u) >> 24u);
    let b1 = f32(i32(packed_u32 << 16u) >> 24u);
    let b2 = f32(i32(packed_u32 << 8u) >> 24u);
    let b3 = f32(i32(packed_u32) >> 24u);
    return vec4<f32>(b0, b1, b2, b3);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_idx = global_id.x;

    // Bounds check to avoid out-of-range buffer reads
    if (vector_idx >= query.vectorCount) {
        return;
    }

    // 1536 Int8 dimensions / 4 = 384 u32 words per vector
    let u32_per_vector = query.dimension / 4u;
    let base_u32_offset = vector_idx * u32_per_vector;

    var dot_product: f32 = 0.0;
    var candidate_norm_sq: f32 = 0.0;

    // Unrolled SIMD inner product loop over vec4 hardware registers
    for (var i: u32 = 0u; i < u32_per_vector; i = i + 1u) {
        let packed_u32 = vectorDatabase[base_u32_offset + i];
        let candidate_i8_x4 = unpack_i8_to_f32_x4(packed_u32);
        let query_f32_x4 = query.queryComponents[i];

        // Hardware 4-component vector dot product
        dot_product += dot(query_f32_x4, candidate_i8_x4);
        candidate_norm_sq += dot(candidate_i8_x4, candidate_i8_x4);
    }

    let candidate_norm = sqrt(candidate_norm_sq);
    let denominator = query.queryNorm * candidate_norm;

    // Output normalized Cosine Similarity score [-1.0, 1.0]
    if (denominator > 0.0) {
        outputScores[vector_idx] = dot_product / denominator;
    } else {
        outputScores[vector_idx] = 0.0;
    }
}
