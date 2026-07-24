/**
 * VectorCompute.ts - Bare-Metal WebGPU Compute Engine for High-Throughput Parallel Vector Search.
 * Manages GPU buffers, WGSL pipeline bindings, workgroup dispatches, and async readbacks.
 */

// Embedded raw WGSL compute shader source code (fallback if external file fetch is unavailable)
const DEFAULT_WGSL_SHADER = `
struct QueryUniform {
    vectorCount: u32,
    dimension: u32,
    queryNorm: f32,
    pad: u32,
    queryComponents: array<vec4<f32>, 384>,
};

@group(0) @binding(0) var<storage, read> vectorDatabase: array<u32>;
@group(0) @binding(1) var<uniform> query: QueryUniform;
@group(0) @binding(2) var<storage, read_write> outputScores: array<f32>;

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

    if (vector_idx >= query.vectorCount) {
        return;
    }

    let u32_per_vector = query.dimension / 4u;
    let base_u32_offset = vector_idx * u32_per_vector;

    var dot_product: f32 = 0.0;
    var candidate_norm_sq: f32 = 0.0;

    for (var i: u32 = 0u; i < u32_per_vector; i = i + 1u) {
        let packed_u32 = vectorDatabase[base_u32_offset + i];
        let candidate_i8_x4 = unpack_i8_to_f32_x4(packed_u32);
        let query_f32_x4 = query.queryComponents[i];

        dot_product += dot(query_f32_x4, candidate_i8_x4);
        candidate_norm_sq += dot(candidate_i8_x4, candidate_i8_x4);
    }

    let candidate_norm = sqrt(candidate_norm_sq);
    let denominator = query.queryNorm * candidate_norm;

    if (denominator > 0.0) {
        outputScores[vector_idx] = dot_product / denominator;
    } else {
        outputScores[vector_idx] = 0.0;
    }
}
`;

export class VectorComputeEngine {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private bindGroup: GPUBindGroup | null = null;

  // GPU Buffers
  private vectorDbBuffer: GPUBuffer | null = null;
  private queryUniformBuffer: GPUBuffer | null = null;
  private outputScoreBuffer: GPUBuffer | null = null;
  private readbackBuffer: GPUBuffer | null = null;

  private vectorCount = 0;
  private dimension = 1536;
  private isInitialized = false;

  /**
   * Initializes the WebGPU Device and compiles the WGSL Compute Pipeline.
   */
  public async init(shaderSource?: string): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('WebGPU is not supported in this environment or browser.');
    }

    console.log('[WebGPU Engine] Requesting GPU Adapter...');
    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });

    if (!this.adapter) {
      throw new Error('Failed to acquire a suitable WebGPU hardware adapter.');
    }

    console.log('[WebGPU Engine] Requesting GPUDevice...');
    this.device = await this.adapter.requestDevice();

    const shaderModule = this.device.createShaderModule({
      label: 'Vector Similarity Shader',
      code: shaderSource || DEFAULT_WGSL_SHADER,
    });

    // Create explicit Bind Group Layout according to WGSL bindings
    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: 'Vector Search Bind Group Layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = this.device.createComputePipeline({
      label: 'Vector Cosine Similarity Pipeline',
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    this.isInitialized = true;
    console.log('[WebGPU Engine] WebGPU Compute Pipeline compiled successfully.');
  }

  /**
   * Uploads flat Int8 vector dataset into WebGPU Storage Buffer.
   *
   * @param int8ArrayData - Flat Int8Array containing all packed vector embeddings.
   * @param vectorCount - Total number of vectors.
   * @param dimension - Dimensionality of vectors (default: 1536).
   */
  public uploadVectorDatabase(int8ArrayData: Int8Array, vectorCount: number, dimension = 1536): void {
    if (!this.isInitialized || !this.device || !this.bindGroupLayout) {
      throw new Error('WebGPU Engine is not initialized. Call init() first.');
    }

    this.vectorCount = vectorCount;
    this.dimension = dimension;

    const totalBytes = int8ArrayData.byteLength;
    console.log(`[WebGPU Engine] Allocating GPU Storage Buffer for ${vectorCount} vectors (${(totalBytes / (1024 * 1024)).toFixed(2)} MB)...`);

    // Clean up old buffers if re-uploading
    this.destroyBuffers();

    // 1. Vector Database Storage Buffer (STORAGE | COPY_DST)
    this.vectorDbBuffer = this.device.createBuffer({
      label: 'Int8 Vector Database Buffer',
      size: totalBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Write Int8 buffer directly into GPU memory
    this.device.queue.writeBuffer(
      this.vectorDbBuffer,
      0,
      int8ArrayData.buffer,
      int8ArrayData.byteOffset,
      int8ArrayData.byteLength
    );

    // 2. Query Uniform Buffer (UNIFORM | COPY_DST)
    // Size = 16 bytes header (vectorCount, dimension, queryNorm, pad) + (1536 * 4) bytes query components
    const uniformSize = 16 + (dimension / 4) * 16;
    this.queryUniformBuffer = this.device.createBuffer({
      label: 'Query Vector Uniform Buffer',
      size: uniformSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 3. Output Scores Storage Buffer (STORAGE | COPY_SRC)
    const outputSizeBytes = vectorCount * Float32Array.BYTES_PER_ELEMENT;
    this.outputScoreBuffer = this.device.createBuffer({
      label: 'Output Similarity Scores Buffer',
      size: outputSizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // 4. Staging Readback Buffer (MAP_READ | COPY_DST)
    this.readbackBuffer = this.device.createBuffer({
      label: 'CPU Readback Buffer',
      size: outputSizeBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Bind Group Linking bindings (0: DB, 1: Query, 2: Output)
    this.bindGroup = this.device.createBindGroup({
      label: 'Vector Search Bind Group',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.vectorDbBuffer } },
        { binding: 1, resource: { buffer: this.queryUniformBuffer } },
        { binding: 2, resource: { buffer: this.outputScoreBuffer } },
      ],
    });

    console.log('[WebGPU Engine] Vector Database uploaded to WebGPU VRAM.');
  }

  /**
   * Dispatches parallel WebGPU compute shader to compute cosine similarity across 100,000+ vectors.
   *
   * @param queryFloat32 - Float32 query vector (1536 dimensions).
   * @returns Float32Array containing similarity score per vector.
   */
  public async computeSimilarity(queryFloat32: Float32Array): Promise<Float32Array> {
    if (
      !this.device ||
      !this.pipeline ||
      !this.bindGroup ||
      !this.vectorDbBuffer ||
      !this.queryUniformBuffer ||
      !this.outputScoreBuffer ||
      !this.readbackBuffer
    ) {
      throw new Error('WebGPU buffers or pipeline not configured. Call uploadVectorDatabase() first.');
    }

    if (queryFloat32.length !== this.dimension) {
      throw new Error(`Query vector dimension mismatch: expected ${this.dimension}, got ${queryFloat32.length}`);
    }

    // Precompute Query L2 Norm
    let queryNormSq = 0;
    for (let i = 0; i < queryFloat32.length; i++) {
      queryNormSq += queryFloat32[i] * queryFloat32[i];
    }
    const queryNorm = Math.sqrt(queryNormSq);

    // Prepare Uniform Buffer Memory Layout (6,160 bytes)
    const uniformSize = 16 + (this.dimension / 4) * 16;
    const uniformArrayBuffer = new ArrayBuffer(uniformSize);
    const headerView = new Uint32Array(uniformArrayBuffer, 0, 4);
    const floatHeaderView = new Float32Array(uniformArrayBuffer, 0, 4);

    headerView[0] = this.vectorCount;
    headerView[1] = this.dimension;
    floatHeaderView[2] = queryNorm;
    headerView[3] = 0; // Padding

    const queryComponentsView = new Float32Array(uniformArrayBuffer, 16);
    queryComponentsView.set(queryFloat32);

    // Upload Query to GPU Uniform Buffer
    this.device.queue.writeBuffer(this.queryUniformBuffer, 0, uniformArrayBuffer);

    // Encode GPU Compute Pass Commands
    const commandEncoder = this.device.createCommandEncoder({
      label: 'Vector Search Command Encoder',
    });

    const computePass = commandEncoder.beginComputePass({
      label: 'Vector Search Compute Pass',
    });
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, this.bindGroup);

    // Calculate Workgroup Dispatch Count (Workgroup Size = 64)
    const workgroupCount = Math.ceil(this.vectorCount / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    // Copy Output Storage Buffer -> Readback Buffer
    const outputSizeBytes = this.vectorCount * Float32Array.BYTES_PER_ELEMENT;
    commandEncoder.copyBufferToBuffer(
      this.outputScoreBuffer,
      0,
      this.readbackBuffer,
      0,
      outputSizeBytes
    );

    // Submit GPU Command Buffer
    this.device.queue.submit([commandEncoder.finish()]);

    // Asynchronously Readback Scores from GPU VRAM to CPU System RAM
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const mappedRange = this.readbackBuffer.getMappedRange();
    
    // Copy out results into Float32Array
    const scoresResult = new Float32Array(mappedRange.slice(0));
    this.readbackBuffer.unmap();

    return scoresResult;
  }

  /**
   * Retrieves Top-K vector matches given a score array.
   */
  public getTopK(scores: Float32Array, topK = 10): Array<{ id: number; score: number }> {
    const topMatches: Array<{ id: number; score: number }> = [];

    for (let i = 0; i < scores.length; i++) {
      const score = scores[i];
      if (topMatches.length < topK) {
        topMatches.push({ id: i, score });
        topMatches.sort((a, b) => b.score - a.score);
      } else if (score > topMatches[topMatches.length - 1].score) {
        topMatches[topMatches.length - 1] = { id: i, score };
        topMatches.sort((a, b) => b.score - a.score);
      }
    }

    return topMatches;
  }

  /**
   * Destroys allocated WebGPU buffers.
   */
  private destroyBuffers(): void {
    if (this.vectorDbBuffer) {
      this.vectorDbBuffer.destroy();
      this.vectorDbBuffer = null;
    }
    if (this.queryUniformBuffer) {
      this.queryUniformBuffer.destroy();
      this.queryUniformBuffer = null;
    }
    if (this.outputScoreBuffer) {
      this.outputScoreBuffer.destroy();
      this.outputScoreBuffer = null;
    }
    if (this.readbackBuffer) {
      this.readbackBuffer.destroy();
      this.readbackBuffer = null;
    }
  }

  /**
   * Releases WebGPU resources and device.
   */
  public dispose(): void {
    this.destroyBuffers();
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.isInitialized = false;
  }
}
