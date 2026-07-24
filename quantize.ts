import * as fs from 'fs';
import * as path from 'path';

/**
 * Interface representing metadata for a single vector in the index.
 */
export interface VectorMetadata {
  id: number;
  offset: number;     // Byte offset in the binary file
  length: number;     // Length in bytes (equal to dimension for Int8)
  min: number;        // Minimum original Float32 value for dequantization
  max: number;        // Maximum original Float32 value for dequantization
}

/**
 * Interface representing the complete metadata index JSON format.
 */
export interface VectorMetadataIndex {
  vectorCount: number;
  dimension: number;
  quantizationType: 'int8';
  vectors: Record<string, VectorMetadata>;
}

/**
 * Quantizes a Float32Array to an Int8Array using Min-Max linear mapping.
 * Maps range [min, max] to [-128, 127].
 *
 * @param floatVector - The original Float32 embedding vector.
 * @returns Object containing quantized Int8Array, min value, and max value.
 */
export function quantizeFloat32ToInt8(floatVector: Float32Array): {
  int8Vector: Int8Array;
  min: number;
  max: number;
} {
  const len = floatVector.length;
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < len; i++) {
    const val = floatVector[i];
    if (val < min) min = val;
    if (val > max) max = val;
  }

  const int8Vector = new Int8Array(len);
  const range = max - min;

  // Handle edge case of constant vector (min === max)
  if (range === 0) {
    int8Vector.fill(0);
    return { int8Vector, min, max };
  }

  const scale = 255 / range;

  for (let i = 0; i < len; i++) {
    // Map [min, max] -> [0, 255] -> [-128, 127]
    const normalized = (floatVector[i] - min) * scale;
    const clamped = Math.min(255, Math.max(0, Math.round(normalized)));
    int8Vector[i] = clamped - 128;
  }

  return { int8Vector, min, max };
}

/**
 * Generates a mock dataset of standard normalized AI embeddings (Float32Array).
 *
 * @param count - Number of vectors to generate.
 * @param dimension - Dimensionality of each vector (e.g. 1536).
 */
function generateMockEmbeddings(count: number, dimension: number): Float32Array[] {
  console.log(`Generating ${count} mock vectors with dimension ${dimension}...`);
  const embeddings: Float32Array[] = [];

  for (let i = 0; i < count; i++) {
    const vec = new Float32Array(dimension);
    let normSq = 0;
    
    for (let d = 0; d < dimension; d++) {
      // Gaussian random generator using Box-Muller transform
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      const val = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      vec[d] = val;
      normSq += val * val;
    }

    // Normalize vector to unit length (L2 norm = 1.0)
    const norm = Math.sqrt(normSq);
    for (let d = 0; d < dimension; d++) {
      vec[d] /= norm;
    }

    embeddings.push(vec);
  }

  return embeddings;
}

/**
 * Main execution script to quantize embeddings and output .bin and metadata .json files.
 */
async function main() {
  const VECTOR_COUNT = 50000;
  const DIMENSION = 1536;
  const OUTPUT_DIR = path.resolve(__dirname, '../dist_data');

  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const binPath = path.join(OUTPUT_DIR, 'vectors.bin');
  const jsonPath = path.join(OUTPUT_DIR, 'metadata.json');

  console.log(`Starting Quantizer Pipeline...`);
  console.log(`Target Output Binary: ${binPath}`);
  console.log(`Target Output Index:  ${jsonPath}`);

  // Step 1: Generate Mock Vectors
  const rawVectors = generateMockEmbeddings(VECTOR_COUNT, DIMENSION);

  // Step 2: Open Write Stream for Binary Vector Storage
  const writeStream = fs.createWriteStream(binPath);
  const metadataIndex: VectorMetadataIndex = {
    vectorCount: VECTOR_COUNT,
    dimension: DIMENSION,
    quantizationType: 'int8',
    vectors: {},
  };

  let currentByteOffset = 0;

  console.log(`Quantizing and writing binary stream...`);
  for (let i = 0; i < VECTOR_COUNT; i++) {
    const floatVec = rawVectors[i];
    const { int8Vector, min, max } = quantizeFloat32ToInt8(floatVec);

    const buffer = Buffer.from(int8Vector.buffer, int8Vector.byteOffset, int8Vector.byteLength);
    writeStream.write(buffer);

    metadataIndex.vectors[`vec_${i}`] = {
      id: i,
      offset: currentByteOffset,
      length: DIMENSION,
      min,
      max,
    };

    currentByteOffset += DIMENSION;

    if ((i + 1) % 10000 === 0) {
      console.log(`Processed ${i + 1} / ${VECTOR_COUNT} vectors...`);
    }
  }

  await new Promise<void>((resolve, reject) => {
    writeStream.end((err?: Error | null) => {
      if (err) reject(err);
      else resolve();
    });
  });

  // Step 3: Write Metadata Index JSON
  fs.writeFileSync(jsonPath, JSON.stringify(metadataIndex, null, 2), 'utf-8');

  const binSizeBytes = fs.statSync(binPath).size;
  const originalSizeBytes = VECTOR_COUNT * DIMENSION * 4;

  console.log('\n--- QUANTIZATION COMPLETED SUCCESSFULLY ---');
  console.log(`Original Float32 Dataset Size: ${(originalSizeBytes / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`Quantized Int8 Binary File Size: ${(binSizeBytes / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`Memory Compression Ratio: ${(originalSizeBytes / binSizeBytes).toFixed(2)}x smaller`);
}

if (require.main === module) {
  main().catch((err) => {
    console.error('Quantization failed:', err);
    process.exit(1);
  });
}
