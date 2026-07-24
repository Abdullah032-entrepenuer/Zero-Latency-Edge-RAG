/**
 * memory.worker.ts - Dedicated Web Worker for Zero-Copy Vector Database Operations.
 * Streams binary vector data into SharedArrayBuffer and provides Int8Array typed memory views.
 */

import { VectorDB } from './VectorDB';

// Worker Command Protocol Types
export type WorkerCommandType = 'INIT' | 'GET_VECTOR' | 'QUERY_KNN';

export interface InitCommandPayload {
  dbName?: string;
  storeName?: string;
  key?: string;
}

export interface GetVectorCommandPayload {
  id: number;
  offset: number;
  length: number;
}

export interface QueryKnnCommandPayload {
  queryVector: Int8Array; // Int8 quantized query embedding
  topK: number;
  dimension: number;
}

export interface WorkerMessageRequest {
  id: string; // Unique Request Correlation ID
  type: WorkerCommandType;
  payload: InitCommandPayload | GetVectorCommandPayload | QueryKnnCommandPayload;
}

export interface WorkerMessageResponse {
  id: string; // Corresponds to Request Correlation ID
  type: WorkerCommandType;
  success: boolean;
  sab?: SharedArrayBuffer;
  byteLength?: number;
  vectorChunk?: Int8Array;
  topMatches?: Array<{ id: number; score: number }>;
  error?: string;
}

// Global Shared State inside Worker Scope
let sharedBuffer: SharedArrayBuffer | null = null;
let vectorView: Int8Array | null = null;
let isInitialized = false;

/**
 * Loads vector dataset from IndexedDB into a SharedArrayBuffer.
 */
async function handleInit(payload: InitCommandPayload): Promise<{ sab: SharedArrayBuffer; byteLength: number }> {
  const dbName = payload.dbName || 'EdgeVectorDB';
  const storeName = payload.storeName || 'vector_store';
  const key = payload.key || 'primary_vectors';

  const vectorDB = new VectorDB(dbName, storeName);
  const data = await vectorDB.get(key);

  if (!data || !data.binBuffer) {
    throw new Error(`No vector data found in IndexedDB under key "${key}". Ensure fetchAndStore was executed.`);
  }

  const rawArrayBuffer = data.binBuffer;
  const totalBytes = rawArrayBuffer.byteLength;

  console.log(`[Worker] Allocating ${totalBytes} bytes in SharedArrayBuffer...`);

  // Allocate SharedArrayBuffer in unmanaged worker memory
  sharedBuffer = new SharedArrayBuffer(totalBytes);
  
  // Copy raw ArrayBuffer into SharedArrayBuffer using Int8Array view
  const destinationView = new Int8Array(sharedBuffer);
  const sourceView = new Int8Array(rawArrayBuffer);
  destinationView.set(sourceView);

  // Maintain persistent typed memory view over entire buffer
  vectorView = destinationView;
  isInitialized = true;

  console.log(`[Worker] SharedArrayBuffer initialized successfully (${(totalBytes / (1024 * 1024)).toFixed(2)} MB).`);

  return { sab: sharedBuffer, byteLength: totalBytes };
}

/**
 * Returns a zero-copy Int8Array slice view of a specific vector byte range.
 */
function handleGetVector(payload: GetVectorCommandPayload): Int8Array {
  if (!isInitialized || !vectorView) {
    throw new Error('Memory worker is not initialized. Run INIT first.');
  }

  const { offset, length } = payload;
  if (offset < 0 || offset + length > vectorView.byteLength) {
    throw new Error(`Out of bounds memory access: offset=${offset}, length=${length}, totalBytes=${vectorView.byteLength}`);
  }

  // Create zero-copy view slice backed by the existing SharedArrayBuffer
  return new Int8Array(sharedBuffer!, offset, length);
}

/**
 * Executes zero-latency Top-K KNN similarity search across all vectors in SharedArrayBuffer.
 * Operates purely on Int8 typed arrays with 32-bit integer SIMD-like accumulation.
 */
function handleQueryKnn(payload: QueryKnnCommandPayload): Array<{ id: number; score: number }> {
  if (!isInitialized || !vectorView) {
    throw new Error('Memory worker is not initialized. Run INIT first.');
  }

  const { queryVector, topK, dimension } = payload;
  const totalVectors = Math.floor(vectorView.byteLength / dimension);

  // Min-Heap style tracking for top-K results
  const topMatches: Array<{ id: number; score: number }> = [];

  for (let vecIdx = 0; vecIdx < totalVectors; vecIdx++) {
    const baseOffset = vecIdx * dimension;
    let dotProduct = 0;

    // Fast inner-loop Int8 dot product in unmanaged SAB memory
    for (let d = 0; d < dimension; d++) {
      dotProduct += queryVector[d] * vectorView[baseOffset + d];
    }

    // Maintain Top-K ranking
    if (topMatches.length < topK) {
      topMatches.push({ id: vecIdx, score: dotProduct });
      topMatches.sort((a, b) => b.score - a.score);
    } else if (dotProduct > topMatches[topMatches.length - 1].score) {
      topMatches[topMatches.length - 1] = { id: vecIdx, score: dotProduct };
      topMatches.sort((a, b) => b.score - a.score);
    }
  }

  return topMatches;
}

// Worker Incoming Message Event Handler
self.onmessage = async (event: MessageEvent<WorkerMessageRequest>) => {
  const { id, type, payload } = event.data;

  try {
    switch (type) {
      case 'INIT': {
        const { sab, byteLength } = await handleInit(payload as InitCommandPayload);
        const response: WorkerMessageResponse = {
          id,
          type,
          success: true,
          sab,
          byteLength,
        };
        self.postMessage(response);
        break;
      }

      case 'GET_VECTOR': {
        const vectorChunk = handleGetVector(payload as GetVectorCommandPayload);
        const response: WorkerMessageResponse = {
          id,
          type,
          success: true,
          vectorChunk,
        };
        self.postMessage(response);
        break;
      }

      case 'QUERY_KNN': {
        const topMatches = handleQueryKnn(payload as QueryKnnCommandPayload);
        const response: WorkerMessageResponse = {
          id,
          type,
          success: true,
          topMatches,
        };
        self.postMessage(response);
        break;
      }

      default:
        throw new Error(`Unknown command type: ${type}`);
    }
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    console.error(`[Worker Error] Command ${type} failed:`, errorMessage);
    const response: WorkerMessageResponse = {
      id,
      type,
      success: false,
      error: errorMessage,
    };
    self.postMessage(response);
  }
};
