/**
 * ClientPipeline.ts - Main Thread Client API for Edge Vector Database.
 * Communicates with memory.worker.ts using SharedArrayBuffer without cloning vector data.
 */

import { quantizeFloat32ToInt8, VectorMetadataIndex } from './quantize';

export class ClientVectorEngine {
  private worker: Worker | null = null;
  private sharedBuffer: SharedArrayBuffer | null = null;
  private memoryView: Int8Array | null = null;
  private metadata: VectorMetadataIndex | null = null;
  private pendingRequests = new Map<string, (response: any) => void>();

  constructor(workerScriptUrl: string) {
    this.worker = new Worker(workerScriptUrl, { type: 'module' });
    this.worker.onmessage = this.handleWorkerMessage.bind(this);
  }

  private handleWorkerMessage(event: MessageEvent): void {
    const { id, success, error, ...data } = event.data;
    const resolver = this.pendingRequests.get(id);

    if (resolver) {
      this.pendingRequests.delete(id);
      if (success) {
        resolver(data);
      } else {
        throw new Error(`Worker Task Execution Failed: ${error}`);
      }
    }
  }

  private sendCommand<T>(type: string, payload: unknown): Promise<T> {
    return new Promise((resolve, reject) => {
      const requestId = `${type}_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
      this.pendingRequests.set(requestId, resolve);

      if (!this.worker) {
        reject(new Error('Web Worker is not active.'));
        return;
      }

      this.worker.postMessage({
        id: requestId,
        type,
        payload,
      });
    });
  }

  /**
   * Initializes the Web Worker and connects to the SharedArrayBuffer created from IndexedDB.
   */
  public async initialize(dbName = 'EdgeVectorDB', key = 'primary_vectors'): Promise<void> {
    console.log('[ClientEngine] Requesting SharedArrayBuffer streaming from Web Worker...');
    
    const response = await this.sendCommand<{ sab: SharedArrayBuffer; byteLength: number }>('INIT', {
      dbName,
      key,
    });

    this.sharedBuffer = response.sab;
    // Create direct Int8Array view on main thread over SharedArrayBuffer
    this.memoryView = new Int8Array(this.sharedBuffer);

    console.log(`[ClientEngine] SharedArrayBuffer attached to Main Thread (${(response.byteLength / (1024 * 1024)).toFixed(2)} MB).`);
  }

  /**
   * Loads vector metadata index JSON.
   */
  public setMetadataIndex(metadata: VectorMetadataIndex): void {
    this.metadata = metadata;
  }

  /**
   * Reads vector Int8 data directly from SharedArrayBuffer at zero cost (no structured cloning).
   */
  public getVectorInt8View(vectorId: number): Int8Array {
    if (!this.memoryView || !this.metadata) {
      throw new Error('Engine not initialized or metadata not loaded.');
    }

    const meta = this.metadata.vectors[`vec_${vectorId}`];
    if (!meta) {
      throw new Error(`Vector ID ${vectorId} not found in metadata.`);
    }

    // Zero-copy slice view over shared memory
    return new Int8Array(this.sharedBuffer!, meta.offset, meta.length);
  }

  /**
   * Dequantizes an Int8 vector view back to original Float32 values without V8 array allocation overhead.
   */
  public dequantizeVector(vectorId: number, targetFloat32: Float32Array): void {
    if (!this.metadata) throw new Error('Metadata missing.');
    const meta = this.metadata.vectors[`vec_${vectorId}`];
    const int8View = this.getVectorInt8View(vectorId);

    const range = meta.max - meta.min;
    const invScale = range / 255;

    for (let i = 0; i < int8View.length; i++) {
      targetFloat32[i] = (int8View[i] + 128) * invScale + meta.min;
    }
  }

  /**
   * Submits a float query vector for KNN similarity search. Quantizes on the fly into Int8 typed array.
   */
  public async queryKNN(
    queryFloat32: Float32Array,
    topK = 10
  ): Promise<Array<{ id: number; score: number }>> {
    if (!this.metadata) throw new Error('Metadata index missing.');

    const { int8Vector } = quantizeFloat32ToInt8(queryFloat32);

    const response = await this.sendCommand<{ topMatches: Array<{ id: number; score: number }> }>('QUERY_KNN', {
      queryVector: int8Vector,
      topK,
      dimension: this.metadata.dimension,
    });

    return response.topMatches;
  }

  /**
   * Destroys worker and cleans up handles.
   */
  public terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.sharedBuffer = null;
    this.memoryView = null;
  }
}
