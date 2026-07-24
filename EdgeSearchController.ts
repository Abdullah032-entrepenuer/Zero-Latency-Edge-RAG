/**
 * EdgeSearchController.ts - Main Thread Central Orchestrator
 * Connects WebGPU Vector Compute Engine with the 3D OffscreenCanvas Rendering Worker.
 */

import { VectorComputeEngine } from './VectorCompute';

export interface TopMatchResult {
  id: number;
  score: number;
}

export interface HighlightClusterPayload {
  topMatches: TopMatchResult[];
}

export type MainToWorkerMessageType = 
  | 'INIT_CANVAS' 
  | 'HIGHLIGHT_CLUSTER' 
  | 'RESET_HIGHLIGHTS' 
  | 'RESIZE';

export interface MainToWorkerMessage<T = unknown> {
  type: MainToWorkerMessageType;
  payload: T;
}

export class EdgeSearchController {
  private computeEngine: VectorComputeEngine;
  private spatialWorker: Worker | null = null;
  private isInitialized = false;

  constructor(spatialWorkerScriptUrl: string) {
    this.computeEngine = new VectorComputeEngine();
    this.spatialWorker = new Worker(spatialWorkerScriptUrl, { type: 'module' });
  }

  /**
   * Initializes WebGPU compute engine and transfers canvas control to 3D Offscreen Worker.
   *
   * @param canvas - HTMLCanvasElement from DOM.
   * @param int8VectorDb - Flat Int8Array of all 100,000 vector embeddings.
   * @param vectorCount - Total number of vectors.
   * @param dimension - Dimensionality of vectors (default 1536).
   */
  public async init(
    canvas: HTMLCanvasElement,
    int8VectorDb: Int8Array,
    vectorCount = 100000,
    dimension = 1536
  ): Promise<void> {
    console.log('[Orchestrator] Initializing Edge Search Controller...');

    // 1. OffscreenCanvas Transfer to Worker (Zero Main Thread Draw Overhead)
    const offscreenCanvas = canvas.transferControlToOffscreen();
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;

    const initMessage: MainToWorkerMessage<{
      canvas: OffscreenCanvas;
      width: number;
      height: number;
      vectorCount: number;
    }> = {
      type: 'INIT_CANVAS',
      payload: {
        canvas: offscreenCanvas,
        width,
        height,
        vectorCount,
      },
    };

    // Transfer OffscreenCanvas ownership to 3D worker
    this.spatialWorker?.postMessage(initMessage, [offscreenCanvas]);

    // 2. Initialize WebGPU Compute Pipeline & Upload VRAM Storage Buffer
    await this.computeEngine.init();
    this.computeEngine.uploadVectorDatabase(int8VectorDb, vectorCount, dimension);

    this.isInitialized = true;
    console.log('[Orchestrator] WebGPU Engine & 3D Worker initialized successfully.');
  }

  /**
   * Executes zero-latency vector search on WebGPU and updates 3D scene highlights.
   *
   * @param queryFloat32 - Float32 query embedding vector.
   * @param topK - Number of top matches to highlight (default 20).
   */
  public async executeSearch(queryFloat32: Float32Array, topK = 20): Promise<TopMatchResult[]> {
    if (!this.isInitialized || !this.spatialWorker) {
      throw new Error('EdgeSearchController is not initialized.');
    }

    // 1. Dispatch WebGPU Cosine Similarity Kernel (Non-blocking GPU execution)
    const scores = await this.computeEngine.computeSimilarity(queryFloat32);

    // 2. Extract Top K Matches
    const topMatches = this.computeEngine.getTopK(scores, topK);

    // 3. Dispatch HIGHLIGHT_CLUSTER payload to 3D Spatial Web Worker
    const highlightMessage: MainToWorkerMessage<HighlightClusterPayload> = {
      type: 'HIGHLIGHT_CLUSTER',
      payload: { topMatches },
    };

    this.spatialWorker.postMessage(highlightMessage);

    return topMatches;
  }

  /**
   * Notifies 3D worker on window resize.
   */
  public handleResize(width: number, height: number): void {
    if (this.spatialWorker) {
      this.spatialWorker.postMessage({
        type: 'RESIZE',
        payload: { width, height },
      });
    }
  }

  /**
   * Disposes compute engine and worker thread.
   */
  public dispose(): void {
    this.computeEngine.dispose();
    if (this.spatialWorker) {
      this.spatialWorker.terminate();
      this.spatialWorker = null;
    }
    this.isInitialized = false;
  }
}
