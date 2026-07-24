/**
 * demo.ts - Complete End-to-End Integration Demo (Phase 1, Phase 2, & Phase 3)
 */

import { VectorDB } from './VectorDB';
import { EdgeSearchController } from './EdgeSearchController';

export async function startFullEdgeRagSystem(canvasElement: HTMLCanvasElement) {
  console.log('--- STARTING ZERO-LATENCY EDGE RAG COMPLETE ARCHITECTURE ---');

  // 1. Phase 1: Initialize IndexedDB Engine & Fetch Vector Dataset
  const vectorDB = new VectorDB('EdgeVectorDB', 'vector_store');
  const hasData = await vectorDB.has('primary_vectors');

  if (!hasData) {
    console.log('[Phase 1] Fetching & storing 100,000 Int8 vectors into IndexedDB...');
    await vectorDB.fetchAndStore('/dist_data/vectors.bin', '/dist_data/metadata.json');
  }

  const storedData = await vectorDB.get('primary_vectors');
  if (!storedData || !storedData.binBuffer) {
    throw new Error('Vector dataset missing from IndexedDB.');
  }

  const binBuffer = storedData.binBuffer;
  const vectorCount = 100000;
  const dimension = 1536;

  // 2. Phase 3 Orchestrator: Connect WebGPU Compute Engine with 3D Spatial Worker
  const controller = new EdgeSearchController('./spatial.worker.ts');

  await controller.init(
    canvasElement,
    new Int8Array(binBuffer),
    vectorCount,
    dimension
  );

  console.log('[System Ready] Zero-Latency Vector DB & 3D Spatial Engine active.');

  // 3. Simulate User Search Query
  const mockQuery = new Float32Array(dimension);
  for (let i = 0; i < dimension; i++) {
    mockQuery[i] = (Math.random() - 0.5) * 2;
  }

  console.log('[Search Executing] Querying WebGPU for Top 20 Nearest Neighbors...');
  console.time('Edge Vector Search Latency');
  const top20Matches = await controller.executeSearch(mockQuery, 20);
  console.timeEnd('Edge Vector Search Latency');

  console.log('Top 20 Matches:', top20Matches);
  return controller;
}
