/**
 * VectorDB.ts - Bare-metal IndexedDB Storage Engine for Edge Vector Embeddings.
 * Persists raw binary ArrayBuffers and metadata locally without third-party dependencies.
 */

export interface StoredVectorData {
  binBuffer: ArrayBuffer;
  metadata: Record<string, unknown>;
  updatedAt: number;
}

export class VectorDB {
  private readonly dbName: string;
  private readonly storeName: string;
  private readonly dbVersion: number;
  private dbPromise: Promise<IDBDatabase> | null = null;

  constructor(dbName = 'EdgeVectorDB', storeName = 'vector_store', dbVersion = 1) {
    this.dbName = dbName;
    this.storeName = storeName;
    this.dbVersion = dbVersion;
  }

  /**
   * Initializes and opens the browser's native IndexedDB instance.
   */
  public async getDB(): Promise<IDBDatabase> {
    if (this.dbPromise) {
      return this.dbPromise;
    }

    this.dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      if (typeof indexedDB === 'undefined') {
        reject(new Error('IndexedDB is not supported in this environment.'));
        return;
      }

      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onupgradeneeded = (event: IDBVersionChangeEvent) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName);
        }
      };

      request.onsuccess = (event: Event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        db.onclose = () => {
          this.dbPromise = null;
        };

        resolve(db);
      };

      request.onerror = (event: Event) => {
        this.dbPromise = null;
        const err = (event.target as IDBOpenDBRequest).error;
        reject(new Error(`IndexedDB Open Error: ${err?.message || 'Unknown error'}`));
      };
    });

    return this.dbPromise;
  }

  /**
   * Fetches binary vector data and JSON metadata index from server, storing directly in IDB.
   *
   * @param binUrl - URL pointing to the raw .bin vector file.
   * @param metadataUrl - URL pointing to the metadata index JSON file.
   * @param key - Storage key identifier (default: 'primary_vectors').
   * @param onProgress - Optional callback reporting download progress percentage.
   */
  public async fetchAndStore(
    binUrl: string,
    metadataUrl: string,
    key = 'primary_vectors',
    onProgress?: (percent: number) => void
  ): Promise<void> {
    console.log(`[VectorDB] Fetching vector binary dataset from ${binUrl}...`);

    // Fetch JSON metadata
    const metadataResponse = await fetch(metadataUrl);
    if (!metadataResponse.ok) {
      throw new Error(`Failed to fetch metadata index: ${metadataResponse.statusText}`);
    }
    const metadata = await metadataResponse.json();

    // Fetch binary vector buffer with progress streaming
    const binResponse = await fetch(binUrl);
    if (!binResponse.ok || !binResponse.body) {
      throw new Error(`Failed to fetch vector binary file: ${binResponse.statusText}`);
    }

    const contentLength = binResponse.headers.get('Content-Length');
    const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;
    
    const reader = binResponse.body.getReader();
    const chunks: Uint8Array[] = [];
    let receivedBytes = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      if (value) {
        chunks.push(value);
        receivedBytes += value.byteLength;
        if (totalBytes > 0 && onProgress) {
          onProgress((receivedBytes / totalBytes) * 100);
        }
      }
    }

    // Assemble flat Uint8Array buffer without memory fragmentation
    const combinedBuffer = new Uint8Array(receivedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      combinedBuffer.set(chunk, offset);
      offset += chunk.byteLength;
    }

    const binBuffer = combinedBuffer.buffer;

    console.log(`[VectorDB] Received ${binBuffer.byteLength} bytes. Persisting to IndexedDB...`);

    // Persist into IndexedDB
    await this.put(key, {
      binBuffer,
      metadata,
      updatedAt: Date.now(),
    });

    console.log(`[VectorDB] Dataset persisted successfully under key "${key}".`);
  }

  /**
   * Directly stores a raw ArrayBuffer payload into IndexedDB.
   */
  public async put(key: string, value: StoredVectorData): Promise<void> {
    const db = await this.getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      const request = store.put(value, key);

      request.onsuccess = () => resolve();
      request.onerror = (event: Event) => {
        const err = (event.target as IDBRequest).error;
        reject(new Error(`IndexedDB Put Failure: ${err?.message}`));
      };
    });
  }

  /**
   * Retrieves stored binary vector data and metadata from IndexedDB.
   */
  public async get(key = 'primary_vectors'): Promise<StoredVectorData | null> {
    const db = await this.getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readonly');
      const store = tx.objectStore(this.storeName);
      const request = store.get(key);

      request.onsuccess = (event: Event) => {
        const result = (event.target as IDBRequest).result;
        resolve(result || null);
      };

      request.onerror = (event: Event) => {
        const err = (event.target as IDBRequest).error;
        reject(new Error(`IndexedDB Get Failure: ${err?.message}`));
      };
    });
  }

  /**
   * Checks whether the dataset already exists locally in IndexedDB.
   */
  public async has(key = 'primary_vectors'): Promise<boolean> {
    const data = await this.get(key);
    return data !== null;
  }

  /**
   * Removes a stored dataset from IndexedDB to free local browser storage.
   */
  public async delete(key = 'primary_vectors'): Promise<void> {
    const db = await this.getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = (event: Event) => {
        const err = (event.target as IDBRequest).error;
        reject(new Error(`IndexedDB Delete Failure: ${err?.message}`));
      };
    });
  }
}
