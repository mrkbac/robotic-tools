import { useRef, useCallback } from "react";
import { openDB, type IDBPDatabase } from "idb";
import type {
  McapRawData,
  ProgressCallback,
  ReadResult,
} from "../mcap/reader.ts";
import { readMcapFile } from "../mcap/reader.ts";
import type { ScanMode } from "../mcap/types.ts";
import type { ThumbnailMap } from "../mcap/image.ts";
import type { TfTreeData } from "../mcap/tf.ts";

interface FileIdentity {
  name: string;
  size: number;
}

function isSameFile(file: File, identity: FileIdentity): boolean {
  return file.name === identity.name && file.size === identity.size;
}

export const MODE_LEVEL: Record<ScanMode, number> = {
  summary: 0,
  rebuild: 1,
  exact: 2,
};

function deriveRawData(
  raw: McapRawData,
  from: ScanMode,
  to: ScanMode,
): McapRawData {
  if (from === to) return raw;

  const fromLevel = MODE_LEVEL[from];
  const toLevel = MODE_LEVEL[to];

  if (toLevel > fromLevel) {
    throw new Error(`Cannot derive ${to} from ${from}`);
  }

  // Both rebuild and exact produce identical data now (both always have channelSizes).
  // Summary only lacks chunkInformation. We keep channelSizes when deriving down
  // since exact/rebuild sizes are strictly better than estimated.
  if (from === "exact" && to === "rebuild") {
    return raw;
  }
  if ((from === "exact" || from === "rebuild") && to === "summary") {
    return { ...raw, chunkInformation: null };
  }

  throw new Error(`Unhandled: ${from} → ${to}`);
}

interface CacheEntry {
  fileIdentity: FileIdentity;
  mode: ScanMode;
  rawData: McapRawData;
  thumbnails: ThumbnailMap;
  tfData: TfTreeData | null;
}

// ---------------------------------------------------------------------------
// IndexedDB helpers (best-effort, failures are silent)
// ---------------------------------------------------------------------------

const DB_NAME = "mcap-inspector-cache";
const STORE_NAME = "raw-data";
const DB_VERSION = 3;

interface IDBCacheEntry extends CacheEntry {
  key: string;
}

function getDB(): Promise<IDBPDatabase> {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (db.objectStoreNames.contains(STORE_NAME)) {
        db.deleteObjectStore(STORE_NAME);
      }
      db.createObjectStore(STORE_NAME, { keyPath: "key" });
    },
  });
}

function cacheKey(identity: FileIdentity): string {
  return `${identity.name}:${identity.size}`;
}

async function loadFromIDB(identity: FileIdentity): Promise<CacheEntry | null> {
  try {
    const db = await getDB();
    const stored = (await db.get(STORE_NAME, cacheKey(identity))) as
      | IDBCacheEntry
      | undefined;
    return stored ?? null;
  } catch {
    return null;
  }
}

async function saveToIDB(entry: CacheEntry): Promise<void> {
  try {
    const db = await getDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    await tx.store.clear();
    await tx.store.put({ ...entry, key: cacheKey(entry.fileIdentity) });
    await tx.done;
  } catch {
    // persistence is best-effort
  }
}

async function clearIDB(): Promise<void> {
  try {
    const db = await getDB();
    await db.clear(STORE_NAME);
  } catch {
    // best-effort
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useMcapCache() {
  const cacheRef = useRef<CacheEntry | null>(null);

  /** Check in-memory cache then IDB; hydrate in-memory on IDB hit. */
  const getOrHydrate = async (file: File): Promise<CacheEntry | null> => {
    const entry = cacheRef.current;
    if (entry && isSameFile(file, entry.fileIdentity)) return entry;

    const idbEntry = await loadFromIDB({ name: file.name, size: file.size });
    if (idbEntry) {
      cacheRef.current = idbEntry;
      return idbEntry;
    }
    return null;
  };

  const tryGetCachedRaw = useCallback(
    async (file: File, mode: ScanMode): Promise<McapRawData | null> => {
      const entry = await getOrHydrate(file);
      if (entry && MODE_LEVEL[entry.mode] >= MODE_LEVEL[mode]) {
        return deriveRawData(entry.rawData, entry.mode, mode);
      }
      return null;
    },
    [],
  );

  const readAndCache = useCallback(
    async (
      file: File,
      mode: ScanMode,
      onProgress?: ProgressCallback,
    ): Promise<ReadResult> => {
      const result = await readMcapFile(file, mode, onProgress);

      const entry: CacheEntry = {
        fileIdentity: {
          name: file.name,
          size: file.size,
        },
        mode,
        rawData: result.rawData,
        thumbnails: result.thumbnails,
        tfData: result.tfData,
      };

      cacheRef.current = entry;

      // Fire-and-forget persist to IndexedDB
      saveToIDB(entry);

      return result;
    },
    [],
  );

  const getCachedMode = useCallback(
    async (file: File): Promise<ScanMode | null> => {
      const entry = await getOrHydrate(file);
      return entry?.mode ?? null;
    },
    [],
  );

  const getThumbnails = useCallback(
    async (file: File): Promise<ThumbnailMap> => {
      const entry = await getOrHydrate(file);
      return entry?.thumbnails ?? new Map();
    },
    [],
  );

  const getTfData = useCallback(
    async (file: File): Promise<TfTreeData | null> => {
      const entry = await getOrHydrate(file);
      return entry?.tfData ?? null;
    },
    [],
  );

  const getTfDataByIdentity = useCallback(
    async (name: string, size: number): Promise<TfTreeData | null> => {
      const identity = { name, size };
      const entry = cacheRef.current;
      if (
        entry &&
        entry.fileIdentity.name === name &&
        entry.fileIdentity.size === size
      ) {
        return entry.tfData ?? null;
      }
      const idbEntry = await loadFromIDB(identity);
      if (idbEntry) {
        cacheRef.current = idbEntry;
        return idbEntry.tfData ?? null;
      }
      return null;
    },
    [],
  );

  const invalidate = useCallback(() => {
    cacheRef.current = null;
    clearIDB();
  }, []);

  return {
    tryGetCachedRaw,
    readAndCache,
    getCachedMode,
    getThumbnails,
    getTfData,
    getTfDataByIdentity,
    invalidate,
  };
}
