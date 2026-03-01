import { openDB } from "idb";
import type { ScanMode } from "../mcap/types.ts";

const DB_NAME = "mcap-history";
const STORE_NAME = "entries";
const DB_VERSION = 2;

export interface HistoryEntry {
  fileId: string;
  fileName: string;
  fileSize: number;
  scanMode: ScanMode;
  hash: string;
  scannedAt: number;
  /** Data URL of a representative thumbnail image (if available). */
  thumbnailUrl?: string;
}

function getDB() {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      // Recreate store on any version bump (thumbnailUrl added in v2)
      if (db.objectStoreNames.contains(STORE_NAME)) {
        db.deleteObjectStore(STORE_NAME);
      }
      const store = db.createObjectStore(STORE_NAME, { keyPath: "fileId" });
      store.createIndex("scannedAt", "scannedAt");
    },
  });
}

export async function saveHistoryEntry(entry: HistoryEntry): Promise<void> {
  try {
    const db = await getDB();
    await db.put(STORE_NAME, entry);
  } catch {
    // best-effort
  }
}

export async function loadHistory(): Promise<HistoryEntry[]> {
  try {
    const db = await getDB();
    const all = await db.getAllFromIndex(STORE_NAME, "scannedAt");
    return all.reverse(); // newest first
  } catch {
    return [];
  }
}

export async function removeHistoryEntry(fileId: string): Promise<void> {
  try {
    const db = await getDB();
    await db.delete(STORE_NAME, fileId);
  } catch {
    // best-effort
  }
}

export async function clearHistory(): Promise<void> {
  try {
    const db = await getDB();
    await db.clear(STORE_NAME);
  } catch {
    // best-effort
  }
}
