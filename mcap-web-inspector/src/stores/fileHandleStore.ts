import { openDB } from "idb";

const DB_NAME = "mcap-file-handles";
const STORE_NAME = "handles";
const DB_VERSION = 1;

interface HandleEntry {
  fileId: string;
  handle: FileSystemFileHandle;
}

function getDB() {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "fileId" });
      }
    },
  });
}

export async function saveFileHandle(
  fileId: string,
  handle: FileSystemFileHandle,
): Promise<void> {
  try {
    const db = await getDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    await tx.store.clear();
    await tx.store.put({ fileId, handle } satisfies HandleEntry);
    await tx.done;
  } catch {
    // best-effort
  }
}

export async function loadFileHandle(
  fileId: string,
): Promise<FileSystemFileHandle | null> {
  try {
    const db = await getDB();
    const entry = (await db.get(STORE_NAME, fileId)) as HandleEntry | undefined;
    return entry?.handle ?? null;
  } catch {
    return null;
  }
}
