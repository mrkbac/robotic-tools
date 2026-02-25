import { useRef, useCallback } from "react";
import type { McapRawData, ProgressCallback } from "../mcap/reader.ts";
import { readMcapFile } from "../mcap/reader.ts";
import type { ScanMode } from "../mcap/types.ts";

interface FileIdentity {
  name: string;
  size: number;
  lastModified: number;
}

function isSameFile(file: File, identity: FileIdentity): boolean {
  return (
    file.name === identity.name &&
    file.size === identity.size &&
    file.lastModified === identity.lastModified
  );
}

const MODE_LEVEL: Record<ScanMode, number> = {
  summary: 0,
  rebuild: 1,
  exact: 2,
};

function deriveRawData(raw: McapRawData, from: ScanMode, to: ScanMode): McapRawData {
  if (from === to) return raw;

  const fromLevel = MODE_LEVEL[from];
  const toLevel = MODE_LEVEL[to];

  if (toLevel > fromLevel) {
    throw new Error(`Cannot derive ${to} from ${from}`);
  }

  if (from === "exact" && to === "rebuild") {
    return { ...raw, channelSizes: null };
  }
  if (from === "exact" && to === "summary") {
    return { ...raw, channelSizes: null, chunkInformation: null };
  }
  if (from === "rebuild" && to === "summary") {
    return { ...raw, chunkInformation: null };
  }

  return raw;
}

interface CacheEntry {
  fileIdentity: FileIdentity;
  mode: ScanMode;
  rawData: McapRawData;
}

export function useMcapCache() {
  const cacheRef = useRef<CacheEntry | null>(null);

  const tryGetCachedRaw = useCallback(
    (file: File, mode: ScanMode): McapRawData | null => {
      const entry = cacheRef.current;
      if (!entry) return null;
      if (!isSameFile(file, entry.fileIdentity)) return null;

      const cachedLevel = MODE_LEVEL[entry.mode];
      const requestedLevel = MODE_LEVEL[mode];

      if (cachedLevel < requestedLevel) return null;

      return deriveRawData(entry.rawData, entry.mode, mode);
    },
    [],
  );

  const readAndCache = useCallback(
    async (
      file: File,
      mode: ScanMode,
      onProgress?: ProgressCallback,
    ): Promise<McapRawData> => {
      const raw = await readMcapFile(file, mode, onProgress);

      cacheRef.current = {
        fileIdentity: {
          name: file.name,
          size: file.size,
          lastModified: file.lastModified,
        },
        mode,
        rawData: raw,
      };

      return raw;
    },
    [],
  );

  const invalidate = useCallback(() => {
    cacheRef.current = null;
  }, []);

  return { tryGetCachedRaw, readAndCache, invalidate };
}
