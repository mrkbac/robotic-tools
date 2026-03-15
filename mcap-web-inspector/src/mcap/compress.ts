import lz4 from "lz4js";
import type { CompressionAlgorithm } from "./types.ts";

let zstdCompressReady: Promise<void> | null = null;
let zstdCompress: ((data: Uint8Array, level: number) => Uint8Array) | null =
  null;

/** Lazy-load and initialize @bokuweb/zstd-wasm for compression. */
export async function ensureZstdCompressInit(): Promise<void> {
  if (zstdCompressReady) {
    await zstdCompressReady;
    return;
  }
  zstdCompressReady = (async () => {
    const mod = await import("@bokuweb/zstd-wasm");
    await mod.init();
    zstdCompress = mod.compress;
  })();
  await zstdCompressReady;
}

type CompressChunkFn = (data: Uint8Array) => {
  compression: string;
  compressedData: Uint8Array;
};

/**
 * Build the `compressChunk` function for McapWriter.
 * Returns `undefined` for "none" (no compression).
 */
export function buildCompressChunk(
  algorithm: CompressionAlgorithm,
): CompressChunkFn | undefined {
  switch (algorithm) {
    case "zstd":
      return (data: Uint8Array) => ({
        compression: "zstd",
        compressedData: zstdCompress!(data, 3),
      });
    case "lz4":
      return (data: Uint8Array) => ({
        compression: "lz4",
        compressedData: new Uint8Array(lz4.compress(data)),
      });
    case "none":
      return undefined;
  }
}
