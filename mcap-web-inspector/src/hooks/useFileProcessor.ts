import { useState, useCallback, useRef } from "react";
import { useNavigate } from "@tanstack/react-router";
import type { McapInfoOutput, ScanMode } from "../mcap/types.ts";
import { computeStats } from "../mcap/stats.ts";
import { pickRepresentativeThumbnailUrl } from "../mcap/image.ts";
import { useMcapCache } from "./useMcapCache.ts";
import { useHistory } from "./useHistory.ts";
import { encodeToHash } from "../url/codec.ts";
import { createFileId } from "../url/fileId.ts";
import { setFileRef } from "../url/fileRef.ts";
import { setThumbnailRef } from "../url/thumbnailRef.ts";
import { createMicroThumbFromMap } from "../url/thumbnail.ts";
import { saveFileHandle } from "../stores/fileHandleStore.ts";

export function useFileProcessor() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [scanMode, setScanMode] = useState<ScanMode>("summary");
  const generationRef = useRef(0);
  const { tryGetCachedRaw, readAndCache, getCachedMode, getThumbnails } =
    useMcapCache();
  const history = useHistory();

  const navigateToView = useCallback(
    async (
      data: McapInfoOutput,
      mode: ScanMode,
      file: File,
      handle?: FileSystemFileHandle,
      thumbnailUrl?: string,
      microThumb?: string | null,
    ) => {
      setFileRef(file);
      const fileId = createFileId(file);
      if (handle) {
        saveFileHandle(fileId, handle);
      }
      const hash = await encodeToHash(data, mode, fileId, microThumb);
      history.addEntry({
        fileId,
        fileName: file.name,
        fileSize: file.size,
        scanMode: mode,
        hash,
        scannedAt: Date.now(),
        thumbnailUrl,
      });
      navigate({ to: "/view", hash });
    },
    [navigate, history.addEntry],
  );

  const processFile = useCallback(
    async (
      selectedFile: File,
      mode: ScanMode,
      handle?: FileSystemFileHandle,
    ) => {
      setError(null);
      setScanMode(mode);

      const cachedRaw = await tryGetCachedRaw(selectedFile, mode);
      if (cachedRaw) {
        const thumbs = await getThumbnails(selectedFile);
        setThumbnailRef(thumbs);
        const thumbUrl = pickRepresentativeThumbnailUrl(thumbs);
        const microThumb = await createMicroThumbFromMap(thumbs);
        const result = computeStats(
          cachedRaw,
          selectedFile.name,
          selectedFile.size,
        );
        await navigateToView(
          result,
          mode,
          selectedFile,
          handle,
          thumbUrl,
          microThumb,
        );
        return;
      }

      const gen = ++generationRef.current;
      setLoading(true);
      setProgress(0);

      try {
        const { rawData: raw, thumbnails: thumbs } = await readAndCache(
          selectedFile,
          mode,
          (bytesRead, total) => {
            setProgress(Math.round((bytesRead / total) * 100));
          },
        );

        if (generationRef.current !== gen) return;

        setThumbnailRef(thumbs);
        const thumbUrl = pickRepresentativeThumbnailUrl(thumbs);
        const microThumb = await createMicroThumbFromMap(thumbs);
        const result = computeStats(raw, selectedFile.name, selectedFile.size);
        await navigateToView(
          result,
          mode,
          selectedFile,
          handle,
          thumbUrl,
          microThumb,
        );
      } catch (err) {
        if (generationRef.current !== gen) return;
        setError(
          err instanceof Error ? err.message : "Failed to read MCAP file",
        );
      } finally {
        if (generationRef.current === gen) {
          setLoading(false);
        }
      }
    },
    [tryGetCachedRaw, readAndCache, getThumbnails, navigateToView],
  );

  const handleFileSelect = useCallback(
    async (selectedFile: File, handle?: FileSystemFileHandle) => {
      const mode = (await getCachedMode(selectedFile)) ?? "summary";
      processFile(selectedFile, mode, handle);
    },
    [processFile, getCachedMode],
  );

  /** Process multiple files. All get cached/historied; the first one navigates to /view. */
  const handleFilesSelect = useCallback(
    async (files: File[], handles?: Map<File, FileSystemFileHandle>) => {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]!;
        const handle = handles?.get(file);
        const mode = (await getCachedMode(file)) ?? "summary";

        if (i === 0) {
          // First file — navigate to it
          await processFile(file, mode, handle);
        } else {
          // Remaining files — just process into cache + history without navigating
          setError(null);
          setScanMode(mode);

          const cachedRaw = await tryGetCachedRaw(file, mode);
          if (cachedRaw) {
            const thumbs = await getThumbnails(file);
            setThumbnailRef(thumbs);
            const thumbUrl = pickRepresentativeThumbnailUrl(thumbs);
            const microThumb = await createMicroThumbFromMap(thumbs);
            const result = computeStats(cachedRaw, file.name, file.size);
            const fileId = createFileId(file);
            if (handle) saveFileHandle(fileId, handle);
            const hash = await encodeToHash(result, mode, fileId, microThumb);
            history.addEntry({
              fileId,
              fileName: file.name,
              fileSize: file.size,
              scanMode: mode,
              hash,
              scannedAt: Date.now(),
              thumbnailUrl: thumbUrl,
            });
            continue;
          }

          const gen = ++generationRef.current;
          setLoading(true);
          setProgress(0);

          try {
            const { rawData: raw, thumbnails: thumbs } = await readAndCache(
              file,
              mode,
              (bytesRead, total) => {
                setProgress(Math.round((bytesRead / total) * 100));
              },
            );
            if (generationRef.current !== gen) return;
            setThumbnailRef(thumbs);
            const thumbUrl = pickRepresentativeThumbnailUrl(thumbs);
            const microThumb = await createMicroThumbFromMap(thumbs);
            const result = computeStats(raw, file.name, file.size);
            const fileId = createFileId(file);
            if (handle) saveFileHandle(fileId, handle);
            const hash = await encodeToHash(result, mode, fileId, microThumb);
            history.addEntry({
              fileId,
              fileName: file.name,
              fileSize: file.size,
              scanMode: mode,
              hash,
              scannedAt: Date.now(),
              thumbnailUrl: thumbUrl,
            });
          } catch (err) {
            if (generationRef.current !== gen) return;
            setError(
              err instanceof Error ? err.message : "Failed to read MCAP file",
            );
          } finally {
            if (generationRef.current === gen) {
              setLoading(false);
            }
          }
        }
      }
    },
    [
      processFile,
      getCachedMode,
      tryGetCachedRaw,
      readAndCache,
      getThumbnails,
      history.addEntry,
    ],
  );

  return {
    loading,
    progress,
    error,
    scanMode,
    handleFileSelect,
    handleFilesSelect,
    history,
  };
}
