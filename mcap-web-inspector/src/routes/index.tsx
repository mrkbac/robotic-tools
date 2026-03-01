import { useState, useCallback, useRef } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Stack, Progress, Alert, Text, Skeleton } from "@mantine/core";
import type { McapInfoOutput, ScanMode } from "../mcap/types.ts";
import { computeStats } from "../mcap/stats.ts";
import { pickRepresentativeThumbnailUrl } from "../mcap/image.ts";
import { FileDropzone } from "../components/FileDropzone.tsx";
import { RecentFiles } from "../components/RecentFiles.tsx";
import { useMcapCache } from "../hooks/useMcapCache.ts";
import { useHistory } from "../hooks/useHistory.ts";
import { encodeToHash } from "../url/codec.ts";
import { createFileId } from "../url/fileId.ts";
import { setFileRef } from "../url/fileRef.ts";
import { setThumbnailRef } from "../url/thumbnailRef.ts";
import { saveFileHandle } from "../stores/fileHandleStore.ts";

function IndexPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [scanMode, setScanMode] = useState<ScanMode>("summary");
  const generationRef = useRef(0);
  const { tryGetCachedRaw, readAndCache, getCachedMode, getThumbnails } = useMcapCache();
  const { entries, addEntry, removeEntry, clearAll } = useHistory();

  const navigateToView = useCallback(
    async (
      data: McapInfoOutput,
      mode: ScanMode,
      file: File,
      handle?: FileSystemFileHandle,
      thumbnailUrl?: string,
    ) => {
      setFileRef(file);
      const fileId = createFileId(file);
      if (handle) {
        saveFileHandle(fileId, handle); // fire-and-forget
      }
      const hash = await encodeToHash(data, mode, fileId);
      addEntry({
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
    [navigate, addEntry],
  );

  const processFile = useCallback(
    async (selectedFile: File, mode: ScanMode, handle?: FileSystemFileHandle) => {
      setError(null);
      setScanMode(mode);

      const cachedRaw = await tryGetCachedRaw(selectedFile, mode);
      if (cachedRaw) {
        const thumbs = await getThumbnails(selectedFile);
        setThumbnailRef(thumbs);
        const thumbUrl = pickRepresentativeThumbnailUrl(thumbs);
        const result = computeStats(cachedRaw, selectedFile.name, selectedFile.size);
        await navigateToView(result, mode, selectedFile, handle, thumbUrl);
        return;
      }

      const gen = ++generationRef.current;
      setLoading(true);
      setProgress(0);

      try {
        const { rawData: raw, thumbnails: thumbs } = await readAndCache(selectedFile, mode, (bytesRead, total) => {
          setProgress(Math.round((bytesRead / total) * 100));
        });

        if (generationRef.current !== gen) return;

        setThumbnailRef(thumbs);
        const thumbUrl = pickRepresentativeThumbnailUrl(thumbs);
        const result = computeStats(raw, selectedFile.name, selectedFile.size);
        await navigateToView(result, mode, selectedFile, handle, thumbUrl);
      } catch (err) {
        if (generationRef.current !== gen) return;
        setError(err instanceof Error ? err.message : "Failed to read MCAP file");
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

  return (
    <>
      <FileDropzone onFileSelect={handleFileSelect} loading={loading} currentFile={null} />

      {loading && (() => {
        const isIndeterminate = scanMode === "summary" && progress === 0;
        return (
          <Stack gap="xs">
            <Text size="sm" c="dimmed">
              {isIndeterminate ? "Reading summary..." : `Scanning file... ${progress}%`}
            </Text>
            <Progress
              value={isIndeterminate ? 100 : progress}
              animated={isIndeterminate}
            />
          </Stack>
        );
      })()}

      {error && (
        <Alert color="red" title="Error">
          {error}
        </Alert>
      )}

      {loading && (
        <Stack gap="lg">
          <Skeleton height={300} radius="md" />
          <Skeleton height={200} radius="md" />
          <Skeleton height={250} radius="md" />
          <Skeleton height={150} radius="md" />
        </Stack>
      )}

      {!loading && (
        <RecentFiles entries={entries} onRemove={removeEntry} onClearAll={clearAll} />
      )}
    </>
  );
}

export const Route = createFileRoute("/")({
  component: IndexPage,
});
