import { useState, useCallback, useEffect, useRef } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Stack, Progress, Alert, Text, Skeleton, Button } from "@mantine/core";
import { IconInfoCircle, IconPlugConnected } from "@tabler/icons-react";
import type { McapInfoOutput, ScanMode } from "../mcap/types.ts";
import { computeStats } from "../mcap/stats.ts";
import { FileInfo } from "../components/FileInfo.tsx";
import { CompressionTable } from "../components/CompressionTable.tsx";
import { ChannelsTable } from "../components/ChannelsTable.tsx";
import { SchemasTable } from "../components/SchemasTable.tsx";
import { MetadataTable } from "../components/MetadataTable.tsx";
import { AttachmentsTable } from "../components/AttachmentsTable.tsx";
import { ExportPanel } from "../components/ExportPanel.tsx";
import { UnifiedDistributionChart } from "../components/UnifiedDistributionChart.tsx";
import { ScanStepper } from "../components/ScanStepper.tsx";
import { FileDropzone } from "../components/FileDropzone.tsx";
import { useMcapCache, MODE_LEVEL } from "../hooks/useMcapCache.ts";
import { decodeFromHash, encodeToHash } from "../url/codec.ts";
import { createFileId, fileMatchesId } from "../url/fileId.ts";
import { getFileRef, setFileRef } from "../url/fileRef.ts";
import { useFileHandleRecovery } from "../hooks/useFileHandleRecovery.ts";
import { saveFileHandle } from "../stores/fileHandleStore.ts";
import { saveHistoryEntry } from "../stores/historyStore.ts";

function ViewPage() {
  const navigate = useNavigate();
  const [data, setData] = useState<McapInfoOutput | null>(null);
  const [scanMode, setScanMode] = useState<ScanMode>("summary");
  const [fileId, setFileId] = useState<string>("");
  const [localFile, setLocalFile] = useState<File | undefined>(getFileRef());

  const [loading, setLoading] = useState(false);
  const [scanTarget, setScanTarget] = useState<ScanMode | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [decodeError, setDecodeError] = useState(false);

  const generationRef = useRef(0);
  const { tryGetCachedRaw, readAndCache, getCachedMode } = useMcapCache();
  const decodedRef = useRef(false);

  const { status: recoveryStatus, file: recoveredFile, requestAccess } =
    useFileHandleRecovery(fileId || null);

  // Apply recovered file
  useEffect(() => {
    if (recoveredFile && fileId && fileMatchesId(recoveredFile, fileId)) {
      setFileRef(recoveredFile);
      setLocalFile(recoveredFile);
    }
  }, [recoveredFile, fileId]);

  // Decode hash on mount
  useEffect(() => {
    if (decodedRef.current) return;
    decodedRef.current = true;

    const rawHash = location.hash.slice(1);
    if (!rawHash) {
      setDecodeError(true);
      return;
    }

    decodeFromHash(rawHash)
      .then((result) => {
        setData(result.data);
        setScanMode(result.scanMode);
        setFileId(result.fileId);

        // Check if we have the file via WeakRef
        const ref = getFileRef();
        if (ref && fileMatchesId(ref, result.fileId)) {
          setLocalFile(ref);
        }
      })
      .catch(() => {
        setDecodeError(true);
      });
  }, []);

  const isSharedView = localFile == null;
  const fileUnavailable = isSharedView && recoveryStatus !== "granted";

  const handleFileAssociation = useCallback(
    async (file: File, handle?: FileSystemFileHandle) => {
      // Matching file: associate for scan upgrades
      if (fileId && fileMatchesId(file, fileId)) {
        setFileRef(file);
        setLocalFile(file);
        if (handle) {
          saveFileHandle(fileId, handle);
        }
        return;
      }

      // Different file: process as a brand new scan
      const newFileId = createFileId(file);
      setFileRef(file);
      setLocalFile(file);
      setError(null);
      if (handle) {
        saveFileHandle(newFileId, handle);
      }

      try {
        const cachedMode = await getCachedMode(file);
        const initialMode: ScanMode = cachedMode ?? "summary";

        const cachedRaw = await tryGetCachedRaw(file, initialMode);
        if (cachedRaw) {
          const result = computeStats(cachedRaw, file.name, file.size);
          setData(result);
          setScanMode(initialMode);
          setFileId(newFileId);
          const newHash = await encodeToHash(result, initialMode, newFileId);
          saveHistoryEntry({
            fileId: newFileId,
            fileName: file.name,
            fileSize: file.size,
            scanMode: initialMode,
            hash: newHash,
            scannedAt: Date.now(),
          });
          navigate({ to: "/view", hash: newHash });
          return;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load cached scan");
        return;
      }

      const gen = ++generationRef.current;
      setLoading(true);
      setProgress(0);

      try {
        const raw = await readAndCache(file, "summary", (bytesRead, total) => {
          setProgress(Math.round((bytesRead / total) * 100));
        });

        if (generationRef.current !== gen) return;

        const result = computeStats(raw, file.name, file.size);
        setData(result);
        setScanMode("summary");
        setFileId(newFileId);
        const newHash = await encodeToHash(result, "summary", newFileId);
        saveHistoryEntry({
          fileId: newFileId,
          fileName: file.name,
          fileSize: file.size,
          scanMode: "summary",
          hash: newHash,
          scannedAt: Date.now(),
        });
        navigate({ to: "/view", hash: newHash });
      } catch (err) {
        if (generationRef.current !== gen) return;
        setError(err instanceof Error ? err.message : "Failed to scan MCAP file");
      } finally {
        if (generationRef.current === gen) {
          setLoading(false);
          setScanTarget(null);
        }
      }
    },
    [fileId, getCachedMode, tryGetCachedRaw, readAndCache, navigate],
  );

  const handleScanTo = useCallback(
    async (mode: ScanMode) => {
      if (!localFile || loading) return;
      if (MODE_LEVEL[mode] <= MODE_LEVEL[scanMode]) return;

      setError(null);
      setScanTarget(mode);

      const updateHash = async (result: McapInfoOutput, m: ScanMode) => {
        const fid = createFileId(localFile);
        const newHash = await encodeToHash(result, m, fid);
        saveHistoryEntry({
          fileId: fid,
          fileName: localFile.name,
          fileSize: localFile.size,
          scanMode: m,
          hash: newHash,
          scannedAt: Date.now(),
        });
        navigate({ to: "/view", hash: newHash, replace: true });
      };

      const cachedRaw = await tryGetCachedRaw(localFile, mode);
      if (cachedRaw) {
        const result = computeStats(cachedRaw, localFile.name, localFile.size);
        setData(result);
        setScanMode(mode);
        setScanTarget(null);
        await updateHash(result, mode);
        return;
      }

      const gen = ++generationRef.current;
      setLoading(true);
      setProgress(0);

      try {
        const raw = await readAndCache(localFile, mode, (bytesRead, total) => {
          setProgress(Math.round((bytesRead / total) * 100));
        });

        if (generationRef.current !== gen) return;

        const result = computeStats(raw, localFile.name, localFile.size);
        setData(result);
        setScanMode(mode);
        await updateHash(result, mode);
      } catch (err) {
        if (generationRef.current !== gen) return;
        setError(err instanceof Error ? err.message : "Failed to scan MCAP file");
      } finally {
        if (generationRef.current === gen) {
          setLoading(false);
          setScanTarget(null);
        }
      }
    },
    [localFile, loading, scanMode, tryGetCachedRaw, readAndCache, navigate],
  );

  if (decodeError) {
    return (
      <Alert color="red" title="Invalid URL">
        Could not decode data from the URL. The link may be corrupted or incomplete.
      </Alert>
    );
  }

  if (!data) {
    return (
      <Stack gap="lg">
        <Skeleton height={300} radius="md" />
        <Skeleton height={200} radius="md" />
        <Skeleton height={250} radius="md" />
      </Stack>
    );
  }

  return (
    <>
      {recoveryStatus === "loading" && isSharedView && (
        <Skeleton height={60} radius="md" />
      )}

      {recoveryStatus === "prompt" && isSharedView && (
        <Alert
          variant="light"
          color="blue"
          icon={<IconPlugConnected size={18} />}
          title="File available — permission needed"
        >
          <Text size="sm" mb="xs">
            Click below to reconnect the file and enable scan upgrades.
          </Text>
          <Button size="xs" variant="light" onClick={requestAccess}>
            Reconnect file
          </Button>
        </Alert>
      )}

      {recoveryStatus === "idle" && isSharedView && (
        <Alert
          variant="light"
          color="yellow"
          icon={<IconInfoCircle size={18} />}
          title="Viewing shared data"
        >
          Drop the .mcap file to enable scan upgrades, or drop a different file to inspect it.
        </Alert>
      )}

      <FileDropzone
        onFileSelect={handleFileAssociation}
        loading={false}
        currentFile={localFile ?? null}
        compact
      />

      <ScanStepper
        scannedMode={scanMode}
        loading={loading}
        scanTarget={scanTarget}
        onScanTo={handleScanTo}
        disabled={fileUnavailable}
      />

      {loading && (
        <Stack gap="xs">
          <Text size="sm" c="dimmed">
            Scanning file... {progress}%
          </Text>
          <Progress value={progress} />
        </Stack>
      )}

      {error && (
        <Alert color="red" title="Error">
          {error}
        </Alert>
      )}

      <Stack gap="lg">
        <FileInfo data={data} />
        <MetadataTable metadata={data.metadata} />
        <CompressionTable data={data} />
        <UnifiedDistributionChart
          channels={data.channels}
          globalDistribution={data.messageDistribution}
        />
        <SchemasTable schemas={data.schemas} />
        <AttachmentsTable
          attachments={data.attachments}
          localFile={localFile}
        />
        <ChannelsTable
          channels={data.channels}
          bucketDurationNs={data.messageDistribution.bucketDurationNs}
          fileSize={data.file.sizeBytes}
        />
      </Stack>

      {localFile && data && <ExportPanel file={localFile} data={data} />}
    </>
  );
}

export const Route = createFileRoute("/view")({
  component: ViewPage,
});
