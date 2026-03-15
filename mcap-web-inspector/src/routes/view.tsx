import { useState, useCallback, useEffect, useRef } from "react";
import {
  createFileRoute,
  useNavigate,
  useLocation,
} from "@tanstack/react-router";
import {
  Stack,
  Alert,
  Text,
  Skeleton,
  Button,
  Tabs,
  Group,
} from "@mantine/core";
import { nprogress } from "@mantine/nprogress";
import {
  IconInfoCircle,
  IconPlugConnected,
  IconFileAnalytics,
  IconDownload,
} from "@tabler/icons-react";
import type { McapInfoOutput, ScanMode } from "../mcap/types.ts";
import type { ThumbnailMap } from "../mcap/image.ts";
import type { TfTreeData } from "../mcap/tf.ts";
import { pickRepresentativeThumbnailUrl } from "../mcap/image.ts";
import { computeStats } from "../mcap/stats.ts";
import { FileHeader } from "../components/FileHeader.tsx";
import { FileInfo } from "../components/FileInfo.tsx";
import { DetailModal, type DetailSection } from "../components/DetailModal.tsx";
import { ChannelsTable } from "../components/channels-table/index.ts";
import { ExportPanel } from "../components/ExportPanel.tsx";
import { TfTreeViewer } from "../components/TfTreeViewer.tsx";
import { TimelapseViewer } from "../components/TimelapseViewer.tsx";
import { ScanLevelIndicator } from "../components/ScanStepper.tsx";
import { useMcapCache, MODE_LEVEL } from "../hooks/useMcapCache.ts";
import { decodeFromHash, encodeToHash } from "../url/codec.ts";
import { createFileId, fileMatchesId } from "../url/fileId.ts";
import { getFileRef, setFileRef } from "../url/fileRef.ts";
import { getThumbnailRef, setThumbnailRef } from "../url/thumbnailRef.ts";
import {
  createMicroThumbFromMap,
  thumbnailBase64ToDataUrl,
} from "../url/thumbnail.ts";
import { useFileHandleRecovery } from "../hooks/useFileHandleRecovery.ts";
import { saveHistoryEntry } from "../stores/historyStore.ts";
import type { ScanProgress } from "../format.ts";
import { formatScanStats } from "../format.ts";

function ViewPage() {
  const navigate = useNavigate();
  const locationHash = useLocation({ select: (l) => l.hash });
  const [data, setData] = useState<McapInfoOutput | null>(null);
  const [scanMode, setScanMode] = useState<ScanMode>("summary");
  const [fileId, setFileId] = useState<string>("");
  const [localFile, setLocalFile] = useState<File | undefined>(getFileRef());

  const [thumbnails, setThumbnails] = useState<ThumbnailMap>(
    () => getThumbnailRef() ?? new Map(),
  );
  const [loading, setLoading] = useState(false);
  const [scanTarget, setScanTarget] = useState<ScanMode | null>(null);
  const [progress, setProgress] = useState(0);
  const [scanStats, setScanStats] = useState<ScanProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [decodeError, setDecodeError] = useState(false);
  const [detailSection, setDetailSection] = useState<DetailSection | null>(
    null,
  );

  const [tfData, setTfData] = useState<TfTreeData | null>(null);
  const [timelapseVideos, setTimelapseVideos] = useState<Map<number, Blob>>(
    () => new Map(),
  );

  const generationRef = useRef(0);
  const scanStartRef = useRef(0);
  const {
    tryGetCachedRaw,
    readAndCache,
    getThumbnails,
    getTfData,
    getTfDataByIdentity,
    getTimelapseVideos,
  } = useMcapCache();
  const [decodedHash, setDecodedHash] = useState<string>("");

  const {
    status: recoveryStatus,
    file: recoveredFile,
    requestAccess,
  } = useFileHandleRecovery(fileId || null);

  // Apply recovered file
  useEffect(() => {
    if (recoveredFile && fileId && fileMatchesId(recoveredFile, fileId)) {
      setFileRef(recoveredFile);
      setLocalFile(recoveredFile);
    }
  }, [recoveredFile, fileId]);

  // Decode hash on mount and when hash changes (e.g. file dropped on /view)
  useEffect(() => {
    const rawHash = locationHash.startsWith("#")
      ? locationHash.slice(1)
      : locationHash;
    if (!rawHash) {
      setDecodeError(true);
      return;
    }
    if (rawHash === decodedHash) return;

    decodeFromHash(rawHash)
      .then((result) => {
        setDecodedHash(rawHash);
        setData(result.data);
        setScanMode(result.scanMode);
        setFileId(result.fileId);
        setDecodeError(false);

        // Reset thumbnails from module-level ref (set during processing)
        const thumbRef = getThumbnailRef();
        if (thumbRef && thumbRef.size > 0) {
          setThumbnails(thumbRef);
        } else {
          setThumbnails(new Map());
        }

        // Check if we have the file via WeakRef
        const ref = getFileRef();
        if (ref && fileMatchesId(ref, result.fileId)) {
          setLocalFile(ref);
        } else {
          setLocalFile(undefined);
        }
      })
      .catch(() => {
        setDecodeError(true);
      });
  }, [locationHash, decodedHash]);

  // Load thumbnails from IDB cache if not already available from ref
  useEffect(() => {
    if (!localFile || thumbnails.size > 0) return;
    getThumbnails(localFile).then((t) => {
      if (t.size > 0) setThumbnails(t);
    });
  }, [localFile, thumbnails.size, getThumbnails]);

  // Load TF data from cache if not already available
  useEffect(() => {
    if (tfData !== null) return;
    if (localFile) {
      getTfData(localFile).then((tf) => {
        if (tf) setTfData(tf);
      });
    } else if (data) {
      getTfDataByIdentity(data.file.path, data.file.size_bytes).then((tf) => {
        if (tf) setTfData(tf);
      });
    }
  }, [localFile, data, tfData, getTfData, getTfDataByIdentity]);

  // Load timelapse videos from cache if not already available
  useEffect(() => {
    if (!localFile || timelapseVideos.size > 0) return;
    getTimelapseVideos(localFile).then((vids) => {
      if (vids.size > 0) setTimelapseVideos(vids);
    });
  }, [localFile, timelapseVideos.size, getTimelapseVideos]);

  const isSharedView = localFile == null;
  const fileUnavailable = isSharedView && recoveryStatus !== "granted";

  const handleScanTo = useCallback(
    async (mode: ScanMode) => {
      if (!localFile || loading) return;
      if (MODE_LEVEL[mode] <= MODE_LEVEL[scanMode]) return;

      setError(null);
      setScanTarget(mode);

      const updateHash = async (
        result: McapInfoOutput,
        m: ScanMode,
        thumbs?: ThumbnailMap,
      ) => {
        const fid = createFileId(localFile);
        const microThumb = thumbs
          ? await createMicroThumbFromMap(thumbs)
          : (data?.thumbnail ?? null);
        const newHash = await encodeToHash(result, m, fid, microThumb);
        saveHistoryEntry({
          fileId: fid,
          fileName: localFile.name,
          fileSize: localFile.size,
          scanMode: m,
          hash: newHash,
          scannedAt: Date.now(),
          thumbnailUrl: pickRepresentativeThumbnailUrl(thumbnails),
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
      scanStartRef.current = Date.now();
      setLoading(true);
      setProgress(0);
      setScanStats(null);
      nprogress.start();

      try {
        const {
          rawData: raw,
          thumbnails: newThumbs,
          tfData: newTfData,
          timelapseVideos: newTimelapseVideos,
        } = await readAndCache(localFile, mode, (bytesRead, total) => {
          const pct = Math.round((bytesRead / total) * 100);
          setProgress(pct);
          setScanStats({
            bytesRead,
            totalBytes: total,
            startTime: scanStartRef.current,
          });
          nprogress.set(pct);
        });

        if (generationRef.current !== gen) return;

        if (newThumbs.size > 0) {
          setThumbnails(newThumbs);
          setThumbnailRef(newThumbs);
        }
        if (newTfData) {
          setTfData(newTfData);
        }
        if (newTimelapseVideos.size > 0) {
          setTimelapseVideos(newTimelapseVideos);
        }
        const result = computeStats(raw, localFile.name, localFile.size);
        setData(result);
        setScanMode(mode);
        await updateHash(
          result,
          mode,
          newThumbs.size > 0 ? newThumbs : undefined,
        );
      } catch (err) {
        if (generationRef.current !== gen) return;
        setError(
          err instanceof Error ? err.message : "Failed to scan MCAP file",
        );
      } finally {
        if (generationRef.current === gen) {
          setLoading(false);
          setScanTarget(null);
          setScanStats(null);
          nprogress.complete();
        }
      }
    },
    [
      localFile,
      loading,
      scanMode,
      data,
      thumbnails,
      tryGetCachedRaw,
      readAndCache,
      navigate,
    ],
  );

  if (decodeError) {
    return (
      <Alert color="red" title="Invalid URL">
        Could not decode data from the URL. The link may be corrupted or
        incomplete.
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
          Go to the home page to open an .mcap file and enable scan upgrades.
        </Alert>
      )}

      {loading &&
        (() => {
          const stats = scanStats ? formatScanStats(scanStats) : "";
          return (
            <Text size="sm" c="dimmed">
              Scanning file... {progress}%{stats ? ` · ${stats}` : ""}
            </Text>
          );
        })()}

      {error && (
        <Alert color="red" title="Error">
          {error}
        </Alert>
      )}

      {(() => {
        const inspectContent = (
          <Stack gap="lg">
            <Group justify="space-between" align="center">
              <FileHeader fileName={data.file.path} />
              <ScanLevelIndicator
                scannedMode={scanMode}
                loading={loading}
                scanTarget={scanTarget}
                onScanTo={handleScanTo}
                disabled={fileUnavailable}
              />
            </Group>
            <FileInfo
              data={data}
              onCountClick={setDetailSection}
              thumbnails={thumbnails}
              fallbackThumbnailUrl={
                data.thumbnail
                  ? thumbnailBase64ToDataUrl(data.thumbnail)
                  : undefined
              }
            />
            <ChannelsTable
              channels={data.channels}
              bucketDurationNs={data.message_distribution.bucket_duration_ns}
              fileSize={data.file.size_bytes}
            />
            {tfData !== null && tfData.transforms.size > 0 && (
              <TfTreeViewer data={tfData} />
            )}
            {timelapseVideos.size > 0 && (
              <TimelapseViewer
                videos={timelapseVideos}
                channelNames={
                  new Map(data.channels.map((ch) => [ch.id, ch.topic]))
                }
              />
            )}
          </Stack>
        );

        return localFile ? (
          <Tabs defaultValue="inspect">
            <Tabs.List>
              <Tabs.Tab
                value="inspect"
                leftSection={<IconFileAnalytics size={16} />}
              >
                Inspect
              </Tabs.Tab>
              <Tabs.Tab value="export" leftSection={<IconDownload size={16} />}>
                Export
              </Tabs.Tab>
            </Tabs.List>
            <Tabs.Panel value="inspect" pt="md">
              {inspectContent}
            </Tabs.Panel>
            <Tabs.Panel value="export" pt="md">
              <ExportPanel file={localFile} data={data} />
            </Tabs.Panel>
          </Tabs>
        ) : (
          inspectContent
        );
      })()}

      <DetailModal
        section={detailSection}
        onClose={() => setDetailSection(null)}
        data={data}
        localFile={localFile}
      />
    </>
  );
}

export const Route = createFileRoute("/view")({
  component: ViewPage,
});
