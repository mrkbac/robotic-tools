import { useState, useCallback, useRef } from "react";
import {
  Container,
  Title,
  Stack,
  SegmentedControl,
  Progress,
  Alert,
  Group,
  Text,
} from "@mantine/core";
import type { McapInfoOutput, ScanMode } from "./mcap/types.ts";
import { computeStats } from "./mcap/stats.ts";
import { FileDropzone } from "./components/FileDropzone.tsx";
import { FileInfo } from "./components/FileInfo.tsx";
import { CompressionTable } from "./components/CompressionTable.tsx";
import { ChannelsTable } from "./components/ChannelsTable.tsx";
import { SchemasTable } from "./components/SchemasTable.tsx";
import { UnifiedDistributionChart } from "./components/UnifiedDistributionChart.tsx";
import { useMcapCache } from "./hooks/useMcapCache.ts";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [scanMode, setScanMode] = useState<ScanMode>("summary");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<McapInfoOutput | null>(null);
  const generationRef = useRef(0);
  const { tryGetCachedRaw, readAndCache } = useMcapCache();

  const processFile = useCallback(
    async (selectedFile: File, mode: ScanMode) => {
      setFile(selectedFile);
      setError(null);

      const cachedRaw = tryGetCachedRaw(selectedFile, mode);
      if (cachedRaw) {
        const result = computeStats(cachedRaw, selectedFile.name, selectedFile.size);
        setData(result);
        return;
      }

      const gen = ++generationRef.current;
      setLoading(true);
      setProgress(0);
      setData(null);

      try {
        const raw = await readAndCache(selectedFile, mode, (bytesRead, total) => {
          setProgress(Math.round((bytesRead / total) * 100));
        });

        if (generationRef.current !== gen) return;

        const result = computeStats(raw, selectedFile.name, selectedFile.size);
        setData(result);
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
    [tryGetCachedRaw, readAndCache],
  );

  const handleModeChange = useCallback(
    (value: string) => {
      const mode = value as ScanMode;
      setScanMode(mode);
      if (file) {
        processFile(file, mode);
      }
    },
    [file, processFile],
  );

  return (
    <Container size="xl" py="xl">
      <Stack gap="lg">
        <Group justify="space-between" align="flex-end">
          <Title order={2}>MCAP Web Inspector</Title>
          <SegmentedControl
            value={scanMode}
            onChange={handleModeChange}
            data={[
              {
                value: "summary",
                label: "Summary (fast)",
              },
              {
                value: "rebuild",
                label: "Rebuild (scan)",
              },
              {
                value: "exact",
                label: "Exact (slow)",
              },
            ]}
          />
        </Group>

        <FileDropzone
          onFileSelect={processFile}
          loading={loading}
          currentFile={file}
        />

        {loading && (
          <Stack gap="xs">
            <Text size="sm" c="dimmed">
              {scanMode === "summary"
                ? "Reading summary..."
                : `Scanning file... ${progress}%`}
            </Text>
            <Progress
              value={scanMode === "summary" ? 100 : progress}
              animated={scanMode === "summary"}
            />
          </Stack>
        )}

        {error && (
          <Alert color="red" title="Error">
            {error}
          </Alert>
        )}

        {data && (
          <Stack gap="lg">
            <FileInfo data={data} />
            <CompressionTable data={data} />
            <UnifiedDistributionChart
              channels={data.channels}
              globalDistribution={data.messageDistribution}
            />
            <SchemasTable schemas={data.schemas} />
            <ChannelsTable channels={data.channels} bucketDurationNs={data.messageDistribution.bucketDurationNs} fileSize={data.file.sizeBytes} />
          </Stack>
        )}
      </Stack>
    </Container>
  );
}
