import { useState, useCallback, useRef } from "react";
import {
  Container,
  Title,
  Stack,
  Progress,
  Alert,
  Group,
  Text,
  ActionIcon,
  Skeleton,
  useMantineColorScheme,
  useComputedColorScheme,
} from "@mantine/core";
import { IconSun, IconMoon } from "@tabler/icons-react";
import type { McapInfoOutput, ScanMode } from "./mcap/types.ts";
import { computeStats } from "./mcap/stats.ts";
import { FileDropzone } from "./components/FileDropzone.tsx";
import { FileInfo } from "./components/FileInfo.tsx";
import { CompressionTable } from "./components/CompressionTable.tsx";
import { ChannelsTable } from "./components/ChannelsTable.tsx";
import { SchemasTable } from "./components/SchemasTable.tsx";
import { UnifiedDistributionChart } from "./components/UnifiedDistributionChart.tsx";
import { ScanUpgradeBar } from "./components/ScanUpgradeBar.tsx";
import { useMcapCache } from "./hooks/useMcapCache.ts";

function ColorSchemeToggle() {
  const { setColorScheme } = useMantineColorScheme();
  const computed = useComputedColorScheme("light");
  return (
    <ActionIcon
      variant="default"
      size="lg"
      onClick={() => setColorScheme(computed === "light" ? "dark" : "light")}
      aria-label="Toggle color scheme"
    >
      {computed === "light" ? <IconMoon size={18} /> : <IconSun size={18} />}
    </ActionIcon>
  );
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [scannedMode, setScannedMode] = useState<ScanMode>("summary");
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
        setScannedMode(mode);
        return;
      }

      const gen = ++generationRef.current;
      setLoading(true);
      setProgress(0);

      try {
        const raw = await readAndCache(selectedFile, mode, (bytesRead, total) => {
          setProgress(Math.round((bytesRead / total) * 100));
        });

        if (generationRef.current !== gen) return;

        const result = computeStats(raw, selectedFile.name, selectedFile.size);
        setData(result);
        setScannedMode(mode);
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

  const handleFileSelect = useCallback(
    (selectedFile: File) => {
      setScannedMode("summary");
      processFile(selectedFile, "summary");
    },
    [processFile],
  );

  const nextMode: Record<string, ScanMode> = { summary: "rebuild", rebuild: "exact" };

  const handleUpgrade = useCallback(() => {
    if (!file || scannedMode === "exact") return;
    processFile(file, nextMode[scannedMode]!);
  }, [file, scannedMode, processFile]);

  return (
    <Container size="xl" py="xl">
      <Stack gap="lg">
        <Group justify="space-between" align="flex-end">
          <Title order={2}>MCAP Web Inspector</Title>
          <ColorSchemeToggle />
        </Group>

        <FileDropzone
          onFileSelect={handleFileSelect}
          loading={loading}
          currentFile={file}
        />

        {loading && (
          <Stack gap="xs">
            <Text size="sm" c="dimmed">
              {scannedMode === "summary" && progress === 0
                ? "Reading summary..."
                : `Scanning file... ${progress}%`}
            </Text>
            <Progress
              value={scannedMode === "summary" && progress === 0 ? 100 : progress}
              animated={scannedMode === "summary" && progress === 0}
            />
          </Stack>
        )}

        {error && (
          <Alert color="red" title="Error">
            {error}
          </Alert>
        )}

        {data && (
          <ScanUpgradeBar
            scannedMode={scannedMode}
            loading={loading}
            onUpgrade={handleUpgrade}
          />
        )}

        {loading && !data && (
          <Stack gap="lg">
            <Skeleton height={300} radius="md" />
            <Skeleton height={200} radius="md" />
            <Skeleton height={250} radius="md" />
            <Skeleton height={150} radius="md" />
          </Stack>
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
