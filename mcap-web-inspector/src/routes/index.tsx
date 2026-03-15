import { createFileRoute } from "@tanstack/react-router";
import { Stack, Alert, Text, Skeleton } from "@mantine/core";
import { FileDropzone } from "../components/FileDropzone.tsx";
import { RecentFiles } from "../components/RecentFiles.tsx";
import { useFileProcessorContext } from "../contexts/FileProcessorContext.tsx";
import { formatScanStats } from "../format.ts";

function IndexPage() {
  const {
    loading,
    progress,
    scanStats,
    error,
    scanMode,
    handleFileSelect,
    history,
  } = useFileProcessorContext();

  return (
    <>
      <FileDropzone
        onFileSelect={handleFileSelect}
        loading={loading}
        currentFile={null}
      />

      {loading &&
        (() => {
          const isIndeterminate = scanMode === "summary" && progress === 0;
          const stats = scanStats ? formatScanStats(scanStats) : "";
          return (
            <Text size="sm" c="dimmed">
              {isIndeterminate
                ? "Reading summary..."
                : `Scanning file... ${progress}%${stats ? ` · ${stats}` : ""}`}
            </Text>
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
        <RecentFiles
          entries={history.entries}
          onRemove={history.removeEntry}
          onClearAll={history.clearAll}
        />
      )}
    </>
  );
}

export const Route = createFileRoute("/")({
  component: IndexPage,
});
