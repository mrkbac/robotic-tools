import { Group, Text, rem } from "@mantine/core";
import type { ScanMode } from "../mcap/types.ts";

interface FileDropzoneProps {
  onFileSelect: (file: File, mode: ScanMode) => void;
  loading: boolean;
  currentFile: File | null;
}

export function FileDropzone({
  onFileSelect,
  loading,
  currentFile,
}: FileDropzoneProps) {
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".mcap")) {
      onFileSelect(file, "summary");
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleClick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".mcap";
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) {
        onFileSelect(file, "summary");
      }
    };
    input.click();
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={loading ? undefined : handleClick}
      style={{
        border: `${rem(2)} dashed var(--mantine-color-dimmed)`,
        borderRadius: rem(12),
        padding: rem(32),
        textAlign: "center",
        cursor: loading ? "wait" : "pointer",
        transition: "border-color 150ms ease",
        opacity: loading ? 0.6 : 1,
      }}
    >
      <Group justify="center" gap="sm">
        <Text size="lg" fw={500} c="dimmed">
          {currentFile
            ? currentFile.name
            : "Drop an .mcap file here or click to browse"}
        </Text>
      </Group>
      {currentFile && (
        <Text size="sm" c="dimmed" mt={4}>
          {formatBytes(currentFile.size)}
        </Text>
      )}
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}
