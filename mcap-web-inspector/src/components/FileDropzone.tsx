import { Group, Text } from "@mantine/core";
import { Dropzone } from "@mantine/dropzone";
import { formatBytes } from "../format.ts";

interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  loading: boolean;
  currentFile: File | null;
}

export function FileDropzone({
  onFileSelect,
  loading,
  currentFile,
}: FileDropzoneProps) {
  return (
    <Dropzone
      onDrop={(files) => {
        const file = files[0];
        if (file) onFileSelect(file);
      }}
      accept={{ "application/octet-stream": [".mcap"] }}
      multiple={false}
      loading={loading}
    >
      <Group justify="center" gap="sm" mih={currentFile ? 60 : 300} style={{ pointerEvents: "none" }}>
        <div>
          <Text size="lg" fw={500} c="dimmed" ta="center">
            {currentFile
              ? currentFile.name
              : "Drop an .mcap file here or click to browse"}
          </Text>
          {currentFile && (
            <Text size="sm" c="dimmed" ta="center" mt={4}>
              {formatBytes(currentFile.size)}
            </Text>
          )}
        </div>
      </Group>
    </Dropzone>
  );
}
