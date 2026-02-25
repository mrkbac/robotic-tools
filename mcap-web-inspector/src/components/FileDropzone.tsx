import { Group, Text } from "@mantine/core";
import { Dropzone } from "@mantine/dropzone";
import { fromEvent } from "file-selector";
import { formatBytes } from "../format.ts";

interface FileDropzoneProps {
  onFileSelect: (file: File, handle?: FileSystemFileHandle) => void;
  loading: boolean;
  currentFile: File | null;
  compact?: boolean;
}

// Module-level variable to stash the handle captured during getFilesFromEvent,
// consumed synchronously in the subsequent onDrop callback.
let capturedHandle: FileSystemFileHandle | undefined;

async function getFilesFromEvent(
  event: unknown,
): Promise<(File | DataTransferItem)[]> {
  // File picker with useFsAccessApi — Mantine/react-dropzone passes an array
  // of FileSystemFileHandle directly. file-selector's getFsHandleFiles converts
  // them to Files but doesn't attach .handle, so stash it here.
  if (
    Array.isArray(event) &&
    event.length > 0 &&
    event[0] instanceof FileSystemFileHandle
  ) {
    capturedHandle = event[0] as FileSystemFileHandle;
  }

  // Delegate to file-selector for all event processing:
  // - Drag events (dragenter/dragover): returns DataTransferItem objects
  //   which bypass react-dropzone's accept filter via isDataTransferItemWithEmptyType().
  // - Drop events: calls getAsFileSystemHandle() and sets file.handle = h.
  // - File picker: converts handles to Files via getFile().
  return fromEvent(event as Event);
}

export function FileDropzone({
  onFileSelect,
  loading,
  currentFile,
  compact,
}: FileDropzoneProps) {
  return (
    <Dropzone
      onDrop={(files) => {
        const file = files[0];
        if (file) {
          // File picker: handle stashed in capturedHandle
          // Drag-and-drop: handle attached by file-selector as file.handle
          const handle =
            capturedHandle ??
            ((file as unknown as { handle?: FileSystemFileHandle }).handle ??
              undefined);
          capturedHandle = undefined;
          onFileSelect(file, handle);
        }
      }}
      getFilesFromEvent={getFilesFromEvent}
      accept={{ "application/octet-stream": [".mcap"] }}
      multiple={false}
      loading={loading}
    >
      <Group justify="center" gap="sm" mih={compact ? 60 : currentFile ? 60 : 300} style={{ pointerEvents: "none" }}>
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
