import { useEffect, useRef, useState, useCallback } from "react";
import { Overlay, Text, Stack } from "@mantine/core";
import { IconFileUpload } from "@tabler/icons-react";

interface GlobalDropZoneProps {
  onFilesSelect: (
    files: File[],
    handles?: Map<File, FileSystemFileHandle>,
  ) => void;
}

export function GlobalDropZone({ onFilesSelect }: GlobalDropZoneProps) {
  const [active, setActive] = useState(false);
  const dragCounterRef = useRef(0);

  const handleDragEnter = useCallback((e: DragEvent) => {
    e.preventDefault();
    if (!e.dataTransfer?.types.includes("Files")) return;
    dragCounterRef.current++;
    if (dragCounterRef.current === 1) {
      setActive(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    dragCounterRef.current--;
    if (dragCounterRef.current === 0) {
      setActive(false);
    }
  }, []);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback(
    async (e: DragEvent) => {
      e.preventDefault();
      dragCounterRef.current = 0;
      setActive(false);

      // Skip if drop target is inside the Mantine Dropzone (let it handle it)
      if (
        e.target instanceof Element &&
        e.target.closest(".mantine-Dropzone-root")
      ) {
        return;
      }

      const items = e.dataTransfer?.items;
      if (!items) return;

      const files: File[] = [];
      const handles = new Map<File, FileSystemFileHandle>();

      const promises: Promise<void>[] = [];

      for (let i = 0; i < items.length; i++) {
        const item = items[i]!;
        if (item.kind !== "file") continue;

        promises.push(
          (async () => {
            // Try to get FileSystemFileHandle (Chromium only)
            let handle: FileSystemFileHandle | undefined;
            if ("getAsFileSystemHandle" in item) {
              try {
                const h = await (
                  item as DataTransferItem & {
                    getAsFileSystemHandle: () => Promise<FileSystemHandle>;
                  }
                ).getAsFileSystemHandle();
                if (h?.kind === "file") {
                  handle = h as FileSystemFileHandle;
                }
              } catch {
                // fallback to getAsFile
              }
            }

            const file = handle
              ? await handle.getFile()
              : item.getAsFile();
            if (!file) return;
            if (!file.name.endsWith(".mcap")) return;

            files.push(file);
            if (handle) handles.set(file, handle);
          })(),
        );
      }

      await Promise.all(promises);

      if (files.length > 0) {
        onFilesSelect(files, handles.size > 0 ? handles : undefined);
      }
    },
    [onFilesSelect],
  );

  useEffect(() => {
    window.addEventListener("dragenter", handleDragEnter);
    window.addEventListener("dragleave", handleDragLeave);
    window.addEventListener("dragover", handleDragOver);
    window.addEventListener("drop", handleDrop);

    return () => {
      window.removeEventListener("dragenter", handleDragEnter);
      window.removeEventListener("dragleave", handleDragLeave);
      window.removeEventListener("dragover", handleDragOver);
      window.removeEventListener("drop", handleDrop);
    };
  }, [handleDragEnter, handleDragLeave, handleDragOver, handleDrop]);

  if (!active) return null;

  return (
    <Overlay fixed backgroundOpacity={0.7} zIndex={1000}>
      <Stack
        align="center"
        justify="center"
        h="100%"
        style={{ pointerEvents: "none" }}
      >
        <IconFileUpload size={64} color="var(--mantine-color-blue-4)" />
        <Text size="xl" fw={600} c="white">
          Drop .mcap files to open
        </Text>
        <Text size="sm" c="dimmed">
          Multiple files supported
        </Text>
      </Stack>
    </Overlay>
  );
}
