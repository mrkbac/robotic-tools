/**
 * Type augmentations for Chromium-only File System Access APIs.
 * Firefox/Safari do not support these — all usage must be guarded at runtime.
 */

interface FileSystemHandlePermissionDescriptor {
  mode?: "read" | "readwrite";
}

interface FileSystemHandle {
  queryPermission?(
    descriptor?: FileSystemHandlePermissionDescriptor,
  ): Promise<PermissionState>;
  requestPermission?(
    descriptor?: FileSystemHandlePermissionDescriptor,
  ): Promise<PermissionState>;
}

interface DataTransferItem {
  getAsFileSystemHandle?(): Promise<FileSystemHandle | null>;
}
