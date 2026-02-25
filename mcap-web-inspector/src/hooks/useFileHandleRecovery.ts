import { useState, useEffect, useCallback, useRef } from "react";
import { loadFileHandle } from "../stores/fileHandleStore.ts";
import { supportsFileSystemAccess } from "../stores/fileHandleSupport.ts";

export type RecoveryStatus = "idle" | "loading" | "prompt" | "granted";

interface FileHandleRecovery {
  status: RecoveryStatus;
  file: File | null;
  requestAccess: () => Promise<void>;
}

export function useFileHandleRecovery(
  fileId: string | null,
): FileHandleRecovery {
  const [status, setStatus] = useState<RecoveryStatus>(
    fileId && supportsFileSystemAccess() ? "loading" : "idle",
  );
  const [file, setFile] = useState<File | null>(null);
  const handleRef = useRef<FileSystemFileHandle | null>(null);

  useEffect(() => {
    if (!fileId || !supportsFileSystemAccess()) return;

    let cancelled = false;

    (async () => {
      const handle = await loadFileHandle(fileId);
      if (cancelled) return;

      if (!handle || !handle.queryPermission) {
        setStatus("idle");
        return;
      }

      handleRef.current = handle;

      const perm = await handle.queryPermission({ mode: "read" });
      if (cancelled) return;

      if (perm === "granted") {
        try {
          const f = await handle.getFile();
          if (cancelled) return;
          setFile(f);
          setStatus("granted");
        } catch {
          setStatus("idle");
        }
      } else if (perm === "prompt") {
        setStatus("prompt");
      } else {
        setStatus("idle");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [fileId]);

  const requestAccess = useCallback(async () => {
    const handle = handleRef.current;
    if (!handle?.requestPermission) {
      setStatus("idle");
      return;
    }

    try {
      const perm = await handle.requestPermission({ mode: "read" });
      if (perm === "granted") {
        const f = await handle.getFile();
        setFile(f);
        setStatus("granted");
      } else {
        setStatus("idle");
      }
    } catch {
      setStatus("idle");
    }
  }, []);

  return { status, file, requestAccess };
}
