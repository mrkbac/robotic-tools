import { createContext, useContext, type ReactNode } from "react";
import { useFileProcessor } from "../hooks/useFileProcessor.ts";

type FileProcessorContextValue = ReturnType<typeof useFileProcessor>;

const FileProcessorContext = createContext<FileProcessorContextValue | null>(
  null,
);

export function FileProcessorProvider({ children }: { children: ReactNode }) {
  const value = useFileProcessor();
  return (
    <FileProcessorContext.Provider value={value}>
      {children}
    </FileProcessorContext.Provider>
  );
}

export function useFileProcessorContext(): FileProcessorContextValue {
  const ctx = useContext(FileProcessorContext);
  if (!ctx) {
    throw new Error(
      "useFileProcessorContext must be used within FileProcessorProvider",
    );
  }
  return ctx;
}
