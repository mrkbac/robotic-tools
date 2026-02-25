import { useState, useEffect, useCallback } from "react";
import {
  type HistoryEntry,
  loadHistory,
  saveHistoryEntry,
  removeHistoryEntry,
  clearHistory,
} from "../stores/historyStore.ts";

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    loadHistory().then(setEntries);
  }, []);

  const addEntry = useCallback(async (entry: HistoryEntry) => {
    await saveHistoryEntry(entry);
    setEntries((prev) => {
      const filtered = prev.filter((e) => e.fileId !== entry.fileId);
      return [entry, ...filtered];
    });
  }, []);

  const removeEntry = useCallback(async (fileId: string) => {
    await removeHistoryEntry(fileId);
    setEntries((prev) => prev.filter((e) => e.fileId !== fileId));
  }, []);

  const clearAll = useCallback(async () => {
    await clearHistory();
    setEntries([]);
  }, []);

  return { entries, addEntry, removeEntry, clearAll };
}
