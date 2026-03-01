import type { SortingState, VisibilityState } from "@tanstack/react-table";

const STORAGE_KEY = "mcap-channels-table-state";

interface PersistedTableState {
  columnVisibility?: VisibilityState;
  sorting?: SortingState;
  columnOrder?: string[];
}

export function loadTableState(): PersistedTableState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedTableState;
  } catch {
    return null;
  }
}

export function saveTableState(state: PersistedTableState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // localStorage full or unavailable — silently ignore
  }
}
