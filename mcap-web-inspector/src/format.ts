const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];

export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const i = Math.floor(Math.log(Math.abs(bytes)) / Math.log(k));
  const idx = Math.min(i, BYTE_UNITS.length - 1);
  return `${(bytes / Math.pow(k, idx)).toFixed(1)} ${BYTE_UNITS[idx]}`;
}

/** Format nanoseconds duration to human-readable string. */
export function formatDuration(ns: number): string {
  if (ns <= 0) return "0 ms";

  const totalMs = ns / 1_000_000;
  if (totalMs < 1000) return `${totalMs.toFixed(0)} ms`;

  const totalSec = totalMs / 1000;
  if (totalSec < 60) return `${totalSec.toFixed(1)} s`;

  const totalMin = totalSec / 60;
  if (totalMin < 60) return `${totalMin.toFixed(1)} min`;

  const hours = Math.floor(totalMin / 60);
  const mins = Math.floor(totalMin % 60);
  const secs = Math.floor(totalSec % 60);

  if (hours > 0) {
    return `${hours}h ${mins}m ${secs}s`;
  }
  return `${mins}m ${secs}s`;
}

/** Format a nanosecond timestamp to a date string. */
export function formatTimestamp(nsTimestamp: number): string {
  const ms = nsTimestamp / 1_000_000;
  return new Date(ms).toLocaleString();
}

/** Format a number with comma separators. */
export function formatNumber(n: number): string {
  return n.toLocaleString();
}

/** Format a nanosecond offset into a short time label for chart axes. */
export function formatBucketTime(ns: number): string {
  const totalSec = ns / 1_000_000_000;
  if (totalSec < 1) return `${(totalSec * 1000).toFixed(0)}ms`;
  if (totalSec < 60) return `${totalSec.toFixed(1)}s`;
  const min = totalSec / 60;
  if (min < 60) return `${min.toFixed(1)}m`;
  const hr = min / 60;
  return `${hr.toFixed(1)}h`;
}

/** Format Hz value. */
export function formatHz(hz: number): string {
  if (hz >= 1000) return `${(hz / 1000).toFixed(1)}k`;
  if (hz >= 100) return hz.toFixed(0);
  if (hz >= 10) return hz.toFixed(1);
  return hz.toFixed(2);
}
