import type { ChannelInfo, MessageDistribution } from "../../mcap/types.ts";

export function channelToDistribution(
  channel: ChannelInfo,
  bucketDurationNs: number,
): MessageDistribution {
  const counts = channel.message_distribution;
  return {
    bucket_count: counts.length,
    bucket_duration_ns: bucketDurationNs,
    message_counts: counts,
    max_count: counts.reduce((m, v) => (v > m ? v : m), 0),
  };
}

export function formatPercent(part: number, total: number): string {
  if (total <= 0) return "0%";
  return `${((part / total) * 100).toFixed(1)}%`;
}

export function nsToDate(ns: number): Date {
  return new Date(ns / 1_000_000);
}

/** Format nanoseconds jitter to human-readable units. */
export function formatJitterNs(ns: number): string {
  if (ns < 1_000) return `${ns.toFixed(0)} ns`;
  if (ns < 1_000_000) return `${(ns / 1_000).toFixed(1)} \u00B5s`;
  if (ns < 1_000_000_000) return `${(ns / 1_000_000).toFixed(1)} ms`;
  return `${(ns / 1_000_000_000).toFixed(2)} s`;
}

/** Get jitter severity color based on CV percentage. */
export function jitterColor(cv: number): string {
  if (cv < 0.05) return "green";
  if (cv < 0.15) return "yellow";
  return "red";
}

/** Check if a single row matches a search filter (case-insensitive). */
export function matchesFilter(
  row: { topic: string; schema_name: string | null; id: number },
  lower: string,
): boolean {
  return (
    row.topic.toLowerCase().includes(lower) ||
    (row.schema_name?.toLowerCase().includes(lower) ?? false) ||
    String(row.id).includes(lower)
  );
}

/** Recursively filter tree rows, keeping ancestors of matching children. */
export function filterTree<
  T extends {
    _kind: string;
    topic: string;
    schema_name: string | null;
    id: number;
    subRows?: T[];
  },
>(rows: T[], lower: string): T[] {
  const result: T[] = [];
  for (const row of rows) {
    if (row._kind === "group" && row.subRows) {
      const filteredChildren = filterTree(row.subRows, lower);
      if (filteredChildren.length > 0) {
        result.push({ ...row, subRows: filteredChildren });
      }
    } else if (matchesFilter(row, lower)) {
      result.push(row);
    }
  }
  return result;
}

/** Hash a string to a deterministic color. */
export function stringToColor(s: string): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = s.charCodeAt(i) + ((hash << 5) - hash);
  }
  const colors = [
    "#4a90d9",
    "#00bcd4",
    "#4caf50",
    "#9c27b0",
    "#ff5722",
    "#ffc107",
    "#e91e63",
    "#3f51b5",
    "#009688",
    "#ff9800",
    "#795548",
    "#607d8b",
  ];
  return colors[Math.abs(hash) % colors.length]!;
}
