import type { ChannelInfo } from "../../mcap/types.ts";
import type { ChannelRow } from "./types.ts";

interface TopicTreeNode {
  segment: string;
  fullPath: string;
  children: Map<string, TopicTreeNode>;
  channels: ChannelInfo[];
}

function buildTopicTree(channels: ChannelInfo[]): TopicTreeNode {
  const root: TopicTreeNode = {
    segment: "",
    fullPath: "",
    children: new Map(),
    channels: [],
  };

  for (const ch of channels) {
    const parts = ch.topic.split("/").filter(Boolean);
    let node = root;

    for (let i = 0; i < parts.length; i++) {
      const segment = parts[i]!;
      const path = "/" + parts.slice(0, i + 1).join("/");

      if (!node.children.has(segment)) {
        node.children.set(segment, {
          segment,
          fullPath: path,
          children: new Map(),
          channels: [],
        });
      }
      node = node.children.get(segment)!;
    }

    node.channels.push(ch);
  }

  return root;
}

function aggregateNode(node: TopicTreeNode): { totalMessages: number; minHz: number; maxHz: number } {
  let totalMessages = 0;
  let minHz = Infinity;
  let maxHz = -Infinity;

  for (const ch of node.channels) {
    totalMessages += ch.message_count;
    minHz = Math.min(minHz, ch.hz_stats.average);
    maxHz = Math.max(maxHz, ch.hz_stats.average);
  }

  for (const child of node.children.values()) {
    const agg = aggregateNode(child);
    totalMessages += agg.totalMessages;
    minHz = Math.min(minHz, agg.minHz);
    maxHz = Math.max(maxHz, agg.maxHz);
  }

  return { totalMessages, minHz: minHz === Infinity ? 0 : minHz, maxHz: maxHz === -Infinity ? 0 : maxHz };
}

function channelToRow(ch: ChannelInfo): ChannelRow {
  return { ...ch, _kind: "channel", _segment: "", _fullPath: "" };
}

function nodeToGroupRow(node: TopicTreeNode, subRows: ChannelRow[]): ChannelRow {
  const agg = aggregateNode(node);
  return {
    id: -Math.abs(hashString(node.fullPath)),
    topic: node.fullPath,
    schema_id: 0,
    schema_name: null,
    message_count: agg.totalMessages,
    size_bytes: null,
    duration_ns: null,
    hz_stats: {
      average: agg.minHz === agg.maxHz ? agg.minHz : (agg.minHz + agg.maxHz) / 2,
      minimum: agg.minHz,
      maximum: agg.maxHz,
      median: null,
    },
    hz_channel: null,
    bytes_per_second_stats: null,
    bytes_per_message: null,
    message_distribution: [],
    message_start_time: null,
    message_end_time: null,
    estimated_sizes: false,
    jitter_ns: null,
    jitter_cv: null,
    _kind: "group",
    _segment: node.segment,
    _fullPath: node.fullPath,
    subRows,
  };
}

function hashString(s: string): number {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = s.charCodeAt(i) + ((hash << 5) - hash);
  }
  return hash;
}

/**
 * Convert a tree node's children into ChannelRow[].
 * Each child becomes a group whose subRows are its own channels + recursed children.
 * Pass-through nodes (no channels and only one child) are collapsed into their child.
 * Empty leaf nodes (no channels, no children) are skipped entirely.
 */
function convertChildren(node: TopicTreeNode): ChannelRow[] {
  const rows: ChannelRow[] = [];

  const sortedChildren = [...node.children.values()].sort((a, b) =>
    a.segment.localeCompare(b.segment),
  );

  for (const child of sortedChildren) {
    const subRows: ChannelRow[] = [
      ...child.channels.map(channelToRow),
      ...convertChildren(child),
    ];

    // Skip empty nodes
    if (subRows.length === 0) continue;

    // Flatten single-child groups — a group wrapping just one item
    // (e.g. /tf with a single /tf topic) adds no value, so promote
    // the child directly.
    if (subRows.length === 1) {
      rows.push(subRows[0]!);
      continue;
    }

    rows.push(nodeToGroupRow(child, subRows));
  }

  return rows;
}

/** Build tree data with subRows for TanStack Table's expanding model. */
export function buildTreeData(channels: ChannelInfo[]): ChannelRow[] {
  const tree = buildTopicTree(channels);
  const rows = convertChildren(tree);
  // Root-level channels (topics without any '/' segments, if any)
  for (const ch of tree.channels) {
    rows.push(channelToRow(ch));
  }
  return rows;
}

/** Convert channels to flat rows (no subRows). */
export function toFlatRows(channels: ChannelInfo[]): ChannelRow[] {
  return channels.map(channelToRow);
}
