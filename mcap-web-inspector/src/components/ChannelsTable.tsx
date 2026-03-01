import { useState, useMemo, Fragment } from "react";
import {
  Table,
  Title,
  Paper,
  Text,
  HoverCard,
  Stack,
  Group,
  Collapse,
  Progress,
  SimpleGrid,
  ScrollArea,
  SegmentedControl,
  Box,
} from "@mantine/core";
import type { ChannelInfo, PartialStats } from "../mcap/types.ts";
import {
  formatBytes,
  formatHz,
  formatNumber,
  formatDuration,
} from "../format.ts";
import { Sparkline } from "@mantine/charts";
import { TimeValue } from "@mantine/dates";
import { DistributionChart } from "./DistributionChart.tsx";
import type { MessageDistribution } from "../mcap/types.ts";

type SortField =
  | "topic"
  | "id"
  | "schema"
  | "msgs"
  | "hz"
  | "jitter"
  | "size"
  | "bps"
  | "bPerMsg";

type ViewMode = "flat" | "tree";

function channelToDistribution(channel: ChannelInfo, bucketDurationNs: number): MessageDistribution {
  const counts = channel.message_distribution;
  return {
    bucket_count: counts.length,
    bucket_duration_ns: bucketDurationNs,
    message_counts: counts,
    max_count: counts.reduce((m, v) => (v > m ? v : m), 0),
  };
}

function formatPercent(part: number, total: number): string {
  if (total <= 0) return "0%";
  return `${((part / total) * 100).toFixed(1)}%`;
}

function nsToDate(ns: number): Date {
  return new Date(ns / 1_000_000);
}

/** Format nanoseconds jitter to human-readable units. */
function formatJitterNs(ns: number): string {
  if (ns < 1_000) return `${ns.toFixed(0)} ns`;
  if (ns < 1_000_000) return `${(ns / 1_000).toFixed(1)} \u00B5s`;
  if (ns < 1_000_000_000) return `${(ns / 1_000_000).toFixed(1)} ms`;
  return `${(ns / 1_000_000_000).toFixed(2)} s`;
}

/** Get jitter severity color based on CV percentage. */
function jitterColor(cv: number): string {
  if (cv < 0.05) return "green";
  if (cv < 0.15) return "yellow";
  return "red";
}

// ── Topic Tree ──

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

/** Collect aggregate stats for a tree node and all descendants. */
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

interface ChannelsTableProps {
  channels: ChannelInfo[];
  bucketDurationNs: number;
  fileSize: number;
}

export function ChannelsTable({ channels, bucketDurationNs, fileSize }: ChannelsTableProps) {
  const [sortField, setSortField] = useState<SortField>("topic");
  const [sortReverse, setSortReverse] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  const [viewMode, setViewMode] = useState<ViewMode>("flat");
  const [collapsedPaths, setCollapsedPaths] = useState<Set<string>>(new Set());
  const estimatePrefix = (ch: ChannelInfo) => ch.estimated_sizes ? "~" : "";

  const hasSizeData = channels.some((ch) => ch.size_bytes !== null);
  const hasEstimatedSizes = channels.some((ch) => ch.estimated_sizes && ch.size_bytes !== null);
  const hasDistribution = channels.some(
    (ch) => ch.message_distribution.length > 0,
  );
  const hasJitter = channels.some((ch) => ch.jitter_cv !== null);

  const totalColumns =
    4 + // ID, Topic, Schema, Msgs
    1 + // Hz
    (hasJitter ? 1 : 0) + // Jitter
    (hasSizeData ? 3 : 0) + // Size, B/s, B/msg
    (hasDistribution ? 1 : 0); // Distribution

  const sorted = useMemo(() => {
    const arr = [...channels];
    arr.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case "topic":
          cmp = a.topic.localeCompare(b.topic);
          break;
        case "id":
          cmp = a.id - b.id;
          break;
        case "schema":
          cmp = (a.schema_name ?? "").localeCompare(b.schema_name ?? "");
          break;
        case "msgs":
          cmp = a.message_count - b.message_count;
          break;
        case "hz":
          cmp = a.hz_stats.average - b.hz_stats.average;
          break;
        case "jitter":
          cmp = (a.jitter_cv ?? 0) - (b.jitter_cv ?? 0);
          break;
        case "size":
          cmp = (a.size_bytes ?? 0) - (b.size_bytes ?? 0);
          break;
        case "bps":
          cmp =
            (a.bytes_per_second_stats?.average ?? 0) -
            (b.bytes_per_second_stats?.average ?? 0);
          break;
        case "bPerMsg":
          cmp = (a.bytes_per_message ?? 0) - (b.bytes_per_message ?? 0);
          break;
      }
      return sortReverse ? -cmp : cmp;
    });
    return arr;
  }, [channels, sortField, sortReverse]);

  const topicTree = useMemo(() => buildTopicTree(sorted), [sorted]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortReverse(!sortReverse);
    } else {
      setSortField(field);
      setSortReverse(false);
    }
  };

  const sortIndicator = (field: SortField) => {
    if (sortField !== field) return "";
    return sortReverse ? " \u25BC" : " \u25B2";
  };

  const toggleExpanded = (id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleCollapsed = (path: string) => {
    setCollapsedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const headerStyle = { cursor: "pointer", userSelect: "none" as const };
  const rightAligned = {
    ...headerStyle,
    textAlign: "right" as const,
  };

  const renderChannelRow = (ch: ChannelInfo, indent = 0) => {
    const expanded = expandedIds.has(ch.id);
    return (
      <Fragment key={ch.id}>
        <Table.Tr
          onClick={() => toggleExpanded(ch.id)}
          style={{ cursor: "pointer" }}
        >
          <Table.Td>
            <Group gap={4} style={indent > 0 ? { paddingLeft: indent * 16 } : undefined}>
              <Text
                size="xs"
                c="dimmed"
                style={{
                  transition: "transform 150ms",
                  transform: expanded
                    ? "rotate(90deg)"
                    : "rotate(0deg)",
                }}
              >
                ▶
              </Text>
              <Text size="sm" c="dimmed">
                {ch.id}
              </Text>
            </Group>
          </Table.Td>
          <Table.Td>
            <TopicDisplay topic={ch.topic} />
          </Table.Td>
          <Table.Td>
            <SchemaDisplay name={ch.schema_name} />
          </Table.Td>
          <Table.Td style={{ textAlign: "right" }}>
            {formatNumber(ch.message_count)}
          </Table.Td>
          <Table.Td style={{ textAlign: "right" }}>
            <HzDisplay
              stats={ch.hz_stats}
              hzChannel={ch.hz_channel}
            />
          </Table.Td>
          {hasJitter && (
            <Table.Td style={{ textAlign: "right" }}>
              <JitterDisplay jitterNs={ch.jitter_ns} jitterCv={ch.jitter_cv} />
            </Table.Td>
          )}
          {hasSizeData && (
            <>
              <Table.Td style={{ textAlign: "right" }}>
                {ch.size_bytes !== null ? (
                  <Text size="sm">
                    {estimatePrefix(ch)}{formatBytes(ch.size_bytes)}{" "}
                    <Text span size="xs" c="dimmed">
                      ({estimatePrefix(ch)}{formatPercent(ch.size_bytes, fileSize)})
                    </Text>
                  </Text>
                ) : (
                  "-"
                )}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                <BpsDisplay stats={ch.bytes_per_second_stats} estimated={ch.estimated_sizes} />
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {ch.bytes_per_message !== null
                  ? `${estimatePrefix(ch)}${formatBytes(ch.bytes_per_message)}`
                  : "-"}
              </Table.Td>
            </>
          )}
          {hasDistribution && (
            <Table.Td>
              {ch.message_distribution.length > 0 ? (
                <Sparkline
                  w={120}
                  h={20}
                  data={ch.message_distribution}
                  curveType="monotone"
                  color="blue"
                  fillOpacity={0.2}
                  strokeWidth={1.5}
                />
              ) : null}
            </Table.Td>
          )}
        </Table.Tr>
        <Table.Tr
          key={`${ch.id}-detail`}
          style={{ backgroundColor: "transparent" }}
        >
          <Table.Td
            colSpan={totalColumns}
            style={{ padding: 0, border: expanded ? undefined : "none" }}
          >
            <Collapse in={expanded}>
              {expanded && (
                <ChannelDetail
                  channel={ch}
                  bucketDurationNs={bucketDurationNs}
                  fileSize={fileSize}
                />
              )}
            </Collapse>
          </Table.Td>
        </Table.Tr>
      </Fragment>
    );
  };

  const renderTreeNode = (node: TopicTreeNode, depth: number): React.ReactNode[] => {
    const rows: React.ReactNode[] = [];
    const isCollapsed = collapsedPaths.has(node.fullPath);

    // Render group header for non-root nodes that have children
    if (depth > 0 && (node.children.size > 0 || node.channels.length === 0)) {
      const agg = aggregateNode(node);
      rows.push(
        <Table.Tr
          key={`group-${node.fullPath}`}
          onClick={() => toggleCollapsed(node.fullPath)}
          style={{ cursor: "pointer", backgroundColor: "var(--mantine-color-default-hover)" }}
        >
          <Table.Td colSpan={2}>
            <Group gap={4} style={{ paddingLeft: (depth - 1) * 16 }}>
              <Text
                size="xs"
                c="dimmed"
                style={{
                  transition: "transform 150ms",
                  transform: isCollapsed ? "rotate(0deg)" : "rotate(90deg)",
                }}
              >
                ▶
              </Text>
              <Text size="sm" fw={600} style={{ color: stringToColor(node.segment) }}>
                /{node.segment}
              </Text>
            </Group>
          </Table.Td>
          <Table.Td />
          <Table.Td style={{ textAlign: "right" }}>
            <Text size="sm" c="dimmed">{formatNumber(agg.totalMessages)}</Text>
          </Table.Td>
          <Table.Td style={{ textAlign: "right" }}>
            <Text size="sm" c="dimmed">
              {agg.minHz === agg.maxHz
                ? formatHz(agg.minHz)
                : `${formatHz(agg.minHz)}-${formatHz(agg.maxHz)}`}
            </Text>
          </Table.Td>
          {hasJitter && <Table.Td />}
          {hasSizeData && (
            <>
              <Table.Td />
              <Table.Td />
              <Table.Td />
            </>
          )}
          {hasDistribution && <Table.Td />}
        </Table.Tr>,
      );
    }

    if (depth > 0 && isCollapsed) return rows;

    // Render leaf channels at this node
    for (const ch of node.channels) {
      rows.push(renderChannelRow(ch, viewMode === "tree" ? depth : 0));
    }

    // Render child nodes
    const sortedChildren = [...node.children.values()].sort((a, b) =>
      a.segment.localeCompare(b.segment),
    );
    for (const child of sortedChildren) {
      rows.push(...renderTreeNode(child, depth + 1));
    }

    return rows;
  };

  return (
    <Paper p="md" withBorder>
      <Group justify="space-between" mb="md">
        <Title order={4}>Channels</Title>
        {channels.length > 0 && (
          <SegmentedControl
            size="xs"
            value={viewMode}
            onChange={(v) => setViewMode(v as ViewMode)}
            data={[
              { label: "Flat", value: "flat" },
              { label: "Tree", value: "tree" },
            ]}
          />
        )}
      </Group>
      {channels.length === 0 ? (
        <Text c="dimmed">No channels found</Text>
      ) : (
        <ScrollArea scrollbars="x">
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th
                  style={headerStyle}
                  onClick={() => handleSort("id")}
                >
                  ID{sortIndicator("id")}
                </Table.Th>
                <Table.Th
                  style={headerStyle}
                  onClick={() => handleSort("topic")}
                >
                  Topic{sortIndicator("topic")}
                </Table.Th>
                <Table.Th
                  style={headerStyle}
                  onClick={() => handleSort("schema")}
                >
                  Schema{sortIndicator("schema")}
                </Table.Th>
                <Table.Th
                  style={rightAligned}
                  onClick={() => handleSort("msgs")}
                >
                  Msgs{sortIndicator("msgs")}
                </Table.Th>
                <Table.Th
                  style={rightAligned}
                  onClick={() => handleSort("hz")}
                >
                  Hz{sortIndicator("hz")}
                </Table.Th>
                {hasJitter && (
                  <Table.Th
                    style={rightAligned}
                    onClick={() => handleSort("jitter")}
                  >
                    Jitter{sortIndicator("jitter")}
                  </Table.Th>
                )}
                {hasSizeData && (
                  <>
                    <Table.Th
                      style={rightAligned}
                      onClick={() => handleSort("size")}
                      title={hasEstimatedSizes ? "Estimated from message index offsets" : undefined}
                    >
                      {hasEstimatedSizes ? "~" : ""}Size{sortIndicator("size")}
                    </Table.Th>
                    <Table.Th
                      style={rightAligned}
                      onClick={() => handleSort("bps")}
                      title={hasEstimatedSizes ? "Estimated from message index offsets" : undefined}
                    >
                      {hasEstimatedSizes ? "~" : ""}B/s{sortIndicator("bps")}
                    </Table.Th>
                    <Table.Th
                      style={rightAligned}
                      onClick={() => handleSort("bPerMsg")}
                      title={hasEstimatedSizes ? "Estimated from message index offsets" : undefined}
                    >
                      {hasEstimatedSizes ? "~" : ""}B/msg{sortIndicator("bPerMsg")}
                    </Table.Th>
                  </>
                )}
                {hasDistribution && <Table.Th>Distribution</Table.Th>}
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {viewMode === "flat"
                ? sorted.map((ch) => renderChannelRow(ch))
                : renderTreeNode(topicTree, 0)}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      )}
    </Paper>
  );
}

// ── Detail panel ──

function ChannelDetail({
  channel,
  bucketDurationNs,
  fileSize,
}: {
  channel: ChannelInfo;
  bucketDurationNs: number;
  fileSize: number;
}) {
  const hasTime =
    channel.message_start_time !== null && channel.message_end_time !== null;
  const hasSize = channel.size_bytes !== null;
  const hasHz =
    channel.hz_stats.minimum !== null || channel.hz_stats.maximum !== null;

  return (
    <div style={{ padding: "12px 16px" }}>
      <SimpleGrid cols={3} spacing="xl">
        {/* Time section */}
        <Stack gap={4}>
          <Text size="sm" fw={600}>
            Time
          </Text>
          <StatsRow
            label="Duration"
            value={
              channel.duration_ns !== null
                ? formatDuration(channel.duration_ns)
                : "-"
            }
          />
          {hasTime && (
            <>
              <Group gap={4}>
                <Text size="xs" c="dimmed">
                  Start
                </Text>
                <TimestampDisplay ns={channel.message_start_time!} />
              </Group>
              <Group gap={4}>
                <Text size="xs" c="dimmed">
                  End
                </Text>
                <TimestampDisplay ns={channel.message_end_time!} />
              </Group>
            </>
          )}
        </Stack>

        {/* Frequency section */}
        <Stack gap={4}>
          <Text size="sm" fw={600}>
            Frequency
          </Text>
          <StatsRow
            label="Average Hz"
            value={formatHz(channel.hz_stats.average)}
            bold
          />
          {hasHz && (
            <>
              {channel.hz_stats.minimum !== null && (
                <StatsRow
                  label="Min Hz"
                  value={formatHz(channel.hz_stats.minimum)}
                />
              )}
              {channel.hz_stats.maximum !== null && (
                <StatsRow
                  label="Max Hz"
                  value={formatHz(channel.hz_stats.maximum)}
                />
              )}
              {channel.hz_stats.median !== null && (
                <StatsRow
                  label="Median Hz"
                  value={formatHz(channel.hz_stats.median)}
                />
              )}
            </>
          )}
          {channel.hz_channel !== null && (
            <StatsRow
              label="Channel Hz"
              value={formatHz(channel.hz_channel)}
            />
          )}
          {channel.jitter_cv !== null && channel.jitter_ns !== null && (
            <>
              <Box mt={4}>
                <Text size="xs" fw={600} c="dimmed">
                  Jitter
                </Text>
              </Box>
              <StatsRow
                label="CV"
                value={`${(channel.jitter_cv * 100).toFixed(1)}%`}
              />
              <StatsRow
                label="Stddev"
                value={formatJitterNs(channel.jitter_ns)}
              />
            </>
          )}
        </Stack>

        {/* Size section */}
        <Stack gap={4}>
          <Text size="sm" fw={600}>
            Size{channel.estimated_sizes ? " (estimated)" : ""}
          </Text>
          {hasSize && (
            <>
              <StatsRow
                label="Total"
                value={`${channel.estimated_sizes ? "~" : ""}${formatBytes(channel.size_bytes!)} (${channel.estimated_sizes ? "~" : ""}${formatPercent(channel.size_bytes!, fileSize)})`}
                bold
              />
              <Progress
                value={(channel.size_bytes! / fileSize) * 100}
                size="sm"
                mt={2}
                mb={2}
              />
              {channel.bytes_per_second_stats && (
                <>
                  <StatsRow
                    label="Avg B/s"
                    value={`${formatBytes(channel.bytes_per_second_stats.average)}/s`}
                  />
                  {channel.bytes_per_second_stats.minimum !== null && (
                    <StatsRow
                      label="Min B/s"
                      value={`${formatBytes(channel.bytes_per_second_stats.minimum)}/s`}
                    />
                  )}
                  {channel.bytes_per_second_stats.maximum !== null && (
                    <StatsRow
                      label="Max B/s"
                      value={`${formatBytes(channel.bytes_per_second_stats.maximum)}/s`}
                    />
                  )}
                  {channel.bytes_per_second_stats.median !== null && (
                    <StatsRow
                      label="Median B/s"
                      value={`${formatBytes(channel.bytes_per_second_stats.median)}/s`}
                    />
                  )}
                </>
              )}
              {channel.bytes_per_message !== null && (
                <StatsRow
                  label="B/msg"
                  value={formatBytes(channel.bytes_per_message)}
                />
              )}
            </>
          )}
          {!hasSize && (
            <Text size="xs" c="dimmed">
              -
            </Text>
          )}
        </Stack>
      </SimpleGrid>

      {/* Distribution chart */}
      {channel.message_distribution.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <Text size="sm" fw={600} mb={4}>
            Message distribution
          </Text>
          <DistributionChart
            distribution={channelToDistribution(channel, bucketDurationNs)}
            height={200}
          />
        </div>
      )}
    </div>
  );
}

// ── Timestamp with TimeValue ──

function TimestampDisplay({ ns }: { ns: number }) {
  const date = nsToDate(ns);
  return (
    <Text size="xs">
      {date.toLocaleDateString()}{" "}
      <TimeValue value={date} withSeconds />
    </Text>
  );
}

// ── Helpers ──

/** Hash a string to a deterministic color. */
function stringToColor(s: string): string {
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

function TopicDisplay({ topic }: { topic: string }) {
  const parts = topic.split("/").filter(Boolean);
  return (
    <Text size="sm" style={{ fontFamily: "monospace" }}>
      {parts.map((part, i) => (
        <span key={i}>
          {i > 0 && <span style={{ color: "#999" }}>/</span>}
          <span style={{ color: stringToColor(part) }}>{part}</span>
        </span>
      ))}
    </Text>
  );
}

function SchemaDisplay({ name }: { name: string | null | undefined }) {
  if (!name) {
    return (
      <Text size="sm" c="dimmed">
        unknown
      </Text>
    );
  }
  return <Text size="sm">{name}</Text>;
}

function StatsRow({
  label,
  value,
  bold,
}: {
  label: string;
  value: string;
  bold?: boolean;
}) {
  return (
    <Group justify="space-between" gap="lg">
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text size="xs" fw={bold ? 600 : 400}>
        {value}
      </Text>
    </Group>
  );
}

function HzDisplay({
  stats,
  hzChannel,
}: {
  stats: PartialStats;
  hzChannel: number | null | undefined;
}) {
  const hasDetails =
    stats.minimum !== null || stats.maximum !== null || hzChannel != null;

  if (!hasDetails) {
    return <Text size="sm">{formatHz(stats.average)}</Text>;
  }

  return (
    <HoverCard openDelay={200} position="top" withArrow shadow="sm">
      <HoverCard.Target>
        <Text
          size="sm"
          style={{ textDecoration: "underline dotted", cursor: "default" }}
        >
          {formatHz(stats.average)}
        </Text>
      </HoverCard.Target>
      <HoverCard.Dropdown>
        <Stack gap={2} style={{ minWidth: 140 }}>
          <StatsRow label="Average" value={formatHz(stats.average)} bold />
          {stats.minimum !== null && (
            <StatsRow label="Min" value={formatHz(stats.minimum)} />
          )}
          {stats.maximum !== null && (
            <StatsRow label="Max" value={formatHz(stats.maximum)} />
          )}
          {stats.median !== null && (
            <StatsRow label="Median" value={formatHz(stats.median)} />
          )}
          {hzChannel != null && (
            <StatsRow label="Channel Hz" value={formatHz(hzChannel)} />
          )}
        </Stack>
      </HoverCard.Dropdown>
    </HoverCard>
  );
}

function JitterDisplay({
  jitterNs,
  jitterCv,
}: {
  jitterNs: number | null;
  jitterCv: number | null;
}) {
  if (jitterCv === null || jitterNs === null) {
    return <Text size="sm">-</Text>;
  }

  const cvPercent = (jitterCv * 100).toFixed(1);
  const color = jitterColor(jitterCv);

  return (
    <HoverCard openDelay={200} position="top" withArrow shadow="sm">
      <HoverCard.Target>
        <Text
          size="sm"
          c={color}
          style={{ textDecoration: "underline dotted", cursor: "default" }}
        >
          {cvPercent}%
        </Text>
      </HoverCard.Target>
      <HoverCard.Dropdown>
        <Stack gap={2} style={{ minWidth: 160 }}>
          <StatsRow label="CV" value={`${cvPercent}%`} bold />
          <StatsRow label="Stddev" value={formatJitterNs(jitterNs)} />
          <Text size="xs" c="dimmed" mt={4}>
            {jitterCv < 0.05
              ? "Stable timing"
              : jitterCv < 0.15
                ? "Moderate jitter"
                : "High jitter"}
          </Text>
        </Stack>
      </HoverCard.Dropdown>
    </HoverCard>
  );
}

function BpsDisplay({ stats, estimated }: { stats: PartialStats | null | undefined; estimated?: boolean }) {
  if (!stats) {
    return <Text size="sm">-</Text>;
  }

  const prefix = estimated ? "~" : "";
  const hasDetails = stats.minimum !== null || stats.maximum !== null;

  if (!hasDetails) {
    return <Text size="sm">{prefix}{formatBytes(stats.average)}</Text>;
  }

  return (
    <HoverCard openDelay={200} position="top" withArrow shadow="sm">
      <HoverCard.Target>
        <Text
          size="sm"
          style={{ textDecoration: "underline dotted", cursor: "default" }}
        >
          {prefix}{formatBytes(stats.average)}
        </Text>
      </HoverCard.Target>
      <HoverCard.Dropdown>
        <Stack gap={2} style={{ minWidth: 140 }}>
          <StatsRow
            label="Average"
            value={`${prefix}${formatBytes(stats.average)}/s`}
            bold
          />
          {stats.minimum !== null && (
            <StatsRow
              label="Min"
              value={`${prefix}${formatBytes(stats.minimum)}/s`}
            />
          )}
          {stats.maximum !== null && (
            <StatsRow
              label="Max"
              value={`${prefix}${formatBytes(stats.maximum)}/s`}
            />
          )}
          {stats.median !== null && (
            <StatsRow
              label="Median"
              value={`${prefix}${formatBytes(stats.median)}/s`}
            />
          )}
        </Stack>
      </HoverCard.Dropdown>
    </HoverCard>
  );
}
