import { useState, useMemo } from "react";
import {
  Table,
  Title,
  Paper,
  Text,
  HoverCard,
  Stack,
  Group,
  Collapse,
} from "@mantine/core";
import type { ChannelInfo, PartialStats } from "../mcap/types.ts";
import {
  formatBytes,
  formatHz,
  formatNumber,
  formatDuration,
  formatTimestamp,
} from "../format.ts";
import { Sparkline } from "@mantine/charts";
import { DistributionChart } from "./DistributionChart.tsx";
import type { MessageDistribution } from "../mcap/types.ts";

type SortField =
  | "topic"
  | "id"
  | "schema"
  | "msgs"
  | "hz"
  | "size"
  | "bps"
  | "bPerMsg";

function channelToDistribution(channel: ChannelInfo, bucketDurationNs: number): MessageDistribution {
  const counts = channel.messageDistribution;
  return {
    bucketCount: counts.length,
    bucketDurationNs,
    messageCounts: counts,
    maxCount: Math.max(0, ...counts),
  };
}

interface ChannelsTableProps {
  channels: ChannelInfo[];
  bucketDurationNs: number;
}

export function ChannelsTable({ channels, bucketDurationNs }: ChannelsTableProps) {
  const [sortField, setSortField] = useState<SortField>("topic");
  const [sortReverse, setSortReverse] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const hasSizeData = channels.some((ch) => ch.sizeBytes !== null);
  const hasDistribution = channels.some(
    (ch) => ch.messageDistribution.length > 0,
  );

  const totalColumns =
    4 + // ID, Topic, Schema, Msgs
    1 + // Hz
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
          cmp = (a.schemaName ?? "").localeCompare(b.schemaName ?? "");
          break;
        case "msgs":
          cmp = a.messageCount - b.messageCount;
          break;
        case "hz":
          cmp = a.hzStats.average - b.hzStats.average;
          break;
        case "size":
          cmp = (a.sizeBytes ?? 0) - (b.sizeBytes ?? 0);
          break;
        case "bps":
          cmp =
            (a.bytesPerSecondStats?.average ?? 0) -
            (b.bytesPerSecondStats?.average ?? 0);
          break;
        case "bPerMsg":
          cmp = (a.bytesPerMessage ?? 0) - (b.bytesPerMessage ?? 0);
          break;
      }
      return sortReverse ? -cmp : cmp;
    });
    return arr;
  }, [channels, sortField, sortReverse]);

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

  const headerStyle = { cursor: "pointer", userSelect: "none" as const };
  const rightAligned = {
    ...headerStyle,
    textAlign: "right" as const,
  };

  return (
    <Paper p="md" withBorder>
      <Title order={4} mb="md">
        Channels
      </Title>
      {channels.length === 0 ? (
        <Text c="dimmed">No channels found</Text>
      ) : (
        <div style={{ overflowX: "auto" }}>
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
                {hasSizeData && (
                  <>
                    <Table.Th
                      style={rightAligned}
                      onClick={() => handleSort("size")}
                    >
                      Size{sortIndicator("size")}
                    </Table.Th>
                    <Table.Th
                      style={rightAligned}
                      onClick={() => handleSort("bps")}
                    >
                      B/s{sortIndicator("bps")}
                    </Table.Th>
                    <Table.Th
                      style={rightAligned}
                      onClick={() => handleSort("bPerMsg")}
                    >
                      B/msg{sortIndicator("bPerMsg")}
                    </Table.Th>
                  </>
                )}
                {hasDistribution && <Table.Th>Distribution</Table.Th>}
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {sorted.map((ch) => {
                const expanded = expandedIds.has(ch.id);
                return (
                  <>
                    <Table.Tr
                      key={ch.id}
                      onClick={() => toggleExpanded(ch.id)}
                      style={{ cursor: "pointer" }}
                    >
                      <Table.Td>
                        <Group gap={4}>
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
                        <SchemaDisplay name={ch.schemaName} />
                      </Table.Td>
                      <Table.Td style={{ textAlign: "right" }}>
                        {formatNumber(ch.messageCount)}
                      </Table.Td>
                      <Table.Td style={{ textAlign: "right" }}>
                        <HzDisplay
                          stats={ch.hzStats}
                          hzChannel={ch.hzChannel}
                        />
                      </Table.Td>
                      {hasSizeData && (
                        <>
                          <Table.Td style={{ textAlign: "right" }}>
                            {ch.sizeBytes !== null
                              ? formatBytes(ch.sizeBytes)
                              : "-"}
                          </Table.Td>
                          <Table.Td style={{ textAlign: "right" }}>
                            <BpsDisplay stats={ch.bytesPerSecondStats} />
                          </Table.Td>
                          <Table.Td style={{ textAlign: "right" }}>
                            {ch.bytesPerMessage !== null
                              ? formatBytes(ch.bytesPerMessage)
                              : "-"}
                          </Table.Td>
                        </>
                      )}
                      {hasDistribution && (
                        <Table.Td>
                          {ch.messageDistribution.length > 0 ? (
                            <Sparkline
                              w={120}
                              h={20}
                              data={ch.messageDistribution}
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
                          <ChannelDetail channel={ch} bucketDurationNs={bucketDurationNs} />
                        </Collapse>
                      </Table.Td>
                    </Table.Tr>
                  </>
                );
              })}
            </Table.Tbody>
          </Table>
        </div>
      )}
    </Paper>
  );
}

// ── Detail panel ──

function ChannelDetail({ channel, bucketDurationNs }: { channel: ChannelInfo; bucketDurationNs: number }) {
  return (
    <div style={{ padding: "12px 16px" }}>
      <Group gap="xl" align="flex-start">
        <Stack gap={4}>
          <Text size="sm" fw={600}>
            Duration
          </Text>
          <Text size="sm" c="dimmed">
            {channel.durationNs !== null
              ? formatDuration(Number(channel.durationNs))
              : "-"}
          </Text>
        </Stack>
        <Stack gap={4}>
          <Text size="sm" fw={600}>
            Time range
          </Text>
          <Text size="sm" c="dimmed">
            {channel.messageStartTime !== null &&
            channel.messageEndTime !== null
              ? `${formatTimestamp(channel.messageStartTime)} → ${formatTimestamp(channel.messageEndTime)}`
              : "-"}
          </Text>
        </Stack>
      </Group>
      {channel.messageDistribution.length > 0 && (
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

function SchemaDisplay({ name }: { name: string | null }) {
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
  hzChannel: number | null;
}) {
  const hasDetails =
    stats.minimum !== null || stats.maximum !== null || hzChannel !== null;

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
          {hzChannel !== null && (
            <StatsRow label="Channel Hz" value={formatHz(hzChannel)} />
          )}
        </Stack>
      </HoverCard.Dropdown>
    </HoverCard>
  );
}

function BpsDisplay({ stats }: { stats: PartialStats | null }) {
  if (!stats) {
    return <Text size="sm">-</Text>;
  }

  const hasDetails = stats.minimum !== null || stats.maximum !== null;

  if (!hasDetails) {
    return <Text size="sm">{formatBytes(stats.average)}</Text>;
  }

  return (
    <HoverCard openDelay={200} position="top" withArrow shadow="sm">
      <HoverCard.Target>
        <Text
          size="sm"
          style={{ textDecoration: "underline dotted", cursor: "default" }}
        >
          {formatBytes(stats.average)}
        </Text>
      </HoverCard.Target>
      <HoverCard.Dropdown>
        <Stack gap={2} style={{ minWidth: 140 }}>
          <StatsRow
            label="Average"
            value={`${formatBytes(stats.average)}/s`}
            bold
          />
          {stats.minimum !== null && (
            <StatsRow
              label="Min"
              value={`${formatBytes(stats.minimum)}/s`}
            />
          )}
          {stats.maximum !== null && (
            <StatsRow
              label="Max"
              value={`${formatBytes(stats.maximum)}/s`}
            />
          )}
          {stats.median !== null && (
            <StatsRow
              label="Median"
              value={`${formatBytes(stats.median)}/s`}
            />
          )}
        </Stack>
      </HoverCard.Dropdown>
    </HoverCard>
  );
}
