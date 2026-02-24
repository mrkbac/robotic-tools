import { useState, useMemo } from "react";
import { Table, Title, Paper, Text } from "@mantine/core";
import type { ChannelInfo } from "../mcap/types.ts";
import { formatBytes, formatHz, formatNumber } from "../format.ts";
import { InlineDistributionBar } from "./DistributionBar.tsx";

type SortField =
  | "topic"
  | "id"
  | "schema"
  | "msgs"
  | "hz"
  | "size"
  | "bps"
  | "bPerMsg";

interface ChannelsTableProps {
  channels: ChannelInfo[];
}

export function ChannelsTable({ channels }: ChannelsTableProps) {
  const [sortField, setSortField] = useState<SortField>("topic");
  const [sortReverse, setSortReverse] = useState(false);

  const hasSizeData = channels.some((ch) => ch.sizeBytes !== null);
  const hasDistribution = channels.some(
    (ch) => ch.messageDistribution.length > 0,
  );

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
              {sorted.map((ch) => (
                <Table.Tr key={ch.id}>
                  <Table.Td>
                    <Text size="sm" c="dimmed">
                      {ch.id}
                    </Text>
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
                    <HzDisplay stats={ch.hzStats} />
                  </Table.Td>
                  {hasSizeData && (
                    <>
                      <Table.Td style={{ textAlign: "right" }}>
                        {ch.sizeBytes !== null
                          ? formatBytes(ch.sizeBytes)
                          : "-"}
                      </Table.Td>
                      <Table.Td style={{ textAlign: "right" }}>
                        {ch.bytesPerSecondStats
                          ? formatBytes(ch.bytesPerSecondStats.average)
                          : "-"}
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
                        <InlineDistributionBar
                          counts={ch.messageDistribution}
                        />
                      ) : null}
                    </Table.Td>
                  )}
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </div>
      )}
    </Paper>
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

function HzDisplay({
  stats,
}: {
  stats: { average: number; minimum: number | null; maximum: number | null };
}) {
  if (stats.minimum !== null && stats.maximum !== null) {
    return (
      <Text size="sm" title={`min: ${formatHz(stats.minimum)}, max: ${formatHz(stats.maximum)}`}>
        {formatHz(stats.average)}
      </Text>
    );
  }
  return <Text size="sm">{formatHz(stats.average)}</Text>;
}
