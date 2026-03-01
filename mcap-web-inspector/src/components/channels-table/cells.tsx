import { Text, HoverCard, Stack, Group } from "@mantine/core";
import type { PartialStats } from "../../mcap/types.ts";
import { formatHz, formatBytes } from "../../format.ts";
import { stringToColor, formatJitterNs, jitterColor } from "./utils.ts";

export function StatsRow({
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

export function TopicDisplay({ topic }: { topic: string }) {
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

export function SchemaDisplay({ name }: { name: string | null | undefined }) {
  if (!name) {
    return (
      <Text size="sm" c="dimmed">
        unknown
      </Text>
    );
  }
  return <Text size="sm">{name}</Text>;
}

export function HzDisplay({
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

export function JitterDisplay({
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

export function BpsDisplay({
  stats,
  estimated,
}: {
  stats: PartialStats | null | undefined;
  estimated?: boolean;
}) {
  if (!stats) {
    return <Text size="sm">-</Text>;
  }

  const prefix = estimated ? "~" : "";
  const hasDetails = stats.minimum !== null || stats.maximum !== null;

  if (!hasDetails) {
    return (
      <Text size="sm">
        {prefix}
        {formatBytes(stats.average)}
      </Text>
    );
  }

  return (
    <HoverCard openDelay={200} position="top" withArrow shadow="sm">
      <HoverCard.Target>
        <Text
          size="sm"
          style={{ textDecoration: "underline dotted", cursor: "default" }}
        >
          {prefix}
          {formatBytes(stats.average)}
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
