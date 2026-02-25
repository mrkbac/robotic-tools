import { useMemo } from "react";
import { AreaChart } from "@mantine/charts";
import { Paper, Title } from "@mantine/core";
import type { ChannelInfo, MessageDistribution } from "../mcap/types.ts";
import { formatBucketTime } from "../format.ts";

const CHANNEL_COLORS = [
  "blue.6",
  "teal.6",
  "green.6",
  "violet.6",
  "orange.6",
  "yellow.6",
  "pink.6",
  "indigo.6",
  "cyan.6",
  "red.6",
  "lime.6",
  "grape.6",
];

/** Pick ~maxTicks evenly-spaced tick indices from total buckets. */
function tickInterval(total: number, maxTicks = 10): number {
  if (total <= maxTicks) return 0;
  return Math.ceil(total / maxTicks) - 1;
}

/** Shorten a topic to its last segment for the legend. */
function shortTopic(topic: string): string {
  const parts = topic.split("/").filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1]! : topic;
}

/** Build unique display names for channels, using shortest unambiguous suffix. */
function uniqueNames(channels: ChannelInfo[]): Map<ChannelInfo, string> {
  const result = new Map<ChannelInfo, string>();
  for (const ch of channels) {
    result.set(ch, shortTopic(ch.topic));
  }
  // Find collisions and disambiguate by prepending more segments
  const seen = new Map<string, ChannelInfo[]>();
  for (const [ch, name] of result) {
    if (!seen.has(name)) seen.set(name, []);
    seen.get(name)!.push(ch);
  }
  for (const [, group] of seen) {
    if (group.length <= 1) continue;
    for (const ch of group) {
      const parts = ch.topic.split("/").filter(Boolean);
      for (let n = 2; n <= parts.length; n++) {
        const candidate = parts.slice(-n).join("/");
        const others = group.filter((o) => o !== ch);
        if (others.every((o) => result.get(o) !== candidate)) {
          result.set(ch, candidate);
          break;
        }
      }
    }
  }
  return result;
}

interface UnifiedDistributionChartProps {
  channels: ChannelInfo[];
  globalDistribution: MessageDistribution;
}

export function UnifiedDistributionChart({
  channels,
  globalDistribution,
}: UnifiedDistributionChartProps) {
  const withDist = useMemo(
    () => channels.filter((ch) => ch.messageDistribution.length > 0),
    [channels],
  );

  const bucketCount = globalDistribution.bucketCount;
  const bucketDurationNs = globalDistribution.bucketDurationNs;

  const names = useMemo(() => uniqueNames(withDist), [withDist]);

  const data = useMemo(
    () =>
      Array.from({ length: bucketCount }, (_, i) => {
        const row: Record<string, string | number> = {
          time: formatBucketTime(i * bucketDurationNs),
        };
        for (const ch of withDist) {
          row[names.get(ch)!] = ch.messageDistribution[i] ?? 0;
        }
        return row;
      }),
    [bucketCount, bucketDurationNs, withDist, names],
  );

  const series = useMemo(
    () =>
      withDist.map((ch, i) => ({
        name: names.get(ch)!,
        color: CHANNEL_COLORS[i % CHANNEL_COLORS.length]!,
      })),
    [withDist, names],
  );

  if (withDist.length === 0) return null;

  return (
    <Paper p="md" withBorder>
      <Title order={4} mb="md">
        Channel Distributions
      </Title>
      <AreaChart
        h={400}
        data={data}
        dataKey="time"
        series={series}
        curveType="monotone"
        withDots={false}
        withXAxis
        withYAxis
        withLegend
        gridAxis="xy"
        fillOpacity={0.15}
        xAxisProps={{ interval: tickInterval(bucketCount) }}
        legendProps={{ verticalAlign: "bottom", height: 50 }}
      />
    </Paper>
  );
}
