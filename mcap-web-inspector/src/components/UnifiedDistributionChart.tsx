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

  const data = useMemo(
    () =>
      Array.from({ length: bucketCount }, (_, i) => {
        const row: Record<string, string | number> = {
          time: formatBucketTime(i * bucketDurationNs),
        };
        for (const ch of withDist) {
          row[shortTopic(ch.topic)] = ch.messageDistribution[i] ?? 0;
        }
        return row;
      }),
    [bucketCount, bucketDurationNs, withDist],
  );

  const series = useMemo(
    () =>
      withDist.map((ch, i) => ({
        name: shortTopic(ch.topic),
        color: CHANNEL_COLORS[i % CHANNEL_COLORS.length]!,
      })),
    [withDist],
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
