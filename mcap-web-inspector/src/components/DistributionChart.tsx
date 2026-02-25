import { useMemo } from "react";
import { AreaChart } from "@mantine/charts";
import type { MessageDistribution } from "../mcap/types.ts";
import { formatBucketTime } from "../format.ts";

/** Pick ~maxTicks evenly-spaced tick indices from total buckets. */
function tickInterval(total: number, maxTicks = 10): number {
  if (total <= maxTicks) return 0;
  return Math.ceil(total / maxTicks) - 1;
}

interface DistributionChartProps {
  distribution: MessageDistribution;
  height?: number;
}

export function DistributionChart({
  distribution,
  height = 300,
}: DistributionChartProps) {
  const { messageCounts, bucketDurationNs } = distribution;

  const data = useMemo(
    () =>
      messageCounts.map((count, i) => ({
        time: formatBucketTime(i * bucketDurationNs),
        messages: count,
      })),
    [messageCounts, bucketDurationNs],
  );

  return (
    <AreaChart
      h={height}
      data={data}
      dataKey="time"
      series={[{ name: "messages", color: "blue.6" }]}
      curveType="monotone"
      withDots={false}
      withXAxis
      withYAxis
      gridAxis="xy"
      xAxisProps={{ interval: tickInterval(data.length) }}
    />
  );
}
