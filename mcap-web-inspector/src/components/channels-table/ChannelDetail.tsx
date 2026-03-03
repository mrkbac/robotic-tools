import { Text, Stack, Group, Progress, SimpleGrid, Box } from "@mantine/core";
import { TimeValue } from "@mantine/dates";
import type { ChannelInfo } from "../../mcap/types.ts";
import { formatBytes, formatHz, formatDuration } from "../../format.ts";
import { DistributionChart } from "../DistributionChart.tsx";
import { StatsRow } from "./cells.tsx";
import {
  channelToDistribution,
  formatPercent,
  formatJitterNs,
  nsToDate,
} from "./utils.ts";

function TimestampDisplay({ ns }: { ns: number }) {
  const date = nsToDate(ns);
  return (
    <Text size="xs">
      {date.toLocaleDateString()} <TimeValue value={date} withSeconds />
    </Text>
  );
}

export function ChannelDetail({
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
      <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="xl">
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
            <StatsRow label="Channel Hz" value={formatHz(channel.hz_channel)} />
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
