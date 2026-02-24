import { Grid, Text, Title, Paper, Badge, Group } from "@mantine/core";
import type { McapInfoOutput } from "../mcap/types.ts";
import { formatBytes, formatDuration, formatTimestamp } from "../format.ts";
import { DistributionBar } from "./DistributionBar.tsx";

interface FileInfoProps {
  data: McapInfoOutput;
}

export function FileInfo({ data }: FileInfoProps) {
  const { file, header, statistics, messageDistribution } = data;
  const durationSec = Number(statistics.durationNs) / 1_000_000_000;
  const bytesPerSec = durationSec > 0 ? file.sizeBytes / durationSec : 0;
  const bytesPerHour = bytesPerSec * 3600;

  return (
    <Paper p="md" withBorder>
      <Title order={4} mb="md">
        File Information
      </Title>
      <Grid gutter="xs">
        <InfoRow label="File" value={file.name} />
        <InfoRow
          label="Size"
          value={
            <Group gap="xs">
              <Text span fw={500}>
                {formatBytes(file.sizeBytes)}
              </Text>
              <Badge variant="light" color="red" size="sm">
                {formatBytes(bytesPerSec)}/s
              </Badge>
              <Badge variant="light" color="orange" size="sm">
                {formatBytes(bytesPerHour)}/h
              </Badge>
            </Group>
          }
        />
        <InfoRow label="Library" value={header.library || "N/A"} />
        <InfoRow label="Profile" value={header.profile || "N/A"} />
        <InfoRow
          label="Messages"
          value={statistics.messageCount.toLocaleString()}
        />
        <InfoRow
          label="Chunks"
          value={statistics.chunkCount.toLocaleString()}
        />
        <InfoRow
          label="Duration"
          value={`${(Number(statistics.durationNs) / 1_000_000).toFixed(2)} ms (${formatDuration(Number(statistics.durationNs))})`}
        />
        <InfoRow
          label="Start"
          value={formatTimestamp(statistics.messageStartTime)}
        />
        <InfoRow
          label="End"
          value={formatTimestamp(statistics.messageEndTime)}
        />
        <InfoRow
          label="Channels"
          value={statistics.channelCount.toLocaleString()}
        />
        <InfoRow
          label="Attachments"
          value={statistics.attachmentCount.toLocaleString()}
        />
        <InfoRow
          label="Metadata"
          value={statistics.metadataCount.toLocaleString()}
        />
        {statistics.messageIndexCount !== null && (
          <InfoRow
            label="Indexed Messages"
            value={statistics.messageIndexCount.toLocaleString()}
          />
        )}
      </Grid>

      {messageDistribution.maxCount > 0 && (
        <>
          <Title order={5} mt="lg" mb="xs">
            Message Distribution
          </Title>
          <DistributionBar counts={messageDistribution.messageCounts} />
          <Text size="xs" c="dimmed" mt={4}>
            Max: {messageDistribution.maxCount.toLocaleString()} msgs/bucket |
            Bucket size: {formatDuration(messageDistribution.bucketDurationNs)}
          </Text>
        </>
      )}
    </Paper>
  );
}

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <>
      <Grid.Col span={3}>
        <Text fw={600} c="blue" size="sm">
          {label}:
        </Text>
      </Grid.Col>
      <Grid.Col span={9}>
        {typeof value === "string" ? <Text size="sm">{value}</Text> : value}
      </Grid.Col>
    </>
  );
}
