import { Grid, Text, Title, Paper, Badge, Group } from "@mantine/core";
import type { McapInfoOutput } from "../mcap/types.ts";
import { formatBytes, formatDuration, formatTimestamp } from "../format.ts";
import { DistributionChart } from "./DistributionChart.tsx";

interface FileInfoProps {
  data: McapInfoOutput;
}

export function FileInfo({ data }: FileInfoProps) {
  const { file, header, statistics, message_distribution } = data;
  const durationSec = statistics.duration_ns / 1_000_000_000;
  const bytesPerSec = durationSec > 0 ? file.size_bytes / durationSec : 0;
  const bytesPerHour = bytesPerSec * 3600;

  return (
    <Paper p="md" withBorder>
      <Title order={4} mb="md">
        File Information
      </Title>
      <Grid gutter="xs">
        <InfoRow label="File" value={file.path} />
        <InfoRow
          label="Size"
          value={
            <Group gap="xs">
              <Text span fw={500}>
                {formatBytes(file.size_bytes)}
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
          value={statistics.message_count.toLocaleString()}
        />
        <InfoRow
          label="Chunks"
          value={statistics.chunk_count.toLocaleString()}
        />
        <InfoRow
          label="Duration"
          value={`${(statistics.duration_ns / 1_000_000).toFixed(2)} ms (${formatDuration(statistics.duration_ns)})`}
        />
        <InfoRow
          label="Start"
          value={formatTimestamp(statistics.message_start_time)}
        />
        <InfoRow
          label="End"
          value={formatTimestamp(statistics.message_end_time)}
        />
        <InfoRow
          label="Channels"
          value={statistics.channel_count.toLocaleString()}
        />
        <InfoRow
          label="Attachments"
          value={statistics.attachment_count.toLocaleString()}
        />
        <InfoRow
          label="Metadata"
          value={statistics.metadata_count.toLocaleString()}
        />
        {statistics.message_index_count != null && (
          <InfoRow
            label="Indexed Messages"
            value={statistics.message_index_count.toLocaleString()}
          />
        )}
      </Grid>

      {message_distribution.max_count > 0 && (
        <>
          <Title order={5} mt="lg" mb="xs">
            Message Distribution
          </Title>
          <DistributionChart distribution={message_distribution} />
          <Text size="xs" c="dimmed" mt={4}>
            Max: {message_distribution.max_count.toLocaleString()} msgs/bucket |
            Bucket size: {formatDuration(message_distribution.bucket_duration_ns)}
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
