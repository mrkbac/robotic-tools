import { SimpleGrid, Text, Title, Paper, Badge, Group, Stack, UnstyledButton } from "@mantine/core";
import type { McapInfoOutput } from "../mcap/types.ts";
import type { DetailSection } from "./DetailModal.tsx";
import { formatBytes, formatDuration, formatTimestamp } from "../format.ts";
import { DistributionChart } from "./DistributionChart.tsx";

interface FileInfoProps {
  data: McapInfoOutput;
  onCountClick?: (section: DetailSection) => void;
}

export function FileInfo({ data, onCountClick }: FileInfoProps) {
  const { file, header, statistics, message_distribution } = data;
  const durationSec = statistics.duration_ns / 1_000_000_000;
  const bytesPerSec = durationSec > 0 ? file.size_bytes / durationSec : 0;
  const bytesPerHour = bytesPerSec * 3600;

  const clickableValue = (count: number, section: DetailSection) => {
    const text = count.toLocaleString();
    if (count > 0 && onCountClick) {
      return (
        <UnstyledButton onClick={() => onCountClick(section)}>
          <Text size="sm" fw={500} td="underline" c="blue">
            {text}
          </Text>
        </UnstyledButton>
      );
    }
    return text;
  };

  return (
    <Paper p="md" withBorder>
      <SimpleGrid cols={{ base: 2, sm: 3, lg: 4 }} spacing="sm">
        <InfoItem
          label="Size"
          value={
            <Group gap="xs" wrap="wrap">
              <Text size="sm" fw={500}>
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
        <InfoItem label="Library" value={header.library || "N/A"} />
        <InfoItem label="Profile" value={header.profile || "N/A"} />
        <InfoItem
          label="Messages"
          value={statistics.message_count.toLocaleString()}
        />
        <InfoItem
          label="Chunks"
          value={clickableValue(statistics.chunk_count, "chunks")}
        />
        <InfoItem
          label="Duration"
          value={`${(statistics.duration_ns / 1_000_000).toFixed(2)} ms (${formatDuration(statistics.duration_ns)})`}
        />
        <InfoItem
          label="Start"
          value={formatTimestamp(statistics.message_start_time)}
        />
        <InfoItem
          label="End"
          value={formatTimestamp(statistics.message_end_time)}
        />
        <InfoItem
          label="Channels"
          value={statistics.channel_count.toLocaleString()}
        />
        <InfoItem
          label="Schemas"
          value={clickableValue(data.schemas.length, "schemas")}
        />
        <InfoItem
          label="Attachments"
          value={clickableValue(statistics.attachment_count, "attachments")}
        />
        <InfoItem
          label="Metadata"
          value={clickableValue(statistics.metadata_count, "metadata")}
        />
        {statistics.message_index_count != null && (
          <InfoItem
            label="Indexed Messages"
            value={statistics.message_index_count.toLocaleString()}
          />
        )}
      </SimpleGrid>

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

function InfoItem({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <Stack gap={2}>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      {typeof value === "string" ? (
        <Text size="sm" fw={500}>
          {value}
        </Text>
      ) : (
        value
      )}
    </Stack>
  );
}
