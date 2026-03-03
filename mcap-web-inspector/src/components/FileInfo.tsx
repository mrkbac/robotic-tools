import { useEffect, useMemo, useRef, useState } from "react";
import {
  SimpleGrid,
  Text,
  Title,
  Paper,
  Badge,
  Group,
  Stack,
  Image,
  Modal,
  UnstyledButton,
} from "@mantine/core";
import type { McapInfoOutput } from "../mcap/types.ts";
import type { ThumbnailMap } from "../mcap/image.ts";
import type { DetailSection } from "./DetailModal.tsx";
import { formatBytes, formatDuration, formatTimestamp } from "../format.ts";
import { DistributionChart } from "./DistributionChart.tsx";

function thumbnailToUrl(data: Uint8Array, format: string): string {
  const mime = format.startsWith("image/")
    ? format
    : `image/${format || "jpeg"}`;
  return URL.createObjectURL(new Blob([data as BlobPart], { type: mime }));
}

interface FileInfoProps {
  data: McapInfoOutput;
  onCountClick?: (section: DetailSection) => void;
  thumbnails: ThumbnailMap;
  fallbackThumbnailUrl?: string;
}

export function FileInfo({
  data,
  onCountClick,
  thumbnails,
  fallbackThumbnailUrl,
}: FileInfoProps) {
  const { file, header, statistics, message_distribution } = data;
  const durationSec = statistics.duration_ns / 1_000_000_000;
  const bytesPerSec = durationSec > 0 ? file.size_bytes / durationSec : 0;
  const bytesPerHour = bytesPerSec * 3600;

  const [modalOpen, setModalOpen] = useState(false);
  const entries = useMemo(() => [...thumbnails.values()], [thumbnails]);
  const urlsRef = useRef<string[]>([]);

  useEffect(() => {
    const urls = entries.map((e) => thumbnailToUrl(e.data, e.format));
    urlsRef.current = urls;
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [entries]);

  const hasThumbnail =
    (entries.length > 0 && urlsRef.current.length > 0) ||
    !!fallbackThumbnailUrl;

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

  const statsGrid = (
    <>
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
          value={formatDuration(statistics.duration_ns)}
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
            Bucket size:{" "}
            {formatDuration(message_distribution.bucket_duration_ns)}
          </Text>
        </>
      )}
    </>
  );

  const hasFullThumbnail = entries.length > 0 && urlsRef.current.length > 0;
  const thumbnailSrc = hasFullThumbnail
    ? urlsRef.current[0]
    : fallbackThumbnailUrl;

  return (
    <>
      <Paper p="md" withBorder>
        {hasThumbnail ? (
          <Group align="stretch" gap="md" wrap="nowrap">
            <UnstyledButton
              onClick={hasFullThumbnail ? () => setModalOpen(true) : undefined}
              style={{
                flexShrink: 0,
                display: "flex",
                alignItems: "center",
                cursor: hasFullThumbnail ? "pointer" : "default",
              }}
            >
              <Image
                src={thumbnailSrc}
                alt="Thumbnail"
                h={hasFullThumbnail ? "100%" : undefined}
                miw={hasFullThumbnail ? undefined : 64}
                mih={hasFullThumbnail ? undefined : 48}
                mah={160}
                w="auto"
                maw={200}
                fit={hasFullThumbnail ? "contain" : "cover"}
                radius="sm"
                style={
                  hasFullThumbnail ? undefined : { imageRendering: "pixelated" }
                }
              />
            </UnstyledButton>
            <div style={{ flex: 1, minWidth: 0 }}>{statsGrid}</div>
          </Group>
        ) : (
          statsGrid
        )}
      </Paper>

      <Modal
        opened={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Image Topics"
        size="xl"
      >
        <SimpleGrid cols={{ base: 1, xs: 2, sm: 3, md: 4 }} spacing="sm">
          {entries.map((entry, i) => (
            <Stack key={entry.channelId} gap={4} align="center">
              <Image
                src={urlsRef.current[i]}
                alt={entry.topic}
                h={100}
                w="auto"
                fit="contain"
                radius="sm"
              />
              <Text size="xs" c="dimmed" ta="center" truncate="end" maw="100%">
                {entry.topic}
              </Text>
            </Stack>
          ))}
        </SimpleGrid>
      </Modal>
    </>
  );
}

function InfoItem({ label, value }: { label: string; value: React.ReactNode }) {
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
