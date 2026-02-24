import { Table, Title, Paper, Text, Group } from "@mantine/core";
import type { McapInfoOutput } from "../mcap/types.ts";
import { formatBytes, formatDuration } from "../format.ts";

interface CompressionTableProps {
  data: McapInfoOutput;
}

export function CompressionTable({ data }: CompressionTableProps) {
  const { byCompression } = data.chunks;
  const compressionTypes = Object.entries(byCompression);

  if (compressionTypes.length === 0) return null;

  const hasMessageCounts = compressionTypes.some(
    ([, stats]) => stats.messageCount > 0,
  );
  const { overlaps } = data.chunks;

  return (
    <Paper p="md" withBorder>
      <Title order={4} mb="md">
        Compression
      </Title>
      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Type</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Chunks</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Compressed</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Uncompressed</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Ratio</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Min Size</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Avg Size</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Max Size</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Min Dur</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Avg Dur</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Max Dur</Table.Th>
            {hasMessageCounts && (
              <Table.Th style={{ textAlign: "right" }}>Msgs</Table.Th>
            )}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {compressionTypes.map(([type, stats]) => (
            <Table.Tr key={type}>
              <Table.Td fw={600}>{type}</Table.Td>
              <Table.Td style={{ textAlign: "right" }}>{stats.count}</Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.compressedSize)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.uncompressedSize)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {(stats.compressionRatio * 100).toFixed(1)}%
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.sizeStats.minimum)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.sizeStats.average)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.sizeStats.maximum)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatDuration(stats.durationStats.minimum)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatDuration(stats.durationStats.average)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatDuration(stats.durationStats.maximum)}
              </Table.Td>
              {hasMessageCounts && (
                <Table.Td style={{ textAlign: "right" }}>
                  {stats.messageCount.toLocaleString()}
                </Table.Td>
              )}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>

      {overlaps.maxConcurrent > 1 && (
        <Group mt="sm" gap="xs">
          <Text size="sm" fw={600} c="blue">
            Overlaps:
          </Text>
          <Text size="sm">
            {overlaps.maxConcurrent} max concurrent,{" "}
            {formatBytes(overlaps.maxConcurrentBytes)} max total size at once
          </Text>
        </Group>
      )}
    </Paper>
  );
}
