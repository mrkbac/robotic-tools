import { Table, Title, Paper, Text, Group } from "@mantine/core";
import type { McapInfoOutput } from "../mcap/types.ts";
import { formatBytes, formatDuration } from "../format.ts";
import { CompressionPieChart } from "./CompressionPieChart.tsx";

interface CompressionTableProps {
  data: McapInfoOutput;
}

export function CompressionTable({ data }: CompressionTableProps) {
  const { by_compression } = data.chunks;
  const compressionTypes = Object.entries(by_compression);

  if (compressionTypes.length === 0) return null;

  const hasMessageCounts = compressionTypes.some(
    ([, stats]) => stats.message_count > 0,
  );
  const { overlaps } = data.chunks;

  return (
    <Paper p="md" withBorder>
      <Title order={4} mb="md">
        Compression
      </Title>
      <CompressionPieChart chunks={data.chunks} />
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
                {formatBytes(stats.compressed_size)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.uncompressed_size)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {(stats.compression_ratio * 100).toFixed(1)}%
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.size_stats.minimum)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.size_stats.average)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatBytes(stats.size_stats.maximum)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatDuration(stats.duration_stats.minimum)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatDuration(stats.duration_stats.average)}
              </Table.Td>
              <Table.Td style={{ textAlign: "right" }}>
                {formatDuration(stats.duration_stats.maximum)}
              </Table.Td>
              {hasMessageCounts && (
                <Table.Td style={{ textAlign: "right" }}>
                  {stats.message_count.toLocaleString()}
                </Table.Td>
              )}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>

      {overlaps.max_concurrent > 1 && (
        <Group mt="sm" gap="xs">
          <Text size="sm" fw={600} c="blue">
            Overlaps:
          </Text>
          <Text size="sm">
            {overlaps.max_concurrent} max concurrent,{" "}
            {formatBytes(overlaps.max_concurrent_bytes)} max total size at once
          </Text>
        </Group>
      )}
    </Paper>
  );
}
