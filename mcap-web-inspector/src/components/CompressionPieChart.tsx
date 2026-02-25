import { PieChart } from "@mantine/charts";
import type { ChunksInfo } from "../mcap/types.ts";
import { formatBytes } from "../format.ts";

const COMPRESSION_COLORS: Record<string, string> = {
  zstd: "#4a90d9",
  lz4: "#4caf50",
  none: "#9e9e9e",
  "": "#9e9e9e",
  zlib: "#ff9800",
  bz2: "#9c27b0",
};

const FALLBACK_COLORS = [
  "#00bcd4",
  "#e91e63",
  "#3f51b5",
  "#ff5722",
  "#795548",
  "#607d8b",
];

function colorForType(type: string): string {
  if (type in COMPRESSION_COLORS) return COMPRESSION_COLORS[type]!;
  let hash = 0;
  for (let i = 0; i < type.length; i++) {
    hash = type.charCodeAt(i) + ((hash << 5) - hash);
  }
  return FALLBACK_COLORS[Math.abs(hash) % FALLBACK_COLORS.length]!;
}

interface CompressionPieChartProps {
  chunks: ChunksInfo;
}

export function CompressionPieChart({ chunks }: CompressionPieChartProps) {
  const entries = Object.entries(chunks.byCompression);
  if (entries.length === 0) return null;

  const data = entries.map(([type, stats]) => ({
    name: type || "none",
    value: stats.uncompressedSize,
    color: colorForType(type),
  }));

  return (
    <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
      <PieChart
        data={data}
        withTooltip
        tooltipDataSource="segment"
        withLabelsLine
        labelsType="percent"
        labelsPosition="outside"
        size={240}
        valueFormatter={(value) => formatBytes(value)}
      />
    </div>
  );
}
