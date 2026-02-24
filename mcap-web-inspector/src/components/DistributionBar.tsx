import { useRef, useEffect } from "react";

/**
 * Color gradient for the distribution bar (low -> high):
 * blue -> cyan -> green -> yellow -> red
 */
const LEVEL_COLORS = [
  "#e0e0e0", // 0: empty
  "#4a90d9", // 1: blue
  "#5b9bd5", // 2: bright blue
  "#00bcd4", // 3: cyan
  "#4caf50", // 4: green
  "#ffeb3b", // 5: yellow
  "#ffc107", // 6: bright yellow
  "#ff5722", // 7: red
  "#d32f2f", // 8: bright red (max)
];

function downsampleToWidth(counts: number[], targetWidth: number): number[] {
  const numBuckets = counts.length;
  if (numBuckets <= targetWidth) return counts;

  const bucketsPerChar = numBuckets / targetWidth;
  const scaled: number[] = [];
  for (let i = 0; i < targetWidth; i++) {
    const startIdx = Math.floor(i * bucketsPerChar);
    const endIdx = Math.floor((i + 1) * bucketsPerChar);
    let maxVal = 0;
    for (let j = startIdx; j < endIdx && j < numBuckets; j++) {
      if (counts[j]! > maxVal) maxVal = counts[j]!;
    }
    scaled.push(maxVal);
  }
  return scaled;
}

interface DistributionBarProps {
  counts: number[];
  width?: number;
  height?: number;
}

export function DistributionBar({
  counts,
  width = 400,
  height = 24,
}: DistributionBarProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const maxCount = Math.max(0, ...counts);
    if (maxCount === 0) {
      ctx.fillStyle = "#e0e0e0";
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "#999";
      ctx.font = "11px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("no messages", width / 2, height / 2 + 4);
      return;
    }

    const displayCounts = downsampleToWidth(counts, width);
    const barWidth = width / displayCounts.length;

    for (let i = 0; i < displayCounts.length; i++) {
      const count = displayCounts[i]!;
      if (count === 0) {
        ctx.fillStyle = LEVEL_COLORS[0]!;
        ctx.fillRect(i * barWidth, 0, barWidth + 0.5, height);
      } else {
        const level = Math.min(8, Math.max(1, Math.ceil((count / maxCount) * 8)));
        ctx.fillStyle = LEVEL_COLORS[level]!;
        const barH = Math.max(2, (count / maxCount) * height);
        // Background
        ctx.fillStyle = "#f5f5f5";
        ctx.fillRect(i * barWidth, 0, barWidth + 0.5, height);
        // Bar
        ctx.fillStyle = LEVEL_COLORS[level]!;
        ctx.fillRect(i * barWidth, height - barH, barWidth + 0.5, barH);
      }
    }
  }, [counts, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, borderRadius: 4 }}
    />
  );
}

/** Smaller inline distribution bar for use in table cells. */
export function InlineDistributionBar({ counts }: { counts: number[] }) {
  return <DistributionBar counts={counts} width={120} height={16} />;
}
