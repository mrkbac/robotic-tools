import { Badge, Anchor, Loader, Group, Text, Tooltip } from "@mantine/core";
import type { ScanMode } from "../mcap/types.ts";

interface ScanLevelIndicatorProps {
  scannedMode: ScanMode;
  loading: boolean;
  scanTarget: ScanMode | null;
  onScanTo: (mode: ScanMode) => void;
  disabled?: boolean;
}

const BADGE_COLOR: Record<ScanMode, string> = {
  summary: "gray",
  rebuild: "blue",
  exact: "green",
};

const BADGE_TOOLTIP: Record<ScanMode, string> = {
  summary: "Header-only scan: file overview and schema info",
  rebuild: "Full scan: per-channel stats, timing, and sizes",
  exact: "Exact scan: precise message counts from index data",
};

const NEXT_MODE: Partial<Record<ScanMode, ScanMode>> = {
  summary: "rebuild",
  rebuild: "exact",
};

const UPGRADE_LABEL: Record<string, string> = {
  rebuild: "Upgrade to rebuild",
  exact: "Run exact scan",
};

export function ScanLevelIndicator({
  scannedMode,
  loading,
  scanTarget,
  onScanTo,
  disabled,
}: ScanLevelIndicatorProps) {
  const next = NEXT_MODE[scannedMode];
  const isScanning = loading && scanTarget != null;

  return (
    <Group gap="xs" wrap="nowrap">
      <Tooltip label={BADGE_TOOLTIP[scannedMode]} withArrow>
        <Badge
          color={BADGE_COLOR[scannedMode]}
          variant="light"
          size="sm"
          tt="capitalize"
        >
          {scannedMode}
        </Badge>
      </Tooltip>

      {isScanning ? (
        <Group gap={4} wrap="nowrap">
          <Loader size="xs" />
          <Text size="xs" c="dimmed">
            {scanTarget === "exact" ? "Exact scan..." : "Rebuilding..."}
          </Text>
        </Group>
      ) : (
        next &&
        !disabled && (
          <Anchor
            size="xs"
            c="dimmed"
            component="button"
            onClick={() => onScanTo(next)}
          >
            {UPGRADE_LABEL[next]}
          </Anchor>
        )
      )}
    </Group>
  );
}
