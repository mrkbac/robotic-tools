import {
  Badge,
  Anchor,
  Button,
  Loader,
  Group,
  Text,
  Tooltip,
} from "@mantine/core";
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
            {scanTarget === "exact" ? "Exact scan..." : "Full scan..."}
          </Text>
        </Group>
      ) : (
        !disabled && (
          <Group gap="xs" wrap="nowrap">
            {scannedMode === "summary" && (
              <Tooltip
                label="TF tree, thumbnails, per-channel timing"
                withArrow
              >
                <Button
                  variant="light"
                  size="compact-xs"
                  onClick={() => onScanTo("rebuild")}
                >
                  Full scan
                </Button>
              </Tooltip>
            )}
            {next === "exact" && (
              <Anchor
                size="xs"
                c="dimmed"
                component="button"
                onClick={() => onScanTo("exact")}
              >
                Exact scan
              </Anchor>
            )}
          </Group>
        )
      )}
    </Group>
  );
}
