import { Alert, Button, Group, Text } from "@mantine/core";
import type { ScanMode } from "../mcap/types.ts";

interface ScanUpgradeBarProps {
  scannedMode: ScanMode;
  loading: boolean;
  onUpgrade: () => void;
}

const upgradeConfig = {
  summary: {
    text: "Scan the file to get per-channel distributions, Hz statistics, and timing details.",
    button: "Scan for details",
  },
  rebuild: {
    text: "Run a full scan to get exact byte sizes and bytes-per-second statistics.",
    button: "Full scan for sizes",
  },
} as const;

export function ScanUpgradeBar({
  scannedMode,
  loading,
  onUpgrade,
}: ScanUpgradeBarProps) {
  if (scannedMode === "exact") return null;

  const config = upgradeConfig[scannedMode];

  return (
    <Alert variant="light" color="blue" p="sm">
      <Group justify="space-between" wrap="nowrap">
        <Text size="sm">{config.text}</Text>
        <Button
          size="xs"
          variant="light"
          onClick={onUpgrade}
          loading={loading}
        >
          {config.button}
        </Button>
      </Group>
    </Alert>
  );
}
