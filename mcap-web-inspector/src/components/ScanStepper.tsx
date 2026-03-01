import { Stepper, Anchor, Text, Group } from "@mantine/core";
import { IconFileAnalytics, IconRefresh } from "@tabler/icons-react";
import type { ScanMode } from "../mcap/types.ts";

interface ScanStepperProps {
  scannedMode: ScanMode;
  loading: boolean;
  scanTarget: ScanMode | null;
  onScanTo: (mode: ScanMode) => void;
  disabled?: boolean;
}

const MODES: ScanMode[] = ["summary", "rebuild"];

const STEPS = [
  {
    label: "Summary",
    description: "Overview & schema info",
    icon: <IconFileAnalytics size={18} />,
  },
  {
    label: "Rebuild",
    description: "Per-channel stats & timing",
    icon: <IconRefresh size={18} />,
  },
] as const;

export function ScanStepper({
  scannedMode,
  loading,
  scanTarget,
  onScanTo,
  disabled,
}: ScanStepperProps) {
  const activeIndex = Math.min(MODES.indexOf(scannedMode), MODES.length - 1);
  const rebuildDone = scannedMode === "rebuild" || scannedMode === "exact";
  const exactRunning = loading && scanTarget === "exact";
  const exactDone = scannedMode === "exact";

  return (
    <>
      <Stepper
        active={activeIndex}
        onStepClick={(index) => onScanTo(MODES[index]!)}
        size="sm"
      >
        {STEPS.map((step, i) => (
          <Stepper.Step
            key={step.label}
            label={step.label}
            description={step.description}
            icon={step.icon}
            allowStepSelect={!loading && !disabled && i > activeIndex}
            loading={loading && scanTarget === MODES[i]}
          />
        ))}
      </Stepper>

      {rebuildDone && !disabled && (
        <Group justify="flex-end">
          {exactRunning ? (
            <Text size="xs" c="dimmed">
              Running exact scan...
            </Text>
          ) : exactDone ? (
            <Text size="xs" c="dimmed">
              Exact scan complete
            </Text>
          ) : (
            <Anchor
              size="xs"
              c="dimmed"
              component="button"
              onClick={() => onScanTo("exact")}
            >
              Run exact scan...
            </Anchor>
          )}
        </Group>
      )}
    </>
  );
}
