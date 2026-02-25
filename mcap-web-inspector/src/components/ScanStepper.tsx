import { Stepper } from "@mantine/core";
import {
  IconFileAnalytics,
  IconRefresh,
  IconZoomCheck,
} from "@tabler/icons-react";
import type { ScanMode } from "../mcap/types.ts";

interface ScanStepperProps {
  scannedMode: ScanMode;
  loading: boolean;
  scanTarget: ScanMode | null;
  onScanTo: (mode: ScanMode) => void;
  disabled?: boolean;
}

const MODES: ScanMode[] = ["summary", "rebuild", "exact"];

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
  {
    label: "Exact",
    description: "Byte-level sizes & rates",
    icon: <IconZoomCheck size={18} />,
  },
] as const;

export function ScanStepper({
  scannedMode,
  loading,
  scanTarget,
  onScanTo,
  disabled,
}: ScanStepperProps) {
  const activeIndex = MODES.indexOf(scannedMode);

  return (
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
  );
}
