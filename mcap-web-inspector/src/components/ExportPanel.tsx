import { useState, useCallback, useMemo, useEffect } from "react";
import {
  Paper,
  Title,
  Accordion,
  Alert,
  Stack,
  Group,
  Checkbox,
  Button,
  Switch,
  Text,
  Progress,
  RangeSlider,
  ScrollArea,
} from "@mantine/core";
import { IconDownload } from "@tabler/icons-react";
import type { McapInfoOutput, FilterConfig } from "../mcap/types.ts";
import { exportFilteredMcap, downloadMcap } from "../mcap/writer.ts";
import { formatTimestamp, formatNumber } from "../format.ts";

interface ExportPanelProps {
  file: File;
  data: McapInfoOutput;
}

const SLIDER_MAX = 1000;

export function ExportPanel({ file, data }: ExportPanelProps) {
  const channels = data.channels;
  const hasTimeRange =
    data.statistics.message_start_time < data.statistics.message_end_time;
  const hasMetadata = data.metadata.length > 0;
  const hasAttachments = data.attachments.length > 0;

  // Topic selection: set of included channel IDs
  const allChannelIds = useMemo(
    () => new Set(channels.map((ch) => ch.id)),
    [channels],
  );
  const [selectedIds, setSelectedIds] = useState<Set<number>>(
    () => new Set(allChannelIds),
  );

  // Auto-select newly appeared channels (e.g. after scan upgrade)
  useEffect(() => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (const id of allChannelIds) {
        if (!prev.has(id)) next.add(id);
      }
      return next.size === prev.size ? prev : next;
    });
  }, [allChannelIds]);

  // Time range (slider values 0-1000)
  const [timeRange, setTimeRange] = useState<[number, number]>([0, SLIDER_MAX]);

  // Toggles
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [includeAttachments, setIncludeAttachments] = useState(true);

  // Export state
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [progressPct, setProgressPct] = useState(0);
  const [progressInfo, setProgressInfo] = useState<{
    messagesWritten: number;
    messagesSkipped: number;
  } | null>(null);

  const allSelected = selectedIds.size === allChannelIds.size;
  const noneSelected = selectedIds.size === 0;

  const toggleChannel = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedIds(new Set(allChannelIds));
  }, [allChannelIds]);

  const selectNone = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  // Convert slider values to bigint nanosecond timestamps
  const sliderToTime = useCallback(
    (value: number): bigint => {
      const start = BigInt(Math.round(data.statistics.message_start_time));
      const end = BigInt(Math.round(data.statistics.message_end_time));
      const range = end - start;
      return start + (range * BigInt(value)) / BigInt(SLIDER_MAX);
    },
    [data.statistics.message_start_time, data.statistics.message_end_time],
  );

  const timeFiltered =
    hasTimeRange && (timeRange[0] > 0 || timeRange[1] < SLIDER_MAX);
  const startTime = timeFiltered ? sliderToTime(timeRange[0]) : null;
  const endTime = timeFiltered ? sliderToTime(timeRange[1]) : null;

  // Estimate selected message count
  const estimatedMessages = useMemo(() => {
    let total = 0;
    for (const ch of channels) {
      if (selectedIds.has(ch.id)) {
        total += ch.message_count;
      }
    }
    return total;
  }, [channels, selectedIds]);

  const handleExport = useCallback(async () => {
    setExporting(true);
    setExportError(null);
    setProgressPct(0);
    setProgressInfo(null);

    try {
      const config: FilterConfig = {
        include_channel_ids: allSelected ? null : new Set(selectedIds),
        start_time: startTime,
        end_time: endTime,
        include_metadata: includeMetadata,
        include_attachments: includeAttachments,
      };

      const result = await exportFilteredMcap(file, config, (info) => {
        setProgressPct(
          Math.round((info.bytesRead / info.totalBytes) * 100),
        );
        setProgressInfo({
          messagesWritten: info.messagesWritten,
          messagesSkipped: info.messagesSkipped,
        });
      });

      // Generate output filename
      const baseName = file.name.replace(/\.mcap$/i, "");
      downloadMcap(result, `${baseName}_filtered.mcap`);
    } catch (err) {
      setExportError(
        err instanceof Error ? err.message : "Export failed",
      );
    } finally {
      setExporting(false);
    }
  }, [
    file,
    selectedIds,
    allSelected,
    startTime,
    endTime,
    includeMetadata,
    includeAttachments,
  ]);

  // Convert number ns timestamps to display values
  const displayStartTime = startTime !== null
    ? Number(startTime)
    : data.statistics.message_start_time;
  const displayEndTime = endTime !== null
    ? Number(endTime)
    : data.statistics.message_end_time;

  return (
    <Paper p="md" withBorder>
      <Accordion variant="default" chevronPosition="left">
        <Accordion.Item value="export">
          <Accordion.Control>
            <Title order={4}>Export</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <Stack gap="md">
              {/* Topic selector */}
              <div>
                <Group justify="space-between" mb="xs">
                  <Text size="sm" fw={600}>
                    Topics ({selectedIds.size}/{channels.length})
                  </Text>
                  <Group gap="xs">
                    <Button
                      size="compact-xs"
                      variant="subtle"
                      onClick={selectAll}
                      disabled={allSelected}
                    >
                      All
                    </Button>
                    <Button
                      size="compact-xs"
                      variant="subtle"
                      onClick={selectNone}
                      disabled={noneSelected}
                    >
                      None
                    </Button>
                  </Group>
                </Group>
                <ScrollArea.Autosize mah={200}>
                  <Stack gap={4}>
                    {channels.map((ch) => (
                      <Checkbox
                        key={ch.id}
                        size="xs"
                        checked={selectedIds.has(ch.id)}
                        onChange={() => toggleChannel(ch.id)}
                        label={
                          <Group gap="xs">
                            <Text size="xs" style={{ fontFamily: "monospace" }}>
                              {ch.topic}
                            </Text>
                            <Text size="xs" c="dimmed">
                              ({formatNumber(ch.message_count)} msgs
                              {ch.schema_name ? `, ${ch.schema_name}` : ""})
                            </Text>
                          </Group>
                        }
                      />
                    ))}
                  </Stack>
                </ScrollArea.Autosize>
              </div>

              {/* Time range */}
              {hasTimeRange && (
                <div>
                  <Group justify="space-between" mb="xs">
                    <Text size="sm" fw={600}>
                      Time Range
                    </Text>
                    {timeFiltered && (
                      <Button
                        size="compact-xs"
                        variant="subtle"
                        onClick={() => setTimeRange([0, SLIDER_MAX])}
                      >
                        Reset
                      </Button>
                    )}
                  </Group>
                  <RangeSlider
                    min={0}
                    max={SLIDER_MAX}
                    step={1}
                    value={timeRange}
                    onChange={setTimeRange}
                    label={null}
                    mb="xs"
                  />
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">
                      {formatTimestamp(displayStartTime)}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {formatTimestamp(displayEndTime)}
                    </Text>
                  </Group>
                </div>
              )}

              {/* Toggles */}
              <Group gap="xl">
                {hasMetadata && (
                  <Switch
                    size="xs"
                    label="Include metadata"
                    checked={includeMetadata}
                    onChange={(e) =>
                      setIncludeMetadata(e.currentTarget.checked)
                    }
                  />
                )}
                {hasAttachments && (
                  <Switch
                    size="xs"
                    label="Include attachments"
                    checked={includeAttachments}
                    onChange={(e) =>
                      setIncludeAttachments(e.currentTarget.checked)
                    }
                  />
                )}
              </Group>

              {/* Summary */}
              <Text size="xs" c="dimmed">
                {selectedIds.size} channel{selectedIds.size !== 1 ? "s" : ""}
                {" selected"}
                {" \u00B7 ~"}
                {formatNumber(estimatedMessages)} messages
                {timeFiltered && " (before time filter)"}
              </Text>

              {/* Export button + progress */}
              <div>
                <Button
                  leftSection={<IconDownload size={16} />}
                  onClick={handleExport}
                  loading={exporting}
                  disabled={noneSelected}
                >
                  Export filtered MCAP
                </Button>
              </div>

              {exportError && (
                <Alert color="red" title="Export failed">
                  {exportError}
                </Alert>
              )}

              {exporting && (
                <Stack gap="xs">
                  <Progress value={progressPct} size="sm" />
                  <Text size="xs" c="dimmed">
                    {progressPct}% read
                    {progressInfo &&
                      ` \u00B7 ${formatNumber(progressInfo.messagesWritten)} written, ${formatNumber(progressInfo.messagesSkipped)} skipped`}
                  </Text>
                </Stack>
              )}
            </Stack>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Paper>
  );
}
