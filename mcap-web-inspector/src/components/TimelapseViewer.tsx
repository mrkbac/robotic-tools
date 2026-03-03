import { useState, useMemo, useEffect } from "react";
import {
  Paper,
  Title,
  Group,
  Text,
  Select,
  ActionIcon,
  Tooltip,
} from "@mantine/core";
import { IconDownload } from "@tabler/icons-react";

interface TimelapseViewerProps {
  videos: Map<number, Blob>;
  /** channelId → topic name, for display. */
  channelNames: Map<number, string>;
}

export function TimelapseViewer({ videos, channelNames }: TimelapseViewerProps) {
  const channelIds = useMemo(() => [...videos.keys()], [videos]);
  const [selectedChannel, setSelectedChannel] = useState<number | null>(null);

  // Auto-select first channel
  useEffect(() => {
    if (selectedChannel === null && channelIds.length > 0) {
      setSelectedChannel(channelIds[0]!);
    }
  }, [channelIds, selectedChannel]);

  const blob = selectedChannel !== null ? videos.get(selectedChannel) : undefined;
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!blob) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(blob);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [blob]);

  if (channelIds.length === 0 || !videoUrl) return null;

  const selectData = channelIds.map((id) => ({
    value: String(id),
    label: channelNames.get(id) ?? `Channel ${id}`,
  }));

  const handleDownload = () => {
    if (!blob || selectedChannel === null) return;
    const topic = channelNames.get(selectedChannel) ?? `channel_${selectedChannel}`;
    const safeName = topic.replace(/\//g, "_").replace(/^_/, "") + "_timelapse.webm";
    const a = document.createElement("a");
    a.href = videoUrl;
    a.download = safeName;
    a.click();
  };

  return (
    <Paper p="md" withBorder>
      <Group justify="space-between" mb="md">
        <Title order={4}>Timelapse Preview</Title>
        <Group gap="sm">
          {channelIds.length > 1 && (
            <Select
              size="xs"
              data={selectData}
              value={selectedChannel !== null ? String(selectedChannel) : null}
              onChange={(val) => setSelectedChannel(val ? Number(val) : null)}
              style={{ minWidth: 200 }}
            />
          )}
          {channelIds.length === 1 && (
            <Text size="sm" c="dimmed">
              {channelNames.get(channelIds[0]!) ?? `Channel ${channelIds[0]}`}
            </Text>
          )}
          <Tooltip label="Download WebM">
            <ActionIcon variant="subtle" onClick={handleDownload}>
              <IconDownload size={18} />
            </ActionIcon>
          </Tooltip>
        </Group>
      </Group>

      <video
        key={videoUrl}
        src={videoUrl}
        controls
        loop
        autoPlay
        muted
        style={{
          width: "100%",
          maxHeight: 480,
          borderRadius: "var(--mantine-radius-sm)",
          backgroundColor: "var(--mantine-color-dark-7)",
        }}
      />
    </Paper>
  );
}
