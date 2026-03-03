import { Link } from "@tanstack/react-router";
import {
  Paper,
  Text,
  Badge,
  ActionIcon,
  Stack,
  Group,
  Image,
  UnstyledButton,
} from "@mantine/core";
import { IconHistory, IconTrash, IconClock } from "@tabler/icons-react";
import type { HistoryEntry } from "../stores/historyStore.ts";
import { formatBytes } from "../format.ts";

const MODE_COLORS: Record<string, string> = {
  summary: "blue",
  rebuild: "orange",
  exact: "green",
};

function formatRelativeTime(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString();
}

interface RecentFilesProps {
  entries: HistoryEntry[];
  onRemove: (fileId: string) => void;
  onClearAll: () => void;
}

export function RecentFiles({
  entries,
  onRemove,
  onClearAll,
}: RecentFilesProps) {
  if (entries.length === 0) return null;

  return (
    <Stack gap="xs">
      <Group justify="space-between">
        <Group gap="xs">
          <IconHistory size={18} />
          <Text fw={500} size="sm">
            Recent files
          </Text>
        </Group>
        <ActionIcon
          variant="subtle"
          color="red"
          size="sm"
          onClick={() => {
            if (
              entries.length > 0 &&
              window.confirm("Clear all recent files?")
            ) {
              onClearAll();
            }
          }}
          title="Clear all"
        >
          <IconTrash size={14} />
        </ActionIcon>
      </Group>

      {entries.map((entry) => (
        <UnstyledButton
          key={entry.fileId}
          component={Link}
          to="/view"
          hash={entry.hash}
        >
          <Paper withBorder p="xs" radius="sm">
            <Group justify="space-between" wrap="nowrap">
              <Group gap="sm" wrap="nowrap" style={{ minWidth: 0 }}>
                {entry.thumbnailUrl && (
                  <Image
                    src={entry.thumbnailUrl}
                    alt=""
                    h={40}
                    w={56}
                    fit="cover"
                    radius="xs"
                    style={{ flexShrink: 0 }}
                  />
                )}
                <div style={{ minWidth: 0 }}>
                  <Text size="sm" fw={500} truncate="end">
                    {entry.fileName}
                  </Text>
                  <Group gap="xs">
                    <Text size="xs" c="dimmed">
                      {formatBytes(entry.fileSize)}
                    </Text>
                    <Badge
                      size="xs"
                      variant="light"
                      color={MODE_COLORS[entry.scanMode] ?? "gray"}
                    >
                      {entry.scanMode}
                    </Badge>
                    <Group gap={2}>
                      <IconClock size={12} style={{ opacity: 0.5 }} />
                      <Text size="xs" c="dimmed">
                        {formatRelativeTime(entry.scannedAt)}
                      </Text>
                    </Group>
                  </Group>
                </div>
              </Group>
              <ActionIcon
                variant="subtle"
                color="gray"
                size="sm"
                onClick={(e: React.MouseEvent) => {
                  e.preventDefault();
                  onRemove(entry.fileId);
                }}
                title="Remove"
              >
                <IconTrash size={14} />
              </ActionIcon>
            </Group>
          </Paper>
        </UnstyledButton>
      ))}
    </Stack>
  );
}
