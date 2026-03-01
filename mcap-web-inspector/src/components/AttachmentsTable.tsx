import { useState, useCallback, useRef } from "react";
import {
  Table,
  Title,
  Paper,
  Text,
  Accordion,
  Button,
  Group,
  ScrollArea,
  Code,
  Collapse,
  Box,
} from "@mantine/core";
import { IconDownload, IconEye } from "@tabler/icons-react";
import type { AttachmentInfo } from "../mcap/types.ts";
import { readAttachment } from "../mcap/reader.ts";
import { formatBytes, formatTimestamp } from "../format.ts";

const TEXT_MEDIA_TYPES = new Set([
  "application/json",
  "application/yaml",
  "application/xml",
  "application/toml",
]);

function isTextType(mediaType: string): boolean {
  return mediaType.startsWith("text/") || TEXT_MEDIA_TYPES.has(mediaType);
}

interface AttachmentsTableProps {
  attachments: AttachmentInfo[];
  localFile: File | undefined;
  bare?: boolean;
}

function AttachmentsContent({
  attachments,
  localFile,
}: {
  attachments: AttachmentInfo[];
  localFile: File | undefined;
}) {
  return (
    <ScrollArea scrollbars="x">
      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Name</Table.Th>
            <Table.Th>Media Type</Table.Th>
            <Table.Th style={{ textAlign: "right" }}>Size</Table.Th>
            <Table.Th>Log Time</Table.Th>
            <Table.Th>Create Time</Table.Th>
            {localFile && <Table.Th>Actions</Table.Th>}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {attachments.map((a, idx) => (
            <AttachmentRow
              key={`${a.name}-${idx}`}
              attachment={a}
              localFile={localFile}
            />
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  );
}

export function AttachmentsTable({
  attachments,
  localFile,
  bare,
}: AttachmentsTableProps) {
  if (attachments.length === 0) return null;

  if (bare)
    return (
      <AttachmentsContent attachments={attachments} localFile={localFile} />
    );

  return (
    <Paper p="md" withBorder>
      <Accordion variant="default" chevronPosition="left">
        <Accordion.Item value="attachments">
          <Accordion.Control>
            <Title order={4}>Attachments ({attachments.length})</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <AttachmentsContent
              attachments={attachments}
              localFile={localFile}
            />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Paper>
  );
}

function AttachmentRow({
  attachment,
  localFile,
}: {
  attachment: AttachmentInfo;
  localFile: File | undefined;
}) {
  const [loading, setLoading] = useState(false);
  const [textContent, setTextContent] = useState<string | null>(null);
  const [viewOpen, setViewOpen] = useState(false);
  const cachedData = useRef<{
    name: string;
    mediaType: string;
    data: Uint8Array;
  } | null>(null);

  const isText = isTextType(attachment.media_type);

  const loadData = useCallback(async () => {
    if (cachedData.current) return cachedData.current;
    const result = await readAttachment(localFile!, attachment.name);
    cachedData.current = result;
    return result;
  }, [localFile, attachment.name]);

  const handleView = useCallback(async () => {
    if (textContent != null) {
      setViewOpen((v) => !v);
      return;
    }
    setLoading(true);
    try {
      const result = await loadData();
      setTextContent(new TextDecoder().decode(result.data));
      setViewOpen(true);
    } finally {
      setLoading(false);
    }
  }, [textContent, loadData]);

  const handleDownload = useCallback(async () => {
    if (!localFile) return;
    setLoading(true);
    try {
      const result = await loadData();
      const blob = new Blob([result.data as BlobPart], {
        type: result.mediaType,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = result.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } finally {
      setLoading(false);
    }
  }, [localFile, loadData]);

  return (
    <>
      <Table.Tr>
        <Table.Td>
          <Text size="sm" style={{ fontFamily: "monospace" }}>
            {attachment.name}
          </Text>
        </Table.Td>
        <Table.Td>
          <Text size="sm" c="dimmed">
            {attachment.media_type}
          </Text>
        </Table.Td>
        <Table.Td style={{ textAlign: "right" }}>
          <Text size="sm">{formatBytes(attachment.data_size)}</Text>
        </Table.Td>
        <Table.Td>
          <Text size="xs">
            {attachment.log_time > 0
              ? formatTimestamp(attachment.log_time)
              : "-"}
          </Text>
        </Table.Td>
        <Table.Td>
          <Text size="xs">
            {attachment.create_time > 0
              ? formatTimestamp(attachment.create_time)
              : "-"}
          </Text>
        </Table.Td>
        {localFile && (
          <Table.Td>
            <Group gap="sm">
              {isText && (
                <Button
                  size="xs"
                  variant="subtle"
                  leftSection={<IconEye size={14} />}
                  onClick={handleView}
                  loading={loading}
                >
                  {viewOpen ? "Hide" : "View"}
                </Button>
              )}
              <Button
                size="xs"
                variant="light"
                leftSection={<IconDownload size={14} />}
                onClick={handleDownload}
                loading={loading}
              >
                Download
              </Button>
            </Group>
          </Table.Td>
        )}
      </Table.Tr>
      {isText && (
        <Table.Tr>
          <Table.Td
            colSpan={localFile ? 6 : 5}
            p={0}
            style={{ border: viewOpen ? undefined : "none" }}
          >
            <Collapse in={viewOpen}>
              <Box mah={400} style={{ overflow: "auto" }} p="xs">
                <Code block>{textContent}</Code>
              </Box>
            </Collapse>
          </Table.Td>
        </Table.Tr>
      )}
    </>
  );
}
