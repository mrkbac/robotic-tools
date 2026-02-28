import { useState, useCallback } from "react";
import {
  Table,
  Title,
  Paper,
  Text,
  Accordion,
  Button,
  Image,
  Group,
  ScrollArea,
} from "@mantine/core";
import { IconDownload } from "@tabler/icons-react";
import type { AttachmentInfo } from "../mcap/types.ts";
import { readAttachment } from "../mcap/reader.ts";
import { formatBytes, formatTimestamp } from "../format.ts";

interface AttachmentsTableProps {
  attachments: AttachmentInfo[];
  localFile: File | undefined;
}

export function AttachmentsTable({ attachments, localFile }: AttachmentsTableProps) {
  if (attachments.length === 0) return null;

  return (
    <Paper p="md" withBorder>
      <Accordion variant="default" chevronPosition="left">
        <Accordion.Item value="attachments">
          <Accordion.Control>
            <Title order={4}>Attachments ({attachments.length})</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <ScrollArea scrollbars="x">
              <Table striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Name</Table.Th>
                    <Table.Th>Media Type</Table.Th>
                    <Table.Th style={{ textAlign: "right" }}>Size</Table.Th>
                    <Table.Th>Log Time</Table.Th>
                    <Table.Th>Create Time</Table.Th>
                    {localFile && <Table.Th>Preview / Download</Table.Th>}
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
  const [thumbnailUrl, setThumbnailUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const isImage = attachment.mediaType.startsWith("image/");

  const handleDownload = useCallback(async () => {
    if (!localFile) return;
    setLoading(true);
    try {
      const result = await readAttachment(
        localFile,
        attachment.offset,
        attachment.length,
      );
      const blob = new Blob([result.data as BlobPart], { type: result.mediaType });
      const url = URL.createObjectURL(blob);

      if (isImage && !thumbnailUrl) {
        setThumbnailUrl(url);
        setLoaded(true);
      }

      // Trigger download
      const a = document.createElement("a");
      a.href = url;
      a.download = result.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      if (!isImage) {
        URL.revokeObjectURL(url);
      }
    } finally {
      setLoading(false);
    }
  }, [localFile, attachment, isImage, thumbnailUrl]);

  const handlePreview = useCallback(async () => {
    if (!localFile || loaded) return;
    setLoading(true);
    try {
      const result = await readAttachment(
        localFile,
        attachment.offset,
        attachment.length,
      );
      const blob = new Blob([result.data as BlobPart], { type: result.mediaType });
      setThumbnailUrl(URL.createObjectURL(blob));
      setLoaded(true);
    } finally {
      setLoading(false);
    }
  }, [localFile, attachment, loaded]);

  return (
    <Table.Tr>
      <Table.Td>
        <Text size="sm" style={{ fontFamily: "monospace" }}>
          {attachment.name}
        </Text>
      </Table.Td>
      <Table.Td>
        <Text size="sm" c="dimmed">
          {attachment.mediaType}
        </Text>
      </Table.Td>
      <Table.Td style={{ textAlign: "right" }}>
        <Text size="sm">{formatBytes(attachment.dataSize)}</Text>
      </Table.Td>
      <Table.Td>
        <Text size="xs">{attachment.logTime > 0n ? formatTimestamp(attachment.logTime) : "-"}</Text>
      </Table.Td>
      <Table.Td>
        <Text size="xs">{attachment.createTime > 0n ? formatTimestamp(attachment.createTime) : "-"}</Text>
      </Table.Td>
      {localFile && (
        <Table.Td>
          <Group gap="sm">
            {isImage && !loaded && (
              <Button
                size="xs"
                variant="subtle"
                onClick={handlePreview}
                loading={loading}
              >
                Preview
              </Button>
            )}
            {thumbnailUrl && (
              <Image
                src={thumbnailUrl}
                alt={attachment.name}
                h={60}
                w="auto"
                fit="contain"
                radius="sm"
              />
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
  );
}
