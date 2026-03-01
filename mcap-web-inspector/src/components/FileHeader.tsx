import { useEffect, useMemo, useRef, useState } from "react";
import { Group, Image, Modal, SimpleGrid, Stack, Text, Title, UnstyledButton } from "@mantine/core";
import type { ThumbnailMap } from "../mcap/image.ts";

function thumbnailToUrl(data: Uint8Array, format: string): string {
  const mime = format.startsWith("image/") ? format : `image/${format || "jpeg"}`;
  return URL.createObjectURL(new Blob([data as BlobPart], { type: mime }));
}

interface FileHeaderProps {
  fileName: string;
  thumbnails: ThumbnailMap;
}

export function FileHeader({ fileName, thumbnails }: FileHeaderProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const entries = useMemo(() => [...thumbnails.values()], [thumbnails]);
  const urlsRef = useRef<string[]>([]);

  useEffect(() => {
    const urls = entries.map((e) => thumbnailToUrl(e.data, e.format));
    urlsRef.current = urls;
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [entries]);

  return (
    <>
      <Group gap="md" align="center">
        <Title order={2}>{fileName}</Title>
        {entries.length > 0 && urlsRef.current.length > 0 && (
          <UnstyledButton onClick={() => setModalOpen(true)}>
            <Image
              src={urlsRef.current[0]}
              alt="Thumbnail"
              h={60}
              w="auto"
              fit="contain"
              radius="sm"
              style={{ cursor: "pointer" }}
            />
          </UnstyledButton>
        )}
      </Group>

      <Modal
        opened={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Image Topics"
        size="xl"
      >
        <SimpleGrid cols={{ base: 1, xs: 2, sm: 3, md: 4 }} spacing="sm">
          {entries.map((entry, i) => (
            <Stack key={entry.channelId} gap={4} align="center">
              <Image
                src={urlsRef.current[i]}
                alt={entry.topic}
                h={100}
                w="auto"
                fit="contain"
                radius="sm"
              />
              <Text size="xs" c="dimmed" ta="center" truncate="end" maw="100%">
                {entry.topic}
              </Text>
            </Stack>
          ))}
        </SimpleGrid>
      </Modal>
    </>
  );
}
