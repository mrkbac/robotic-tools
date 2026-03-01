import { Modal, ScrollArea } from "@mantine/core";
import type { McapInfoOutput } from "../mcap/types.ts";
import { CompressionTable } from "./CompressionTable.tsx";
import { SchemasTable } from "./SchemasTable.tsx";
import { MetadataTable } from "./MetadataTable.tsx";
import { AttachmentsTable } from "./AttachmentsTable.tsx";

export type DetailSection = "chunks" | "schemas" | "metadata" | "attachments";

interface DetailModalProps {
  section: DetailSection | null;
  onClose: () => void;
  data: McapInfoOutput;
  localFile?: File;
}

const TITLES: Record<DetailSection, (data: McapInfoOutput) => string> = {
  chunks: (d) => {
    const types = Object.keys(d.chunks.by_compression).length;
    return `Compression (${types} ${types === 1 ? "type" : "types"})`;
  },
  schemas: (d) => `Schemas (${d.schemas.length})`,
  metadata: (d) => `Metadata (${d.metadata.length})`,
  attachments: (d) => `Attachments (${d.attachments.length})`,
};

export function DetailModal({ section, onClose, data, localFile }: DetailModalProps) {
  return (
    <Modal
      opened={section != null}
      onClose={onClose}
      title={section ? TITLES[section](data) : ""}
      size="xl"
    >
      <ScrollArea.Autosize mah="70vh">
        {section === "chunks" && <CompressionTable data={data} bare />}
        {section === "schemas" && <SchemasTable schemas={data.schemas} bare />}
        {section === "metadata" && <MetadataTable metadata={data.metadata} bare />}
        {section === "attachments" && (
          <AttachmentsTable attachments={data.attachments} localFile={localFile} bare />
        )}
      </ScrollArea.Autosize>
    </Modal>
  );
}
