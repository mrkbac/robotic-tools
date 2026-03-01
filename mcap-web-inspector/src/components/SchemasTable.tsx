import { Fragment, useState } from "react";
import { Table, Title, Paper, Text, Accordion, Group } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import type { SchemaInfo } from "../mcap/types.ts";

function encodingToLanguage(encoding: string): string | undefined {
  switch (encoding.toLowerCase()) {
    case "jsonschema":
    case "json":
      return "json";
    case "protobuf":
      return "protobuf";
    case "flatbuffer":
      return "cpp";
    default:
      return undefined;
  }
}

interface SchemasTableProps {
  schemas: SchemaInfo[];
  bare?: boolean;
}

function SchemasContent({ schemas }: { schemas: SchemaInfo[] }) {
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const toggleExpanded = (id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <Table striped highlightOnHover>
      <Table.Thead>
        <Table.Tr>
          <Table.Th>ID</Table.Th>
          <Table.Th>Name</Table.Th>
          <Table.Th>Encoding</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {schemas.map((s) => {
          const expanded = expandedIds.has(s.id);
          const hasData = s.data.length > 0;
          return (
            <Fragment key={s.id}>
              <Table.Tr
                onClick={hasData ? () => toggleExpanded(s.id) : undefined}
                style={hasData ? { cursor: "pointer" } : undefined}
              >
                <Table.Td>
                  <Group gap={4}>
                    {hasData ? (
                      <Text
                        size="xs"
                        c="dimmed"
                        style={{
                          transition: "transform 150ms",
                          transform: expanded
                            ? "rotate(90deg)"
                            : "rotate(0deg)",
                        }}
                      >
                        ▶
                      </Text>
                    ) : (
                      <Text size="xs" c="dimmed" style={{ width: 10 }}>
                        {" "}
                      </Text>
                    )}
                    <Text size="sm" c="dimmed">
                      {s.id}
                    </Text>
                  </Group>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{s.name}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm" c="dimmed">
                    {s.encoding || "-"}
                  </Text>
                </Table.Td>
              </Table.Tr>
              {hasData && expanded && (
                <Table.Tr style={{ backgroundColor: "transparent" }}>
                  <Table.Td colSpan={3} style={{ padding: 0 }}>
                    <CodeHighlight
                      code={s.data}
                      language={encodingToLanguage(s.encoding)}
                      withExpandButton
                      defaultExpanded={false}
                      maxCollapsedHeight="400px"
                    />
                  </Table.Td>
                </Table.Tr>
              )}
            </Fragment>
          );
        })}
      </Table.Tbody>
    </Table>
  );
}

export function SchemasTable({ schemas, bare }: SchemasTableProps) {
  if (schemas.length === 0) return null;

  if (bare) return <SchemasContent schemas={schemas} />;

  return (
    <Paper p="md" withBorder>
      <Accordion variant="default" chevronPosition="left">
        <Accordion.Item value="schemas">
          <Accordion.Control>
            <Title order={4}>Schemas ({schemas.length})</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <SchemasContent schemas={schemas} />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Paper>
  );
}
