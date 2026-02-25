import { useState } from "react";
import {
  Table,
  Title,
  Paper,
  Text,
  Accordion,
  Collapse,
  Group,
  Code,
  ScrollArea,
} from "@mantine/core";
import type { SchemaInfo } from "../mcap/types.ts";

interface SchemasTableProps {
  schemas: SchemaInfo[];
}

export function SchemasTable({ schemas }: SchemasTableProps) {
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  if (schemas.length === 0) return null;

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
    <Paper p="md" withBorder>
      <Accordion variant="default" chevronPosition="left">
        <Accordion.Item value="schemas">
          <Accordion.Control>
            <Title order={4}>Schemas ({schemas.length})</Title>
          </Accordion.Control>
          <Accordion.Panel>
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
                    <>
                      <Table.Tr
                        key={s.id}
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
                      {hasData && (
                        <Table.Tr
                          key={`${s.id}-detail`}
                          style={{ backgroundColor: "transparent" }}
                        >
                          <Table.Td
                            colSpan={3}
                            style={{
                              padding: 0,
                              border: expanded ? undefined : "none",
                            }}
                          >
                            <Collapse in={expanded}>
                              <ScrollArea
                                mah={400}
                                style={{ padding: "8px 16px 12px" }}
                              >
                                <Code block style={{ whiteSpace: "pre" }}>
                                  {s.data}
                                </Code>
                              </ScrollArea>
                            </Collapse>
                          </Table.Td>
                        </Table.Tr>
                      )}
                    </>
                  );
                })}
              </Table.Tbody>
            </Table>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Paper>
  );
}
