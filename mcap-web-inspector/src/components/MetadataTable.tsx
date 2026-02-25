import {
  Table,
  Title,
  Paper,
  Text,
  Accordion,
} from "@mantine/core";
import type { MetadataInfo } from "../mcap/types.ts";

interface MetadataTableProps {
  metadata: MetadataInfo[];
}

export function MetadataTable({ metadata }: MetadataTableProps) {
  if (metadata.length === 0) return null;

  return (
    <Paper p="md" withBorder>
      <Accordion variant="default" chevronPosition="left">
        <Accordion.Item value="metadata">
          <Accordion.Control>
            <Title order={4}>Metadata ({metadata.length})</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <Accordion variant="separated" chevronPosition="left">
              {metadata.map((m, idx) => (
                <Accordion.Item key={`${m.name}-${idx}`} value={`${m.name}-${idx}`}>
                  <Accordion.Control>
                    <Text size="sm" fw={500}>
                      {m.name}
                    </Text>
                  </Accordion.Control>
                  <Accordion.Panel>
                    {Object.keys(m.metadata).length === 0 ? (
                      <Text size="sm" c="dimmed">
                        No key-value pairs
                      </Text>
                    ) : (
                      <Table striped highlightOnHover>
                        <Table.Thead>
                          <Table.Tr>
                            <Table.Th>Key</Table.Th>
                            <Table.Th>Value</Table.Th>
                          </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                          {Object.entries(m.metadata).map(([key, value]) => (
                            <Table.Tr key={key}>
                              <Table.Td>
                                <Text size="sm" style={{ fontFamily: "monospace" }}>
                                  {key}
                                </Text>
                              </Table.Td>
                              <Table.Td>
                                <Text size="sm" style={{ fontFamily: "monospace", wordBreak: "break-all" }}>
                                  {value}
                                </Text>
                              </Table.Td>
                            </Table.Tr>
                          ))}
                        </Table.Tbody>
                      </Table>
                    )}
                  </Accordion.Panel>
                </Accordion.Item>
              ))}
            </Accordion>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Paper>
  );
}
