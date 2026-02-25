import { useState } from "react";
import { Table, Title, Paper, Text, Collapse, Group } from "@mantine/core";
import type { SchemaInfo } from "../mcap/types.ts";

interface SchemasTableProps {
  schemas: SchemaInfo[];
}

export function SchemasTable({ schemas }: SchemasTableProps) {
  const [opened, setOpened] = useState(false);

  if (schemas.length === 0) return null;

  return (
    <Paper p="md" withBorder>
      <Group
        onClick={() => setOpened((o) => !o)}
        style={{ cursor: "pointer", userSelect: "none" }}
        gap="xs"
      >
        <Text
          size="sm"
          c="dimmed"
          style={{
            transition: "transform 150ms",
            transform: opened ? "rotate(90deg)" : "rotate(0deg)",
          }}
        >
          ▶
        </Text>
        <Title order={4}>Schemas ({schemas.length})</Title>
      </Group>
      <Collapse in={opened}>
        <Table striped highlightOnHover mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>ID</Table.Th>
              <Table.Th>Name</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {schemas.map((s) => (
              <Table.Tr key={s.id}>
                <Table.Td>
                  <Text size="sm" c="dimmed">
                    {s.id}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{s.name}</Text>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Collapse>
    </Paper>
  );
}
