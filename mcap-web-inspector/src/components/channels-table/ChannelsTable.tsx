import { useState, useMemo, Fragment, useCallback } from "react";
import {
  Table,
  Title,
  Paper,
  Text,
  Group,
  Collapse,
  ScrollArea,
  SegmentedControl,
  Menu,
  ActionIcon,
  Checkbox,
} from "@mantine/core";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getExpandedRowModel,
  flexRender,
  type SortingState,
  type VisibilityState,
  type ExpandedState,
  type Header,
} from "@tanstack/react-table";
import { IconColumns } from "@tabler/icons-react";
import type { ChannelInfo } from "../../mcap/types.ts";
import type { ChannelRow } from "./types.ts";
import { getColumns } from "./columns.tsx";
import { ChannelDetail } from "./ChannelDetail.tsx";
import { buildTreeData, toFlatRows } from "./tree-data.ts";
import { stringToColor } from "./utils.ts";

type ViewMode = "flat" | "tree";

// Tree rail layout constants
const RAIL_START = 10; // px from cell left edge to first rail
const RAIL_GAP = 14;   // px between rail centers
const RAIL_WIDTH = 3;  // px width of each rail line

/** Build CSS background + paddingLeft for vertical tree rail lines on a cell. */
function getRailStyle(topic: string, depth: number): React.CSSProperties | undefined {
  if (depth <= 0) return undefined;
  const segments = topic.split("/").filter(Boolean);
  const gradients = Array.from({ length: depth }, (_, i) => {
    const color = stringToColor(segments[i] ?? "");
    const x = RAIL_START + i * RAIL_GAP;
    return `linear-gradient(${color},${color}) ${x}px 0/${RAIL_WIDTH}px 100% no-repeat`;
  });
  return {
    background: gradients.join(","),
    paddingLeft: RAIL_START + depth * RAIL_GAP + 4,
  };
}

interface ChannelsTableProps {
  channels: ChannelInfo[];
  bucketDurationNs: number;
  fileSize: number;
}

export function ChannelsTable({ channels, bucketDurationNs, fileSize }: ChannelsTableProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("flat");
  const [sorting, setSorting] = useState<SortingState>([{ id: "topic", desc: false }]);
  const [detailExpandedIds, setDetailExpandedIds] = useState<Set<number>>(new Set());
  const [expanded, setExpanded] = useState<ExpandedState>(true);

  const hasSizeData = channels.some((ch) => ch.size_bytes !== null);
  const hasEstimatedSizes = channels.some((ch) => ch.estimated_sizes && ch.size_bytes !== null);
  const hasDistribution = channels.some((ch) => ch.message_distribution.length > 0);
  const hasJitter = channels.some((ch) => ch.jitter_cv !== null);

  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(() => ({
    jitter: hasJitter,
    size: hasSizeData,
    bps: hasSizeData,
    bPerMsg: hasSizeData,
    distribution: hasDistribution,
  }));

  const columns = useMemo(
    () => getColumns({ fileSize, hasEstimatedSizes, detailExpandedIds }),
    [fileSize, hasEstimatedSizes, detailExpandedIds],
  );

  const data = useMemo(() => {
    if (viewMode === "tree") return buildTreeData(channels);
    return toFlatRows(channels);
  }, [channels, viewMode]);

  const toggleDetail = useCallback((id: number) => {
    setDetailExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const table = useReactTable<ChannelRow>({
    data,
    columns,
    state: { sorting, columnVisibility, expanded },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    onExpandedChange: setExpanded,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    getSubRows: (row) => row.subRows,
    getRowId: (row) => row._kind === "group" ? `g:${row._fullPath}` : `c:${row.id}`,
    enableSortingRemoval: false,
  });

  const visibleCellCount = table.getVisibleLeafColumns().length;
  const isTree = viewMode === "tree";

  return (
    <Paper p="md" withBorder>
      <Group justify="space-between" mb="md">
        <Title order={4}>Channels</Title>
        <Group gap="xs">
          <ColumnsMenu table={table} />
          {channels.length > 0 && (
            <SegmentedControl
              size="xs"
              value={viewMode}
              onChange={(v) => {
                setViewMode(v as ViewMode);
                setExpanded(v === "tree" ? true : {});
              }}
              data={[
                { label: "Flat", value: "flat" },
                { label: "Tree", value: "tree" },
              ]}
            />
          )}
        </Group>
      </Group>
      {channels.length === 0 ? (
        <Text c="dimmed">No channels found</Text>
      ) : (
        <ScrollArea scrollbars="x">
          <Table striped={!isTree} highlightOnHover>
            <Table.Thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <Table.Tr key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <SortableHeader key={header.id} header={header} />
                  ))}
                </Table.Tr>
              ))}
            </Table.Thead>
            <Table.Tbody>
              {table.getRowModel().rows.map((row) => {
                const isGroup = row.original._kind === "group";
                const detailExpanded = !isGroup && detailExpandedIds.has(row.original.id);

                return (
                  <Fragment key={row.id}>
                    <Table.Tr
                      onClick={() =>
                        isGroup
                          ? row.toggleExpanded()
                          : toggleDetail(row.original.id)
                      }
                      style={{ cursor: "pointer" }}
                    >
                      {row.getVisibleCells().map((cell) => {
                        const railStyle = isTree && cell.column.id === "id"
                          ? getRailStyle(row.original.topic, row.depth)
                          : undefined;
                        return (
                          <Table.Td
                            key={cell.id}
                            style={{
                              textAlign: cell.column.columnDef.meta?.align,
                              ...railStyle,
                              ...(isGroup
                                ? { fontWeight: 500, opacity: 0.8 }
                                : {}),
                            }}
                          >
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </Table.Td>
                        );
                      })}
                    </Table.Tr>

                    {!isGroup && (
                      <Table.Tr style={{ backgroundColor: "transparent" }}>
                        <Table.Td
                          colSpan={visibleCellCount}
                          style={{
                            padding: 0,
                            border: detailExpanded ? undefined : "none",
                          }}
                        >
                          <Collapse in={detailExpanded}>
                            {detailExpanded && (
                              <ChannelDetail
                                channel={row.original}
                                bucketDurationNs={bucketDurationNs}
                                fileSize={fileSize}
                              />
                            )}
                          </Collapse>
                        </Table.Td>
                      </Table.Tr>
                    )}
                  </Fragment>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      )}
    </Paper>
  );
}

function SortableHeader({ header }: { header: Header<ChannelRow, unknown> }) {
  const canSort = header.column.getCanSort();
  const sorted = header.column.getIsSorted();
  const meta = header.column.columnDef.meta;

  return (
    <Table.Th
      onClick={canSort ? header.column.getToggleSortingHandler() : undefined}
      style={{
        cursor: canSort ? "pointer" : undefined,
        userSelect: canSort ? "none" : undefined,
        textAlign: meta?.align,
      }}
      title={meta?.headerTitle}
    >
      {flexRender(header.column.columnDef.header, header.getContext())}
      {sorted === "asc" && " \u25B2"}
      {sorted === "desc" && " \u25BC"}
    </Table.Th>
  );
}

function ColumnsMenu({ table }: { table: ReturnType<typeof useReactTable<ChannelRow>> }) {
  return (
    <Menu shadow="md" closeOnItemClick={false}>
      <Menu.Target>
        <ActionIcon variant="subtle" size="sm" title="Toggle columns">
          <IconColumns size={16} />
        </ActionIcon>
      </Menu.Target>
      <Menu.Dropdown>
        {table.getAllLeafColumns().map((column) => (
          <Menu.Item key={column.id}>
            <Checkbox
              size="xs"
              label={typeof column.columnDef.header === "string" ? column.columnDef.header : column.id}
              checked={column.getIsVisible()}
              onChange={column.getToggleVisibilityHandler()}
            />
          </Menu.Item>
        ))}
      </Menu.Dropdown>
    </Menu>
  );
}
