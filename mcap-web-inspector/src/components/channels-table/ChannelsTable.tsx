import { useState, useMemo, useEffect, Fragment, useCallback } from "react";
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
  TextInput,
  CloseButton,
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
  type ColumnOrderState,
  type Header,
} from "@tanstack/react-table";
import { IconColumns, IconSearch, IconArrowUp, IconArrowDown } from "@tabler/icons-react";
import type { ChannelInfo } from "../../mcap/types.ts";
import type { ChannelRow } from "./types.ts";
import { getColumns } from "./columns.tsx";
import { ChannelDetail } from "./ChannelDetail.tsx";
import { buildTreeData, toFlatRows } from "./tree-data.ts";
import { stringToColor, filterTree, matchesFilter } from "./utils.ts";
import { loadTableState, saveTableState } from "./persistence.ts";

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
  const [detailExpandedIds, setDetailExpandedIds] = useState<Set<number>>(new Set());
  const [expanded, setExpanded] = useState<ExpandedState>(true);
  const [globalFilter, setGlobalFilter] = useState("");

  const hasSizeData = channels.some((ch) => ch.size_bytes !== null);
  const hasEstimatedSizes = channels.some((ch) => ch.estimated_sizes && ch.size_bytes !== null);
  const hasDistribution = channels.some((ch) => ch.message_distribution.length > 0);
  const hasJitter = channels.some((ch) => ch.jitter_cv !== null);

  // Data-driven default visibility
  const dataDefaults = useMemo<VisibilityState>(() => ({
    id: false,
    schema_name: false,
    jitter: hasJitter,
    size: hasSizeData,
    bps: hasSizeData,
    bPerMsg: hasSizeData,
    distribution: hasDistribution,
  }), [hasJitter, hasSizeData, hasDistribution]);

  // Load persisted state on mount, merge with data-driven defaults
  const persisted = useMemo(() => loadTableState(), []);

  const [sorting, setSorting] = useState<SortingState>(
    persisted?.sorting ?? [{ id: "topic", desc: false }],
  );
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(
    () => ({ ...dataDefaults, ...persisted?.columnVisibility }),
  );
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>(
    persisted?.columnOrder ?? [],
  );

  // Persist changes
  useEffect(() => {
    saveTableState({ columnVisibility, sorting, columnOrder });
  }, [columnVisibility, sorting, columnOrder]);

  const columns = useMemo(
    () => getColumns({ fileSize, hasEstimatedSizes, detailExpandedIds }),
    [fileSize, hasEstimatedSizes, detailExpandedIds],
  );

  const rawData = useMemo(() => {
    if (viewMode === "tree") return buildTreeData(channels);
    return toFlatRows(channels);
  }, [channels, viewMode]);

  // Apply global filter
  const filterLower = globalFilter.trim().toLowerCase();
  const data = useMemo(() => {
    if (!filterLower) return rawData;
    if (viewMode === "tree") return filterTree(rawData, filterLower);
    return rawData.filter((row) => matchesFilter(row, filterLower));
  }, [rawData, filterLower, viewMode]);

  const toggleDetail = useCallback((id: number) => {
    setDetailExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Force all expanded when filtering in tree mode
  const effectiveExpanded = filterLower && viewMode === "tree" ? true : expanded;

  const table = useReactTable<ChannelRow>({
    data,
    columns,
    state: {
      sorting,
      columnVisibility,
      columnOrder,
      expanded: effectiveExpanded,
    },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    onColumnOrderChange: setColumnOrder,
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
          <TextInput
            size="xs"
            placeholder="Filter channels…"
            leftSection={<IconSearch size={14} />}
            rightSection={
              globalFilter ? (
                <CloseButton size="xs" onClick={() => setGlobalFilter("")} />
              ) : null
            }
            value={globalFilter}
            onChange={(e) => setGlobalFilter(e.currentTarget.value)}
            style={{ width: 200 }}
          />
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
                        const meta = cell.column.columnDef.meta;
                        const railStyle = isTree && cell.column.id === "expand"
                          ? getRailStyle(row.original.topic, row.depth)
                          : undefined;
                        return (
                          <Table.Td
                            key={cell.id}
                            style={{
                              textAlign: meta?.align,
                              width: meta?.width,
                              maxWidth: meta?.width,
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
        width: meta?.width,
        maxWidth: meta?.width,
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
  const hideable = table.getAllLeafColumns().filter(
    (col) => col.columnDef.meta?.enableHiding !== false,
  );

  const moveColumn = (id: string, direction: -1 | 1) => {
    const currentOrder = table.getState().columnOrder.length > 0
      ? table.getState().columnOrder
      : table.getAllLeafColumns().map((c) => c.id);
    const idx = currentOrder.indexOf(id);
    if (idx < 0) return;
    const swapIdx = idx + direction;
    if (swapIdx < 0 || swapIdx >= currentOrder.length) return;
    const next = [...currentOrder];
    [next[idx], next[swapIdx]] = [next[swapIdx]!, next[idx]!];
    table.setColumnOrder(next);
  };

  return (
    <Menu shadow="md" closeOnItemClick={false}>
      <Menu.Target>
        <ActionIcon variant="subtle" size="sm" title="Toggle columns">
          <IconColumns size={16} />
        </ActionIcon>
      </Menu.Target>
      <Menu.Dropdown>
        {hideable.map((column) => (
          <Menu.Item key={column.id}>
            <Group gap={4} wrap="nowrap">
              <Checkbox
                size="xs"
                label={typeof column.columnDef.header === "string" ? column.columnDef.header : column.id}
                checked={column.getIsVisible()}
                onChange={column.getToggleVisibilityHandler()}
                style={{ flex: 1 }}
              />
              <ActionIcon
                variant="subtle"
                size="xs"
                onClick={(e) => { e.stopPropagation(); moveColumn(column.id, -1); }}
                title="Move up"
              >
                <IconArrowUp size={12} />
              </ActionIcon>
              <ActionIcon
                variant="subtle"
                size="xs"
                onClick={(e) => { e.stopPropagation(); moveColumn(column.id, 1); }}
                title="Move down"
              >
                <IconArrowDown size={12} />
              </ActionIcon>
            </Group>
          </Menu.Item>
        ))}
      </Menu.Dropdown>
    </Menu>
  );
}
