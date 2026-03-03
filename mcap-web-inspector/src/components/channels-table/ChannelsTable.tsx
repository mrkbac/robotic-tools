import { useState, useMemo, useEffect, Fragment, useCallback } from "react";
import {
  Table,
  Title,
  Paper,
  Text,
  Group,
  Collapse,
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
  type RowSelectionState,
  type Header,
  type Row,
} from "@tanstack/react-table";
import {
  IconColumns,
  IconSearch,
  IconArrowUp,
  IconArrowDown,
} from "@tabler/icons-react";
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
const RAIL_GAP = 14; // px between rail centers
const RAIL_WIDTH = 3; // px width of each rail line

/** Build CSS background + paddingLeft for vertical tree rail lines on a cell. */
function getRailStyle(
  topic: string,
  depth: number,
): React.CSSProperties | undefined {
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
  selectable?: boolean;
  selectedChannelIds?: Set<number>;
  onSelectedChannelIdsChange?: (ids: Set<number>) => void;
  compact?: boolean;
}

/** Extract channel IDs from a RowSelectionState. */
function rowSelectionToChannelIds(
  state: RowSelectionState,
  rows: Row<ChannelRow>[],
): Set<number> {
  const ids = new Set<number>();
  for (const row of rows) {
    if (row.original._kind === "channel" && state[row.id]) {
      ids.add(row.original.id);
    }
    if (row.subRows.length > 0) {
      for (const id of rowSelectionToChannelIds(state, row.subRows)) {
        ids.add(id);
      }
    }
  }
  return ids;
}

export function ChannelsTable({
  channels,
  bucketDurationNs,
  fileSize,
  selectable,
  selectedChannelIds,
  onSelectedChannelIdsChange,
  compact,
}: ChannelsTableProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("flat");
  const [detailExpandedIds, setDetailExpandedIds] = useState<Set<number>>(
    new Set(),
  );
  const [expanded, setExpanded] = useState<ExpandedState>(true);
  const [globalFilter, setGlobalFilter] = useState("");

  const hasSizeData = channels.some((ch) => ch.size_bytes !== null);
  const hasEstimatedSizes = channels.some(
    (ch) => ch.estimated_sizes && ch.size_bytes !== null,
  );
  const hasDistribution = channels.some(
    (ch) => ch.message_distribution.length > 0,
  );
  const hasJitter = channels.some((ch) => ch.jitter_cv !== null);

  // Data-driven default visibility
  const dataDefaults = useMemo<VisibilityState>(
    () => ({
      id: false,
      schema_name: false,
      jitter: hasJitter,
      size: hasSizeData,
      bps: hasSizeData,
      bPerMsg: hasSizeData,
      distribution: hasDistribution,
    }),
    [hasJitter, hasSizeData, hasDistribution],
  );

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

  const isTreeView = viewMode === "tree";

  const columns = useMemo(
    () =>
      getColumns({
        fileSize,
        hasEstimatedSizes,
        detailExpandedIds,
        selectable,
        compact,
        isTreeView,
      }),
    [
      fileSize,
      hasEstimatedSizes,
      detailExpandedIds,
      selectable,
      compact,
      isTreeView,
    ],
  );

  const rawData = useMemo(() => {
    if (isTreeView) return buildTreeData(channels);
    return toFlatRows(channels);
  }, [channels, isTreeView]);

  // Apply global filter
  const filterLower = globalFilter.trim().toLowerCase();
  const data = useMemo(() => {
    if (!filterLower) return rawData;
    if (isTreeView) return filterTree(rawData, filterLower);
    return rawData.filter((row) => matchesFilter(row, filterLower));
  }, [rawData, filterLower, isTreeView]);

  const toggleDetail = useCallback((id: number) => {
    setDetailExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Force all expanded when filtering in tree mode
  const effectiveExpanded = filterLower && isTreeView ? true : expanded;

  // Selection state: convert between Set<number> and TanStack RowSelectionState
  const rowSelection = useMemo((): RowSelectionState => {
    if (!selectable || !selectedChannelIds) return {};
    // We need the table's row model, but it's not available yet during config.
    // Instead, build selection from data rows directly — the getRowId is deterministic.
    const state: RowSelectionState = {};
    function walkData(rows: ChannelRow[]) {
      for (const row of rows) {
        const rowId =
          row._kind === "group" ? `g:${row._fullPath}` : `c:${row.id}`;
        if (row._kind === "channel" && selectedChannelIds!.has(row.id)) {
          state[rowId] = true;
        }
        if (row.subRows) {
          walkData(row.subRows);
          // Mark group selected if all children are selected
          if (
            row.subRows.length > 0 &&
            row.subRows.every((sub) => {
              const subId =
                sub._kind === "group" ? `g:${sub._fullPath}` : `c:${sub.id}`;
              return state[subId];
            })
          ) {
            state[rowId] = true;
          }
        }
      }
    }
    walkData(data);
    return state;
  }, [selectable, selectedChannelIds, data]);

  const table = useReactTable<ChannelRow>({
    data,
    columns,
    state: {
      sorting,
      columnVisibility,
      columnOrder,
      expanded: effectiveExpanded,
      ...(selectable ? { rowSelection } : {}),
    },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    onColumnOrderChange: setColumnOrder,
    onExpandedChange: setExpanded,
    ...(selectable
      ? {
          enableRowSelection: true,
          enableSubRowSelection: true,
          onRowSelectionChange: (updater) => {
            const next =
              typeof updater === "function" ? updater(rowSelection) : updater;
            if (onSelectedChannelIdsChange) {
              const ids = rowSelectionToChannelIds(
                next,
                table.getRowModel().rows,
              );
              onSelectedChannelIdsChange(ids);
            }
          },
        }
      : {}),
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    getSubRows: (row) => row.subRows,
    getRowId: (row) =>
      row._kind === "group" ? `g:${row._fullPath}` : `c:${row.id}`,
    enableSortingRemoval: false,
  });

  const visibleCellCount = table.getVisibleLeafColumns().length;

  const handleRowClick = useCallback(
    (row: Row<ChannelRow>) => {
      const isGroup = row.original._kind === "group";
      if (selectable) {
        if (isGroup) {
          row.toggleExpanded();
        } else {
          row.toggleSelected();
        }
      } else {
        if (isGroup) {
          row.toggleExpanded();
        } else {
          toggleDetail(row.original.id);
        }
      }
    },
    [selectable, toggleDetail],
  );

  const tableContent =
    channels.length === 0 ? (
      <Text c="dimmed">No channels found</Text>
    ) : (
      <Table.ScrollContainer type="native" minWidth={500}>
        <Table
          striped={!isTreeView}
          highlightOnHover
          stickyHeader
          stickyHeaderOffset={0}
        >
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
              const detailExpanded =
                !compact && !isGroup && detailExpandedIds.has(row.original.id);

              return (
                <Fragment key={row.id}>
                  <Table.Tr
                    onClick={() => handleRowClick(row)}
                    style={{ cursor: "pointer" }}
                  >
                    {row.getVisibleCells().map((cell) => {
                      const meta = cell.column.columnDef.meta;
                      const railStyle =
                        isTreeView && cell.column.id === "expand"
                          ? getRailStyle(row.original.topic, row.depth)
                          : undefined;
                      return (
                        <Table.Td
                          key={cell.id}
                          style={{
                            textAlign: meta?.align,
                            width: meta?.width,
                            maxWidth: meta?.width,
                            ...(meta?.truncate
                              ? {
                                  whiteSpace: "nowrap",
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                  maxWidth: meta?.width ?? 300,
                                }
                              : {}),
                            ...railStyle,
                            ...(isGroup
                              ? { fontWeight: 500, opacity: 0.8 }
                              : {}),
                          }}
                        >
                          {flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext(),
                          )}
                        </Table.Td>
                      );
                    })}
                  </Table.Tr>

                  {!compact && !isGroup && (
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
      </Table.ScrollContainer>
    );

  const handleViewModeChange = useCallback((mode: ViewMode) => {
    setViewMode(mode);
    setExpanded(mode === "tree" ? true : {});
  }, []);

  const toolbar = (
    <Toolbar
      globalFilter={globalFilter}
      onFilterChange={setGlobalFilter}
      table={table}
      viewMode={viewMode}
      onViewModeChange={handleViewModeChange}
      showToggle={channels.length > 0}
    />
  );

  if (selectable) {
    return (
      <>
        <Group justify="flex-end" mb="xs">
          {toolbar}
        </Group>
        {tableContent}
      </>
    );
  }

  return (
    <Paper p="md" withBorder>
      <Group justify="space-between" mb="md">
        <Title order={4}>Channels</Title>
        {toolbar}
      </Group>
      {tableContent}
    </Paper>
  );
}

function Toolbar({
  globalFilter,
  onFilterChange,
  table,
  viewMode,
  onViewModeChange,
  showToggle,
}: {
  globalFilter: string;
  onFilterChange: (value: string) => void;
  table: ReturnType<typeof useReactTable<ChannelRow>>;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  showToggle: boolean;
}) {
  return (
    <Group gap="xs">
      <TextInput
        size="xs"
        placeholder="Filter channels…"
        leftSection={<IconSearch size={14} />}
        rightSection={
          globalFilter ? (
            <CloseButton size="xs" onClick={() => onFilterChange("")} />
          ) : null
        }
        value={globalFilter}
        onChange={(e) => onFilterChange(e.currentTarget.value)}
        style={{ width: 200 }}
      />
      <ColumnsMenu table={table} />
      {showToggle && (
        <SegmentedControl
          size="xs"
          value={viewMode}
          onChange={(v) => onViewModeChange(v as ViewMode)}
          data={[
            { label: "Flat", value: "flat" },
            { label: "Tree", value: "tree" },
          ]}
        />
      )}
    </Group>
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
      {sorted === "asc" && (
        <IconArrowUp
          size={12}
          style={{ verticalAlign: "middle", marginLeft: 2 }}
        />
      )}
      {sorted === "desc" && (
        <IconArrowDown
          size={12}
          style={{ verticalAlign: "middle", marginLeft: 2 }}
        />
      )}
    </Table.Th>
  );
}

function ColumnsMenu({
  table,
}: {
  table: ReturnType<typeof useReactTable<ChannelRow>>;
}) {
  const hideable = table
    .getAllLeafColumns()
    .filter((col) => col.columnDef.meta?.enableHiding !== false);

  const moveColumn = (id: string, direction: -1 | 1) => {
    const currentOrder =
      table.getState().columnOrder.length > 0
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
                label={
                  typeof column.columnDef.header === "string"
                    ? column.columnDef.header
                    : column.id
                }
                checked={column.getIsVisible()}
                onChange={column.getToggleVisibilityHandler()}
                style={{ flex: 1 }}
              />
              <ActionIcon
                variant="subtle"
                size="xs"
                onClick={(e) => {
                  e.stopPropagation();
                  moveColumn(column.id, -1);
                }}
                title="Move up"
              >
                <IconArrowUp size={12} />
              </ActionIcon>
              <ActionIcon
                variant="subtle"
                size="xs"
                onClick={(e) => {
                  e.stopPropagation();
                  moveColumn(column.id, 1);
                }}
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
