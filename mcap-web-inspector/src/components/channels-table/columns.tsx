import { Text, Group, Tooltip, Checkbox } from "@mantine/core";
import { Sparkline } from "@mantine/charts";
import { createColumnHelper } from "@tanstack/react-table";
import type { ColumnDef, RowData } from "@tanstack/react-table";
import { formatNumber, formatBytes, formatHz } from "../../format.ts";
import { TopicDisplay, SchemaDisplay, HzDisplay, JitterDisplay, BpsDisplay } from "./cells.tsx";
import { stringToColor, formatPercent } from "./utils.ts";
import { getSchemaIcon } from "./schema-icons.tsx";
import type { ChannelRow } from "./types.ts";

declare module "@tanstack/react-table" {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ColumnMeta<TData extends RowData, TValue> {
    align?: "left" | "right" | "center";
    headerTitle?: string;
    width?: number;
    enableHiding?: boolean;
  }
}

const col = createColumnHelper<ChannelRow>();

interface ColumnContext {
  fileSize: number;
  hasEstimatedSizes: boolean;
  detailExpandedIds: Set<number>;
  selectable?: boolean;
  compact?: boolean;
}

const COMPACT_HIDDEN = new Set(["hz", "jitter", "size", "bps", "bPerMsg", "distribution"]);

export function getColumns(ctx: ColumnContext): ColumnDef<ChannelRow, unknown>[] {
  const estimatePrefix = (row: ChannelRow) => row.estimated_sizes ? "~" : "";

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cols: ColumnDef<ChannelRow, any>[] = [];

  if (ctx.selectable) {
    cols.push(
      col.display({
        id: "select",
        enableSorting: false,
        meta: { enableHiding: false, width: 32 },
        header: ({ table }) => (
          <Checkbox
            size="xs"
            checked={table.getIsAllRowsSelected()}
            indeterminate={table.getIsSomeRowsSelected()}
            onChange={table.getToggleAllRowsSelectedHandler()}
            onClick={(e) => e.stopPropagation()}
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            size="xs"
            checked={row.getIsSelected()}
            indeterminate={row.getIsSomeSelected()}
            onChange={row.getToggleSelectedHandler()}
            onClick={(e) => e.stopPropagation()}
          />
        ),
      }),
    );
  }

  cols.push(
    col.display({
      id: "expand",
      header: "",
      enableSorting: false,
      meta: { enableHiding: false, width: 32 },
      cell: ({ row }) => {
        const isGroup = row.original._kind === "group";
        const isExpanded = isGroup
          ? row.getIsExpanded()
          : ctx.detailExpandedIds.has(row.original.id);

        return (
          <Group gap={4} wrap="nowrap" align="center">
            <Text
              size="xs"
              c="dimmed"
              style={{
                transition: "transform 150ms",
                transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
                flexShrink: 0,
              }}
            >
              ▶
            </Text>
            {isGroup && (
              <Text size="sm" fw={600} style={{ color: stringToColor(row.original._segment) }}>
                /{row.original._segment}
              </Text>
            )}
          </Group>
        );
      },
    }),

    col.display({
      id: "schemaIcon",
      header: "",
      enableSorting: false,
      meta: { enableHiding: false, width: 32 },
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        const { Icon, color } = getSchemaIcon(row.original.schema_name);
        return (
          <Tooltip label={row.original.schema_name ?? "Unknown"} openDelay={300}>
            <Icon size={16} color={color} style={{ display: "block" }} />
          </Tooltip>
        );
      },
    }),

    col.accessor("id", {
      header: "ID",
      enableSorting: true,
      sortingFn: "basic",
      meta: { width: 52 },
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        return (
          <Text size="sm" c="dimmed">
            {row.original.id}
          </Text>
        );
      },
    }),

    col.accessor("topic", {
      header: "Topic",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        return <TopicDisplay topic={row.original.topic} />;
      },
    }),

    col.accessor("schema_name", {
      header: "Schema",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const a = rowA.original.schema_name ?? "";
        const b = rowB.original.schema_name ?? "";
        return a.localeCompare(b);
      },
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        return <SchemaDisplay name={row.original.schema_name} />;
      },
    }),

    col.accessor("message_count", {
      header: "Msgs",
      enableSorting: true,
      sortingFn: "basic",
      meta: { align: "right" },
      cell: ({ row }) => {
        if (row.original._kind === "group") {
          return <Text size="sm" c="dimmed">{formatNumber(row.original.message_count)}</Text>;
        }
        return formatNumber(row.original.message_count);
      },
    }),

    col.display({
      id: "hz",
      header: "Hz",
      enableSorting: true,
      sortingFn: (rowA, rowB) =>
        rowA.original.hz_stats.average - rowB.original.hz_stats.average,
      meta: { align: "right" },
      cell: ({ row }) => {
        if (row.original._kind === "group") {
          const min = row.original.hz_stats.minimum ?? row.original.hz_stats.average;
          const max = row.original.hz_stats.maximum ?? row.original.hz_stats.average;
          return (
            <Text size="sm" c="dimmed">
              {min === max
                ? formatHz(min)
                : `${formatHz(min)}-${formatHz(max)}`}
            </Text>
          );
        }
        return (
          <HzDisplay
            stats={row.original.hz_stats}
            hzChannel={row.original.hz_channel}
          />
        );
      },
    }),

    col.display({
      id: "jitter",
      header: "Jitter",
      enableSorting: true,
      sortingFn: (rowA, rowB) =>
        (rowA.original.jitter_cv ?? 0) - (rowB.original.jitter_cv ?? 0),
      meta: { align: "right" },
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        return (
          <JitterDisplay
            jitterNs={row.original.jitter_ns}
            jitterCv={row.original.jitter_cv}
          />
        );
      },
    }),

    col.accessor("size_bytes", {
      id: "size",
      header: ctx.hasEstimatedSizes ? "~Size" : "Size",
      enableSorting: true,
      sortingFn: (rowA, rowB) =>
        (rowA.original.size_bytes ?? 0) - (rowB.original.size_bytes ?? 0),
      meta: {
        align: "right",
        headerTitle: ctx.hasEstimatedSizes ? "Estimated from message index offsets" : undefined,
      },
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        const ch = row.original;
        if (ch.size_bytes === null) return "-";
        return (
          <Text size="sm">
            {estimatePrefix(ch)}{formatBytes(ch.size_bytes)}{" "}
            <Text span size="xs" c="dimmed">
              ({estimatePrefix(ch)}{formatPercent(ch.size_bytes, ctx.fileSize)})
            </Text>
          </Text>
        );
      },
    }),

    col.display({
      id: "bps",
      header: ctx.hasEstimatedSizes ? "~B/s" : "B/s",
      enableSorting: true,
      sortingFn: (rowA, rowB) =>
        (rowA.original.bytes_per_second_stats?.average ?? 0) -
        (rowB.original.bytes_per_second_stats?.average ?? 0),
      meta: {
        align: "right",
        headerTitle: ctx.hasEstimatedSizes ? "Estimated from message index offsets" : undefined,
      },
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        return (
          <BpsDisplay
            stats={row.original.bytes_per_second_stats}
            estimated={row.original.estimated_sizes}
          />
        );
      },
    }),

    col.accessor("bytes_per_message", {
      id: "bPerMsg",
      header: ctx.hasEstimatedSizes ? "~B/msg" : "B/msg",
      enableSorting: true,
      sortingFn: (rowA, rowB) =>
        (rowA.original.bytes_per_message ?? 0) - (rowB.original.bytes_per_message ?? 0),
      meta: {
        align: "right",
        headerTitle: ctx.hasEstimatedSizes ? "Estimated from message index offsets" : undefined,
      },
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        const ch = row.original;
        return ch.bytes_per_message !== null
          ? `${estimatePrefix(ch)}${formatBytes(ch.bytes_per_message)}`
          : "-";
      },
    }),

    col.display({
      id: "distribution",
      header: "Distribution",
      enableSorting: false,
      cell: ({ row }) => {
        if (row.original._kind === "group") return null;
        if (row.original.message_distribution.length === 0) return null;
        return (
          <Sparkline
            w={120}
            h={20}
            data={row.original.message_distribution}
            curveType="monotone"
            color="blue"
            fillOpacity={0.2}
            strokeWidth={1.5}
          />
        );
      },
    }),
  );

  if (ctx.compact) {
    return cols.filter((c) => !("id" in c && typeof c.id === "string" && COMPACT_HIDDEN.has(c.id)));
  }
  return cols;
}
