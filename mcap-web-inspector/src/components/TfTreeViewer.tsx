import { useState, useMemo } from "react";
import {
  Paper,
  Title,
  Table,
  Text,
  Badge,
  Alert,
  Group,
  Slider,
} from "@mantine/core";
import { IconAlertTriangle } from "@tabler/icons-react";
import type { TfTreeData, TfTransform } from "../mcap/tf.ts";
import { quaternionToEuler, getTransformsAtTime } from "../mcap/tf.ts";
import { formatBucketTime } from "../format.ts";

interface TfTreeViewerProps {
  data: TfTreeData;
}

interface TreeNode {
  frame: string;
  transform: TfTransform | null; // null for root frames
  count: number;
  children: TreeNode[];
  depth: number;
}

function fmt(n: number): string {
  return n.toFixed(3);
}

function fmtDeg(n: number): string {
  return n.toFixed(1);
}

const SLIDER_MAX = 1000;

function buildTree(
  transforms: Map<string, TfTransform>,
  updateCounts: Map<string, number>,
): TreeNode[] {
  const childFrames = new Set<string>();
  const parentFrames = new Set<string>();
  const childrenOf = new Map<string, TreeNode[]>();

  for (const [key, tf] of transforms) {
    childFrames.add(tf.childFrame);
    parentFrames.add(tf.parentFrame);

    const node: TreeNode = {
      frame: tf.childFrame,
      transform: tf,
      count: updateCounts.get(key) ?? 0,
      children: [],
      depth: 0,
    };

    if (!childrenOf.has(tf.parentFrame)) {
      childrenOf.set(tf.parentFrame, []);
    }
    childrenOf.get(tf.parentFrame)!.push(node);
  }

  // Root frames: frames that are parents but never children
  const rootFrames = [...parentFrames].filter((f) => !childFrames.has(f));

  // Build trees from each root
  function attachChildren(node: TreeNode, depth: number) {
    node.depth = depth;
    const kids = childrenOf.get(node.frame) ?? [];
    kids.sort((a, b) => a.frame.localeCompare(b.frame));
    node.children = kids;
    for (const child of kids) {
      attachChildren(child, depth + 1);
    }
  }

  const roots: TreeNode[] = rootFrames.sort().map((frame) => {
    const root: TreeNode = {
      frame,
      transform: null,
      count: 0,
      children: [],
      depth: 0,
    };
    attachChildren(root, 0);
    return root;
  });

  return roots;
}

function flattenTree(nodes: TreeNode[]): TreeNode[] {
  const result: TreeNode[] = [];
  function walk(node: TreeNode) {
    result.push(node);
    for (const child of node.children) {
      walk(child);
    }
  }
  for (const root of nodes) {
    walk(root);
  }
  return result;
}

export function TfTreeViewer({ data }: TfTreeViewerProps) {
  const [sliderValue, setSliderValue] = useState(SLIDER_MAX);

  const hasTimeRange = data.startTimeNs < data.endTimeNs;

  const sliderToTimeNs = (value: number): bigint => {
    if (!hasTimeRange) return data.endTimeNs;
    const range = data.endTimeNs - data.startTimeNs;
    return data.startTimeNs + (range * BigInt(value)) / BigInt(SLIDER_MAX);
  };

  const currentTimeNs = sliderToTimeNs(sliderValue);
  const isAtEnd = sliderValue === SLIDER_MAX;

  const activeTransforms = useMemo(() => {
    if (isAtEnd) return data.transforms;
    return getTransformsAtTime(data, currentTimeNs);
  }, [data, isAtEnd, currentTimeNs]);

  const tree = useMemo(
    () => buildTree(activeTransforms, data.updateCounts),
    [activeTransforms, data.updateCounts],
  );
  const flatNodes = useMemo(() => flattenTree(tree), [tree]);

  const staticCount = [...data.transforms.values()].filter(
    (t) => t.isStatic,
  ).length;
  const dynamicCount = data.transforms.size - staticCount;
  const frameCount = new Set([
    ...[...data.transforms.values()].map((t) => t.parentFrame),
    ...[...data.transforms.values()].map((t) => t.childFrame),
  ]).size;

  const timeLabel = hasTimeRange
    ? formatBucketTime(Number(currentTimeNs - data.startTimeNs))
    : "";
  const totalLabel = hasTimeRange
    ? formatBucketTime(Number(data.endTimeNs - data.startTimeNs))
    : "";

  return (
    <Paper p="md" withBorder>
      <Group justify="space-between" mb="md">
        <Title order={4}>TF Tree</Title>
        <Text size="sm" c="dimmed">
          {frameCount} frames ({staticCount} static, {dynamicCount} dynamic)
        </Text>
      </Group>

      {hasTimeRange && (
        <Group gap="sm" mb="md" align="center">
          <Slider
            value={sliderValue}
            onChange={setSliderValue}
            min={0}
            max={SLIDER_MAX}
            step={1}
            label={null}
            style={{ flex: 1 }}
          />
          <Text size="sm" c="dimmed" style={{ whiteSpace: "nowrap" }}>
            {timeLabel} / {totalLabel}
          </Text>
        </Group>
      )}

      {data.multipleParents.size > 0 && (
        <Alert
          color="yellow"
          icon={<IconAlertTriangle size={16} />}
          title="Multiple parents detected"
          mb="md"
        >
          {[...data.multipleParents.entries()].map(([child, parents]) => (
            <Text size="sm" key={child}>
              <strong>{child}</strong> has parents: {parents.join(", ")}
            </Text>
          ))}
        </Alert>
      )}

      <Table.ScrollContainer type="native" minWidth={700}>
        <Table striped highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Frame</Table.Th>
              <Table.Th style={{ textAlign: "center" }}>Type</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Count</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>tx</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>ty</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>tz</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>roll&deg;</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>pitch&deg;</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>yaw&deg;</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {flatNodes.map((node, i) => {
              const isRoot = node.transform === null;
              const tf = node.transform;
              const [roll, pitch, yaw] = tf
                ? quaternionToEuler(...tf.rotation)
                : [0, 0, 0];

              return (
                <Table.Tr key={`${node.frame}-${i}`}>
                  <Table.Td
                    style={{
                      paddingLeft: 12 + node.depth * 20,
                      fontWeight: isRoot ? 600 : 400,
                      fontFamily: "monospace",
                      fontSize: "0.85em",
                    }}
                  >
                    {node.depth > 0 && (
                      <Text component="span" c="dimmed" mr={4}>
                        {"└ "}
                      </Text>
                    )}
                    {node.frame}
                  </Table.Td>
                  <Table.Td style={{ textAlign: "center" }}>
                    {!isRoot && (
                      <Badge
                        size="xs"
                        color={tf!.isStatic ? "green" : "red"}
                        variant="light"
                      >
                        {tf!.isStatic ? "static" : "dynamic"}
                      </Badge>
                    )}
                  </Table.Td>
                  <Table.Td style={{ textAlign: "right" }}>
                    {node.count > 0 ? node.count.toLocaleString() : ""}
                  </Table.Td>
                  <Table.Td
                    style={{ textAlign: "right", fontFamily: "monospace" }}
                  >
                    {tf ? fmt(tf.translation[0]) : ""}
                  </Table.Td>
                  <Table.Td
                    style={{ textAlign: "right", fontFamily: "monospace" }}
                  >
                    {tf ? fmt(tf.translation[1]) : ""}
                  </Table.Td>
                  <Table.Td
                    style={{ textAlign: "right", fontFamily: "monospace" }}
                  >
                    {tf ? fmt(tf.translation[2]) : ""}
                  </Table.Td>
                  <Table.Td
                    style={{ textAlign: "right", fontFamily: "monospace" }}
                  >
                    {tf ? fmtDeg(roll) : ""}
                  </Table.Td>
                  <Table.Td
                    style={{ textAlign: "right", fontFamily: "monospace" }}
                  >
                    {tf ? fmtDeg(pitch) : ""}
                  </Table.Td>
                  <Table.Td
                    style={{ textAlign: "right", fontFamily: "monospace" }}
                  >
                    {tf ? fmtDeg(yaw) : ""}
                  </Table.Td>
                </Table.Tr>
              );
            })}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Paper>
  );
}
