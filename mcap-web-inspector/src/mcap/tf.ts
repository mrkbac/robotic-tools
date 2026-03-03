import type { Channel, Schema } from "@mcap/core";
import { parse } from "@foxglove/rosmsg";
import { MessageReader } from "@foxglove/rosmsg2-serialization";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TfTransform {
  parentFrame: string;
  childFrame: string;
  translation: [number, number, number];
  rotation: [number, number, number, number]; // x, y, z, w
  isStatic: boolean;
  timestampNs: bigint;
}

export interface TfTreeData {
  /** Unique transforms keyed by "parent→child". */
  transforms: Map<string, TfTransform>;
  /** How many times each "parent→child" pair was seen. */
  updateCounts: Map<string, number>;
  /** Children that appear under multiple parents (violations). child → parent list */
  multipleParents: Map<string, string[]>;
  /** Per-key arrays sorted by timestampNs for time seeking. */
  transformsByKey: Map<string, TfTransform[]>;
  /** Earliest transform timestamp across all keys. */
  startTimeNs: bigint;
  /** Latest transform timestamp across all keys. */
  endTimeNs: bigint;
}

export interface TfChannelInfo {
  channelId: number;
  topic: string;
  encoding: string;
  schemaData: string;
  isStatic: boolean;
}

// ---------------------------------------------------------------------------
// Schema detection
// ---------------------------------------------------------------------------

const TF_SCHEMAS = new Set(["tf2_msgs/msg/TFMessage", "tf2_msgs/TFMessage"]);

export function isTfSchema(name: string): boolean {
  return TF_SCHEMAS.has(name);
}

const STATIC_TF_TOPICS = new Set(["/tf_static"]);

/** Find all channels whose schema is a TFMessage type. */
export function findTfChannels(
  channelsById: ReadonlyMap<number, Channel & { type: "Channel" }>,
  schemasById: ReadonlyMap<number, Schema & { type: "Schema" }>,
): TfChannelInfo[] {
  const result: TfChannelInfo[] = [];
  for (const [channelId, channel] of channelsById) {
    const schema = schemasById.get(channel.schemaId);
    if (!schema) continue;
    if (isTfSchema(schema.name)) {
      result.push({
        channelId,
        topic: channel.topic,
        encoding: channel.messageEncoding || schema.encoding,
        schemaData: new TextDecoder().decode(schema.data),
        isStatic: STATIC_TF_TOPICS.has(channel.topic),
      });
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Reader creation & message parsing
// ---------------------------------------------------------------------------

/** Create a Foxglove MessageReader from a ROS 2 TFMessage schema. */
export function createTfReader(schemaData: string): MessageReader | null {
  try {
    const defs = parse(schemaData, { ros2: true });
    return new MessageReader(defs);
  } catch {
    return null;
  }
}

interface RawTransformStamped {
  header?: {
    stamp?: { sec: number; nanosec?: number; nsec?: number };
    frame_id?: string;
  };
  child_frame_id?: string;
  transform?: {
    translation?: { x: number; y: number; z: number };
    rotation?: { x: number; y: number; z: number; w: number };
  };
}

/** Parse a CDR-encoded TFMessage into individual transforms. */
export function parseTfMessage(
  reader: MessageReader,
  data: Uint8Array,
  isStatic: boolean,
  logTime: bigint,
): TfTransform[] {
  try {
    const msg = reader.readMessage<{ transforms: RawTransformStamped[] }>(data);
    if (!msg.transforms || !Array.isArray(msg.transforms)) return [];

    return msg.transforms.map((t) => {
      const stamp = t.header?.stamp;
      const sec = stamp?.sec ?? 0;
      const nsec = stamp?.nanosec ?? stamp?.nsec ?? 0;
      const timestampNs = BigInt(sec) * 1_000_000_000n + BigInt(nsec);
      const tr = t.transform?.translation;
      const rot = t.transform?.rotation;

      return {
        parentFrame: t.header?.frame_id ?? "",
        childFrame: t.child_frame_id ?? "",
        translation: [tr?.x ?? 0, tr?.y ?? 0, tr?.z ?? 0],
        rotation: [rot?.x ?? 0, rot?.y ?? 0, rot?.z ?? 0, rot?.w ?? 1],
        isStatic,
        timestampNs: timestampNs > 0n ? timestampNs : logTime,
      };
    });
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Tree building
// ---------------------------------------------------------------------------

/** Find the latest transform with timestampNs <= timeNs via binary search. */
export function binarySearchLatest(
  arr: TfTransform[],
  timeNs: bigint,
): TfTransform | null {
  let lo = 0;
  let hi = arr.length - 1;
  let result: TfTransform | null = null;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid]!.timestampNs <= timeNs) {
      result = arr[mid]!;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return result;
}

/** Reconstruct the transform map at a given point in time. */
export function getTransformsAtTime(
  data: TfTreeData,
  timeNs: bigint,
): Map<string, TfTransform> {
  const result = new Map<string, TfTransform>();
  for (const [key, arr] of data.transformsByKey) {
    const tf = binarySearchLatest(arr, timeNs);
    if (tf) result.set(key, tf);
  }
  return result;
}

/** Build the TF tree data from accumulated transforms and counts. */
export function buildTfTreeData(
  transforms: Map<string, TfTransform>,
  updateCounts: Map<string, number>,
  transformsByKey: Map<string, TfTransform[]>,
): TfTreeData {
  // Sort each per-key array by timestampNs and compute time bounds
  let startTimeNs = 0x7fff_ffff_ffff_ffffn;
  let endTimeNs = 0n;

  for (const arr of transformsByKey.values()) {
    arr.sort((a, b) =>
      a.timestampNs < b.timestampNs
        ? -1
        : a.timestampNs > b.timestampNs
          ? 1
          : 0,
    );
    if (arr.length > 0) {
      const first = arr[0]!.timestampNs;
      const last = arr[arr.length - 1]!.timestampNs;
      if (first < startTimeNs) startTimeNs = first;
      if (last > endTimeNs) endTimeNs = last;
    }
  }

  if (startTimeNs > endTimeNs) {
    startTimeNs = 0n;
    endTimeNs = 0n;
  }

  // Detect multi-parent violations: a child frame with >1 distinct parent
  const childToParents = new Map<string, Set<string>>();
  for (const tf of transforms.values()) {
    if (!childToParents.has(tf.childFrame)) {
      childToParents.set(tf.childFrame, new Set());
    }
    childToParents.get(tf.childFrame)!.add(tf.parentFrame);
  }

  const multipleParents = new Map<string, string[]>();
  for (const [child, parents] of childToParents) {
    if (parents.size > 1) {
      multipleParents.set(child, [...parents]);
    }
  }

  return {
    transforms,
    updateCounts,
    multipleParents,
    transformsByKey,
    startTimeNs,
    endTimeNs,
  };
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/** Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in degrees. */
export function quaternionToEuler(
  x: number,
  y: number,
  z: number,
  w: number,
): [roll: number, pitch: number, yaw: number] {
  const sinr = 2.0 * (w * x + y * z);
  const cosr = 1.0 - 2.0 * (x * x + y * y);
  const roll = Math.atan2(sinr, cosr);

  const sinp = 2.0 * (w * y - z * x);
  const pitch =
    Math.abs(sinp) >= 1 ? (Math.sign(sinp) * Math.PI) / 2 : Math.asin(sinp);

  const siny = 2.0 * (w * z + x * y);
  const cosy = 1.0 - 2.0 * (y * y + z * z);
  const yaw = Math.atan2(siny, cosy);

  const RAD2DEG = 180 / Math.PI;
  return [roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG];
}
