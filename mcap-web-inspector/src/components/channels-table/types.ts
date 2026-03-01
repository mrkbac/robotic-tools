import type { ChannelInfo } from "../../mcap/types.ts";

export type ChannelRow = ChannelInfo & {
  _kind: "channel" | "group";
  _segment: string;
  _fullPath: string;
  subRows?: ChannelRow[];
};
