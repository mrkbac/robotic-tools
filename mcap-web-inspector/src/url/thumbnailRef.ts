import type { ThumbnailMap } from "../mcap/image.ts";

/** Module-level ref to keep thumbnails across route transitions. */

let current: ThumbnailMap | null = null;

export function setThumbnailRef(thumbnails: ThumbnailMap): void {
  current = thumbnails;
}

export function getThumbnailRef(): ThumbnailMap | null {
  return current;
}
