import type { ThumbnailMap } from "../mcap/image.ts";

const THUMB_W = 48;
const THUMB_H = 36;
const JPEG_QUALITY = 0.5;

/**
 * Resize raw image data to a 48x36 JPEG micro-thumbnail (center-crop, cover fit).
 * Returns a base64 string (no data URL prefix), or null on failure.
 */
export async function createMicroThumbnail(
  imageData: Uint8Array,
  format: string,
): Promise<string | null> {
  try {
    const mime = format.startsWith("image/")
      ? format
      : `image/${format || "jpeg"}`;
    const blob = new Blob([imageData as BlobPart], { type: mime });
    const bmp = await createImageBitmap(blob);

    // Center-crop to cover THUMB_W x THUMB_H aspect ratio
    const srcAspect = bmp.width / bmp.height;
    const dstAspect = THUMB_W / THUMB_H;
    let sx: number, sy: number, sw: number, sh: number;

    if (srcAspect > dstAspect) {
      // Source is wider — crop sides
      sh = bmp.height;
      sw = sh * dstAspect;
      sx = (bmp.width - sw) / 2;
      sy = 0;
    } else {
      // Source is taller — crop top/bottom
      sw = bmp.width;
      sh = sw / dstAspect;
      sx = 0;
      sy = (bmp.height - sh) / 2;
    }

    const canvas = new OffscreenCanvas(THUMB_W, THUMB_H);
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, THUMB_W, THUMB_H);
    bmp.close();

    const outBlob = await canvas.convertToBlob({
      type: "image/jpeg",
      quality: JPEG_QUALITY,
    });
    const buf = await outBlob.arrayBuffer();
    const bytes = new Uint8Array(buf);

    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]!);
    }
    return btoa(binary);
  } catch {
    return null;
  }
}

/** Pick the first thumbnail from the map and create a micro-thumbnail. */
export async function createMicroThumbFromMap(
  thumbnails: ThumbnailMap,
): Promise<string | null> {
  const first = thumbnails.values().next();
  if (first.done) return null;
  return createMicroThumbnail(first.value.data, first.value.format);
}

/** Wrap a base64 thumbnail string into a data URL for display. */
export function thumbnailBase64ToDataUrl(base64: string): string {
  return `data:image/jpeg;base64,${base64}`;
}
