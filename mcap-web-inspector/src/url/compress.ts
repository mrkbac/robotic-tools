/** Browser-native deflate-raw compression with base64url encoding. */

export async function compressToBase64url(json: string): Promise<string> {
  const input = new TextEncoder().encode(json);
  const cs = new CompressionStream("deflate-raw");
  const writer = cs.writable.getWriter();
  writer.write(input);
  writer.close();
  const compressed = await new Response(cs.readable).arrayBuffer();
  return arrayBufferToBase64url(new Uint8Array(compressed));
}

export async function decompressFromBase64url(b64: string): Promise<string> {
  const bytes = base64urlToUint8Array(b64);
  const ds = new DecompressionStream("deflate-raw");
  const writer = ds.writable.getWriter();
  writer.write(bytes as ArrayBufferView<ArrayBuffer>);
  writer.close();
  const decompressed = await new Response(ds.readable).arrayBuffer();
  return new TextDecoder().decode(decompressed);
}

function arrayBufferToBase64url(bytes: Uint8Array): string {
  const binary = String.fromCharCode(...bytes);
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

function base64urlToUint8Array(b64: string): Uint8Array {
  const padded = b64.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}
