/** File identity for URL encoding — matches the cache key pattern from useMcapCache. */

export function createFileId(file: File): string {
  const sanitized = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
  return `${sanitized}-${file.size.toString(36)}`;
}

export function fileMatchesId(file: File, fileId: string): boolean {
  return createFileId(file) === fileId;
}
