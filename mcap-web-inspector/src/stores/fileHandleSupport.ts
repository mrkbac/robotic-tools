export function supportsFileSystemAccess(): boolean {
  return (
    typeof DataTransferItem !== "undefined" &&
    "getAsFileSystemHandle" in DataTransferItem.prototype
  );
}
