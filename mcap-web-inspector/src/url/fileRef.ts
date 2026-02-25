/** Module-level WeakRef to keep the File object across route transitions. */

let ref: WeakRef<File> | null = null;

export function setFileRef(file: File): void {
  ref = new WeakRef(file);
}

export function getFileRef(): File | undefined {
  return ref?.deref();
}
