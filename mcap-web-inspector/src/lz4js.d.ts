declare module "lz4js" {
  function decompress(buffer: Uint8Array, maxOutputLength?: number): Uint8Array;
  function compress(buffer: Uint8Array): Uint8Array;
  export default { decompress, compress };
}
