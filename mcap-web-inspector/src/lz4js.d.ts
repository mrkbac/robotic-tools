declare module "lz4js" {
  function decompress(
    buffer: Uint8Array,
    maxOutputLength?: number,
  ): Uint8Array;
  export default { decompress };
}
