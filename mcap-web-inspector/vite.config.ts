import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  base: "/robotic-tools/",
  plugins: [react()],
  optimizeDeps: {
    exclude: ["@bokuweb/zstd-wasm"],
  },
});
