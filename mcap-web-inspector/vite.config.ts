import { copyFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import type { Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { tanstackRouter } from "@tanstack/router-plugin/vite";

function spa404(): Plugin {
  return {
    name: "spa-404",
    closeBundle() {
      const outDir = resolve(__dirname, "dist");
      copyFileSync(resolve(outDir, "index.html"), resolve(outDir, "404.html"));
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  base: "/robotic-tools/",
  plugins: [tanstackRouter({ quoteStyle: "double" }), react(), spa404()],
});
