import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MantineProvider } from "@mantine/core";
import {
  CodeHighlightAdapterProvider,
  createHighlightJsAdapter,
} from "@mantine/code-highlight";
import { RouterProvider } from "@tanstack/react-router";
import hljs from "highlight.js/lib/core";
import json from "highlight.js/lib/languages/json";
import protobuf from "highlight.js/lib/languages/protobuf";
import cpp from "highlight.js/lib/languages/cpp";
import "@mantine/core/styles.css";
import "@mantine/charts/styles.css";
import "@mantine/dates/styles.css";
import "@mantine/dropzone/styles.css";
import "@mantine/code-highlight/styles.css";
import { router } from "./router.ts";

hljs.registerLanguage("json", json);
hljs.registerLanguage("protobuf", protobuf);
hljs.registerLanguage("cpp", cpp);

const highlightJsAdapter = createHighlightJsAdapter(hljs);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MantineProvider defaultColorScheme="auto">
      <CodeHighlightAdapterProvider adapter={highlightJsAdapter}>
        <RouterProvider router={router} />
      </CodeHighlightAdapterProvider>
    </MantineProvider>
  </StrictMode>,
);
