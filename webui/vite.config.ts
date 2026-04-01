import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
      "/infer": "http://127.0.0.1:8000",
      "/defaults": "http://127.0.0.1:8000",
      "/atlas": "http://127.0.0.1:8000",
      "/overlays": "http://127.0.0.1:8000",
    },
  },
});
