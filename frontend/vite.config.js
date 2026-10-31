import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: parseInt(process.env.PORT || "5173", 10),
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
    headers: {
      "Access-Control-Allow-Origin": "*",
    },
  },
  assetsInclude: ["**/*.wasm", "**/*.data", "**/*.binarypb"],
});
