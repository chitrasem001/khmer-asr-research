import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI on :5173 and forwards /api to the FastAPI server on :8000.
// `npm run build` writes the app to ../static, which FastAPI serves — no Node needed to run the demo.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "../static", emptyOutDir: true },
});
