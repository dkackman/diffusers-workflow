import { defineConfig } from "vite";
import { resolve } from "node:path";

// Four entry points, one bundle: the main window shows provisioning and
// then navigates away to the server, so the shell's other screens have to
// be their own pages rather than routes inside a single-page app.
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        connect: resolve(__dirname, "connect.html"),
        developer: resolve(__dirname, "developer.html"),
        logs: resolve(__dirname, "logs.html"),
      },
    },
  },
  clearScreen: false,
  server: { port: 1420, strictPort: true },
});
