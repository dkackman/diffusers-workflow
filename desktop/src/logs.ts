// A live tail of the server's own output, so a failed start is diagnosable
// without a terminal.
import { invoke } from "@tauri-apps/api/core";

const app = document.getElementById("app")!;
app.innerHTML = `<h1>Server logs</h1><div class="panel"><pre id="log"></pre></div>`;

async function refresh() {
  const text = await invoke<string>("log_tail", { lines: 500 });
  document.getElementById("log")!.textContent = text;
}

refresh();
setInterval(refresh, 2000);
