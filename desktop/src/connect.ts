// The MCP config, ready to paste. It deliberately carries no port: the
// Python client reads the live one from server.json.
import { invoke } from "@tauri-apps/api/core";

const app = document.getElementById("app")!;

async function main() {
  const config = await invoke<string>("mcp_config_json");
  app.innerHTML = `
    <h1>Connect to Claude</h1>
    <p class="lede">Add this to your MCP client's configuration. It needs no
    port — the server publishes whichever one it bound, and the client picks
    it up automatically.</p>
    <div class="panel"><pre id="config"></pre></div>
    <button id="copy">Copy</button>
  `;
  document.getElementById("config")!.textContent = config;
  document.getElementById("copy")!.addEventListener("click", async () => {
    await navigator.clipboard.writeText(config);
    (document.getElementById("copy") as HTMLButtonElement).textContent = "Copied";
  });
}

main();
