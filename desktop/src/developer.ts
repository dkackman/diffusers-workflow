// The venv the CLI and REPL live in. The installer never touches PATH, so
// this window is how someone finds them.
import { invoke } from "@tauri-apps/api/core";

const app = document.getElementById("app")!;

async function main() {
  const [venv, documents] = await Promise.all([
    invoke<string>("venv_path"),
    invoke<string>("documents_path"),
  ]);
  app.innerHTML = `
    <h1>Developer</h1>
    <p class="lede">The command line tools are installed in this app's virtual
    environment. Nothing was added to your PATH.</p>
    <div class="panel">
      <p><strong>Environment</strong></p>
      <pre id="venv"></pre>
      <p><strong>Workflows, prompts and outputs</strong></p>
      <pre id="documents"></pre>
      <p>Available there: <code>dw-run</code>, <code>dw-repl</code>,
      <code>dw-validate</code>, <code>dw-serve</code>, <code>dw-mcp</code>.</p>
      <p class="lede">Running <code>dw-repl</code> while this app is open puts
      two processes on one GPU; the REPL will warn you.</p>
      <button id="terminal">Open terminal here</button>
    </div>
    <div class="panel">
      <p><strong>Repair installation</strong></p>
      <p class="lede">Rebuilds the Python environment on the next launch. Your
      workflows, prompts, outputs and downloaded models are untouched.</p>
      <button class="secondary" id="repair">Repair on next launch</button>
    </div>
  `;
  document.getElementById("venv")!.textContent = venv;
  document.getElementById("documents")!.textContent = documents;
  document
    .getElementById("terminal")!
    .addEventListener("click", () => invoke("open_terminal"));
  document.getElementById("repair")!.addEventListener("click", async () => {
    await invoke("repair_installation");
    (document.getElementById("repair") as HTMLButtonElement).textContent =
      "Will rebuild on next launch";
  });
}

main();
