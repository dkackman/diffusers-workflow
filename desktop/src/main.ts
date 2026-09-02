// The first-run and startup screen. Once the server answers, this window
// navigates to it and the SPA in ui/ takes over entirely.
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type Accelerator =
  | { kind: "cuda"; index_url: string; driver: string; name: string }
  | { kind: "cpu"; reason: string };

type Status = {
  needs_provisioning: boolean;
  running: boolean;
  base_url: string | null;
  version: string;
};

const app = document.getElementById("app")!;

function render(html: string) {
  app.innerHTML = html;
}

function escape(text: string): string {
  const node = document.createElement("span");
  node.textContent = text;
  return node.innerHTML;
}

async function showProvisioning(status: Status) {
  const accelerator = await invoke<Accelerator>("detect_accelerator");
  const detected =
    accelerator.kind === "cuda"
      ? `${accelerator.name} (driver ${accelerator.driver})`
      : accelerator.reason;

  render(`
    <h1>Set up diffusers-workflow</h1>
    <p class="lede">Version ${escape(status.version)}. This downloads about 4 GB
    of Python packages and takes 5–15 minutes. Models are downloaded later,
    the first time you run a workflow that needs them.</p>
    <div class="panel">
      <p><strong>Detected:</strong> ${escape(detected)}</p>
      <label>Install
        <select id="choice">
          ${
            accelerator.kind === "cuda"
              ? `<option value="${escape(accelerator.index_url)}">GPU build (CUDA)</option>`
              : ""
          }
          <option value="">CPU build</option>
        </select>
      </label>
    </div>
    <button id="go">Install</button>
    <div class="panel" id="progress" hidden><ol class="steps" id="steps"></ol></div>
    <p class="error" id="error"></p>
  `);

  document.getElementById("go")!.addEventListener("click", async () => {
    const choice = (document.getElementById("choice") as HTMLSelectElement).value;
    (document.getElementById("go") as HTMLButtonElement).disabled = true;
    document.getElementById("progress")!.hidden = false;
    try {
      const url = await invoke<string>("start_provisioning", {
        indexUrl: choice || null,
      });
      window.location.replace(url);
    } catch (error) {
      document.getElementById("error")!.textContent = String(error);
      (document.getElementById("go") as HTMLButtonElement).disabled = false;
    }
  });
}

function showStarting() {
  render(`
    <h1>Starting the engine…</h1>
    <p class="lede">Loading PyTorch and the workflow engine. The first start
    after an update takes longer.</p>
    <p class="error" id="error"></p>
  `);
}

async function boot() {
  const status = await invoke<Status>("shell_status");
  if (status.needs_provisioning) {
    await showProvisioning(status);
    return;
  }
  showStarting();
  try {
    const url = await invoke<string>("start_server");
    window.location.replace(url);
  } catch (error) {
    document.getElementById("error")!.textContent = String(error);
  }
}

listen<{ step: number; of: number; command: string[] }>(
  "provision://progress",
  (event) => {
    const steps = document.getElementById("steps");
    if (!steps) return;
    const labels = ["Creating the environment", "Installing PyTorch", "Installing the engine"];
    steps.innerHTML = labels
      .map((label, i) => {
        const state = i + 1 < event.payload.step ? "done" : i + 1 === event.payload.step ? "active" : "";
        return `<li data-state="${state}">${escape(label)}</li>`;
      })
      .join("");
  },
);

boot();
