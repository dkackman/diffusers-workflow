<script lang="ts">
  import { Globe, Plug, ShieldCheck, TriangleAlert } from '@lucide/svelte'
  import { api } from '../api'
  import CopyButton from '../CopyButton.svelte'
  import {
    addressLabel,
    browserUrl,
    isLoopbackBind,
    isUnauthenticatedPublicBind,
    mcpAddCommand,
    mcpUrl,
  } from '../serverinfo'
  import type { HealthInfo, ServerInfo } from '../types'

  let info = $state<ServerInfo | null>(null)
  let error = $state('')
  let health = $state<HealthInfo | null>(null)

  $effect(() => {
    api
      .server()
      .then((s) => {
        info = s
        error = ''
      })
      .catch((e) => (error = e instanceof Error ? e.message : String(e)))
  })

  // The live half of the page: the same endpoint App.svelte polls for the
  // status bar, fetched here so the worker/queue lines keep up on their own
  $effect(() => {
    const poll = async () => {
      try {
        health = await api.health()
      } catch {
        health = null
      }
    }
    poll()
    const timer = setInterval(poll, 5000)
    return () => clearInterval(timer)
  })

  /** Which interface address the connect snippets are written against. */
  let selected = $state('')
  const addresses = $derived(info?.addresses ?? [])
  const address = $derived(
    addresses.find((a) => a.address === selected)?.address ??
      addresses[0]?.address ??
      '',
  )
  const port = $derived(info?.port ?? 0)
  const mcpPath = $derived(info?.mcp.path ?? '/mcp')
</script>

<div class="head">
  <h1>Server</h1>
  {#if info}
    <span class="muted">{info.hostname} · v{info.version}</span>
  {/if}
</div>

{#if error}
  <p class="warn">{error}</p>
  <p class="muted">
    The server details need the API token — set it with the key icon above.
  </p>
{/if}

{#if info}
  <div class="panel">
    <h2>Status</h2>
    <dl>
      <dt>Hostname</dt>
      <dd>{info.hostname}</dd>
      <dt>Version</dt>
      <dd>{info.version}</dd>
      <dt>Device</dt>
      <dd>{info.device}</dd>
      <dt>Worker</dt>
      <dd>
        {#if health?.worker_alive}
          running
        {:else if health}
          <span class="muted">not started — spawns with the first job</span>
        {:else}
          <span class="warn">unreachable</span>
        {/if}
      </dd>
      <dt>Queue</dt>
      <dd>{health?.queued ?? 0} queued</dd>
      <dt>Current job</dt>
      <dd>
        {#if health?.current_job}
          <a href={'#/jobs/' + health.current_job}>{health.current_job} →</a>
        {:else}
          <span class="muted">idle</span>
        {/if}
      </dd>
    </dl>
  </div>

  <div class="panel">
    <h2>Binding</h2>
    <dl>
      <dt>Bind host</dt>
      <dd><code>{info.bind_host}</code></dd>
      <dt>Port</dt>
      <dd><code>{info.port}</code></dd>
      <dt>Wildcard</dt>
      <dd>{info.wildcard_bind ? 'yes' : 'no'}</dd>
    </dl>
    <p class="muted note">
      {#if info.wildcard_bind}
        A wildcard bind is reachable on every address listed below.
      {:else if isLoopbackBind(info.bind_host)}
        A loopback bind is reachable only from this machine.
      {:else}
        This server answers on {info.bind_host} only.
      {/if}
    </p>
  </div>

  <div class="panel">
    <h2>Addresses</h2>
    {#if addresses.length === 0}
      <p class="muted">
        No non-loopback address — this machine is reachable only from itself.
      </p>
    {:else}
      <div class="tablewrap">
        <table>
          <thead>
            <tr>
              <th></th>
              <th>address</th>
              <th>family</th>
              <th>interface</th>
              <th>url</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {#each addresses as entry (entry.address)}
              <tr class:chosen={entry.address === address}>
                <td>
                  <input
                    type="radio"
                    name="address"
                    value={entry.address}
                    checked={entry.address === address}
                    onchange={() => (selected = entry.address)}
                    aria-label="use {addressLabel(entry)} in the snippets below"
                  />
                </td>
                <td><code>{entry.address}</code></td>
                <td class="muted">{entry.family}</td>
                <td class="muted">{entry.interface}</td>
                <td><code>{browserUrl(entry.address, port)}</code></td>
                <td>
                  <CopyButton
                    text={browserUrl(entry.address, port)}
                    title="copy {browserUrl(entry.address, port)}"
                  />
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>

  <div class="panel">
    <h2><Plug size={15} />MCP</h2>
    {#if info.mcp.mounted}
      <dl>
        <dt>Mounted</dt>
        <dd>yes</dd>
        <dt>Path</dt>
        <dd><code>{info.mcp.path}</code></dd>
        {#if address}
          <dt>Endpoint</dt>
          <dd>
            <code>{mcpUrl(address, port, mcpPath)}</code>
            <CopyButton
              text={mcpUrl(address, port, mcpPath)}
              title="copy the MCP endpoint URL"
            />
          </dd>
        {/if}
      </dl>
    {:else}
      <p class="muted">
        Not mounted — restart with <code>dw-serve --mcp</code> to serve the MCP
        tool surface at <code>{info.mcp.path}</code>.
      </p>
    {/if}
  </div>

  <div class="panel" class:warn-edge={isUnauthenticatedPublicBind(info)}>
    <h2><ShieldCheck size={15} />Authentication</h2>
    <dl>
      <dt>Token required</dt>
      <dd>{info.auth_required ? 'yes' : 'no'}</dd>
    </dl>
    {#if isUnauthenticatedPublicBind(info)}
      <p class="warn">
        <TriangleAlert size={14} />
        No token is set and this server is bound beyond loopback: anyone who can reach
        this address can run workflows on this GPU.
      </p>
    {:else if info.auth_required}
      <p class="muted note">
        The token is never displayed here. It lives in the server's environment;
        the browser keeps its own copy in <code>localStorage</code>.
      </p>
    {/if}
  </div>

  <div class="panel">
    <h2><Globe size={15} />Connect from another machine</h2>
    {#if !address}
      <p class="muted">
        No network address to build a URL from — nothing outside this machine
        can reach it.
      </p>
    {:else}
      {#if addresses.length > 1}
        <label class="picker">
          Address
          <select
            value={address}
            onchange={(e) => (selected = e.currentTarget.value)}
          >
            {#each addresses as entry (entry.address)}
              <option value={entry.address}>{addressLabel(entry)}</option>
            {/each}
          </select>
        </label>
      {/if}

      <h3>Browser</h3>
      <div class="snippet">
        <code>{browserUrl(address, port)}</code>
        <CopyButton
          text={browserUrl(address, port)}
          title="copy the browser URL"
        />
      </div>
      {#if info.auth_required}
        <p class="muted note">
          Paste the token once via the key icon; it is kept in that browser.
        </p>
      {/if}

      <h3>Claude Code</h3>
      {#if info.mcp.mounted}
        <div class="snippet">
          <code>{mcpAddCommand(address, port, mcpPath)}</code>
          <CopyButton
            text={mcpAddCommand(address, port, mcpPath)}
            title="copy the claude mcp add command"
          />
        </div>
        <p class="muted note">
          Replace <code>&lt;token&gt;</code> with the server's API token.
        </p>
      {:else}
        <p class="muted">
          The MCP endpoint is not mounted, so there is no command to run yet —
          restart the server with <code>dw-serve --mcp</code>.
        </p>
      {/if}
    {/if}
  </div>

  <div class="panel">
    <h2>Directories</h2>
    <dl>
      <dt>Workspace</dt>
      <dd>
        {#if info.directories.workspace}
          <code>{info.directories.workspace}</code>
        {:else}
          <span class="muted">none — directories set individually</span>
        {/if}
      </dd>
      <dt>Workflows</dt>
      <dd><code>{info.directories.workflows}</code></dd>
      <dt>Outputs</dt>
      <dd><code>{info.directories.outputs}</code></dd>
      <dt>Prompts</dt>
      <dd>
        {#if info.directories.prompts}
          <code>{info.directories.prompts}</code>
        {:else}
          <span class="muted">none configured</span>
        {/if}
      </dd>
      <dt>Assets</dt>
      <dd>
        {#if info.directories.assets}
          <code>{info.directories.assets}</code>
        {:else}
          <span class="muted">none configured</span>
        {/if}
      </dd>
    </dl>
  </div>
{:else if !error}
  <p class="muted">loading server details…</p>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.4rem 0.8rem;
  }
  .panel {
    margin-bottom: 0.8rem;
  }
  h2 {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
  }
  h3 {
    margin: 0.9rem 0 0.35rem;
    font-size: 0.85rem;
    color: var(--muted);
  }
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--space-2) var(--space-4);
    margin: 0;
    align-items: baseline;
  }
  dt {
    font-weight: 600;
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  dd {
    margin: 0;
  }
  .note {
    margin: 0.6rem 0 0;
    max-width: 80ch;
    font-size: 0.85rem;
  }
  .warn {
    color: var(--warn);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    max-width: 80ch;
  }
  .snippet {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .snippet code {
    overflow-wrap: anywhere;
  }
  .picker {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    color: var(--muted);
  }
  .tablewrap {
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th {
    text-align: left;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    font-weight: 600;
    padding: 0.2rem 0.5rem 0.2rem 0;
  }
  td {
    padding: 0.25rem 0.5rem 0.25rem 0;
    border-top: 1px solid var(--line);
  }
  tr.chosen code {
    color: var(--accent);
  }
</style>
