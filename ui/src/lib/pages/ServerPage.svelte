<script lang="ts">
  import { Globe, Plug, ShieldCheck, TriangleAlert } from '@lucide/svelte'
  import { api } from '../api'
  import CopyButton from '../CopyButton.svelte'
  import ServerHealthList from '../ServerHealthList.svelte'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { SECTION_LABELS } from '../routes'
  import CachesPanel from '../CachesPanel.svelte'
  import {
    addressLabel,
    bindSnippet,
    browserUrl,
    isLoopbackBind,
    isUnauthenticatedPublicBind,
    mcpAddCommand,
    mcpUrl,
    REMOTE_GUIDE,
    runtimeRows,
    runtimeText,
  } from '../serverinfo'
  import type { ServerInfo } from '../types'

  // Static while the server runs; a reconnect or an accepted token raises
  // the epoch, which reads it again, so the page recovers on its own
  const read = resource(() => api.server(), { watch: [] })
  const info = $derived<ServerInfo | null>(read.data ?? null)

  /** Which interface address the connect snippets are written against. */
  let selected = $state('')
  // A wildcard bind answers on every interface; a bind to one address answers
  // on that address only, so it is the only one worth offering
  const addresses = $derived.by(() => {
    if (!info) return []
    if (info.wildcard_bind || isLoopbackBind(info.bind_host)) {
      return info.addresses
    }
    const named = info.addresses.find((a) => a.address === info.bind_host)
    return [
      named ?? {
        address: info.bind_host,
        family: info.bind_host.includes(':') ? 'IPv6' : 'IPv4',
        interface: '',
      },
    ]
  })
  const address = $derived(
    addresses.find((a) => a.address === selected)?.address ??
      addresses[0]?.address ??
      '',
  )
  const port = $derived(info?.port ?? 0)
  const mcpPath = $derived(info?.mcp.path ?? '/mcp')
  const loopback = $derived(info ? isLoopbackBind(info.bind_host) : false)
</script>

<!-- The status page's header: the queue below it is a section of this page -->
<PageHeader
  title={SECTION_LABELS.server.status}
  secondary={info ? `${info.hostname} · v${info.version}` : undefined}
  baseline
/>

<LoadState of={read} label="the server details" />

{#if info}
  <div class="panel">
    <h2>Server</h2>
    <dl class="kv">
      <dt>Hostname</dt>
      <dd>{info.hostname}</dd>
      <dt>Version</dt>
      <dd>{info.version}</dd>
      <dt>Device</dt>
      <dd>{info.device}</dd>
    </dl>
    <ServerHealthList />
  </div>

  <div class="panel runtime">
    <h2>
      Runtime
      <CopyButton text={runtimeText(info)} title="copy the runtime versions" />
    </h2>
    <dl class="kv">
      {#each runtimeRows(info.runtime) as row (row.name)}
        <dt>{row.name}</dt>
        <dd>
          {#if row.value != null}
            <code>{row.value}</code>
          {:else}
            <span class="muted">{row.absent}</span>
          {/if}
        </dd>
      {/each}
    </dl>
  </div>

  <div class="panel">
    <h2>Binding</h2>
    <dl class="kv">
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
      {:else if loopback}
        A loopback bind is reachable only from this machine.
      {:else}
        This server answers on {info.bind_host} only.
      {/if}
    </p>
  </div>

  <div class="panel">
    <h2>Addresses</h2>
    {#if loopback}
      <p class="muted">
        This server is bound to {info.bind_host}, so only this machine can reach
        it.
      </p>
    {:else if addresses.length === 0}
      <p class="muted">
        No non-loopback address — this machine is reachable only from itself.
      </p>
    {:else}
      <div class="tablewrap">
        <table class="datatable">
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
      <dl class="kv">
        <dt>Mounted</dt>
        <dd>yes</dd>
        <dt>Path</dt>
        <dd><code>{info.mcp.path}</code></dd>
        {#if address && !loopback}
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
        Not mounted. Restart the server with --mcp (<code>
          python -m dw.serve --mcp
        </code>) to serve the MCP tools at <code>{info.mcp.path}</code>.
      </p>
    {/if}
  </div>

  <div class="panel" class:warn-edge={isUnauthenticatedPublicBind(info)}>
    <h2><ShieldCheck size={15} />Authentication</h2>
    <dl class="kv">
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
    {#if loopback}
      <p class="muted">
        Only this machine can reach this server. To use it from another, restart
        it bound to the network, with a token:
      </p>
      <div class="snippet">
        <code>{bindSnippet()}</code>
        <CopyButton text={bindSnippet()} title="copy the restart command" />
      </div>
      <p class="muted note">
        <a href={REMOTE_GUIDE} target="_blank" rel="noopener noreferrer">
          How to use dw from another machine
        </a>
      </p>
    {:else if !address}
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
          <code>
            {mcpAddCommand(address, port, mcpPath, info.auth_required)}
          </code>
          <CopyButton
            text={mcpAddCommand(address, port, mcpPath, info.auth_required)}
            title="copy the claude mcp add command"
          />
        </div>
        {#if info.auth_required}
          <p class="muted note">
            Replace <code>&lt;token&gt;</code> with the server's API token.
          </p>
        {/if}
      {:else}
        <p class="muted">
          The MCP endpoint is not mounted, so there is no command to run yet.
          Restart the server with --mcp.
        </p>
      {/if}
    {/if}
  </div>

  <div class="panel">
    <h2>Directories</h2>
    <dl class="kv">
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

  <CachesPanel />
{/if}

<style>
  .panel {
    margin-bottom: 0.8rem;
  }
  h3 {
    margin: 0.9rem 0 0.35rem;
    font-size: 0.85rem;
    color: var(--muted);
  }
  .note {
    margin: 0.6rem 0 0;
    max-width: 80ch;
    font-size: 0.85rem;
  }
  .warn {
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
  tr.chosen code {
    color: var(--accent);
  }
</style>
