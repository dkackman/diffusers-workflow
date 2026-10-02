<script lang="ts">
  import { CircleCheck, Download, Sparkles } from '@lucide/svelte'
  import Suggest from '../ui/Suggest.svelte'
  import type { Enhancer } from '../enhancer.svelte'

  // The prompt editor's "Enhance with AI" panel. Its state is the page's
  // Enhancer, so a view switch that unmounts this keeps a download going
  let { enhancer, onuse }: { enhancer: Enhancer; onuse: () => void } = $props()
</script>

<div class="panel">
  <h2><Sparkles size={15} /> Enhance with AI</h2>
  <p class="muted hint">
    Expand an idea into a full prompt with a local language model. Runs as an
    ordinary job - it waits its turn behind anything generating.
  </p>
  {#if enhancer.down}
    <p class="muted hint">
      Enhancement is unavailable - the server reported no enhancer presets.
      Editing and saving still work.
    </p>
  {/if}
  <div class="metagrid">
    <label for="enhance-preset">preset</label>
    <select
      id="enhance-preset"
      value={enhancer.presetKey}
      onchange={(e) => enhancer.pickPreset(e.currentTarget.value)}
    >
      {#each enhancer.presets as p (p.key)}
        <option value={p.key}>{p.label}</option>
      {/each}
    </select>
    <label for="enhance-model">model</label>
    <span class="modelrow">
      <Suggest
        id="enhance-model"
        suggestions={enhancer.preset?.models ?? []}
        bind:value={enhancer.model}
        placeholder="Hugging Face repo id"
      />
      {#if enhancer.model}
        {#if enhancer.modelCached}
          <span class="chip good" title="already in the local model cache"
            >cached</span
          >
        {:else if enhancer.downloading}
          <span class="chip" title="downloading to the local model cache"
            >downloading…</span
          >
        {:else}
          <button
            class="quiet withicon"
            onclick={() => enhancer.downloadModel()}
            title="download this model to the local cache now - otherwise the first enhancement downloads it"
          >
            <Download size={13} />get
          </button>
        {/if}
      {/if}
    </span>
    <label for="enhance-device">device</label>
    <select
      id="enhance-device"
      bind:value={enhancer.device}
      title="where the language model runs - cpu keeps VRAM free for generation"
    >
      <option value="">preset default (cpu)</option>
      <option value="cuda">cuda</option>
      <option value="mps">mps</option>
      <option value="cpu">cpu</option>
    </select>
    <label for="enhance-idea">idea</label>
    <textarea
      id="enhance-idea"
      rows="3"
      spellcheck="true"
      bind:value={enhancer.idea}
      placeholder={enhancer.preset?.placeholder ?? 'describe what to generate'}
    ></textarea>
  </div>
  <div class="enhanceactions">
    <button
      class="withicon"
      onclick={() => enhancer.generate()}
      disabled={enhancer.busy || !enhancer.presets.length}
      title="expand the idea with the selected model"
    >
      <Sparkles size={14} />Generate
    </button>
    {#if enhancer.busy && enhancer.jobId}
      <button class="quiet" onclick={() => enhancer.cancel()}>cancel</button>
      <a class="muted joblink" href={'#/jobs/' + enhancer.jobId}>watch job</a>
    {/if}
    {#if enhancer.status}
      <span class="muted enhancestatus"
        ><span class="pulse-dot"></span>{enhancer.status}</span
      >
    {/if}
  </div>
  {#if enhancer.error}<p class="error">{enhancer.error}</p>{/if}
  {#if enhancer.result}
    <textarea class="resultbox" rows="8" readonly value={enhancer.result}
    ></textarea>
    <div class="enhanceactions">
      <button class="withicon" onclick={onuse}>
        <CircleCheck size={14} />Use as prompt text
      </button>
      <button class="quiet" onclick={() => (enhancer.result = '')}>
        discard
      </button>
    </div>
  {/if}
</div>

<style>
  .panel {
    margin-bottom: 1rem;
    min-width: 0;
  }
  .panel h2 {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  textarea {
    width: 100%;
    resize: vertical;
  }
  .metagrid {
    display: grid;
    grid-template-columns: minmax(120px, 40%) minmax(0, 1fr);
    gap: 0.5rem 0.8rem;
    align-items: center;
    margin-top: 0.6rem;
  }
  @container (max-width: 420px) {
    .metagrid {
      grid-template-columns: minmax(0, 1fr);
      gap: 0.2rem;
    }
  }
  .metagrid label {
    font-weight: 600;
    color: var(--muted);
    font-size: 0.85rem;
  }
  .metagrid textarea {
    align-self: stretch;
  }
  .modelrow {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .modelrow :global(input) {
    flex: 1;
  }
  .chip.good {
    color: var(--good);
    border-color: var(--good);
  }
  .enhanceactions {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.7rem;
  }
  .enhancestatus {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.85rem;
  }
  .joblink {
    font-size: 0.85rem;
  }
  .resultbox {
    margin-top: 0.7rem;
    font-size: 0.9rem;
  }
  .error {
    color: var(--bad);
  }
  .hint {
    font-size: 0.8rem;
  }
</style>
