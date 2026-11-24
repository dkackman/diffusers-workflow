<script lang="ts">
  // The introduction a newcomer sees on the Overview of any workspace, until
  // "Got it" in this browser. Storage that throws just means it shows again
  // next load: the flag keeps it hidden for the session
  import { FIRST_RUN_KEY, type FirstRunLink } from './firstRun'
  import { storageGet, storageSet } from './storage'

  let { link }: { link: FirstRunLink } = $props()

  let dismissed = $state(storageGet(FIRST_RUN_KEY, false))

  function dismiss() {
    dismissed = true
    storageSet(FIRST_RUN_KEY, true)
  }
</script>

{#if !dismissed}
  <section class="panel firstrun" aria-labelledby="firstrun-title">
    <h2 id="firstrun-title">New here? Make your first picture</h2>
    <p>
      dw runs image, video and audio workflows on this server's GPU. Pick a
      workflow, press Run, and what it makes lands in the Gallery.
    </p>
    <ol>
      <li>
        Open text-to-image. It uses a small model that needs no Hugging Face
        login.
      </li>
      <li>Type a prompt and press Run.</li>
      <li>Watch the job, then find the picture in the Gallery.</li>
    </ol>
    <div class="actions">
      <a class="plain cta" href={link.href}>{link.label}</a>
      <span class="muted">Press <kbd>?</kbd> for keyboard shortcuts.</span>
      <span class="flex"></span>
      <button
        class="quiet"
        type="button"
        aria-label="Dismiss this introduction"
        onclick={dismiss}
      >
        Got it
      </button>
    </div>
  </section>
{/if}

<style>
  .firstrun {
    margin-bottom: var(--space-4);
  }
  ol {
    margin: var(--space-2) 0;
    padding-left: var(--space-5, 1.5rem);
  }
  /* The one filled control, as a link: it goes somewhere */
  .cta {
    background: var(--accent);
    color: var(--accent-ink);
    border: 1px solid var(--accent);
    border-radius: var(--radius-1);
    padding: 0.4rem 0.9rem;
    font-size: var(--t-sm);
    font-weight: 600;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-3);
  }
</style>
