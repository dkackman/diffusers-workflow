<script lang="ts">
  // The introduction a newcomer sees on the Overview of any workspace, until
  // "Got it" in this browser. Storage that throws just means it shows again
  // next load: firstRun.ts keeps it hidden for the session
  import { tick } from 'svelte'
  import {
    dismissFirstRun,
    firstRunDismissed,
    type FirstRunLink,
  } from './firstRun'

  let { link }: { link: FirstRunLink } = $props()

  let dismissed = $state(firstRunDismissed())

  // The button leaves with the card: focus goes to the page's h1 (focusable
  // by script, as PageHeader makes it) rather than falling to the body
  async function dismiss() {
    dismissed = true
    dismissFirstRun()
    await tick()
    document
      .querySelector<HTMLElement>('main h1, h1')
      ?.focus({ preventScroll: true })
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
        aria-label="Got it, dismiss this introduction"
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
