<script lang="ts">
  import { Bell, BellOff } from '@lucide/svelte'
  import { alertMode } from './finishAlert'
  import { alerts, toggleAlerts } from './alerts.svelte'
</script>

<button
  class="bare icon"
  onclick={toggleAlerts}
  aria-pressed={alerts.on}
  aria-label={alerts.on
    ? 'stop job-finished alerts'
    : 'alert me when a job finishes'}
  title={alerts.on
    ? 'alerts on - click to stop'
    : alertMode() === 'chime'
      ? 'chime when a job finishes while this tab is in the background (this page is not a secure origin, so desktop notifications are unavailable)'
      : 'notify me when a job finishes while this tab is in the background'}
>
  {#if alerts.on}<Bell size={15} />{:else}<BellOff size={15} />{/if}
</button>
