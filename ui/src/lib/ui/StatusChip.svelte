<script lang="ts">
  // A job's status as a chip. The look is `.chip.<status>` in app.css, so a
  // status has one colour wherever it is shown: a cancelled job is amber on
  // the Jobs page, the job header, the Overview and as a request take. A
  // request *item's* chip is a different thing (`itemState.ts`, `StateTag`)
  // and stays separate by design.
  let { status, label }: { status: string; label?: string } = $props()
</script>

<span class="chip {status}">{label ?? status}</span>
