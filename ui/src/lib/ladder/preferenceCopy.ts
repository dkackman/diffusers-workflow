// Plain-language copy for the preference pages: a key as a label, a skip
// reason in words, a download note as a before-first-job line. The strings read here
// are the server's (dw/preferences.py `downloads_gate_check`); this only
// rewords them.

/** A preference key as a label: `text_to_image` -> `Text to image` */
export function keyLabel(key: string): string {
  const words = key.replace(/[_-]+/g, ' ').trim()
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/** The device types a target's `devices` names, in the order they are said */
const DEVICES: [string, string][] = [
  ['cuda', 'CUDA'],
  ['mps', 'Apple MPS'],
  ['cpu', 'the CPU'],
]

/** The server's skip for a target the last full check found gated
 * (dw/preferences.py `KNOWN_BLOCKED`) */
const REASONS: Record<string, string> = {
  'gated, no access (last full check)':
    'gated: no access (as of the last full check)',
}

/** `cuda/mps-only` (dw/preferences.py `skip_reason`) in words, in one order
 * whatever order the ladder listed them; null when a device is unknown */
function devicesText(reason: string): string | null {
  const only = /^([a-z]+(?:\/[a-z]+)*)-only$/.exec(reason)
  if (!only) return null
  const named = new Set(only[1].split('/'))
  const known = DEVICES.filter(([device]) => named.has(device))
  if (known.length !== named.size) return null
  const words = known.map(([, word]) => word)
  const last = words.pop()!
  return `needs ${words.length ? `${words.join(', ')} or ${last}` : last}`
}

/** Why a model was passed over; a device or known-blocked reason in words,
 * any other as given */
export function reasonText(reason: string): string {
  return REASONS[reason] ?? devicesText(reason) ?? reason
}

/** The lead a refused tier shows. A refusal that passed models over is a
 * no fit: each model is struck below with its reason, so the lead says only
 * that. Any other refusal (a model not in the catalog, refused arguments) is
 * not the machine's doing: a neutral lead and the server's message */
export function refusalLead(
  refusal: string,
  skipped: readonly unknown[],
): string {
  return skipped.length
    ? 'Nothing fits this machine'
    : `This tier is refused: ${refusal}`
}

const SIZED = /^will download (\d+(?:\.\d+)?) GB$/

/** The line for a model that is not on disk yet, from a resolution's notes;
 * null when none of them says so */
export function downloadNote(notes: string[]): { text: string } | null {
  for (const note of notes) {
    const sized = SIZED.exec(note)
    if (sized) return { text: `downloads ${sized[1]} GB before its first job` }
    if (note === 'will download its weights' || note === 'not downloaded')
      return { text: 'downloads its weights before its first job' }
  }
  return null
}
