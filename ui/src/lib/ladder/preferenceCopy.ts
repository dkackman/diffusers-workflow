// Plain-language copy for the preference pages: a key as a label, a skip
// reason in words, a download note as a first-use line. The strings read here
// are the server's (dw/preferences.py `downloads_gate_check`); this only
// rewords them.

/** A preference key as a label: `text_to_image` -> `Text to image` */
export function keyLabel(key: string): string {
  const words = key.replace(/[_-]+/g, ' ').trim()
  return words.charAt(0).toUpperCase() + words.slice(1)
}

const REASONS: Record<string, string> = {
  'cuda-only': 'needs CUDA',
  'mps-only': 'needs Apple MPS',
  'cpu-only': 'CPU only',
  'cuda/mps-only': 'needs CUDA or Apple MPS',
}

/** Why a model was passed over; a device reason in words, any other as given */
export function reasonText(reason: string): string {
  return REASONS[reason] ?? reason
}

const SIZED = /^will download (\d+(?:\.\d+)?) GB$/

/** The line for a model that is not on disk yet, from a resolution's notes;
 * null when none of them says so */
export function downloadNote(notes: string[]): { text: string } | null {
  for (const note of notes) {
    const sized = SIZED.exec(note)
    if (sized) return { text: `downloads ${sized[1]} GB on first use` }
    if (note === 'will download its weights' || note === 'not downloaded')
      return { text: 'downloads its weights on first use' }
  }
  return null
}
