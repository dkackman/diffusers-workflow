import type { JobExport } from '../types'

const plural = (n: number) => (n === 1 ? '' : 's')

/** What a finished export says: how much was gathered, where, and how many
 * files the job referenced that are no longer on disk. */
export function exportMessage(result: JobExport): string {
  const mb = (result.total_bytes / (1024 * 1024)).toFixed(1)
  const files = result.files.length
  const missing = result.missing.length
  return (
    `Exported ${files} file${plural(files)} (${mb} MB) to ${result.directory}` +
    (missing
      ? ` - ${missing} referenced file${plural(missing)} no longer on disk`
      : '')
  )
}
