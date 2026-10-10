/** Shared renderings for the numbers every page shows: how big a file is and
 * when it was last touched. Four pages had their own copies of both, each
 * rounding a little differently - one page's model size never fell below a
 * whole MB, another's asset size never rose above one decimal past 1 GB -
 * so a byte count read a different way depending which page happened to
 * show it. One function per number, here. */

const KB = 1024
const MB = KB * 1024
const GB = MB * 1024

export function formatBytes(size: number): string {
  if (size >= GB) return (size / GB).toFixed(2) + ' GB'
  if (size < MB) return Math.max(1, Math.round(size / KB)) + ' KB'
  return (size / MB).toFixed(1) + ' MB'
}

export function formatMtime(mtime: number): string {
  return new Date(mtime * 1000).toLocaleString()
}

/** A size in GB to one decimal and no unit - the header meter and the
 * models list put the unit beside it themselves. */
export const gbFromMb = (mb: number): string => (mb / 1024).toFixed(1)
export const gbFromBytes = (bytes: number): string =>
  (bytes / 1024 ** 3).toFixed(1)

/** A span of seconds as people read it: `12.3s` under a minute, `4m 03s`
 * under an hour, `1h 02m` past it. The flow view, the log and the job
 * header all show run time through this, so a step's figure and the job's
 * read the same way. */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const whole = Math.round(seconds)
  const pad = (n: number) => String(n).padStart(2, '0')
  if (whole < 3600) return `${Math.floor(whole / 60)}m ${pad(whole % 60)}s`
  return `${Math.floor(whole / 3600)}h ${pad(Math.floor((whole % 3600) / 60))}m`
}
