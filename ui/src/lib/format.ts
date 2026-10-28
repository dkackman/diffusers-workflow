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

/** A unix-seconds timestamp as a compact date and time in the viewer's
 * locale - 'Oct 26, 2:03 PM' - with the year only when it is not this one.
 * Every timestamp the UI shows goes through here, so a time never appears
 * without the day it belongs to. */
export function formatMtime(mtime: number, now: Date = new Date()): string {
  const date = new Date(mtime * 1000)
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() === now.getFullYear() ? undefined : 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

/** Just the day, in the same compact form - 'Oct 26', or 'Mar 3, 2025'
 * from another year - for a list where the time of day says nothing. */
export function formatDay(mtime: number, now: Date = new Date()): string {
  const date = new Date(mtime * 1000)
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() === now.getFullYear() ? undefined : 'numeric',
  })
}

/** The same timestamp to the second, with the year - for a tooltip behind
 * the compact form, where the detail is wanted but not the width. */
export function formatExact(mtime: number): string {
  return new Date(mtime * 1000).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'medium',
  })
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

/** Minutes for a person: under one is '<1 min', anything else keeps the
 * measured value ('~3 min', '~1.5 min') rather than rounding it away. */
export const formatMinutes = (minutes: number) =>
  minutes < 1 ? '<1 min' : `~${Number(minutes.toFixed(2))} min`
