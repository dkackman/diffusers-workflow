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

/** How long ago a unix-seconds timestamp was, for a list: 'just now'
 * under a minute, '5 min ago', '3 h ago', '2 days ago' within a month, and
 * past that the day itself (formatDay), since '40 days ago' is a sum the
 * reader then has to do. A time a little ahead of `now` (a clock skew)
 * reads as just now. */
export function formatRelative(at: number, now: number): string {
  const ago = now - at
  if (ago < 60) return 'just now'
  if (ago < 3600) return `${Math.floor(ago / 60)} min ago`
  if (ago < 86_400) return `${Math.floor(ago / 3600)} h ago`
  const days = Math.floor(ago / 86_400)
  if (days < 31) return days === 1 ? '1 day ago' : `${days} days ago`
  return formatDay(at, new Date(now * 1000))
}

/** A size in GB to one decimal and no unit - the header meter and the
 * models list put the unit beside it themselves. */
export const gbFromMb = (mb: number): string => (mb / 1024).toFixed(1)
export const gbFromBytes = (bytes: number): string =>
  (bytes / 1024 ** 3).toFixed(1)

/** A span of seconds as people read it: `12.3s` under a minute, `4m 03s`
 * under an hour, `1h 02m` under a day, `1d 02h` past it. The flow view, the
 * log and the job header all show run time through this, so a step's figure
 * and the job's read the same way. */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const whole = Math.round(seconds)
  const pad = (n: number) => String(n).padStart(2, '0')
  if (whole < 3600) return `${Math.floor(whole / 60)}m ${pad(whole % 60)}s`
  if (whole < 86_400)
    return `${Math.floor(whole / 3600)}h ${pad(Math.floor((whole % 3600) / 60))}m`
  return `${Math.floor(whole / 86_400)}d ${pad(Math.floor((whole % 86_400) / 3600))}h`
}

/** An estimate for a person: under one minute is 'under 1 min', anything
 * else keeps the measured value ('about 3 min', 'about 1.5 min') rather
 * than rounding it away. */
export const formatMinutes = (minutes: number) =>
  minutes < 1
    ? 'under 1 min'
    : minutes < 60
      ? `about ${Number(minutes.toFixed(2))} min`
      : formatApprox(minutes * 60)

/** A span of seconds as a rough figure - an ETA: 'under 1 min', 'about 3
 * min' in whole minutes, 'about 1 h 20 min' from an hour up. */
export function formatApprox(seconds: number): string {
  const minutes = Math.round(seconds / 60)
  if (seconds < 60 || minutes < 1) return 'under 1 min'
  if (minutes < 60) return `about ${minutes} min`
  const h = Math.floor(minutes / 60)
  return minutes % 60 === 0
    ? `about ${h} h`
    : `about ${h} h ${minutes % 60} min`
}

/** Minutes that were measured, not estimated, so with no tilde: '38 min
 * spent' is a fact where 'about 38 min' reads as a guess. Under ten keeps one
 * decimal, past it the whole minute. */
export function formatSpent(minutes: number): string {
  if (minutes <= 0) return '0 min'
  if (minutes < 1) return 'under 1 min'
  const shown = minutes < 10 ? Number(minutes.toFixed(1)) : Math.round(minutes)
  return `${shown} min`
}
