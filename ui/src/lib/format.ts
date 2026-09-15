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
