/** A workspace's identity colour: a hue hashed from its name, so it is the
 *  same on every page and every visit without being stored. Lightness and
 *  chroma are fixed at values that read on both the paper and darkroom
 *  panels. It identifies a workspace; it is not state, so it never stands
 *  in for --live or --select. */
export function wsColor(name: string): string {
  // FNV-1a: short names that differ by one letter still land far apart
  let h = 0x811c9dc5
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  const hue = (h >>> 0) % 360
  return `oklch(0.7 0.14 ${hue})`
}
